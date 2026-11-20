from __future__ import annotations
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional
import numpy as np

class AudioPipeline:
    def __init__(self, model_size: str = "base.en", enable_mic: bool = True) -> None:
        self.enable_mic = enable_mic
        self.tts_queue: queue.Queue[str] = queue.Queue()
        self.tts_worker_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_worker_thread.start()

        self.whisper_model = None
        self._init_whisper(model_size)

        self.mic_stream = None
        self.audio_buffer: List[np.ndarray] = []
        self.is_recording = False
        self.silence_threshold = 0.015
        self.speech_callback: Optional[Callable[[str], None]] = None

        if self.enable_mic:
            self._init_microphone()

    def _init_whisper(self, model_size: str) -> None:
        try:
            from faster_whisper import WhisperModel
            # Try CPU INT8 for ultra-fast <100ms response
            self.whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        except Exception as e:
            self.whisper_model = None

    def _init_microphone(self) -> None:
        try:
            import sounddevice as sd
            def audio_callback(indata, frames, time_info, status):
                if status:
                    return
                volume = float(np.sqrt(np.mean(indata**2)))
                if volume > self.silence_threshold:
                    self.audio_buffer.append(indata.copy())
                    self.is_recording = True
                elif self.is_recording and len(self.audio_buffer) > 0:
                    # Trailing silence after speech detected
                    if len(self.audio_buffer) >= 15:  # ~0.5s of speech minimum
                        full_audio = np.concatenate(self.audio_buffer, axis=0)
                        self._process_recorded_speech(full_audio)
                    self.audio_buffer.clear()
                    self.is_recording = False

            self.mic_stream = sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype="float32",
                blocksize=1024,
                callback=audio_callback
            )
            self.mic_stream.start()
        except Exception:
            self.mic_stream = None

    def _process_recorded_speech(self, audio_data: np.ndarray) -> None:
        if self.whisper_model is None:
            return
        try:
            # Flatten to 1D float32 array
            flat = audio_data.flatten()
            segments, _ = self.whisper_model.transcribe(flat, beam_size=1, language="en")
            transcript = " ".join([s.text for s in segments]).strip()
            if len(transcript) > 1 and self.speech_callback is not None:
                self.speech_callback(transcript)
        except Exception:
            pass

    def transcribe_audio_pcm(self, pcm_data: bytes) -> str:
        if self.whisper_model is None:
            return ""
        try:
            audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = self.whisper_model.transcribe(audio_np, beam_size=1, language="en")
            return " ".join([s.text for s in segments]).strip()
        except Exception:
            return ""

    def speak_async(self, text: str) -> None:
        if not text:
            return
        self.tts_queue.put(text)

    def _tts_worker(self) -> None:
        engine = None
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 180)
            engine.setProperty("volume", 0.95)
        except Exception:
            engine = None

        while True:
            text = self.tts_queue.get()
            if engine is not None:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception:
                    pass
            self.tts_queue.task_done()

    def close(self) -> None:
        if self.mic_stream is not None:
            try:
                self.mic_stream.stop()
                self.mic_stream.close()
            except Exception:
                pass
