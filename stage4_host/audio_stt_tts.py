from __future__ import annotations
import math
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import scipy.signal

class AudioPipeline:
    def __init__(self, model_size: str = "base.en", enable_mic: bool = True) -> None:
        self.enable_mic = enable_mic
        self.tts_queue: queue.Queue[str] = queue.Queue()
        self.stt_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.speech_callback: Optional[Callable[[str], None]] = None

        self.tts_worker_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_worker_thread.start()

        self.whisper_model = None
        self._init_whisper(model_size)

        self.stt_worker_thread = threading.Thread(target=self._stt_worker, daemon=True)
        self.stt_worker_thread.start()

        self.mic_stream = None
        self.selected_dev_idx: Optional[int] = None
        self.candidate_dev_indices: List[int] = []
        self.native_samplerate = 44100
        self.channels = 1
        self.gain_boost = 15.0
        self.silence_threshold = 0.0008
        self.audio_buffer: List[np.ndarray] = []
        self.is_recording = False
        self.silence_blocks = 0
        self.max_silence_blocks = 20
        self.last_mic_level_print = 0.0
        self.zero_rms_start: Optional[float] = None
        self.fallback_in_progress = False

        if self.enable_mic:
            self._init_microphone()

    def _init_whisper(self, model_size: str) -> None:
        try:
            from faster_whisper import WhisperModel
            self.whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
            print("[AUDIO] Faster-Whisper base.en INT8 model loaded successfully.")
        except Exception as e:
            print(f"[AUDIO WARN] Faster-Whisper initialization failed: {e}")
            self.whisper_model = None

    def _init_microphone(self) -> None:
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            print("[AUDIO] Enumerating Host Audio Input Devices:")
            self.candidate_dev_indices = []
            for idx, dev in enumerate(devices):
                if dev.get("max_input_channels", 0) > 0:
                    self.candidate_dev_indices.append(idx)
                    is_def = " [DEFAULT]" if idx == sd.default.device[0] else ""
                    print(f"  [{idx:2d}] {dev.get('name', 'Unknown')}{is_def} | {dev.get('max_input_channels')} ch | {dev.get('default_samplerate')} Hz")

            default_idx = sd.default.device[0]
            if default_idx is not None and default_idx >= 0 and default_idx in self.candidate_dev_indices:
                chosen_idx = default_idx
            elif self.candidate_dev_indices:
                chosen_idx = self.candidate_dev_indices[0]
            else:
                chosen_idx = None

            if chosen_idx is None:
                print("[AUDIO WARN] No audio input devices detected.")
                return

            self.selected_dev_idx = chosen_idx
            self._start_stream_on_device(chosen_idx)
        except Exception as e:
            print(f"[AUDIO WARN] Microphone initialization failed: {e}")
            self.mic_stream = None

    def _start_stream_on_device(self, dev_idx: int) -> None:
        import sounddevice as sd
        dev_info = sd.query_devices(dev_idx)
        self.native_samplerate = int(dev_info.get("default_samplerate", 44100))
        self.channels = min(int(dev_info.get("max_input_channels", 1)), 2)
        blocksize = 1024
        self.max_silence_blocks = int(0.55 * self.native_samplerate / blocksize)

        # 1.0s Ambient Noise Floor Calibration
        try:
            calib_rec = sd.rec(int(1.0 * self.native_samplerate), samplerate=self.native_samplerate, channels=self.channels, device=dev_idx, dtype="float32")
            sd.wait()
            raw_noise_rms = float(np.sqrt(np.mean(calib_rec**2)))
        except Exception:
            raw_noise_rms = 0.0001

        # Fallback check if chosen device is completely flatline
        if raw_noise_rms < 1e-7 and len(self.candidate_dev_indices) > 1:
            for alt_idx in self.candidate_dev_indices:
                if alt_idx == dev_idx:
                    continue
                try:
                    alt_info = sd.query_devices(alt_idx)
                    alt_sr = int(alt_info.get("default_samplerate", 44100))
                    alt_ch = min(int(alt_info.get("max_input_channels", 1)), 2)
                    alt_rec = sd.rec(int(0.4 * alt_sr), samplerate=alt_sr, channels=alt_ch, device=alt_idx, dtype="float32")
                    sd.wait()
                    alt_rms = float(np.sqrt(np.mean(alt_rec**2)))
                    if alt_rms > 1e-7:
                        dev_idx = alt_idx
                        dev_info = alt_info
                        self.native_samplerate = alt_sr
                        self.channels = alt_ch
                        raw_noise_rms = alt_rms
                        break
                except Exception:
                    continue

        self.selected_dev_idx = dev_idx
        boosted_noise = raw_noise_rms * self.gain_boost
        self.silence_threshold = max(0.0006, float(boosted_noise * 1.8))

        print(f"[AUDIO] Active Mic: [{dev_idx}] {dev_info.get('name')} @ {self.native_samplerate} Hz ({self.channels} ch)")
        print(f"[AUDIO] Noise Floor: raw_rms={raw_noise_rms:.6f}, boosted={boosted_noise:.6f} | Dynamic Trigger: {self.silence_threshold:.6f} | Pre-Amp: {self.gain_boost:.1f}x")

        def audio_callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            if status:
                return
            if indata.ndim > 1 and indata.shape[1] > 1:
                mono = np.mean(indata, axis=1)
            else:
                mono = indata.flatten()

            mono_boosted = mono * self.gain_boost
            rms = float(np.sqrt(np.mean(mono_boosted**2)))
            now = time.time()

            if rms < 1e-7:
                if self.zero_rms_start is None:
                    self.zero_rms_start = now
                elif (now - self.zero_rms_start > 2.0) and not self.fallback_in_progress:
                    threading.Thread(target=self._auto_fallback_device, daemon=True).start()
            else:
                self.zero_rms_start = None

            # Live responsive VU meter
            if rms > max(0.0002, self.silence_threshold * 0.4):
                if now - self.last_mic_level_print > 0.25:
                    bar_len = min(20, max(1, int(rms * 280)))
                    meter_bar = "█" * bar_len
                    status_lbl = "SPEECH" if rms > self.silence_threshold else "VOICE"
                    print(f"\r[MIC] RMS={rms:.4f} ({status_lbl}) |{meter_bar:<20}| ", end="", flush=True)
                    self.last_mic_level_print = now

            if rms > self.silence_threshold:
                self.audio_buffer.append(mono_boosted)
                self.is_recording = True
                self.silence_blocks = 0
            elif self.is_recording:
                self.silence_blocks += 1
                self.audio_buffer.append(mono_boosted)
                if self.silence_blocks >= self.max_silence_blocks:
                    total_samples = sum(len(c) for c in self.audio_buffer)
                    min_samples = int(0.20 * self.native_samplerate)
                    if total_samples >= min_samples:
                        full_raw = np.concatenate(self.audio_buffer, axis=0)
                        np.clip(full_raw, -1.0, 1.0, out=full_raw)
                        num_16k = int(len(full_raw) * 16000 / self.native_samplerate)
                        if num_16k > 0:
                            audio_16k = scipy.signal.resample(full_raw, num_16k).astype(np.float32)
                            self.stt_queue.put(audio_16k)
                    self.audio_buffer.clear()
                    self.is_recording = False
                    self.silence_blocks = 0

        if self.mic_stream is not None:
            try:
                self.mic_stream.stop()
                self.mic_stream.close()
            except Exception:
                pass

        self.mic_stream = sd.InputStream(
            device=dev_idx,
            samplerate=self.native_samplerate,
            channels=self.channels,
            dtype="float32",
            blocksize=blocksize,
            callback=audio_callback
        )
        self.mic_stream.start()
        print("[AUDIO] Microphone input stream active and listening.")

    def _auto_fallback_device(self) -> None:
        self.fallback_in_progress = True
        try:
            import sounddevice as sd
            for alt_idx in self.candidate_dev_indices:
                if alt_idx == self.selected_dev_idx:
                    continue
                try:
                    alt_info = sd.query_devices(alt_idx)
                    alt_sr = int(alt_info.get("default_samplerate", 44100))
                    alt_ch = min(int(alt_info.get("max_input_channels", 1)), 2)
                    test_rec = sd.rec(int(0.3 * alt_sr), samplerate=alt_sr, channels=alt_ch, device=alt_idx, dtype="float32")
                    sd.wait()
                    if float(np.sqrt(np.mean(test_rec**2))) > 1e-7:
                        print(f"\n[AUDIO] Auto-fallback switching to Device [{alt_idx}] {alt_info.get('name')}")
                        self._start_stream_on_device(alt_idx)
                        break
                except Exception:
                    continue
        finally:
            self.fallback_in_progress = False

    def _stt_worker(self) -> None:
        while True:
            audio_16k = self.stt_queue.get()
            if audio_16k is None:
                self.stt_queue.task_done()
                break
            if self.whisper_model is not None:
                try:
                    flat = audio_16k.flatten()
                    segments, _ = self.whisper_model.transcribe(flat, beam_size=1, language="en", condition_on_previous_text=False)
                    transcript = " ".join([s.text for s in segments]).strip()
                    if len(transcript) >= 2:
                        print(f"\n[HEARD]: \"{transcript}\"", flush=True)
                        if self.speech_callback is not None:
                            self.speech_callback(transcript)
                except Exception as e:
                    print(f"[AUDIO WARN] Whisper transcription error: {e}")
            self.stt_queue.task_done()

    def transcribe_audio_pcm(self, pcm_data: bytes) -> str:
        if self.whisper_model is None:
            return ""
        try:
            audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = self.whisper_model.transcribe(audio_np, beam_size=1, language="en", condition_on_previous_text=False)
            return " ".join([s.text for s in segments]).strip()
        except Exception:
            return ""

    def speak_async(self, text: str) -> None:
        if not text:
            return
        self.tts_queue.put(text)

    def _tts_worker(self) -> None:
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            pass

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
            if not text:
                self.tts_queue.task_done()
                continue

            print(f"\n[SPEAKING]: {text}", flush=True)
            spoken = False
            if engine is not None:
                try:
                    engine.say(text)
                    engine.runAndWait()
                    spoken = True
                except Exception:
                    spoken = False

            if not spoken:
                try:
                    escaped = text.replace('"', '`"').replace("'", "''")
                    ps_cmd = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{escaped}')"
                    subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, timeout=12)
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
        self.stt_queue.put(None)
