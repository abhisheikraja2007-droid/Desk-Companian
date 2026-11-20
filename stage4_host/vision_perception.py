from __future__ import annotations
import math
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

class VisionPerception:
    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self._init_camera()

        self.last_face_time = 0.0
        self.face_detected = False
        self.smile_detected = False
        self.gaze_offset_x = 0.0
        self.gaze_offset_y = 0.0
        self.identified_person = "Owner"
        self.consecutive_unknown_faces = 0

        # Try initializing CascadeClassifier if available
        self.face_cascade = None
        self.smile_cascade = None
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            try:
                face_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                smile_xml = cv2.data.haarcascades + "haarcascade_smile.xml"
                if os.path.exists(face_xml):
                    self.face_cascade = cv2.CascadeClassifier(face_xml)
                if os.path.exists(smile_xml):
                    self.smile_cascade = cv2.CascadeClassifier(smile_xml)
            except Exception:
                self.face_cascade = None
                self.smile_cascade = None

    def _init_camera(self) -> None:
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.cap = None
        except Exception:
            self.cap = None

    def read_frame(self) -> Tuple[bool, np.ndarray]:
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return True, frame

        # Procedural fallback synthetic frame if physical webcam is occupied or not present
        synth = np.zeros((480, 640, 3), dtype=np.uint8)
        synth[:] = (23, 15, 11)  # #0B0F17
        # Draw simulated user avatar
        cv2.circle(synth, (320, 240), 90, (166, 184, 20), -1)
        cv2.ellipse(synth, (320, 390), (140, 100), 0, 0, 180, (212, 182, 6), -1)
        cv2.putText(synth, "LAPTOP HIL SYNTHETIC PERCEPTION CAM", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, (234, 179, 8), 1)
        cv2.putText(synth, f"TIME: {time.strftime('%H:%M:%S')}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return True, synth

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = []
        if self.face_cascade is not None:
            try:
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.15,
                    minNeighbors=5,
                    minSize=(60, 60)
                )
            except Exception:
                faces = []

        if len(faces) == 0:
            # Color-based face candidate segmentation fallback
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([25, 255, 255]))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 4000:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    aspect = bh / float(bw)
                    if 0.8 <= aspect <= 1.8:
                        faces.append((x, y, bw, bh))
                        break

        self.face_detected = len(faces) > 0
        self.smile_detected = False

        if self.face_detected:
            self.last_face_time = time.time()
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            fx, fy, fw, fh = faces[0]

            face_cx = fx + fw / 2.0
            face_cy = fy + fh / 2.0
            self.gaze_offset_x = (face_cx - (w / 2.0)) / (w / 2.0)
            self.gaze_offset_y = (face_cy - (h / 2.0)) / (h / 2.0)

            # Detect smile
            if self.smile_cascade is not None:
                try:
                    roi_gray = gray[fy + int(fh * 0.45): fy + fh, fx: fx + fw]
                    smiles = self.smile_cascade.detectMultiScale(
                        roi_gray,
                        scaleFactor=1.4,
                        minNeighbors=12,
                        minSize=(25, 25)
                    )
                    self.smile_detected = len(smiles) > 0
                except Exception:
                    self.smile_detected = False
            else:
                self.smile_detected = False

            # Check for multiple faces or unfamiliar visitor entering field
            if len(faces) > 1:
                self.identified_person = "Guest"
                self.consecutive_unknown_faces += 1
            else:
                self.identified_person = "Owner"
                self.consecutive_unknown_faces = 0

            # Draw visual tracking HUD
            color = (8, 179, 234) if self.smile_detected else (212, 182, 6)
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color, 2)
            label = f"{self.identified_person} | {'Smiling :)' if self.smile_detected else 'Neutral'}"
            cv2.putText(frame, label, (fx, max(20, fy - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.55, color, 1)

        return {
            "face_detected": self.face_detected,
            "smile_detected": self.smile_detected,
            "person_id": self.identified_person,
            "gaze_offset_x": round(self.gaze_offset_x, 3),
            "gaze_offset_y": round(self.gaze_offset_y, 3),
            "annotated_frame": frame
        }

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
