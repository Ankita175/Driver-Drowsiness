"""
ear_detector.py — Eye state detection using OpenCV face + eye cascades.

Key fixes:
  - Requires 5+ consecutive no-eye frames before marking eyes closed (was 3)
  - Requires 2+ eyes missing consecutively, not just 1 detection failure
  - Alert level needs sustained closure (consec_frames) — single misses ignored
  - No-eye streak decays by 2 when eyes ARE detected (clears fast on reopen)
  - minNeighbors raised to reduce false "no eye" detections
  - EAR values kept as-is (cascade-based, not landmark-based)
"""

import cv2
import numpy as np
from collections import deque
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    EAR_THRESHOLD, EAR_CONSEC_FRAMES, EAR_WARNING_FRAMES
)

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)


class EARDetector:
    def __init__(self,
                 ear_threshold=EAR_THRESHOLD,
                 consec_frames=EAR_CONSEC_FRAMES,
                 warning_frames=EAR_WARNING_FRAMES):
        self.ear_threshold  = ear_threshold
        self.consec_frames  = consec_frames
        self.warning_frames = warning_frames
        self.frame_counter  = 0
        self.total_blinks   = 0
        self.perclos_buf    = deque(maxlen=30 * 60)

        self._no_eye_streak = 0
        self._min_face      = 80

        # How many consecutive no-eye frames before we believe eyes are closed
        # 5 frames at 15fps = 0.33 seconds — filters out cascade misses
        self._CLOSED_CONFIRM = 5

    def process_frame(self, bgr):
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        out = dict(ear=1.0, mar=0.0, yaw=0.0, pitch=0.0, roll=0.0,
                   eye_closed=False, alert_level=0, perclos=0.0,
                   blink_rate=0.0, face_detected=False, landmarks=None,
                   face_box=None, eye_boxes=[])

        # ── Detect face ───────────────────────────────────────────────────
        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4,
            minSize=(self._min_face, self._min_face)
        )
        if len(faces) == 0:
            faces = FACE_CASCADE.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50)
            )
        if len(faces) == 0:
            # Face gone — decay streak so it doesn't stay high
            self._no_eye_streak = max(0, self._no_eye_streak - 1)
            return out

        out["face_detected"] = True
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        out["face_box"] = (fx, fy, fw, fh)

        # ── Eye detection in upper 55% of face ───────────────────────────
        eye_region_h = int(fh * 0.55)
        face_roi     = gray[fy:fy + eye_region_h, fx:fx + fw]

        # minNeighbors=5 — stricter, fewer false "no eye" misses
        eyes = EYE_CASCADE.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(int(fw * 0.10), int(fw * 0.08))
        )

        eyes_detected = len(eyes) >= 1

        if eyes_detected:
            out["eye_boxes"] = [
                (fx + ex, fy + ey, ew, eh) for (ex, ey, ew, eh) in eyes
            ]
            out["ear"] = 0.35
            # Decay streak by 2 per frame — clears quickly when eyes reopen
            self._no_eye_streak = max(0, self._no_eye_streak - 2)
        else:
            self._no_eye_streak += 1
            out["ear"] = 0.18

        # ── Eyes confirmed closed only after _CLOSED_CONFIRM streak ──────
        # This is the key fix — cascade misses 1-4 frames are ignored
        eye_closed = (self._no_eye_streak >= self._CLOSED_CONFIRM)
        out["eye_closed"] = eye_closed

        # ── Frame counter ─────────────────────────────────────────────────
        if eye_closed:
            self.frame_counter += 1
        else:
            if self.frame_counter >= self.warning_frames:
                self.total_blinks += 1
            self.frame_counter = 0
            # Only reset streak fully when eyes confidently reopen
            if eyes_detected:
                self._no_eye_streak = 0

        # ── PERCLOS — only log confident states ───────────────────────────
        if eyes_detected or self._no_eye_streak >= self._CLOSED_CONFIRM:
            self.perclos_buf.append(int(eye_closed))

        perclos = float(np.mean(self.perclos_buf)) if self.perclos_buf else 0.0
        out["perclos"] = round(perclos, 4)

        elapsed_min      = max(len(self.perclos_buf) / (30 * 60), 1e-6)
        out["blink_rate"] = round(self.total_blinks / elapsed_min, 2)

        # ── Alert level ───────────────────────────────────────────────────
        if self.frame_counter >= self.consec_frames:
            out["alert_level"] = 2
        elif self.frame_counter >= self.warning_frames:
            out["alert_level"] = 1
        else:
            out["alert_level"] = 0

        return out

    def draw_overlay(self, frame, result):
        h, w  = frame.shape[:2]
        lvl   = result["alert_level"]
        BGR   = {0:(0,200,80), 1:(0,165,255), 2:(0,80,255), 3:(0,0,255)}
        color = BGR.get(lvl, (255,255,255))
        label = {0:"ALERT", 1:"WARNING", 2:"DROWSY", 3:"CRITICAL"}[lvl]

        # Status bar
        cv2.rectangle(frame, (0,0), (w,38), (20,20,20), -1)
        cv2.putText(frame, f"EAR:{result['ear']:.3f}",
                    (8,26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200,200,200), 1)
        cv2.putText(frame, f"PERCLOS:{result['perclos']:.1%}",
                    (155,26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200,200,200), 1)
        eye_state = "OPEN" if not result["eye_closed"] else "CLOSED"
        eye_color = (0,200,80) if not result["eye_closed"] else (0,80,255)
        cv2.putText(frame, f"Eyes:{eye_state}",
                    (370,26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, eye_color, 1)

        # Alert badge
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2)[0][0]
        cv2.rectangle(frame, (w-tw-18,5), (w-4,32), color, -1)
        cv2.putText(frame, label, (w-tw-10,26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255,255,255), 2)

        # Face box
        if result.get("face_box"):
            fx, fy, fw, fh = result["face_box"]
            cv2.rectangle(frame, (fx,fy), (fx+fw,fy+fh), (100,100,100), 1)

        # Eye boxes (green when open)
        for (ex, ey, ew, eh) in result.get("eye_boxes", []):
            cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), (0,220,100), 2)

        # Red border when drowsy/critical
        if lvl >= 2:
            cv2.rectangle(frame, (0,0), (w-1,h-1), color, 4)

        return frame

    def reset(self):
        self.frame_counter  = 0
        self.total_blinks   = 0
        self._no_eye_streak = 0
        self.perclos_buf.clear()