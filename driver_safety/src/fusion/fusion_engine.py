"""
fusion_engine.py — Weighted Multi-Modal Fusion Engine

Combines:
  • Vision score  : CNN drowsy-prob  (0→alert, 1→drowsy)
                   + EAR/PERCLOS  signals
  • Audio score   : LR intox-prob   (0→sober, 1→intoxicated)

Fusion formula (graceful-degradation):
  if both available : fused = V_weight × V_score + A_weight × A_score
  if vision only    : fused = V_score
  if audio only     : fused = A_score

Temporal smoothing (rolling mean over last 10 frames) reduces jitter.
Critical escalation: if both modalities are >60% confidence → level 3.
"""

import os, sys, time
import numpy as np
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    VISION_WEIGHT, AUDIO_WEIGHT,
    FUSION_WARN_THRESH, FUSION_DROWSY_THRESH, FUSION_CRITICAL_THRESH,
    ALERT_LEVELS, EAR_THRESHOLD, ALERT_COOLDOWN
)


class FusionEngine:
    """
    Usage
    -----
    engine = FusionEngine()
    result = engine.update(
        cnn_drowsy_prob = 0.72,   # from CNN.predict()
        ear_result      = {...},   # from EARDetector.process_frame()
        audio_intox_prob= 0.45    # from predict_intoxication()  [optional]
    )
    print(result["alert_level"], result["message"])
    """

    def __init__(self, vision_weight=VISION_WEIGHT,
                 audio_weight=AUDIO_WEIGHT, smooth_window=10):
        self.vw = vision_weight
        self.aw = audio_weight
        self._v_buf = deque(maxlen=smooth_window)
        self._a_buf = deque(maxlen=smooth_window)
        self._f_buf = deque(maxlen=smooth_window)
        self._level  = 0
        self._alerts = 0
        self._last_t = 0.0
        self._consec_critical = 0
        self.history = []

    # ── Score helpers ─────────────────────────────────────────────────────

    def _vision_score(self, cnn_prob, ear):
        if cnn_prob is None and ear is None:
            return None
        parts = []
        if cnn_prob is not None:
            parts.append(float(cnn_prob) * 0.50)
        if ear is not None:
            parts.append(float(ear.get("perclos", 0.0)) * 0.30)
            raw_ear = float(ear.get("ear", 1.0))
            ear_sc  = max(0.0, (EAR_THRESHOLD * 1.5 - raw_ear) / (EAR_THRESHOLD * 1.5))
            parts.append(ear_sc * 0.10)
            pitch = abs(float(ear.get("pitch", 0.0)))
            parts.append(min(1.0, pitch / 30.0) * 0.10)
        return float(np.clip(sum(parts), 0.0, 1.0))

    @staticmethod
    def _audio_score(intox_prob):
        if intox_prob is None:
            return None
        return float(np.clip(intox_prob, 0.0, 1.0))

    def _fuse(self, vs, as_):
        if vs is not None and as_ is not None:
            return float(np.clip(self.vw * vs + self.aw * as_, 0.0, 1.0))
        return float(vs if vs is not None else (as_ if as_ is not None else 0.0))

    @staticmethod
    def _smooth(buf, val):
        buf.append(val)
        return float(np.mean(buf))

    def _level_from_score(self, fused, vs, as_):
        if fused >= FUSION_CRITICAL_THRESH:
            lvl = 3
        elif fused >= FUSION_DROWSY_THRESH:
            lvl = 2
        elif fused >= FUSION_WARN_THRESH:
            lvl = 1
        else:
            lvl = 0
        # Both-modality escalation
        if vs is not None and as_ is not None:
            if vs > 0.60 and as_ > 0.60:
                lvl = max(lvl, 3)
            elif vs > 0.45 and as_ > 0.45:
                lvl = max(lvl, 2)
        return lvl

    # ── Public API ────────────────────────────────────────────────────────

    def update(self, cnn_drowsy_prob=None, ear_result=None, audio_intox_prob=None):
        """
        Process new sensor readings and return fusion decision.

        Parameters
        ----------
        cnn_drowsy_prob   : float [0,1]  P(drowsy) from CNN
        ear_result        : dict  from EARDetector.process_frame()
        audio_intox_prob  : float [0,1]  P(intoxicated) from audio LR

        Returns
        -------
        dict with keys:
            alert_level, alert_info, fused_score,
            vision_score, audio_score, message,
            should_alert, modalities, consecutive_critical, total_alerts
        """
        vs_raw = self._vision_score(cnn_drowsy_prob, ear_result)
        as_raw = self._audio_score(audio_intox_prob)

        vs = (self._smooth(self._v_buf, vs_raw)
              if vs_raw is not None
              else (float(np.mean(self._v_buf)) if self._v_buf else None))
        as_ = (self._smooth(self._a_buf, as_raw)
               if as_raw is not None
               else (float(np.mean(self._a_buf)) if self._a_buf else None))

        fused_raw    = self._fuse(vs_raw, as_raw)
        fused_smooth = self._smooth(self._f_buf, fused_raw)

        level = self._level_from_score(fused_smooth, vs, as_)
        self._consec_critical = (self._consec_critical + 1
                                  if level >= 3 else 0)
        self._level = level

        now = time.time()
        should_alert = (level >= 1) and (now - self._last_t > ALERT_COOLDOWN)
        if should_alert:
            self._last_t = now
            self._alerts += 1

        info = ALERT_LEVELS[level]
        result = dict(
            alert_level         = level,
            alert_info          = info,
            fused_score         = round(fused_smooth, 4),
            vision_score        = round(vs,  4) if vs  is not None else None,
            audio_score         = round(as_, 4) if as_ is not None else None,
            message             = self._msg(level, vs, as_),
            should_alert        = should_alert,
            modalities          = dict(vision_active=vs_raw is not None,
                                       audio_active =as_raw is not None),
            consecutive_critical= self._consec_critical,
            total_alerts        = self._alerts,
        )
        if len(self.history) < 2000:
            self.history.append(dict(t=now, level=level, fused=fused_smooth,
                                     vision=vs, audio=as_))
        return result

    def _msg(self, lvl, vs, as_):
        if lvl == 0:
            return "Driver is alert."
        if lvl == 1:
            parts = (["vision: eye activity"] if vs and vs > 0.30 else []) + \
                    (["audio: speech irregularity"] if as_ and as_ > 0.30 else [])
            return "Warning: " + (", ".join(parts) or "mild indicator") + "."
        if lvl == 2:
            return ("DROWSY — prolonged eye closure detected. " +
                    ("Speech impairment detected. " if as_ and as_ > 0.50 else "") +
                    "Please take a break!")
        return "CRITICAL — Immediate danger! Pull over NOW!"

    def session_summary(self):
        if not self.history:
            return {}
        dur    = self.history[-1]["t"] - self.history[0]["t"]
        levels = [h["level"] for h in self.history]
        n      = len(levels)
        return dict(
            duration_s       = round(dur, 1),
            total_frames     = n,
            total_alerts     = self._alerts,
            pct_alert        = round(sum(l==0 for l in levels)/n*100, 1),
            pct_warning      = round(sum(l==1 for l in levels)/n*100, 1),
            pct_drowsy_plus  = round(sum(l>=2 for l in levels)/n*100, 1),
            avg_fused        = round(np.mean([h["fused"] for h in self.history]), 4),
            max_fused        = round(max (h["fused"] for h in self.history), 4),
        )

    def reset(self):
        self._v_buf.clear(); self._a_buf.clear(); self._f_buf.clear()
        self._level=0; self._alerts=0; self._last_t=0.0
        self._consec_critical=0; self.history.clear()


if __name__ == "__main__":
    eng = FusionEngine()
    for i in range(25):
        cnn_p = min(0.9, i * 0.04)
        ear_r = dict(perclos=min(0.45, i*0.018), ear=max(0.16, 0.40-i*0.01),
                     pitch=i*0.7)
        r = eng.update(cnn_drowsy_prob=cnn_p, ear_result=ear_r,
                       audio_intox_prob=0.3 + (i>15)*0.3)
        print(f"f{i+1:02d} fused={r['fused_score']:.3f}  "
              f"lvl={r['alert_level']} {r['alert_info']['emoji']}  "
              f"{r['message'][:55]}")
    print("\n", eng.session_summary())
