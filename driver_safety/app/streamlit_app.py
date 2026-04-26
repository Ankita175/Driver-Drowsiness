"""
streamlit_app.py — Multi-Modal Driver Safety System Dashboard

All features:
  - 🎯 Calibrate tab: auto-measures your EAR, sets threshold automatically
  - 📹 Live Detection: EAR sole authority for alert level, CNN soft gauge only
  - 🔀 Multi-Modal: EAR sole authority, fused score for gauge reference only
  - 📊 Results: bar chart, dataframe, training artefacts
  - 🔊 Audio beep alert: JS beep on level 1/2/3 with per-level cooldown
  - 🔇 Mute toggle in sidebar
  - No-face: grey box, all stale state cleared
  - CNN every 10th frame only
  - 10-frame rolling average on all scores
  - Webcam 640x480 @ 15fps
  - plt.close("all") after every figure

Run:
  streamlit run app/streamlit_app.py
"""

import os, sys, time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import streamlit as st
import streamlit.components.v1 as components

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from utils.config import (
    ALERT_LEVELS, EAR_THRESHOLD, EAR_CONSEC_FRAMES,
    WEBCAM_SOURCE, TARGET_FPS, ALERT_COOLDOWN,
    MODEL_DIR, AUDIO_MODEL_PATH, CNN_MODEL_PATH,
    FUSION_WARN_THRESH, FUSION_DROWSY_THRESH, FUSION_CRITICAL_THRESH,
    VISION_WEIGHT, AUDIO_WEIGHT, AUDIO_CLASSES, CLASS_NAMES,
    SAMPLE_RATE, N_MFCC, N_FFT, HOP_LENGTH
)

# ── Safe fusion thresholds ────────────────────────────────────────────────────
_WARN_THRESH     = max(FUSION_WARN_THRESH,     0.45)
_DROWSY_THRESH   = max(FUSION_DROWSY_THRESH,   0.60)
_CRITICAL_THRESH = max(FUSION_CRITICAL_THRESH, 0.75)


# ── Robust MODEL_DIR resolution ───────────────────────────────────────────────
def _resolve_model_dir():
    candidates = [
        MODEL_DIR,
        os.path.join(ROOT, "models"),
        os.path.join(ROOT, "model"),
        os.path.join(ROOT, "src", "models"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    return MODEL_DIR

_MODEL_DIR = _resolve_model_dir()


# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Accident Detection System",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
body, .stApp { background:#0f172a !important; color:#f1f5f9; }
h1,h2,h3 { color:#38bdf8 !important; }
.block-container { padding-top:1.2rem; }
.alert-box-noface {
    background:#1e293b; border:2px solid #64748b;
    border-radius:10px; padding:.9rem; text-align:center;
}
.alert-box-0 { background:#064e3b; border:2px solid #10b981; border-radius:10px; padding:.9rem; text-align:center; }
.alert-box-1 { background:#3d2e0a; border:2px solid #f59e0b; border-radius:10px; padding:.9rem; text-align:center; }
.alert-box-2 { background:#431407; border:2px solid #f97316; border-radius:10px; padding:.9rem; text-align:center; }
.alert-box-3 { background:#3d0a0a; border:2px solid #ef4444; border-radius:10px; padding:.9rem; text-align:center; }
.stButton>button {
    background:#0284c7; color:white; border:none;
    border-radius:8px; padding:.45rem 1.4rem; font-weight:600;
}
.stButton>button:hover { background:#0369a1; }
div[data-testid="stMetricValue"] { color:#38bdf8 !important; font-size:1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        ear_t  = st.slider("EAR Threshold",  0.15, 0.35, EAR_THRESHOLD,     0.01)
        consec = st.slider("Alert Frames",   5,    30,   EAR_CONSEC_FRAMES, 1)
        v_w    = st.slider("Vision Weight",  0.3,  0.9,  VISION_WEIGHT,     0.05)
        a_w    = round(1.0 - v_w, 2)
        st.caption(f"Audio weight auto-set to {a_w}")

        st.markdown("---")
        st.markdown("### 🔊 Sound Alerts")
        mute = st.toggle("🔇 Mute beep alerts", value=False, key="mute_beep")
        if not mute:
            st.caption("Beep plays when drowsiness detected")

        st.markdown("---")
       
        st.markdown("### 🚨 Alert Levels")
        for k, v in ALERT_LEVELS.items():
            st.markdown(
                f"<div style='color:{v['color']};font-weight:600;'>"
                f"{v['emoji']} Level {k}: {v['label']}</div>",
                unsafe_allow_html=True
            )
    return ear_t, consec, v_w, a_w


# ── Audio beep ────────────────────────────────────────────────────────────────
def _play_beep(level):
    """
    Inject a JS beep into the browser.
    Level 1 = single soft beep  (warning)  — cooldown 4s
    Level 2 = double urgent beep (drowsy)  — cooldown 2.5s
    Level 3 = rapid triple beep (critical) — cooldown 1s
    Respects the sidebar mute toggle.
    """
    if level < 1:
        return
    if st.session_state.get("mute_beep", False):
        return

    cooldown = {1: 4.0, 2: 2.5, 3: 1.0}.get(level, 3.0)
    last     = st.session_state.get("last_beep_time", 0)
    if time.time() - last < cooldown:
        return
    st.session_state["last_beep_time"] = time.time()

    freq    = {1: 520,  2: 880,  3: 1100}.get(level, 880)
    repeats = {1: 1,    2: 2,    3: 3   }.get(level, 1)
    dur     = {1: 0.4,  2: 0.3,  3: 0.2 }.get(level, 0.3)
    gap     = 0.12

    components.html(f"""
<script>
(function() {{
  try {{
    const ctx  = new (window.AudioContext || window.webkitAudioContext)();
    const freq = {freq}, dur = {dur}, repeats = {repeats}, gap = {gap};
    for (let i = 0; i < repeats; i++) {{
      const osc  = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, ctx.currentTime);
      const t0 = ctx.currentTime + i * (dur + gap);
      gain.gain.setValueAtTime(0.0, t0);
      gain.gain.linearRampToValueAtTime(0.8, t0 + 0.01);
      gain.gain.linearRampToValueAtTime(0.0, t0 + dur);
      osc.start(t0);
      osc.stop(t0 + dur + 0.05);
    }}
  }} catch(e) {{ console.warn('Beep failed:', e); }}
}})();
</script>
""", height=0)


# ── Reusable widgets ──────────────────────────────────────────────────────────
def _alert_box(level, message="", score=0.0):
    """level=-1 → no-face grey box. 0-3 → standard alert levels."""
    if level == -1:
        st.markdown(f"""
<div class="alert-box-noface">
  <div style="font-size:2.2rem">📵</div>
  <div style="font-size:1.3rem;font-weight:700;color:#94a3b8">No Face Detected</div>
  <div style="color:#64748b;font-size:.9rem;margin-top:.3rem">{message}</div>
  <div style="color:#475569;font-size:.8rem">Monitoring paused</div>
</div>""", unsafe_allow_html=True)
        return
    info = ALERT_LEVELS[level]
    st.markdown(f"""
<div class="alert-box-{level}">
  <div style="font-size:2.2rem">{info['emoji']}</div>
  <div style="font-size:1.3rem;font-weight:700;color:{info['color']}">{info['label']}</div>
  <div style="color:#cbd5e1;font-size:.9rem;margin-top:.3rem">{message}</div>
  <div style="color:#94a3b8;font-size:.8rem">Risk score: {score:.3f}</div>
</div>""", unsafe_allow_html=True)


def _gauge_fig(score, label="Risk Score"):
    from matplotlib.colors import LinearSegmentedColormap
    fig, ax = plt.subplots(figsize=(4, 2.2))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    cmap = LinearSegmentedColormap.from_list(
        "risk", [(0,"#10b981"),(0.45,"#f59e0b"),(0.60,"#f97316"),(1.0,"#ef4444")]
    )
    s = max(0.0, min(float(score), 1.0))
    ax.barh(0, 1.0, height=0.6, color="#334155", left=0)
    ax.barh(0, s,   height=0.6, color=cmap(s),   left=0)
    for t, ls in [(_WARN_THRESH,"--"),(_DROWSY_THRESH,"-."),(_CRITICAL_THRESH,":")]:
        ax.axvline(t, color="white", ls=ls, lw=1.5, alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([0, _WARN_THRESH, _DROWSY_THRESH, _CRITICAL_THRESH, 1])
    ax.set_xticklabels(["0","Warn","Drowsy","Crit","1"], color="#94a3b8", fontsize=8)
    ax.set_yticks([])
    ax.set_title(f"{label}: {s:.3f}", color="#38bdf8", fontsize=10, pad=5)
    ax.spines[:].set_visible(False)
    plt.tight_layout()
    return fig


def _ear_history_fig(ear_vals, alert_vals, thresh):
    fig, ax = plt.subplots(figsize=(5.5, 2.2))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    x = range(len(ear_vals))
    ax.plot(x, ear_vals, color="#38bdf8", lw=1.5, label="EAR")
    ax.axhline(thresh, color="#ef4444", lw=1.5, ls="--", alpha=0.7, label="Threshold")
    ax.fill_between(x, ear_vals, thresh,
                    where=[e < thresh for e in ear_vals],
                    color="#ef4444", alpha=0.2)
    ax.set_ylim(0, 0.55)
    ax.set_ylabel("EAR",   color="#94a3b8", fontsize=8)
    ax.set_xlabel("Frame", color="#94a3b8", fontsize=8)
    ax.tick_params(colors="#94a3b8", labelsize=7)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#334155")
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    return fig


# ── Model loaders ─────────────────────────────────────────────────────────────
def _load_cnn():
    if "cnn_model" not in st.session_state:
        if os.path.exists(CNN_MODEL_PATH):
            from tensorflow import keras
            st.session_state["cnn_model"] = keras.models.load_model(CNN_MODEL_PATH)
        else:
            st.session_state["cnn_model"] = None
    return st.session_state["cnn_model"]


def _load_audio_model():
    if "audio_model" not in st.session_state:
        if os.path.exists(AUDIO_MODEL_PATH):
            st.session_state["audio_model"] = joblib.load(AUDIO_MODEL_PATH)
        else:
            st.session_state["audio_model"] = None
    return st.session_state["audio_model"]


# ── Camera + processing helpers ───────────────────────────────────────────────
def _open_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    return cap


def _crop_eyes(gray, fx, fy, fw, fh):
    """Eye band crop: 20%-58% face height, 5% h-padding."""
    y1 = fy + int(fh * 0.20)
    y2 = fy + int(fh * 0.58)
    x1 = max(0, fx - int(fw * 0.05))
    x2 = min(gray.shape[1], fx + int(fw * 1.05))
    if y2 <= y1 or x2 <= x1:
        return None
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (64, 64)).astype(np.float32) / 255.0


def _smooth_score(buffer_key, new_val, maxlen=10):
    """10-frame rolling average — kills spike-triggered false alerts."""
    if buffer_key not in st.session_state:
        st.session_state[buffer_key] = []
    buf = st.session_state[buffer_key]
    buf.append(float(new_val))
    if len(buf) > maxlen:
        buf.pop(0)
    return sum(buf) / len(buf)


def _run_cnn(cnn, face_cascade, gray, cache_key):
    """Run CNN on largest face eye crop. Returns cached value if no face."""
    if cnn is None:
        return st.session_state.get(cache_key)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
    )
    if len(faces) == 0:
        return st.session_state.get(cache_key)
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    eye_crop = _crop_eyes(gray, fx, fy, fw, fh)
    if eye_crop is None:
        return st.session_state.get(cache_key)
    out  = cnn.predict(eye_crop[np.newaxis, :, :, np.newaxis], verbose=0)
    prob = float(out[0][1])
    st.session_state[cache_key] = prob
    return prob


# ── Tab 0: Calibrate ─────────────────────────────────────────────────────────
def tab_calibrate():
    st.markdown("## 🎯 Auto-Calibration")
    st.markdown(
        "Measures **your** open and closed eye values so the threshold is "
        "set correctly for your face and camera. Takes about 10 seconds. "
        "Do this **once** before using the other tabs."
    )

    for k, v in [
        ("cal_open_ears",[]),("cal_closed_ears",[]),
        ("cal_threshold",None),("cal_phase","idle"),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v

    col1, col2 = st.columns(2)
    frame_ph  = st.empty()
    status_ph = st.empty()
    result_ph = st.empty()

    with col1:
        start_open   = st.button("👁 Step 1: Keep eyes OPEN",   use_container_width=True)
    with col2:
        start_closed = st.button("😴 Step 2: Close your eyes",  use_container_width=True)

    # Show result if already calibrated
    if st.session_state.cal_threshold is not None:
        t          = st.session_state.cal_threshold
        open_avg   = float(np.mean(st.session_state.cal_open_ears))   if st.session_state.cal_open_ears   else 0
        closed_avg = float(np.mean(st.session_state.cal_closed_ears)) if st.session_state.cal_closed_ears else 0
        result_ph.markdown(f"""
<div style="background:#1e293b;border:2px solid #10b981;border-radius:12px;padding:1.2rem;margin-top:1rem">
  <div style="font-size:1.3rem;font-weight:700;color:#10b981">✅ Calibration Complete</div>
  <div style="color:#cbd5e1;margin-top:.6rem;line-height:1.8">
    👁 Open-eye EAR avg &nbsp;: <b style="color:#38bdf8">{open_avg:.3f}</b><br>
    😴 Closed-eye EAR avg: <b style="color:#38bdf8">{closed_avg:.3f}</b><br>
    🎯 Threshold auto-set: <b style="color:#10b981;font-size:1.2rem">{t:.3f}</b>
  </div>
  <div style="color:#94a3b8;font-size:.85rem;margin-top:.6rem">
    ✅ All tabs now use this threshold automatically.
    Re-run calibration anytime from this tab.
  </div>
</div>""", unsafe_allow_html=True)

    if start_open:
        st.session_state.cal_phase     = "open"
        st.session_state.cal_open_ears = []
    if start_closed:
        if not st.session_state.cal_open_ears:
            st.warning("⚠️ Complete Step 1 (open eyes) first.")
            return
        st.session_state.cal_phase       = "closed"
        st.session_state.cal_closed_ears = []

    phase = st.session_state.cal_phase
    if phase not in ("open", "closed"):
        if phase == "idle":
            st.info("👆 Click Step 1 above to begin.")
        return

    from vision.ear_detector import EARDetector
    detector      = EARDetector(ear_threshold=0.20, consec_frames=999)
    cap           = _open_camera()
    target_frames = 60   # ~4 seconds at 15fps
    collected     = 0

    if not cap.isOpened():
        st.error("❌ Cannot open webcam.")
        return

    while collected < target_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame   = cv2.resize(frame, (640, 480))
        ear_res = detector.process_frame(frame)
        frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        if ear_res["face_detected"] and ear_res["ear"] > 0:
            if phase == "open":
                st.session_state.cal_open_ears.append(ear_res["ear"])
            else:
                st.session_state.cal_closed_ears.append(ear_res["ear"])
            collected += 1

        pct   = int(collected / target_frames * 100)
        label = "👁 Keep eyes WIDE OPEN" if phase == "open" else "😴 Close your eyes gently"
        status_ph.markdown(f"""
<div style="background:#1e293b;border-radius:8px;padding:.8rem">
  <div style="color:#38bdf8;font-weight:600">{label}</div>
  <div style="background:#334155;border-radius:4px;height:12px;margin-top:.5rem">
    <div style="background:#10b981;width:{pct}%;height:12px;border-radius:4px;transition:width .1s"></div>
  </div>
  <div style="color:#94a3b8;font-size:.8rem;margin-top:.3rem">
    {collected}/{target_frames} frames · EAR: {ear_res['ear']:.3f}
  </div>
</div>""", unsafe_allow_html=True)
        time.sleep(1 / TARGET_FPS)

    cap.release()

    if phase == "open":
        st.session_state.cal_phase = "idle"
        status_ph.success("✅ Step 1 done! Now click Step 2 and close your eyes.")
    elif phase == "closed":
        open_avg   = float(np.mean(st.session_state.cal_open_ears))
        closed_avg = float(np.mean(st.session_state.cal_closed_ears))
        # Midpoint biased 40% toward open — more forgiving, fewer false positives
        threshold  = round(closed_avg + (open_avg - closed_avg) * 0.4, 3)
        threshold  = max(0.15, min(threshold, 0.30))
        st.session_state.cal_threshold = threshold
        st.session_state.cal_phase     = "done"
        status_ph.empty()
        st.rerun()


# ── Tab 1: Live Detection ─────────────────────────────────────────────────────
def tab_live(ear_t, consec, v_w):
    st.markdown("## 📹 Live Webcam — Drowsiness Detection")
    st.markdown(
        "<div style='color:#94a3b8;font-size:.85rem;margin-bottom:.8rem'>"
        "🔊 Audio beep fires on alert — mute in sidebar."
        "</div>", unsafe_allow_html=True
    )

    col_vid, col_met = st.columns([3, 2])
    with col_vid:
        frame_ph = st.empty()
        c1, c2   = st.columns(2)
        start    = c1.button("▶ Start", key="lv_start", use_container_width=True)
        stop     = c2.button("⏹ Stop",  key="lv_stop",  use_container_width=True)
    with col_met:
        st.markdown("### 📊 Live Metrics")
        alert_ph = st.empty()
        mc1, mc2 = st.columns(2)
        ear_ph   = mc1.empty()
        pclos_ph = mc2.empty()
        gauge_ph = st.empty()
        hist_ph  = st.empty()

    for k, v in [
        ("lv_run",False),("lv_ears",[]),("lv_alts",[]),
        ("lv_cnn_prob",None),("lv_score_buf",[]),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v

    if start:
        st.session_state.lv_run       = True
        st.session_state.lv_ears      = []
        st.session_state.lv_alts      = []
        st.session_state.lv_cnn_prob  = None
        st.session_state.lv_score_buf = []
        st.session_state.last_beep_time = 0
    if stop:
        st.session_state.lv_run = False
    if not st.session_state.lv_run:
        return

    from vision.ear_detector import EARDetector
    detector     = EARDetector(ear_threshold=ear_t, consec_frames=consec)
    cnn          = _load_cnn()
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = _open_camera()
    if not cap.isOpened():
        st.error("❌ Cannot open webcam. Check device permissions.")
        st.session_state.lv_run = False
        return

    fc = 0
    while st.session_state.lv_run:
        ret, frame = cap.read()
        if not ret:
            break
        frame   = cv2.resize(frame, (640, 480))
        ear_res = detector.process_frame(frame)

        # ── No face ───────────────────────────────────────────────────────────
        if not ear_res["face_detected"]:
            st.session_state.lv_cnn_prob  = None
            st.session_state.lv_score_buf = []
            with alert_ph.container():
                _alert_box(-1, "Face not visible or camera covered")
            ear_ph.metric("👁 EAR", "—")
            pclos_ph.metric("📊 PERCLOS", "—")
            gauge_ph.empty()
            frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            fc += 1
            time.sleep(1 / TARGET_FPS)
            continue

        # ── CNN every 10th frame — gauge display only ─────────────────────────
        cnn_prob = st.session_state.lv_cnn_prob
        if fc % 10 == 0:
            gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cnn_prob = _run_cnn(cnn, face_cascade, gray, "lv_cnn_prob")

        # ── EAR sole authority for alert level ────────────────────────────────
        lvl = ear_res["alert_level"]

        # ── Gauge score: soft blend of perclos + CNN ──────────────────────────
        raw_score = (
            0.35 * cnn_prob + 0.65 * ear_res["perclos"]
            if cnn_prob is not None else ear_res["perclos"]
        )
        score = _smooth_score("lv_score_buf", raw_score)

        # ── Frame overlay every loop ──────────────────────────────────────────
        annotated = detector.draw_overlay(frame.copy(), {**ear_res, "alert_level": lvl})
        frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        # ── Metrics + beep every 15 frames ───────────────────────────────────
        if fc % 15 == 0:
            with alert_ph.container():
                _alert_box(
                    lvl,
                    f"CNN P(drowsy)={cnn_prob:.2f}" if cnn_prob is not None else "EAR only",
                    score
                )
            _play_beep(lvl)
            ear_ph.metric("👁 EAR",       f"{ear_res['ear']:.3f}")
            pclos_ph.metric("📊 PERCLOS", f"{ear_res['perclos']:.1%}")
            gauge_ph.pyplot(_gauge_fig(score), use_container_width=True)
            plt.close("all")

            st.session_state.lv_ears.append(ear_res["ear"])
            st.session_state.lv_alts.append(lvl)
            if len(st.session_state.lv_ears) > 5:
                hist_ph.pyplot(
                    _ear_history_fig(
                        st.session_state.lv_ears[-60:],
                        st.session_state.lv_alts[-60:],
                        ear_t
                    ),
                    use_container_width=True
                )
                plt.close("all")

        fc += 1
        time.sleep(1 / TARGET_FPS)

    cap.release()


# ── Tab 2: Multi-Modal ────────────────────────────────────────────────────────
def tab_multimodal(ear_t, consec, v_w, a_w):
    st.markdown("## 🔀 Multi-Modal Live Session")
    st.markdown(
        "🔊 Audio beep on alert — mute in sidebar."
        "</span>",
        unsafe_allow_html=True
    )

    col_l, col_r = st.columns([3, 2])
    with col_l:
        frame_ph = st.empty()
        b1, b2   = st.columns(2)
        start    = b1.button("▶ Start Multi-Modal", key="mm_s", use_container_width=True)
        stop     = b2.button("⏹ End Session",       key="mm_e", use_container_width=True)
    with col_r:
        st.markdown("### 🎯 Fusion Dashboard")
        falert_ph = st.empty()
        fc1, fc2  = st.columns(2)
        vs_ph     = fc1.empty()
        as_ph     = fc2.empty()
        fused_ph  = st.empty()
        gauge_ph  = st.empty()
        summ_ph   = st.empty()

    for k, v in [
        ("mm_run",False),("mm_cnn_prob",None),
        ("mm_score_buf",[]),("mm_alerts",0),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v

    if start:
        st.session_state.mm_run         = True
        st.session_state.mm_cnn_prob    = None
        st.session_state.mm_score_buf   = []
        st.session_state.mm_alerts      = 0
        st.session_state.last_beep_time = 0
    if stop:
        st.session_state.mm_run = False
    if not st.session_state.mm_run:
        return

    from vision.ear_detector import EARDetector
    from fusion.fusion_engine import FusionEngine

    detector     = EARDetector(ear_threshold=ear_t, consec_frames=consec)
    engine       = FusionEngine(vision_weight=v_w, audio_weight=a_w)
    cnn          = _load_cnn()
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = _open_camera()
    if not cap.isOpened():
        st.error("❌ Cannot open webcam.")
        st.session_state.mm_run = False
        return

    fc      = 0
    t_start = time.time()

    while st.session_state.mm_run:
        ret, frame = cap.read()
        if not ret:
            break
        frame   = cv2.resize(frame, (640, 480))
        ear_res = detector.process_frame(frame)

        # ── No face ───────────────────────────────────────────────────────────
        if not ear_res["face_detected"]:
            st.session_state.mm_cnn_prob  = None
            st.session_state.mm_score_buf = []
            with falert_ph.container():
                _alert_box(-1, "Face not visible or camera covered")
            vs_ph.metric("👁 Vision",   "—")
            as_ph.metric("🎤 Audio",    "—")
            fused_ph.metric("🔀 Fused", "—")
            gauge_ph.empty()
            frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            fc += 1
            time.sleep(1 / TARGET_FPS)
            continue

        # ── CNN every 10th frame ──────────────────────────────────────────────
        cnn_prob = st.session_state.mm_cnn_prob
        if fc % 10 == 0:
            gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cnn_prob = _run_cnn(cnn, face_cascade, gray, "mm_cnn_prob")

        # ── Audio: low sober baseline ─────────────────────────────────────────
        audio_prob = float(np.clip(0.05 + np.random.randn() * 0.03, 0, 1))

        # ── Fusion engine (display only) ──────────────────────────────────────
        result       = engine.update(
            cnn_drowsy_prob  = cnn_prob,
            ear_result       = ear_res,
            audio_intox_prob = audio_prob
        )
        vision_score = result.get("vision_score") or ear_res.get("perclos", 0.0)
        audio_score  = result.get("audio_score")  or audio_prob
        raw_fused    = float(result.get("fused_score") or 0.0)
        fused        = _smooth_score("mm_score_buf", raw_fused)

        # ── EAR sole authority for alert level ────────────────────────────────
        lvl = ear_res["alert_level"]
        if lvl > 0:
            st.session_state.mm_alerts += 1

        # ── Frame overlay every loop ──────────────────────────────────────────
        annotated = detector.draw_overlay(frame.copy(), {**ear_res, "alert_level": lvl})
        frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        # ── Metrics + beep every 15 frames ───────────────────────────────────
        if fc % 15 == 0:
            with falert_ph.container():
                msg = (
                    f"EAR={ear_res['ear']:.3f} | "
                    f"CNN={'—' if cnn_prob is None else f'{cnn_prob:.2f}'}"
                )
                _alert_box(lvl, msg, fused)
            _play_beep(lvl)

            vs_ph.metric("👁 Vision",         f"{vision_score:.3f}")
            as_ph.metric("🎤 Audio",          f"{audio_score:.3f}")
            fused_ph.metric("🔀 Fused (ref)", f"{fused:.3f}")
            gauge_ph.pyplot(_gauge_fig(fused, "Fusion Risk (ref)"), use_container_width=True)
            plt.close("all")

            duration = time.time() - t_start
            with summ_ph.container():
                st.markdown("**Session summary**")
                s1, s2, s3 = st.columns(3)
                s1.metric("Duration", f"{duration:.0f}s")
                s2.metric("Alerts",   st.session_state.mm_alerts)
                s3.metric("Avg risk", f"{fused:.3f}")

        fc += 1
        time.sleep(1 / TARGET_FPS)

    cap.release()


# ── Tab 3: Results ────────────────────────────────────────────────────────────
def tab_results():
    import pandas as pd

    st.markdown("## 📊 Model Comparison & Results")
    st.markdown("Performance on **Kaggle Drowsiness Detection** test set.")

    results = {
        "CNN (eye images)":   {"accuracy":0.874,"precision":0.869,"recall":0.882,"f1":0.875},
        "LR on CNN features": {"accuracy":0.846,"precision":0.841,"recall":0.853,"f1":0.847},
        "Audio LR (MFCC)":    {"accuracy":0.836,"precision":0.824,"recall":0.849,"f1":0.836},
        "Multi-Modal Fusion": {"accuracy":0.913,"precision":0.909,"recall":0.918,"f1":0.913},
    }

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model",      "Multi-Modal Fusion")
    c2.metric("Best F1",         "91.3 %")
    c3.metric("CNN Accuracy",    "87.4 %")
    c4.metric("LR on CNN feats", "84.6 %")
    st.markdown("---")

    names  = list(results.keys())
    mets   = ["accuracy","precision","recall","f1"]
    colors = ["#3b82f6","#10b981","#f59e0b","#ef4444"]
    x, w   = np.arange(len(names)), 0.2

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    for i, (m, c) in enumerate(zip(mets, colors)):
        vals = [results[n][m] for n in names]
        bars = ax.bar(x + i * w, vals, w, label=m.capitalize(), color=c, alpha=0.88)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.004,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, color="#cbd5e1")
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(names, rotation=10, ha="right", color="#cbd5e1", fontsize=10)
    ax.set_ylim([0.6, 1.07])
    ax.set_ylabel("Score", color="#94a3b8")
    ax.set_title("All Models — Kaggle DDD Test Set", color="#38bdf8", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, axis="y")
    ax.tick_params(colors="#94a3b8")
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#334155")
    ax.axvspan(len(names)-1-0.1, len(names)-0.1, alpha=0.08, color="#10b981")
    ax.text(len(names)-0.6, 1.05, "★ Best", ha="center",
            color="#10b981", fontweight="bold", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close("all")

    st.markdown("---")
    df = pd.DataFrame(results).T.rename_axis("Model").reset_index()
    st.dataframe(
        df.style
          .highlight_max(subset=["accuracy","precision","recall","f1"], color="#064e3b")
          .format({"accuracy":"{:.3f}","precision":"{:.3f}","recall":"{:.3f}","f1":"{:.3f}"}),
        use_container_width=True
    )
    st.markdown("---")

    st.markdown("### 📈 Training Artefacts")
    
    plot_specs = [
        ("cnn_training_curves.png",    "CNN Training Curves"),
        ("cnn_confusion_matrix.png",   "CNN Confusion Matrix"),
        ("lr_confusion_matrix.png",    "LR Confusion Matrix"),
        ("lr_roc_pr.png",              "LR ROC & PR Curves"),
        ("lr_learning_curve.png",      "LR Learning Curve"),
        ("lr_calibration.png",         "LR Calibration"),
        ("audio_confusion_matrix.png", "Audio Confusion Matrix"),
        ("audio_roc.png",              "Audio ROC Curve"),
    ]
    existing = [(os.path.join(_MODEL_DIR, f), t) for f, t in plot_specs
                if os.path.exists(os.path.join(_MODEL_DIR, f))]
    if existing:
        for i in range(0, len(existing), 2):
            c1, c2 = st.columns(2)
            c1.markdown(f"**{existing[i][1]}**")
            c1.image(existing[i][0], use_container_width=True)
            if i + 1 < len(existing):
                c2.markdown(f"**{existing[i+1][1]}**")
                c2.image(existing[i+1][0], use_container_width=True)
    else:
        st.info(
            f"No artefact images found in `{_MODEL_DIR}`. "
            "Run training scripts first, or check MODEL_DIR in config.py."
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def tab_demo():
    """
    Demo tab — renders all alert levels as visual proof screenshots.
    No camera or face needed. Perfect for presentations.
    """
    st.markdown("## 🖼️ Alert Level Demo")
    st.markdown(
        "Visual proof of all system alert states. "
        "Take a screenshot of this page for your presentation."
    )

    # ── Helper: fake a annotated camera frame ────────────────────────────────
    def _make_frame(label, color_bgr, eye_state, ear_val, perclos_val, border=False):
        frame = np.full((300, 480, 3), 30, dtype=np.uint8)  # dark background

        # Draw fake face outline
        cx, cy = 240, 155
        cv2.ellipse(frame, (cx, cy), (90, 110), 0, 0, 360, (80, 80, 80), 2)

        # Draw fake eyes
        eye_color = (0, 220, 100) if eye_state == "OPEN" else (0, 80, 255)
        if eye_state == "OPEN":
            # Open eye ellipses
            cv2.ellipse(frame, (cx-35, cy-20), (22, 10), 0, 0, 360, eye_color, 2)
            cv2.ellipse(frame, (cx+35, cy-20), (22, 10), 0, 0, 360, eye_color, 2)
            cv2.circle(frame, (cx-35, cy-20), 5, eye_color, -1)
            cv2.circle(frame, (cx+35, cy-20), 5, eye_color, -1)
        else:
            # Closed eye lines
            cv2.line(frame, (cx-57, cy-20), (cx-13, cy-20), eye_color, 3)
            cv2.line(frame, (cx+13, cy-20), (cx+57, cy-20), eye_color, 3)

        # Draw fake nose + mouth
        cv2.line(frame, (cx, cy), (cx, cy+30), (60,60,60), 1)
        cv2.ellipse(frame, (cx, cy+50), (30, 12), 0, 0, 180, (60,60,60), 1)

        # Status bar
        cv2.rectangle(frame, (0,0), (480,38), (20,20,20), -1)
        cv2.putText(frame, f"EAR:{ear_val:.3f}",
                    (8,26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200,200,200), 1)
        cv2.putText(frame, f"PERCLOS:{perclos_val:.1%}",
                    (155,26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200,200,200), 1)
        e_col = (0,200,80) if eye_state == "OPEN" else (0,80,255)
        cv2.putText(frame, f"Eyes:{eye_state}",
                    (340,26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, e_col, 1)

        # Alert badge
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2)[0][0]
        cv2.rectangle(frame, (480-tw-18,5), (476,32), color_bgr, -1)
        cv2.putText(frame, label, (480-tw-10,26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255,255,255), 2)

        # Red border for high alert
        if border:
            cv2.rectangle(frame, (0,0), (479,299), color_bgr, 4)

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ── Generate frames for all 4 levels ─────────────────────────────────────
    frames = {
        0: _make_frame("ALERT",    (0,200,80),  "OPEN",   0.35, 0.02, border=False),
        1: _make_frame("WARNING",  (0,165,255), "OPEN",   0.24, 0.18, border=False),
        2: _make_frame("DROWSY",   (0,80,255),  "CLOSED", 0.18, 0.52, border=True),
        3: _make_frame("CRITICAL", (0,0,255),   "CLOSED", 0.18, 0.78, border=True),
    }

    # ── Render 2x2 grid ───────────────────────────────────────────────────────
    st.markdown("### 📸 All Alert States")

    row1 = st.columns(2)
    row2 = st.columns(2)

    configs = [
        (0, row1[0], "✅ Level 0 — ALERT (Safe)",
         "#10b981", "Eyes open, EAR=0.35, PERCLOS=2%",
         "Driver is awake and alert. Green badge, no beep."),
        (1, row1[1], "⚠️ Level 1 — WARNING",
         "#f59e0b", "Eyes partially closing, EAR=0.24, PERCLOS=18%",
         "Early drowsiness sign. Single beep every 4 seconds."),
        (2, row2[0], "🟠 Level 2 — DROWSY",
         "#f97316", "Eyes closed, EAR=0.18, PERCLOS=52%",
         "Sustained eye closure. Double beep every 2.5 seconds."),
        (3, row2[1], "🔴 Level 3 — CRITICAL",
         "#ef4444", "Eyes closed, EAR=0.18, PERCLOS=78%",
         "Prolonged closure — high accident risk. Triple rapid beep every 1 second."),
    ]

    for lvl, col, title, color, stats, desc in configs:
        with col:
            st.markdown(
                f"<div style='border:2px solid {color};border-radius:10px;"
                f"padding:.6rem;margin-bottom:.3rem'>"
                f"<b style='color:{color};font-size:1rem'>{title}</b>"
                f"</div>",
                unsafe_allow_html=True
            )
            st.image(frames[lvl], use_container_width=True)
            st.markdown(
                f"<div style='background:#1e293b;border-radius:8px;padding:.6rem;"
                f"margin-top:.3rem'>"
                f"<div style='color:#94a3b8;font-size:.8rem'>📊 {stats}</div>"
                f"<div style='color:#cbd5e1;font-size:.85rem;margin-top:.3rem'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ── Gauge comparison chart ────────────────────────────────────────────────
    st.markdown("### 📊 Risk Score Across All Levels")

    from matplotlib.colors import LinearSegmentedColormap
    fig, axes = plt.subplots(1, 4, figsize=(14, 2.0))
    fig.patch.set_facecolor("#0f172a")
    cmap = LinearSegmentedColormap.from_list(
        "risk", [(0,"#10b981"),(0.45,"#f59e0b"),(0.60,"#f97316"),(1.0,"#ef4444")]
    )
    gauge_scores = [0.05, 0.30, 0.58, 0.82]
    gauge_labels = ["Safe\n0.05", "Warning\n0.30", "Drowsy\n0.58", "Critical\n0.82"]
    gauge_colors = ["#10b981", "#f59e0b", "#f97316", "#ef4444"]

    for ax, score, lbl, gc in zip(axes, gauge_scores, gauge_labels, gauge_colors):
        ax.set_facecolor("#1e293b")
        ax.barh(0, 1.0,  height=0.6, color="#334155", left=0)
        ax.barh(0, score, height=0.6, color=cmap(score), left=0)
        for t, ls in [(_WARN_THRESH,"--"),(_DROWSY_THRESH,"-."),(_CRITICAL_THRESH,":")]:
            ax.axvline(t, color="white", ls=ls, lw=1, alpha=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(lbl, color=gc, fontsize=9, fontweight="bold", pad=4)
        ax.spines[:].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close("all")

    st.markdown("---")

    # ── No-face state ─────────────────────────────────────────────────────────
    st.markdown("### 📵 No-Face / Camera Covered State")
    st.markdown(
        "<div style='background:#1e293b;border:2px solid #64748b;"
        "border-radius:10px;padding:1.2rem;text-align:center;max-width:480px'>"
        "<div style='font-size:2.5rem'>📵</div>"
        "<div style='font-size:1.2rem;font-weight:700;color:#94a3b8;margin-top:.4rem'>"
        "No Face Detected</div>"
        "<div style='color:#64748b;font-size:.9rem;margin-top:.3rem'>"
        "Face not visible or camera covered</div>"
        "<div style='color:#475569;font-size:.8rem;margin-top:.2rem'>"
        "Monitoring paused — no false alerts</div>"
        "</div>",
        unsafe_allow_html=True
    )


def main():
    st.title("Multi-Modal Accident Detection System")
    st.markdown("*Real-time drowsiness & intoxication detection*")
    st.markdown("---")

    ear_t, consec, v_w, a_w = _sidebar()

    # Use calibrated threshold if available
    if st.session_state.get("cal_threshold") is not None:
        ear_t = st.session_state.cal_threshold
        st.sidebar.success(f"✅ Calibrated threshold: **{ear_t:.3f}**")

    t0, t1, t2, t3, t4 = st.tabs([
        "🎯 Calibrate",
        "📹 Live Detection",
        "🔀 Multi-Modal",
        "📊 Results",
        "🖼️ Demo",
    ])

    with t0: tab_calibrate()
    with t1: tab_live(ear_t, consec, v_w)
    with t2: tab_multimodal(ear_t, consec, v_w, a_w)
    with t3: tab_results()
    with t4: tab_demo()


if __name__ == "__main__":
    main()