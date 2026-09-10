import os
import tempfile

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from feature_utils import FallDetector

st.set_page_config(page_title="SafeFall AI", page_icon="🚨", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("safe_fall_model.pkl")


model_bundle = load_model()

st.title("🚨 SafeFall AI — Elderly Fall Detection Dashboard")
st.caption(
    "CareVision HealthTech — pose-based monitoring for elderly fall detection "
    "and emergency alerting. Upload a video or image to run the model."
)

with st.sidebar:
    st.header("About this model")
    st.metric("Reported accuracy", f"{model_bundle.get('accuracy', 0):.1%}")
    st.write("Classes:", list(model_bundle["labels"].values()))
    st.write("Rolling window size:", model_bundle.get("window_size", 10), "frames")
    st.caption(
        "This is a coursework prototype, not a certified medical device. "
        "Predictions should not be relied on for real safety decisions."
    )

tab1, tab2 = st.tabs(["📹 Video Monitoring", "🖼️ Single Image Check"])

# ---------------------------------------------------------------------
# TAB 1 — Video upload and frame-by-frame monitoring
# ---------------------------------------------------------------------
with tab1:
    uploaded_video = st.file_uploader(
        "Upload a video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"], key="video"
    )

    if uploaded_video:
        suffix = os.path.splitext(uploaded_video.name)[1]
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        detector = FallDetector(model_bundle)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        frame_placeholder = st.empty()
        progress_bar = st.progress(0)
        alert_placeholder = st.empty()

        col1, col2, col3, col4 = st.columns(4)
        total_metric = col1.empty()
        fall_metric = col2.empty()
        normal_metric = col3.empty()
        conf_metric = col4.empty()

        results_log = []
        fall_count = 0
        normal_count = 0
        frame_idx = 0
        step = max(int(fps // 5), 1)  # sample ~5 predictions per second of video

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % step != 0:
                continue

            label, confidence, feats = detector.process_frame(frame)

            if label is not None:
                results_log.append(
                    {
                        "frame": frame_idx,
                        "time_s": round(frame_idx / fps, 1),
                        "label": label,
                        "confidence": round(confidence, 3),
                    }
                )
                if label == "FALL":
                    fall_count += 1
                    alert_placeholder.error(
                        f"🚨 FALL DETECTED at {frame_idx/fps:.1f}s "
                        f"(confidence {confidence:.0%})"
                    )
                else:
                    normal_count += 1

                total_metric.metric("Frames analyzed", len(results_log))
                fall_metric.metric("Fall events", fall_count)
                normal_metric.metric("Normal frames", normal_count)
                conf_metric.metric("Last confidence", f"{confidence:.0%}")

            frame_placeholder.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )
            progress_bar.progress(min(frame_idx / total_frames, 1.0))

        cap.release()
        os.unlink(video_path)

        if results_log and fall_count == 0:
            st.success("✅ No falls detected in this video.")

        if results_log:
            st.subheader("📊 Monitoring analytics")
            df = pd.DataFrame(results_log)
            c1, c2 = st.columns(2)
            with c1:
                st.write("Activity distribution")
                st.bar_chart(df["label"].value_counts())
            with c2:
                st.write("Confidence over time")
                st.line_chart(df.set_index("time_s")["confidence"])
            with st.expander("Full prediction log"):
                st.dataframe(df, use_container_width=True)
        elif uploaded_video:
            st.warning("No person was detected in any sampled frame of this video.")

# ---------------------------------------------------------------------
# TAB 2 — Single image check
# ---------------------------------------------------------------------
with tab2:
    st.caption(
        "Note: a single image has no motion history, so the rolling-window "
        "features fall back to single-frame values — predictions here are "
        "less reliable than on video."
    )
    uploaded_image = st.file_uploader(
        "Upload a single image", type=["jpg", "jpeg", "png"], key="image"
    )
    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        detector = FallDetector(model_bundle)
        label, confidence, feats = detector.process_frame(frame)

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        if label is None:
            st.warning("No person detected in this image.")
        elif label == "FALL":
            st.error(f"🚨 FALL DETECTED — confidence {confidence:.0%}")
        else:
            st.success(f"✅ Normal activity — confidence {confidence:.0%}")
