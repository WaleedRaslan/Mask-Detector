"""
app.py
======
Mask Detector Dashboard - a single Streamlit app that provides:
1) Live camera feed (real-time video stream in the browser)
2) Image upload for mask/no-mask verification
3) Interactive statistics using matplotlib

Everything is pure Python (no separate API to build/maintain).
"""

import os
import time

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

from detector import (
    detect_and_annotate,
    clear_log,
    ensure_log_file,
    LOG_PATH,
)

# ==========================
# Page config
# ==========================
st.set_page_config(
    page_title="Mask Detector Dashboard",
    page_icon="😷",
    layout="wide",
)

ensure_log_file()

st.title("Mask Detector Dashboard")
st.caption("Live camera detection, uploaded-image analysis, and detailed statistics — powered by MediaPipe and TensorFlow")

tab_cam, tab_upload, tab_stats = st.tabs(
    ["📷 Live Camera", "🖼️ Upload Image", "📊 Statistics"]
)

# ==========================
# Tab 1: Live camera (WebRTC)
# ==========================
with tab_cam:
    st.subheader("Live Camera Detection")
    st.write("Click **START** to turn on the camera. Faces will be detected automatically and classified (Mask / No Mask) in real time.")

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    class MaskVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.last_log_time = 0.0
            self.log_interval = 2.0  # seconds between log entries to avoid flooding the log

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            now = time.time()
            do_log = (now - self.last_log_time) > self.log_interval

            annotated, detections = detect_and_annotate(img, source="camera", do_log=do_log)

            if do_log and detections:
                self.last_log_time = now

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="mask-detection-stream",
        video_processor_factory=MaskVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

    st.info("💡 If the camera doesn't start, make sure you allow camera access in your browser.")

# ==========================
# Tab 2: Upload image
# ==========================
with tab_upload:
    st.subheader("Upload an Image to Check Mask Status")

    uploaded_file = st.file_uploader(
        "Choose an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        with st.spinner("Analyzing image..."):
            annotated_bgr, detections = detect_and_annotate(
                frame_bgr.copy(), source="upload", do_log=True
            )
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(annotated_rgb, caption="Detection Result", use_container_width=True)

        if detections:
            st.markdown("### Results")
            for i, d in enumerate(detections, start=1):
                icon = "✅" if d["label"] == "mask" else "🚫"
                st.write(
                    f"{icon} **Face {i}:** {d['label_text']} — Confidence: {d['confidence'] * 100:.1f}%"
                )
        else:
            st.warning("⚠️ No face was detected in the image. Try uploading a clearer photo.")

# ==========================
# Tab 3: Statistics
# ==========================
with tab_stats:
    st.subheader("Detection Statistics")

    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    else:
        df = pd.DataFrame(columns=["timestamp", "source", "label", "confidence"])

    if df.empty:
        st.info("No data yet. Try the live camera or upload an image first to see statistics here.")
    else:
        sources = sorted(df["source"].unique().tolist())
        selected_sources = st.multiselect(
            "Filter by source", options=sources, default=sources
        )
        filtered = df[df["source"].isin(selected_sources)]

        if filtered.empty:
            st.warning("No data matches this filter.")
        else:
            # ---- Summary cards ----
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Checks", len(filtered))
            c2.metric("✅ Mask", int((filtered["label"] == "mask").sum()))
            c3.metric("🚫 No Mask", int((filtered["label"] == "no_mask").sum()))

            colors_map = {"mask": "#2ecc71", "no_mask": "#e74c3c"}
            labels_en = {"mask": "Mask", "no_mask": "No Mask"}

            g1, g2 = st.columns(2)

            # ---- Pie chart: status distribution ----
            with g1:
                counts = filtered["label"].value_counts()
                fig1, ax1 = plt.subplots()
                ax1.pie(
                    counts.values,
                    labels=[labels_en.get(l, l) for l in counts.index],
                    autopct="%1.1f%%",
                    colors=[colors_map.get(l, "#3498db") for l in counts.index],
                    startangle=90,
                )
                ax1.set_title("Status Distribution (Mask / No Mask)")
                ax1.axis("equal")
                st.pyplot(fig1)

            # ---- Bar chart: checks per day ----
            with g2:
                tmp = filtered.copy()
                tmp["date"] = tmp["timestamp"].dt.date
                daily = tmp.groupby(["date", "label"]).size().unstack(fill_value=0)
                daily = daily.reindex(columns=["mask", "no_mask"], fill_value=0)

                fig2, ax2 = plt.subplots()
                daily.rename(columns=labels_en).plot(
                    kind="bar",
                    stacked=True,
                    ax=ax2,
                    color=[colors_map["mask"], colors_map["no_mask"]],
                )
                ax2.set_title("Checks Per Day")
                ax2.set_xlabel("Date")
                ax2.set_ylabel("Count")
                ax2.legend(title="")
                plt.xticks(rotation=45)
                st.pyplot(fig2)

            # ---- Confidence distribution ----
            fig3, ax3 = plt.subplots()
            ax3.hist(filtered["confidence"], bins=20, color="#3498db", edgecolor="white")
            ax3.set_title("Classification Confidence Distribution")
            ax3.set_xlabel("Confidence")
            ax3.set_ylabel("Count")
            st.pyplot(fig3)

            # ---- Raw table ----
            st.markdown("### Detailed Log")
            st.dataframe(
                filtered.sort_values("timestamp", ascending=False),
                use_container_width=True,
            )

            if st.button("🗑️ Clear Entire Log"):
                clear_log()
                st.rerun()