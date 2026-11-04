import tempfile
import threading
from pathlib import Path

import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer

from src.inference import ASLSignRecognizer

st.set_page_config(page_title="ASL Sign Recognition", page_icon="🤟", layout="wide")


@st.cache_resource(show_spinner="Loading ASL model...")
def load_recognizer():
    return ASLSignRecognizer()


class SignVideoRecorder(VideoProcessorBase):
    """Collect camera frames in the browser session for later inference."""

    def __init__(self):
        self.frames = []
        self.lock = threading.Lock()
        self.max_frames = 384

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        with self.lock:
            if len(self.frames) < self.max_frames:
                self.frames.append(image.copy())
        return frame

    def snapshot(self):
        with self.lock:
            return [frame.copy() for frame in self.frames]


st.title("🤟 ASL Sign Recognition")
st.caption("Record one isolated ASL sign using your camera, then recognize it.")
st.info(
    "This model classifies one isolated sign from a 250-class ASL vocabulary. "
    "It does not translate continuous signing into sentences."
)

st.subheader("Record with your camera")
st.write("Keep your face, upper body, and both hands visible. Record one sign, then stop.")

ctx = webrtc_streamer(
    key="asl-camera-recorder",
    mode=WebRtcMode.SENDONLY,
    video_processor_factory=SignVideoRecorder,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

processor = ctx.video_processor
frames = processor.snapshot() if processor is not None else []

if frames:
    st.caption(f"Captured {len(frames)} frames (maximum 384).")
    if len(frames) >= 384:
        st.warning("Frame limit reached. The recording has been capped; use a shorter sign recording.")

if not ctx.state.playing and frames:
    if st.button("Recognize recorded sign", type="primary"):
        video_path = None
        try:
            height, width = frames[0].shape[:2]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                video_path = tmp.name

            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                24.0,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError("Could not create a temporary video from the camera recording.")
            try:
                for image in frames:
                    writer.write(image)
            finally:
                writer.release()

            recognizer = load_recognizer()
            with st.spinner("Extracting landmarks and recognizing the sign..."):
                results = recognizer.predict_video(video_path)

            st.subheader("Prediction")
            best = results[0]
            st.metric("Predicted sign", best["label"], f'{best["confidence"]:.1%} confidence')
            st.subheader("Top 5 predictions")
            st.dataframe(
                [
                    {"Sign": item["label"], "Confidence": f'{item["confidence"]:.2%}'}
                    for item in results
                ],
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.error(f"Could not process this recording: {exc}")
            st.exception(exc)
        finally:
            if video_path:
                Path(video_path).unlink(missing_ok=True)
elif not ctx.state.playing and not frames:
    st.caption("Start the camera, record a sign, and stop the camera to enable recognition.")

with st.expander("Alternative: upload a video"):
    uploaded = st.file_uploader("Upload a signing video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded and st.button("Recognize uploaded video"):
        suffix = Path(uploaded.name).suffix or ".mp4"
        video_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                video_path = tmp.name
            recognizer = load_recognizer()
            with st.spinner("Extracting landmarks and running inference..."):
                results = recognizer.predict_video(video_path)
            st.subheader("Upload prediction")
            best = results[0]
            st.metric("Predicted sign", best["label"], f'{best["confidence"]:.1%} confidence')
            st.dataframe(
                [
                    {"Sign": item["label"], "Confidence": f'{item["confidence"]:.2%}'}
                    for item in results
                ],
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.error(f"Could not process this video: {exc}")
            st.exception(exc)
        finally:
            if video_path:
                Path(video_path).unlink(missing_ok=True)

with st.expander("About this model"):
    st.write(
        "The app uses the Kaggle Google Isolated Sign Language Recognition first-place "
        "TensorFlow Lite model. Its input is MediaPipe Holistic landmark sequences, not raw RGB frames."
    )
    st.write(
        "The model recognizes isolated signs from a 250-class vocabulary. Accuracy can vary "
        "with camera angle, lighting, signing style, and video framing."
    )
