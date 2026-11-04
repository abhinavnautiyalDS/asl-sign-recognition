import tempfile
import threading
import time
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
    """Collect camera frames and track recording duration."""

    def __init__(self):
        self.frames = []
        self.lock = threading.Lock()
        self.max_frames = 384
        self.started_at = None
        self.last_frame_at = None

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        now = time.monotonic()
        with self.lock:
            if self.started_at is None:
                self.started_at = now
            self.last_frame_at = now
            if len(self.frames) < self.max_frames:
                self.frames.append(image.copy())
        return frame

    def snapshot(self):
        with self.lock:
            return [frame.copy() for frame in self.frames]

    def elapsed_seconds(self):
        with self.lock:
            if self.started_at is None:
                return 0.0
            end = self.last_frame_at or self.started_at
            return max(0.0, end - self.started_at)


def format_duration(seconds):
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"


st.title("🤟 ASL Sign Recognition")
st.caption("Record one isolated ASL sign using your camera, then recognize it.")
st.info(
    "This model classifies one isolated sign from a 250-class ASL vocabulary. "
    "It does not translate continuous signing into sentences."
)

st.subheader("1. Record with your camera")
st.write("Start the camera, perform one sign, then stop the camera.")
ctx = webrtc_streamer(
    key="asl-camera-recorder",
    mode=WebRtcMode.SENDONLY,
    video_processor_factory=SignVideoRecorder,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

processor = ctx.video_processor
# Do not block Streamlit's script thread with a timer loop. The duration is
# calculated from captured frames and displayed after recording stops.

frames = processor.snapshot() if processor is not None else []
duration = processor.elapsed_seconds() if processor is not None else 0.0

if frames:
    st.success(f"Recording captured · {format_duration(duration)} · {len(frames)} frames")
    if len(frames) >= 384:
        st.warning("The 384-frame limit was reached. Keep recordings short (about 16 seconds at 24 FPS).")

st.subheader("2. Classify recording")
if not ctx.state.playing and frames:
    if st.button("Classify sign", type="primary"):
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
            with st.spinner("Extracting landmarks and classifying the sign..."):
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
    st.caption("Start the camera to begin recording. Stop it when you finish the sign.")

with st.expander("Alternative: upload a video"):
    uploaded = st.file_uploader("Upload a signing video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded and st.button("Classify uploaded video"):
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
