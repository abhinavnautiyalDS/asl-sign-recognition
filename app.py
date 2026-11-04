import streamlit as st
import tempfile
from pathlib import Path

from src.inference import ASLSignRecognizer

st.set_page_config(page_title="ASL Sign Recognition", page_icon="🤟", layout="wide")

@st.cache_resource(show_spinner="Loading ASL model...")
def load_recognizer():
    return ASLSignRecognizer()

st.title("🤟 ASL Sign Recognition")
st.caption("Isolated American Sign Language (ASL) sign classification · 250 classes")
st.info("Upload a short video containing one isolated sign. This demo predicts a vocabulary label; it does not translate continuous signing into sentences.")

uploaded = st.file_uploader("Upload a signing video", type=["mp4", "mov", "avi", "mkv"])
if uploaded:
    st.video(uploaded)
    if st.button("Recognize sign", type="primary"):
        suffix = Path(uploaded.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            video_path = tmp.name
        try:
            recognizer = load_recognizer()
            with st.spinner("Extracting landmarks and running inference..."):
                results = recognizer.predict_video(video_path)
            st.subheader("Prediction")
            best = results[0]
            st.metric("Predicted sign", best["label"], f'{best["confidence"]:.1%} confidence')
            st.subheader("Top 5 predictions")
            st.dataframe(
                [{"Sign": item["label"], "Confidence": f'{item["confidence"]:.2%}'} for item in results],
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.error(f"Could not process this video: {exc}")
            st.exception(exc)
        finally:
            Path(video_path).unlink(missing_ok=True)

with st.expander("About this model"):
    st.write("The app uses the Kaggle Google Isolated Sign Language Recognition first-place TensorFlow Lite model. Its input is MediaPipe Holistic landmark sequences, not raw RGB frames.")
    st.write("The model recognizes isolated signs from a 250-class vocabulary. Accuracy can vary with camera angle, lighting, signing style, and video framing.")
