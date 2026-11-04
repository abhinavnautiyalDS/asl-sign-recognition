import tempfile
from pathlib import Path

import streamlit as st

from src.inference import ASLSignRecognizer

st.set_page_config(page_title="ASL Sign Recognition", page_icon="🤟", layout="wide")


@st.cache_resource(show_spinner="Loading ASL model...")
def load_recognizer():
    return ASLSignRecognizer()


def show_predictions(results, title="Prediction"):
    st.subheader(title)
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


st.title("🤟 ASL Sign Recognition")
st.caption("Upload a video containing one isolated ASL sign, then recognize it.")
st.info(
    "This model classifies one isolated sign from a 250-class ASL vocabulary. "
    "It does not translate continuous signing into sentences."
)

st.subheader("Upload a signing video")
uploaded = st.file_uploader(
    "Choose a video",
    type=["mp4", "mov", "avi", "mkv"],
)

if uploaded and st.button("Classify sign", type="primary"):
    video_path = None
    try:
        suffix = Path(uploaded.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            video_path = tmp.name

        recognizer = load_recognizer()
        with st.spinner("Extracting landmarks and running inference..."):
            results = recognizer.predict_video(video_path)

        show_predictions(results)
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
