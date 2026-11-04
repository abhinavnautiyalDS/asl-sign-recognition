# ASL Sign Recognition

A Streamlit demo for **isolated American Sign Language (ASL) sign classification** using the Kaggle Google Isolated Sign Language Recognition first-place TensorFlow Lite model.

## What it does

1. Accepts a short video containing one isolated sign.
2. Extracts MediaPipe Holistic landmarks from sampled frames.
3. Reorders landmarks into the competition layout: face (468), left hand (21), pose (33), right hand (21).
4. Sends raw landmark sequences to the published TensorFlow Lite model.
5. Displays the top five vocabulary predictions.

> This is not continuous ASL sentence translation. It classifies one isolated sign from a 250-class vocabulary.

## Run locally

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/abhinavnautiyalDS/asl-sign-recognition.git
cd asl-sign-recognition

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The model (about 11 MB) and class map are downloaded from Hugging Face on first run and cached in the local `assets/` directory. They are intentionally not committed to Git.

## Deployment

After local inference is verified, deploy this repository using Streamlit Community Cloud with:
- Main file: `app.py`
- Python version: 3.10 or 3.11 (select in Advanced settings if needed)

The first cloud run downloads the model assets. MediaPipe and TensorFlow dependencies may require adjustment based on the selected Cloud Python runtime.

## Model and references

- Model: [sign/kaggle-asl-signs-1st-place](https://huggingface.co/sign/kaggle-asl-signs-1st-place)
- Original first-place inference/training code: [hoyso48 GitHub repository](https://github.com/hoyso48/Google---Isolated-Sign-Language-Recognition-1st-place-solution)
- Competition: [Google - Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs)

The competition dataset and the model have separate licensing terms. Review the upstream model card and competition terms before redistribution or commercial use.
