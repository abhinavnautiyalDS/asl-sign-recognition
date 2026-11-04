import json
import os
from pathlib import Path

import numpy as np
import requests
import tensorflow as tf

from src.landmarks import extract_video_landmarks

MODEL_URL = "https://huggingface.co/sign/kaggle-asl-signs-1st-place/resolve/main/model.tflite"
LABELS_URL = "https://huggingface.co/sign/kaggle-asl-signs-1st-place/resolve/main/sign_to_prediction_index_map.json"
ASSET_DIR = Path("assets")
MODEL_PATH = ASSET_DIR / "model.tflite"
LABELS_PATH = ASSET_DIR / "sign_to_prediction_index_map.json"

def _download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        destination.write_bytes(response.content)

class ASLSignRecognizer:
    def __init__(self):
        _download(MODEL_URL, MODEL_PATH)
        _download(LABELS_URL, LABELS_PATH)
        self.labels = self._load_labels()
        self.interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH), num_threads=max(1, os.cpu_count() or 1))
        self.interpreter.allocate_tensors()
        self.signature_list = self.interpreter.get_signature_list()
        if "serving_default" not in self.signature_list:
            raise RuntimeError(f"Expected 'serving_default' signature. Found: {self.signature_list}")
        self.predict_fn = self.interpreter.get_signature_runner("serving_default")

    def _load_labels(self):
        raw = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        # Kaggle mapping is sign string -> integer class index.
        if raw and all(isinstance(v, int) for v in raw.values()):
            return {int(index): label for label, index in raw.items()}
        return {int(index): str(label) for index, label in raw.items()}

    def predict_landmarks(self, frames, top_k=5):
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim != 3 or frames.shape[1:] != (543, 3):
            raise ValueError(f"Expected [frames, 543, 3] landmarks; received {frames.shape}")
        # The published competition model accepts raw landmarks and includes its
        # learned preprocessing in the TFLite graph.
        output = self.predict_fn(inputs=frames)
        scores = np.asarray(output["outputs"]).reshape(-1)
        # Softmax only if the exported model returns logits rather than probabilities.
        if np.any(scores < 0) or not np.isclose(scores.sum(), 1.0, atol=1e-3):
            shifted = scores - np.max(scores)
            exp = np.exp(shifted)
            scores = exp / exp.sum()
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            {"label": self.labels.get(int(i), f"class_{int(i)}"), "confidence": float(scores[i])}
            for i in indices
        ]

    def predict_video(self, video_path, top_k=5):
        landmarks = extract_video_landmarks(video_path)
        return self.predict_landmarks(landmarks, top_k=top_k)
