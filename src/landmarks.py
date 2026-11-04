import cv2
import mediapipe as mp
import numpy as np

ROWS_PER_FRAME = 543

def _landmark_array(landmarks, count):
    if landmarks is None:
        return np.full((count, 3), np.nan, dtype=np.float32)
    return np.asarray([[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float32)

def extract_video_landmarks(video_path, max_frames=384):
    """Return [frames, 543, 3] in Kaggle ASL dataset order:
    face(468), left_hand(21), pose(33), right_hand(21).
    """
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("The uploaded video could not be opened.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    stride = max(1, int(np.ceil(frame_count / max_frames))) if frame_count else 1
    frames = []
    frame_index = 0

    try:
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % stride == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = holistic.process(rgb)
                    face = _landmark_array(result.face_landmarks, 468)
                    left = _landmark_array(result.left_hand_landmarks, 21)
                    pose = _landmark_array(result.pose_landmarks, 33)
                    right = _landmark_array(result.right_hand_landmarks, 21)
                    frames.append(np.concatenate([face, left, pose, right], axis=0))
                frame_index += 1
                if len(frames) >= max_frames:
                    break
    finally:
        cap.release()

    if not frames:
        raise ValueError("No frames could be read from this video.")
    return np.stack(frames).astype(np.float32)
