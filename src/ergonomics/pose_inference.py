from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd

from .paths import MEDIAPIPE_TASK_MODEL_PATH


LANDMARK_IDS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}

SKELETON_SEGMENTS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
]


def _mediapipe_imports():
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
    except ImportError as exc:
        raise ImportError(
            "Falta mediapipe. Instala la dependencia con `python -m pip install mediapipe` antes de ejecutar esta fase."
        ) from exc
    return mp, vision, BaseOptions


@dataclass(frozen=True)
class MediaPipePoseConfig:
    model_path: Path = MEDIAPIPE_TASK_MODEL_PATH
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    min_visibility: float = 0.3


class MediaPipePoseEstimator:
    def __init__(self, config: MediaPipePoseConfig | None = None):
        self.config = config or MediaPipePoseConfig()
        self._model = None
        self._mp = None

    def __enter__(self) -> "MediaPipePoseEstimator":
        mp, vision, BaseOptions = _mediapipe_imports()
        if not self.config.model_path.exists():
            raise FileNotFoundError(
                f"No se encontro el modelo de MediaPipe en {self.config.model_path}."
            )

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.config.model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=self.config.min_pose_detection_confidence,
            min_pose_presence_confidence=self.config.min_pose_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._mp = mp
        self._model = vision.PoseLandmarker.create_from_options(options)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._model is not None:
            self._model.close()
        self._model = None

    def infer_image(self, image_path: str | Path, metadata: dict | None = None) -> dict:
        if self._model is None or self._mp is None:
            raise RuntimeError("El estimador de MediaPipe no esta inicializado.")

        image_path = Path(image_path)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self._model.detect(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image_rgb)
        )

        row = {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "group": metadata.get("group") if metadata else None,
            "split": metadata.get("split") if metadata else None,
            "image_width": int(image_rgb.shape[1]),
            "image_height": int(image_rgb.shape[0]),
            "pose_detected": bool(result.pose_landmarks),
            "pose_landmarks_count": int(len(result.pose_landmarks[0])) if result.pose_landmarks else 0,
            "visible_landmarks_count": 0,
        }

        for landmark_name in LANDMARK_IDS:
            row[f"{landmark_name}_x"] = None
            row[f"{landmark_name}_y"] = None
            row[f"{landmark_name}_z"] = None
            row[f"{landmark_name}_visibility"] = None

        if not result.pose_landmarks:
            return row

        landmarks = result.pose_landmarks[0]
        visible_count = 0
        for landmark_name, landmark_id in LANDMARK_IDS.items():
            landmark = landmarks[landmark_id]
            visibility = float(getattr(landmark, "visibility", 0.0))
            row[f"{landmark_name}_x"] = float(landmark.x)
            row[f"{landmark_name}_y"] = float(landmark.y)
            row[f"{landmark_name}_z"] = float(landmark.z)
            row[f"{landmark_name}_visibility"] = visibility
            if visibility >= self.config.min_visibility:
                visible_count += 1

        row["visible_landmarks_count"] = visible_count
        return row


def run_mediapipe_pose_batch(
    sample_df: pd.DataFrame,
    config: MediaPipePoseConfig | None = None,
) -> pd.DataFrame:
    if sample_df.empty:
        return pd.DataFrame()

    rows = []
    with MediaPipePoseEstimator(config=config) as estimator:
        for item in sample_df.to_dict(orient="records"):
            rows.append(
                estimator.infer_image(
                    item["image_path"],
                    metadata={"group": item.get("group"), "split": item.get("split")},
                )
            )
    return pd.DataFrame(rows)
