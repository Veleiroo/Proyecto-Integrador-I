from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd

from .paths import YOLO_POSE_WEIGHTS_PATH


YOLO_LANDMARK_IDS = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
}


@dataclass(frozen=True)
class YoloPoseConfig:
    """Configuracion minima para ejecutar YOLO Pose en imagen estatica."""

    weights_path: Path = YOLO_POSE_WEIGHTS_PATH
    device: str | int = "auto"
    min_confidence: float = 0.3


class YoloPoseEstimator:
    """Wrapper ligero de Ultralytics YOLO Pose con salida compatible con el pipeline."""

    def __init__(self, config: YoloPoseConfig | None = None):
        self.config = config or YoloPoseConfig()
        self._model = None
        self._runtime_device: str | int | None = None

    def __enter__(self) -> "YoloPoseEstimator":
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Falta ultralytics/torch. Activa el entorno vpc2 o instala `ultralytics` y `torch`."
            ) from exc

        if not self.config.weights_path.exists():
            raise FileNotFoundError(f"No se encontraron pesos YOLO Pose en {self.config.weights_path}.")

        if self.config.device == "auto":
            self._runtime_device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self._runtime_device = self.config.device

        self._model = YOLO(str(self.config.weights_path))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._model = None

    @property
    def backend(self) -> str:
        if self._runtime_device is None:
            return "yolo_pose_uninitialized"
        return f"torch_cuda:{self._runtime_device}" if self._runtime_device != "cpu" else "torch_cpu"

    def infer_image(self, image_path: str | Path, metadata: dict | None = None) -> dict:
        if self._model is None:
            raise RuntimeError("El estimador YOLO Pose no esta inicializado.")

        image_path = Path(image_path)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
        height, width = image_bgr.shape[:2]

        result = self._model.predict(source=str(image_path), verbose=False, device=self._runtime_device)[0]
        row = {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "group": metadata.get("group") if metadata else None,
            "split": metadata.get("split") if metadata else None,
            "image_width": int(width),
            "image_height": int(height),
            "pose_model": "yolo_pose",
            "pose_backend": self.backend,
            "pose_detected": False,
            "pose_landmarks_count": 0,
            "visible_landmarks_count": 0,
        }

        for landmark_name in YOLO_LANDMARK_IDS:
            row[f"{landmark_name}_x"] = None
            row[f"{landmark_name}_y"] = None
            row[f"{landmark_name}_z"] = None
            row[f"{landmark_name}_visibility"] = None

        if result.keypoints is None:
            return row

        keypoint_data = getattr(result.keypoints, "data", None)
        if keypoint_data is None or len(keypoint_data) == 0:
            return row

        selected_index = 0
        if getattr(result, "boxes", None) is not None and getattr(result.boxes, "conf", None) is not None:
            selected_index = int(result.boxes.conf.argmax().item())

        keypoints = keypoint_data[selected_index].cpu().numpy()
        row["pose_detected"] = True
        row["pose_landmarks_count"] = int(len(keypoints))

        visible_count = 0
        for landmark_name, landmark_id in YOLO_LANDMARK_IDS.items():
            if landmark_id >= len(keypoints):
                continue
            point = keypoints[landmark_id]
            confidence = float(point[2]) if len(point) > 2 else 1.0
            row[f"{landmark_name}_x"] = float(point[0]) / float(width)
            row[f"{landmark_name}_y"] = float(point[1]) / float(height)
            row[f"{landmark_name}_z"] = None
            row[f"{landmark_name}_visibility"] = confidence
            if confidence >= self.config.min_confidence:
                visible_count += 1

        row["visible_landmarks_count"] = visible_count
        return row


def run_yolo_pose_batch(
    sample_df: pd.DataFrame,
    config: YoloPoseConfig | None = None,
) -> pd.DataFrame:
    """Procesa un lote de imagenes con YOLO Pose y devuelve landmarks normalizados."""
    if sample_df.empty:
        return pd.DataFrame()

    rows = []
    with YoloPoseEstimator(config=config) as estimator:
        for item in sample_df.to_dict(orient="records"):
            rows.append(
                estimator.infer_image(
                    item["image_path"],
                    metadata={"group": item.get("group"), "split": item.get("split")},
                )
            )
    return pd.DataFrame(rows)
