from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .lateral_rules import analyze_lateral_pose_row
from .paths import MEDIAPIPE_TASK_MODEL_PATH, YOLO_POSE_WEIGHTS_PATH
from .pose_inference import MediaPipePoseConfig, MediaPipePoseEstimator
from .posture_rules import analyze_pose_row
from .yolo_pose_inference import YoloPoseConfig, YoloPoseEstimator


STATUS_LABELS = {
    "adequate": "Adecuada",
    "improvable": "Mejorable",
    "risk": "Riesgo",
    "insufficient_data": "Datos insuficientes",
}


FRONT_METRICS = [
    "shoulder_tilt_deg",
    "shoulder_height_diff_ratio",
    "head_lateral_offset_ratio",
    "neck_tilt_deg",
    "trunk_tilt_deg",
    "left_elbow_angle_deg",
    "right_elbow_angle_deg",
]


LATERAL_METRICS = [
    "head_forward_offset_ratio",
    "neck_forward_tilt_deg",
    "trunk_forward_tilt_deg",
    "shoulder_hip_offset_ratio",
    "lateral_elbow_angle_deg",
]


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 4)
    return value


def _pick_metrics(analysis: dict, names: list[str]) -> dict[str, Any]:
    return {name: _clean_value(analysis.get(name)) for name in names}


def _pick_statuses(analysis: dict) -> dict[str, str]:
    return {
        key: value
        for key, value in analysis.items()
        if key.endswith("_status") and key != "overall_status" and isinstance(value, str)
    }


def _review_payload(
    *,
    view: str,
    model: str,
    backend: str | None,
    analysis: dict,
    metrics: dict[str, Any],
    visible_landmarks_count: int | None,
    pose_detected: bool,
) -> dict:
    status = str(analysis.get("overall_status", "insufficient_data"))
    return {
        "view": view,
        "model": model,
        "backend": backend,
        "pose_detected": bool(pose_detected),
        "visible_landmarks_count": visible_landmarks_count,
        "status": status,
        "status_label": STATUS_LABELS.get(status, status),
        "feedback": analysis.get("feedback"),
        "metrics": metrics,
        "components": _pick_statuses(analysis),
    }


@dataclass
class PostureAnalyzer:
    """Servicio de aplicacion reutilizable por API, CLI o interfaz local."""

    front_visibility_threshold: float = 0.35
    lateral_visibility_threshold: float = 0.3
    yolo_device: str | int = "auto"

    def __post_init__(self) -> None:
        self._front_estimator: MediaPipePoseEstimator | None = None
        self._lateral_estimator: YoloPoseEstimator | None = None

    def close(self) -> None:
        if self._front_estimator is not None:
            self._front_estimator.__exit__(None, None, None)
            self._front_estimator = None
        if self._lateral_estimator is not None:
            self._lateral_estimator.__exit__(None, None, None)
            self._lateral_estimator = None

    def _get_front_estimator(self) -> MediaPipePoseEstimator:
        if self._front_estimator is None:
            config = MediaPipePoseConfig(
                model_path=MEDIAPIPE_TASK_MODEL_PATH,
                min_visibility=self.front_visibility_threshold,
            )
            self._front_estimator = MediaPipePoseEstimator(config=config)
            self._front_estimator.__enter__()
        return self._front_estimator

    def _get_lateral_estimator(self) -> YoloPoseEstimator:
        if self._lateral_estimator is None:
            config = YoloPoseConfig(
                weights_path=YOLO_POSE_WEIGHTS_PATH,
                device=self.yolo_device,
                min_confidence=self.lateral_visibility_threshold,
            )
            self._lateral_estimator = YoloPoseEstimator(config=config)
            self._lateral_estimator.__enter__()
        return self._lateral_estimator

    def analyze_front_image(self, image_path: str | Path) -> dict:
        pose_row = self._get_front_estimator().infer_image(image_path)
        analysis = analyze_pose_row(
            pose_row,
            visibility_threshold=self.front_visibility_threshold,
        )
        return _review_payload(
            view="front",
            model="MediaPipe Pose",
            backend="mediapipe_tasks",
            analysis=analysis,
            metrics=_pick_metrics(analysis, FRONT_METRICS),
            visible_landmarks_count=pose_row.get("visible_landmarks_count"),
            pose_detected=bool(pose_row.get("pose_detected")),
        )

    def analyze_lateral_image(self, image_path: str | Path) -> dict:
        estimator = self._get_lateral_estimator()
        pose_row = estimator.infer_image(image_path)
        analysis = analyze_lateral_pose_row(
            pose_row,
            visibility_threshold=self.lateral_visibility_threshold,
        )
        return _review_payload(
            view="lateral",
            model="YOLO Pose",
            backend=pose_row.get("pose_backend"),
            analysis=analysis,
            metrics=_pick_metrics(analysis, LATERAL_METRICS),
            visible_landmarks_count=pose_row.get("visible_landmarks_count"),
            pose_detected=bool(pose_row.get("pose_detected")),
        )
