from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from .lateral_rules import analyze_lateral_pose_row
from .paths import MEDIAPIPE_TASK_MODEL_PATH, YOLO_POSE_WEIGHTS_PATH
from .pose_inference import MediaPipePoseConfig, MediaPipePoseEstimator, SKELETON_SEGMENTS
from .posture_rules import analyze_pose_row
from .yolo_pose_inference import YoloPoseConfig, YoloPoseEstimator


STATUS_LABELS = {
    "adequate": "Adecuada",
    "improvable": "Mejorable",
    "risk": "Riesgo",
    "insufficient_data": "Datos insuficientes",
}

SEVERITY_ORDER = {
    "insufficient_data": 0,
    "adequate": 1,
    "improvable": 2,
    "risk": 3,
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


DEBUG_COLORS = {
    "adequate": (47, 158, 68),
    "improvable": (245, 159, 0),
    "risk": (224, 49, 49),
    "insufficient_data": (134, 142, 150),
}


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _point_from_pose(pose_row: dict, name: str, width: int, height: int, min_visibility: float = 0.3) -> tuple[int, int] | None:
    visibility = _finite_float(pose_row.get(f"{name}_visibility"))
    x = _finite_float(pose_row.get(f"{name}_x"))
    y = _finite_float(pose_row.get(f"{name}_y"))
    if visibility is None or x is None or y is None:
        return None
    if visibility < min_visibility:
        return None
    return int(x * width), int(y * height)


def _component_status(analysis: dict, *keys: str) -> str:
    for key in keys:
        value = analysis.get(key)
        if isinstance(value, str):
            return value
    return "insufficient_data"


def _debug_rule_lines(view: str, pose_row: dict, analysis: dict, width: int, height: int) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []

    def add(label: str, start: str, end: str, status: str) -> None:
        start_point = _point_from_pose(pose_row, start, width, height)
        end_point = _point_from_pose(pose_row, end, width, height)
        if start_point is None or end_point is None:
            return
        lines.append(
            {
                "label": label,
                "from": start,
                "to": end,
                "status": status,
                "points": [start_point, end_point],
            }
        )

    if view == "lateral":
        side = analysis.get("lateral_side") or "left"
        shoulder = f"{side}_shoulder"
        elbow = f"{side}_elbow"
        wrist = f"{side}_wrist"
        hip = f"{side}_hip"
        add("Cabeza-cuello", "nose", shoulder, _component_status(analysis, "head_neck_status", "neck_status"))
        if analysis.get("lateral_torso_valid"):
            add("Tronco lateral", hip, shoulder, _component_status(analysis, "trunk_status", "trunk_tilt_status"))
        if analysis.get("lateral_elbow_chain_valid"):
            add("Codo lateral", shoulder, elbow, _component_status(analysis, "lateral_elbow_status"))
            add("Antebrazo lateral", elbow, wrist, _component_status(analysis, "lateral_elbow_status"))
        return lines

    add("Linea de hombros", "left_shoulder", "right_shoulder", _component_status(analysis, "shoulder_status", "shoulder_tilt_status"))
    add("Eje cervical", "nose", "left_shoulder", _component_status(analysis, "head_status", "neck_tilt_status"))
    add("Eje cervical", "nose", "right_shoulder", _component_status(analysis, "head_status", "neck_tilt_status"))
    add("Tronco izquierdo", "left_shoulder", "left_hip", _component_status(analysis, "trunk_status"))
    add("Tronco derecho", "right_shoulder", "right_hip", _component_status(analysis, "trunk_status"))
    return lines


def _draw_debug_overlay(image_path: str | Path, pose_row: dict, analysis: dict, view: str, output_path: str | Path) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
    height, width = image.shape[:2]
    keypoints: list[dict[str, Any]] = []
    point_lookup: dict[str, tuple[int, int]] = {}

    for key in pose_row:
        if not key.endswith("_x"):
            continue
        name = key[:-2]
        point = _point_from_pose(pose_row, name, width, height)
        visibility = _finite_float(pose_row.get(f"{name}_visibility"))
        if point is None:
            continue
        point_lookup[name] = point
        keypoints.append(
            {
                "name": name,
                "x": point[0],
                "y": point[1],
                "visibility": round(visibility, 3) if visibility is not None else None,
            }
        )

    for start, end in SKELETON_SEGMENTS:
        if start in point_lookup and end in point_lookup:
            cv2.line(image, point_lookup[start], point_lookup[end], (255, 190, 11), 2, cv2.LINE_AA)

    rule_lines = _debug_rule_lines(view, pose_row, analysis, width, height)
    for line in rule_lines:
        color = DEBUG_COLORS.get(line["status"], DEBUG_COLORS["insufficient_data"])
        start_point, end_point = line["points"]
        cv2.line(image, start_point, end_point, color, 4, cv2.LINE_AA)

    for point in keypoints:
        cv2.circle(image, (point["x"], point["y"]), 5, (0, 180, 216), -1, cv2.LINE_AA)

    status = analysis.get("overall_status", "insufficient_data")
    cv2.rectangle(image, (14, 14), (430, 76), (15, 23, 42), -1)
    cv2.putText(image, f"{view.upper()} | {STATUS_LABELS.get(status, status)}", (28, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, DEBUG_COLORS.get(status, (255, 255, 255)), 2, cv2.LINE_AA)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return {"keypoints": keypoints, "rule_lines": rule_lines}


def _image_data_url(image_path: str | Path) -> str:
    import base64

    path = Path(image_path)
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


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


def _worst_status(statuses: list[str]) -> str:
    available = [status for status in statuses if status != "insufficient_data"]
    if not available:
        return "insufficient_data"
    return max(available, key=lambda item: SEVERITY_ORDER.get(item, 0))


def _prefix_components(prefix: str, components: dict[str, str]) -> dict[str, str]:
    return {f"{prefix}_{key}": value for key, value in components.items()}


def combine_review_payloads(front: dict, lateral: dict | None = None) -> dict:
    if lateral is None:
        return {
            **front,
            "view": "combined",
            "model": front.get("model", "MediaPipe Pose"),
            "backend": front.get("backend"),
            "feedback": f"Evaluación frontal: {front.get('feedback')}",
        }

    status = _worst_status([str(front.get("status")), str(lateral.get("status"))])
    front_feedback = front.get("feedback") or "Sin feedback frontal."
    lateral_feedback = lateral.get("feedback") or "Sin feedback lateral."
    return {
        "view": "combined",
        "model": f"{front.get('model', 'MediaPipe Pose')} + {lateral.get('model', 'YOLO Pose')}",
        "backend": "combined",
        "pose_detected": bool(front.get("pose_detected")) and bool(lateral.get("pose_detected")),
        "visible_landmarks_count": (front.get("visible_landmarks_count") or 0) + (lateral.get("visible_landmarks_count") or 0),
        "status": status,
        "status_label": STATUS_LABELS.get(status, status),
        "feedback": f"Frontal: {front_feedback} Lateral: {lateral_feedback}",
        "metrics": {
            **(front.get("metrics") or {}),
            **(lateral.get("metrics") or {}),
        },
        "components": {
            **_prefix_components("front", front.get("components") or {}),
            **_prefix_components("lateral", lateral.get("components") or {}),
        },
        "views": {
            "front": front,
            "lateral": lateral,
        },
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

    def _reset_lateral_estimator(self) -> None:
        if self._lateral_estimator is not None:
            self._lateral_estimator.__exit__(None, None, None)
            self._lateral_estimator = None

    def _get_front_estimator(self) -> MediaPipePoseEstimator:
        if self._front_estimator is None or getattr(self._front_estimator, "_model", None) is None:
            config = MediaPipePoseConfig(
                model_path=MEDIAPIPE_TASK_MODEL_PATH,
                min_visibility=self.front_visibility_threshold,
            )
            self._front_estimator = MediaPipePoseEstimator(config=config)
            self._front_estimator.__enter__()
        return self._front_estimator

    def _get_lateral_estimator(self) -> YoloPoseEstimator:
        if self._lateral_estimator is None or getattr(self._lateral_estimator, "_model", None) is None:
            config = YoloPoseConfig(
                weights_path=YOLO_POSE_WEIGHTS_PATH,
                device=self.yolo_device,
                min_confidence=self.lateral_visibility_threshold,
            )
            self._lateral_estimator = YoloPoseEstimator(config=config)
            self._lateral_estimator.__enter__()
        return self._lateral_estimator

    def _infer_lateral_pose_with_recovery(self, image_path: str | Path) -> dict:
        try:
            return self._get_lateral_estimator().infer_image(image_path)
        except RuntimeError as exc:
            if "no esta inicializado" not in str(exc).lower():
                raise
            self._reset_lateral_estimator()
            return self._get_lateral_estimator().infer_image(image_path)

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
        model = "YOLO Pose"
        backend = None
        try:
            pose_row = self._infer_lateral_pose_with_recovery(image_path)
            backend = pose_row.get("pose_backend")
        except ImportError:
            pose_row = self._get_front_estimator().infer_image(image_path)
            model = "MediaPipe Pose (fallback lateral)"
            backend = "mediapipe_tasks_lateral_fallback"
        analysis = analyze_lateral_pose_row(
            pose_row,
            visibility_threshold=self.lateral_visibility_threshold,
        )
        return _review_payload(
            view="lateral",
            model=model,
            backend=backend,
            analysis=analysis,
            metrics=_pick_metrics(analysis, LATERAL_METRICS),
            visible_landmarks_count=pose_row.get("visible_landmarks_count"),
            pose_detected=bool(pose_row.get("pose_detected")),
        )

    def analyze_combined_images(self, front_image_path: str | Path, lateral_image_path: str | Path | None = None) -> dict:
        front = self.analyze_front_image(front_image_path)
        lateral = self.analyze_lateral_image(lateral_image_path) if lateral_image_path is not None else None
        return combine_review_payloads(front, lateral)

    def analyze_debug_image(self, image_path: str | Path, *, view: str, output_dir: str | Path) -> dict:
        if view not in {"front", "lateral"}:
            raise ValueError(f"Vista de depuracion no soportada: {view}")

        image_path = Path(image_path)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        safe_stem = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in image_path.stem)[:48]
        run_dir = Path(output_dir) / f"{timestamp}_{view}_{safe_stem}"
        run_dir.mkdir(parents=True, exist_ok=True)

        original_path = run_dir / f"original{image_path.suffix or '.jpg'}"
        shutil.copy2(image_path, original_path)
        annotated_path = run_dir / "annotated.jpg"

        if view == "front":
            pose_row = self._get_front_estimator().infer_image(original_path)
            analysis = analyze_pose_row(
                pose_row,
                visibility_threshold=self.front_visibility_threshold,
            )
            result = _review_payload(
                view="front",
                model="MediaPipe Pose",
                backend="mediapipe_tasks",
                analysis=analysis,
                metrics=_pick_metrics(analysis, FRONT_METRICS),
                visible_landmarks_count=pose_row.get("visible_landmarks_count"),
                pose_detected=bool(pose_row.get("pose_detected")),
            )
        else:
            model = "YOLO Pose"
            backend = None
            try:
                pose_row = self._infer_lateral_pose_with_recovery(original_path)
                backend = pose_row.get("pose_backend")
            except ImportError:
                pose_row = self._get_front_estimator().infer_image(original_path)
                model = "MediaPipe Pose (fallback lateral)"
                backend = "mediapipe_tasks_lateral_fallback"
            analysis = analyze_lateral_pose_row(
                pose_row,
                visibility_threshold=self.lateral_visibility_threshold,
            )
            result = _review_payload(
                view="lateral",
                model=model,
                backend=backend,
                analysis=analysis,
                metrics=_pick_metrics(analysis, LATERAL_METRICS),
                visible_landmarks_count=pose_row.get("visible_landmarks_count"),
                pose_detected=bool(pose_row.get("pose_detected")),
            )

        debug = _draw_debug_overlay(original_path, pose_row, analysis, view, annotated_path)
        return {
            "ok": True,
            "result": result,
            "debug": {
                **debug,
                "saved_dir": str(run_dir),
                "original_path": str(original_path),
                "annotated_path": str(annotated_path),
                "original_image_data_url": _image_data_url(original_path),
                "annotated_image_data_url": _image_data_url(annotated_path),
            },
        }
