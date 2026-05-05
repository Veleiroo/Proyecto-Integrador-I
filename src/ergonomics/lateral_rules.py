from __future__ import annotations

import math
import pandas as pd

from .posture_rules import SEVERITY_ORDER


LATERAL_SIDES = ("left", "right")


def _is_visible(row: dict | pd.Series, landmark_name: str, threshold: float = 0.3) -> bool:
    value = row.get(f"{landmark_name}_visibility")
    return value is not None and not pd.isna(value) and float(value) >= threshold


def _point(row: dict | pd.Series, landmark_name: str, threshold: float = 0.3) -> tuple[float, float] | None:
    if not _is_visible(row, landmark_name, threshold=threshold):
        return None
    x = row.get(f"{landmark_name}_x")
    y = row.get(f"{landmark_name}_y")
    if x is None or y is None or pd.isna(x) or pd.isna(y):
        return None
    return float(x), float(y)


def _distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])


def _line_tilt_from_vertical(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    dx = abs(point_b[0] - point_a[0])
    dy = abs(point_b[1] - point_a[1])
    if dy == 0.0:
        return 90.0
    return math.degrees(math.atan2(dx, dy))


def _joint_angle(point_a: tuple[float, float], point_b: tuple[float, float], point_c: tuple[float, float]) -> float:
    vector_ba = (point_a[0] - point_b[0], point_a[1] - point_b[1])
    vector_bc = (point_c[0] - point_b[0], point_c[1] - point_b[1])
    norm_ba = math.hypot(*vector_ba)
    norm_bc = math.hypot(*vector_bc)
    if norm_ba == 0.0 or norm_bc == 0.0:
        return math.nan
    dot_product = vector_ba[0] * vector_bc[0] + vector_ba[1] * vector_bc[1]
    cosine = max(-1.0, min(1.0, dot_product / (norm_ba * norm_bc)))
    return math.degrees(math.acos(cosine))


def _severity_from_max(value: float | None, adequate_max: float, improvable_max: float) -> str:
    if value is None or pd.isna(value):
        return "insufficient_data"
    if value <= adequate_max:
        return "adequate"
    if value <= improvable_max:
        return "improvable"
    return "risk"


def _severity_from_target_deviation(
    value: float | None,
    target: float,
    adequate_delta: float,
    improvable_delta: float,
) -> str:
    if value is None or pd.isna(value):
        return "insufficient_data"
    deviation = abs(value - target)
    if deviation <= adequate_delta:
        return "adequate"
    if deviation <= improvable_delta:
        return "improvable"
    return "risk"


def _choose_lateral_side(row: dict | pd.Series, visibility_threshold: float) -> str | None:
    scores: dict[str, float] = {}
    for side in LATERAL_SIDES:
        names = ["nose", f"{side}_shoulder", f"{side}_elbow", f"{side}_hip"]
        score = 0.0
        for name in names:
            value = row.get(f"{name}_visibility")
            if value is not None and not pd.isna(value):
                score += float(value)
        visible_required = sum(_is_visible(row, name, threshold=visibility_threshold) for name in names)
        scores[side] = score + visible_required

    best_side = max(scores, key=scores.get)
    if scores[best_side] <= 0.0:
        return None
    return best_side


def extract_lateral_posture_metrics(row: dict | pd.Series, visibility_threshold: float = 0.3) -> dict:
    """
    Extrae variables para vista lateral.
    La vista lateral no exige simetria izquierda/derecha: usa el lado corporal con mejor visibilidad.
    """
    side = _choose_lateral_side(row, visibility_threshold=visibility_threshold)
    if side is None:
        return {
            "lateral_side": None,
            "lateral_chain_ready": False,
            "lateral_torso_length": None,
            "head_forward_offset_ratio": None,
            "neck_forward_tilt_deg": None,
            "trunk_forward_tilt_deg": None,
            "shoulder_hip_offset_ratio": None,
            "lateral_elbow_angle_deg": None,
        }

    nose = _point(row, "nose", threshold=visibility_threshold)
    shoulder = _point(row, f"{side}_shoulder", threshold=visibility_threshold)
    elbow = _point(row, f"{side}_elbow", threshold=visibility_threshold)
    wrist = _point(row, f"{side}_wrist", threshold=visibility_threshold)
    hip = _point(row, f"{side}_hip", threshold=visibility_threshold)

    torso_length = None
    if shoulder is not None and hip is not None:
        torso_length = _distance(shoulder, hip)

    lateral_chain_ready = nose is not None and shoulder is not None and elbow is not None and hip is not None

    head_forward_offset_ratio = None
    neck_forward_tilt_deg = None
    if nose is not None and shoulder is not None and torso_length and torso_length > 0:
        head_forward_offset_ratio = abs(nose[0] - shoulder[0]) / torso_length
        neck_forward_tilt_deg = _line_tilt_from_vertical(shoulder, nose)

    trunk_forward_tilt_deg = None
    shoulder_hip_offset_ratio = None
    if shoulder is not None and hip is not None and torso_length and torso_length > 0:
        trunk_forward_tilt_deg = _line_tilt_from_vertical(hip, shoulder)
        shoulder_hip_offset_ratio = abs(shoulder[0] - hip[0]) / torso_length

    lateral_elbow_angle_deg = None
    if shoulder is not None and elbow is not None and wrist is not None:
        lateral_elbow_angle_deg = _joint_angle(shoulder, elbow, wrist)

    return {
        "lateral_side": side,
        "lateral_chain_ready": lateral_chain_ready,
        "lateral_torso_length": torso_length,
        "head_forward_offset_ratio": head_forward_offset_ratio,
        "neck_forward_tilt_deg": neck_forward_tilt_deg,
        "trunk_forward_tilt_deg": trunk_forward_tilt_deg,
        "shoulder_hip_offset_ratio": shoulder_hip_offset_ratio,
        "lateral_elbow_angle_deg": lateral_elbow_angle_deg,
    }


def evaluate_lateral_posture_metrics(metrics: dict) -> dict:
    """
    Reglas laterales calibradas parcialmente con MultiPosture.
    MultiPosture valida umbrales de tronco; cabeza/cuello se mantienen como senal auxiliar
    hasta disponer de etiquetas expertas especificas de cabeza adelantada.
    """
    head_offset_status = _severity_from_max(
        metrics.get("head_forward_offset_ratio"),
        adequate_max=0.20,
        improvable_max=0.35,
    )
    neck_status = _severity_from_max(
        metrics.get("neck_forward_tilt_deg"),
        adequate_max=12.0,
        improvable_max=22.0,
    )
    head_neck_status = max(
        [head_offset_status, neck_status],
        key=lambda item: SEVERITY_ORDER[item],
    )

    trunk_tilt_status = _severity_from_max(
        metrics.get("trunk_forward_tilt_deg"),
        adequate_max=7.2,
        improvable_max=8.7,
    )
    shoulder_hip_status = _severity_from_max(
        metrics.get("shoulder_hip_offset_ratio"),
        adequate_max=0.12,
        improvable_max=0.15,
    )
    trunk_status = max(
        [trunk_tilt_status, shoulder_hip_status],
        key=lambda item: SEVERITY_ORDER[item],
    )

    lateral_elbow_status = _severity_from_target_deviation(
        metrics.get("lateral_elbow_angle_deg"),
        target=95.0,
        adequate_delta=20.0,
        improvable_delta=40.0,
    )

    statuses = {
        "head_offset_status": head_offset_status,
        "neck_status": neck_status,
        "head_neck_status": head_neck_status,
        "trunk_tilt_status": trunk_tilt_status,
        "shoulder_hip_status": shoulder_hip_status,
        "trunk_status": trunk_status,
        "lateral_elbow_status": lateral_elbow_status,
    }

    available_statuses = [
        status
        for key, status in statuses.items()
        if key in {"trunk_status", "lateral_elbow_status"} and status != "insufficient_data"
    ]
    if not available_statuses:
        overall_status = "insufficient_data"
    else:
        overall_status = max(available_statuses, key=lambda item: SEVERITY_ORDER[item])

    feedback = []
    if head_neck_status in {"improvable", "risk"}:
        feedback.append("Acerca la cabeza al eje del tronco y evita adelantar el cuello.")
    if trunk_status in {"improvable", "risk"}:
        feedback.append("Endereza el tronco para reducir la flexion mantenida.")
    if lateral_elbow_status in {"improvable", "risk"}:
        feedback.append("Ajusta la distancia al escritorio para acercar el codo a una posicion comoda.")

    return {
        **statuses,
        "overall_status": overall_status,
        "feedback": " ".join(feedback) if feedback else "Sin alertas principales en la vista lateral.",
    }


def analyze_lateral_pose_row(
    pose_row: dict | pd.Series,
    *,
    visibility_threshold: float = 0.3,
) -> dict:
    metrics = extract_lateral_posture_metrics(pose_row, visibility_threshold=visibility_threshold)
    evaluation = evaluate_lateral_posture_metrics(metrics)
    return {
        "image_path": pose_row.get("image_path"),
        "image_name": pose_row.get("image_name"),
        "group": pose_row.get("group"),
        "split": pose_row.get("split"),
        "pose_detected": pose_row.get("pose_detected"),
        **metrics,
        **evaluation,
    }


def analyze_lateral_pose_dataframe(
    pose_df: pd.DataFrame,
    *,
    visibility_threshold: float = 0.3,
) -> pd.DataFrame:
    if pose_df.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            analyze_lateral_pose_row(row, visibility_threshold=visibility_threshold)
            for row in pose_df.to_dict(orient="records")
        ]
    )
