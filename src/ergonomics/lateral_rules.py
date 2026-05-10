from __future__ import annotations

import math
import pandas as pd

from .posture_rules import SEVERITY_ORDER


LATERAL_SIDES = ("left", "right")


def _visibility(row: dict | pd.Series, landmark_name: str) -> float | None:
    value = row.get(f"{landmark_name}_visibility")
    if value is None or pd.isna(value):
        return None
    return float(value)


def _is_visible(row: dict | pd.Series, landmark_name: str, threshold: float = 0.3) -> bool:
    value = _visibility(row, landmark_name)
    return value is not None and value >= threshold


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


def _cap_severity(status: str, max_status: str) -> str:
    return status if SEVERITY_ORDER[status] <= SEVERITY_ORDER[max_status] else max_status


def _segment_ratio(
    point_a: tuple[float, float] | None,
    point_b: tuple[float, float] | None,
    reference_length: float | None,
) -> float | None:
    if point_a is None or point_b is None or reference_length is None or reference_length <= 0:
        return None
    return _distance(point_a, point_b) / reference_length


def _valid_lateral_torso(
    row: dict | pd.Series,
    side: str,
    shoulder: tuple[float, float] | None,
    hip: tuple[float, float] | None,
    torso_length: float | None,
    visibility_threshold: float,
) -> bool:
    if shoulder is None or hip is None or torso_length is None:
        return False
    shoulder_visibility = _visibility(row, f"{side}_shoulder") or 0.0
    hip_visibility = _visibility(row, f"{side}_hip") or 0.0
    if shoulder_visibility < max(visibility_threshold, 0.45) or hip_visibility < max(visibility_threshold, 0.45):
        return False
    vertical_delta = hip[1] - shoulder[1]
    if vertical_delta < 0.12:
        return False
    if not 0.16 <= torso_length <= 0.46:
        return False
    # Cuando la supuesta cadera queda demasiado desplazada horizontalmente, suele ser silla, mesa o respaldo.
    if abs(shoulder[0] - hip[0]) / torso_length > 0.38:
        return False
    return True


def _valid_lateral_elbow_chain(
    row: dict | pd.Series,
    side: str,
    shoulder: tuple[float, float] | None,
    elbow: tuple[float, float] | None,
    wrist: tuple[float, float] | None,
    torso_length: float | None,
    visibility_threshold: float,
) -> bool:
    if shoulder is None or elbow is None or wrist is None or torso_length is None:
        return False
    elbow_visibility = _visibility(row, f"{side}_elbow") or 0.0
    wrist_visibility = _visibility(row, f"{side}_wrist") or 0.0
    if elbow_visibility < max(visibility_threshold, 0.45) or wrist_visibility < max(visibility_threshold, 0.45):
        return False
    upper_arm_ratio = _segment_ratio(shoulder, elbow, torso_length)
    forearm_ratio = _segment_ratio(elbow, wrist, torso_length)
    if upper_arm_ratio is None or forearm_ratio is None:
        return False
    if not 0.28 <= upper_arm_ratio <= 1.35:
        return False
    if not 0.28 <= forearm_ratio <= 1.55:
        return False
    # Evita aceptar puntos claramente fuera de la cadena corporal por silla/mesa.
    if abs(upper_arm_ratio - forearm_ratio) > 1.0:
        return False
    return True


def _choose_lateral_side(row: dict | pd.Series, visibility_threshold: float) -> str | None:
    scores: dict[str, float] = {}
    for side in LATERAL_SIDES:
        names = ["nose", f"{side}_shoulder", f"{side}_hip"]
        score = 0.0
        for name in names:
            value = row.get(f"{name}_visibility")
            if value is not None and not pd.isna(value):
                weight = 2.0 if name.endswith(("_shoulder", "_hip")) else 1.0
                score += float(value) * weight
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
            "lateral_torso_valid": False,
            "lateral_elbow_chain_valid": False,
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

    lateral_torso_valid = _valid_lateral_torso(
        row,
        side,
        shoulder,
        hip,
        torso_length,
        visibility_threshold,
    )

    lateral_chain_ready = nose is not None and shoulder is not None and hip is not None and lateral_torso_valid

    head_forward_offset_ratio = None
    neck_forward_tilt_deg = None
    if nose is not None and shoulder is not None:
        neck_forward_tilt_deg = _line_tilt_from_vertical(shoulder, nose)
        if torso_length and torso_length > 0 and lateral_torso_valid:
            head_forward_offset_ratio = abs(nose[0] - shoulder[0]) / torso_length

    trunk_forward_tilt_deg = None
    shoulder_hip_offset_ratio = None
    if shoulder is not None and hip is not None and torso_length and torso_length > 0 and lateral_torso_valid:
        trunk_forward_tilt_deg = _line_tilt_from_vertical(hip, shoulder)
        shoulder_hip_offset_ratio = abs(shoulder[0] - hip[0]) / torso_length

    lateral_elbow_chain_valid = _valid_lateral_elbow_chain(
        row,
        side,
        shoulder,
        elbow,
        wrist,
        torso_length,
        visibility_threshold,
    )
    lateral_elbow_angle_deg = None
    if shoulder is not None and elbow is not None and wrist is not None and lateral_elbow_chain_valid:
        lateral_elbow_angle_deg = _joint_angle(shoulder, elbow, wrist)

    return {
        "lateral_side": side,
        "lateral_chain_ready": lateral_chain_ready,
        "lateral_torso_length": torso_length,
        "lateral_torso_valid": lateral_torso_valid,
        "lateral_elbow_chain_valid": lateral_elbow_chain_valid,
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
    # Cabeza adelantada/cuello en lateral se interpreta como aviso preventivo, no como riesgo por sí solo.
    head_neck_status = _cap_severity(head_neck_status, "improvable")

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

    raw_lateral_elbow_status = _severity_from_target_deviation(
        metrics.get("lateral_elbow_angle_deg"),
        target=95.0,
        adequate_delta=20.0,
        improvable_delta=40.0,
    )
    # En perfil, muñeca/codo se contaminan fácilmente con mesa, silla o teclado. El codo queda como señal auxiliar:
    # puede explicar una captura, pero no debe elevar por sí solo el diagnóstico global a riesgo.
    lateral_elbow_status = _cap_severity(raw_lateral_elbow_status, "improvable")

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
        if key in {"head_neck_status", "trunk_status"} and status != "insufficient_data"
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
    if lateral_elbow_status == "improvable":
        feedback.append("El angulo del codo se usa como senal auxiliar porque puede verse afectado por oclusiones de mesa o silla.")
    if not metrics.get("lateral_torso_valid"):
        feedback.append("La cadera lateral no supera el filtro geometrico de calidad; puede haber oclusion por silla o mesa.")
    if not metrics.get("lateral_elbow_chain_valid") and metrics.get("lateral_elbow_angle_deg") is None:
        feedback.append("El brazo lateral queda fuera del diagnostico porque codo o muñeca no son fiables.")

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
