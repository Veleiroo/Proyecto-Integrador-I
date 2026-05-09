from __future__ import annotations

import math
import pandas as pd

# --- CONFIGURACIÓN DE SEVERIDAD ---
# Definimos un orden numérico para comparar niveles de riesgo. 
# Esto permite que, si hay varios problemas, el sistema detecte cuál es el peor.
SEVERITY_ORDER = {
    "insufficient_data": -1,
    "adequate": 0,
    "improvable": 1,
    "risk": 2,
}

# --- UTILIDADES DE VISIBILIDAD Y PUNTOS ---

def _is_visible(row: dict | pd.Series, landmark_name: str, threshold: float = 0.35) -> bool:
    """Verifica si un punto específico fue detectado con suficiente confianza por el modelo."""
    value = row.get(f"{landmark_name}_visibility")
    return value is not None and not pd.isna(value) and float(value) >= threshold


def _point(row: dict | pd.Series, landmark_name: str, threshold: float = 0.35) -> tuple[float, float] | None:
    """Extrae las coordenadas (x, y) de un punto solo si es visible."""
    if not _is_visible(row, landmark_name, threshold=threshold):
        return None
    x = row.get(f"{landmark_name}_x")
    y = row.get(f"{landmark_name}_y")
    if x is None or y is None or pd.isna(x) or pd.isna(y):
        return None
    return float(x), float(y)


def _midpoint(point_a: tuple[float, float] | None, point_b: tuple[float, float] | None):
    """
    Calcula el punto medio entre dos coordenadas.
    Se usa para hallar el centro de los hombros o de la cadera mediante la fórmula:
    M = left((x_1 + x_2)/ 2, (y_1 + y_2)/2)
    """
    if point_a is None or point_b is None:
        return None
    return ((point_a[0] + point_b[0]) / 2.0, (point_a[1] + point_b[1]) / 2.0)

# --- CÁLCULOS GEOMÉTRICOS ---

def _angle_from_horizontal(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    """Calcula el ángulo de una línea respecto al eje horizontal"""
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    return math.degrees(math.atan2(dy, dx))


def _angle_from_vertical(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    """Calcula el ángulo de una línea respecto al eje vertical"""
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    return math.degrees(math.atan2(dx, dy))


def _joint_angle(point_a: tuple[float, float], point_b: tuple[float, float], point_c: tuple[float, float]) -> float:
    """
    Calcula el ángulo interno de una articulación (ej. el codo).
    Usa el producto escalar de vectores y el arcocoseno para hallar el ángulo
    """
    vector_ba = (point_a[0] - point_b[0], point_a[1] - point_b[1])
    vector_bc = (point_c[0] - point_b[0], point_c[1] - point_b[1])
    norm_ba = math.hypot(*vector_ba)
    norm_bc = math.hypot(*vector_bc)
    if norm_ba == 0.0 or norm_bc == 0.0:
        return math.nan
    dot_product = vector_ba[0] * vector_bc[0] + vector_ba[1] * vector_bc[1]
    cosine = max(-1.0, min(1.0, dot_product / (norm_ba * norm_bc)))
    return math.degrees(math.acos(cosine))


def _line_tilt_from_horizontal(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    """Normaliza la inclinación de una línea respecto a la horizontal (0-90 grados)"""
    raw_angle = abs(_angle_from_horizontal(point_a, point_b))
    while raw_angle > 180.0:
        raw_angle -= 180.0
    return min(raw_angle, abs(180.0 - raw_angle))


def _line_tilt_from_vertical(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    """Calcula cuánto se desvía una línea respecto a la vertical absoluta"""
    dx = abs(point_b[0] - point_a[0])
    dy = abs(point_b[1] - point_a[1])
    if dy == 0.0:
        return 90.0
    return math.degrees(math.atan2(dx, dy))

# --- CLASIFICACIÓN DE RIESGO ---

def _severity_from_max(value: float | None, adequate_max: float, improvable_max: float) -> str:
    """Determina la gravedad comparando el valor con límites máximos permitidos"""
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
    """Determina la gravedad basándose en cuánto se aleja el valor de un objetivo ideal"""
    if value is None or pd.isna(value):
        return "insufficient_data"
    deviation = abs(value - target)
    if deviation <= adequate_delta:
        return "adequate"
    if deviation <= improvable_delta:
        return "improvable"
    return "risk"

# --- PROCESAMIENTO PRINCIPAL ---

def extract_posture_metrics(row: dict | pd.Series, visibility_threshold: float = 0.35) -> dict:
    """
    EXTRACTOR DE MÉTRICAS:
    Traduce los landmarks del modelo en medidas físicas (grados y proporciones).
    """
    # 1. Obtención de puntos anatómicos visibles
    nose = _point(row, "nose", threshold=visibility_threshold)
    left_shoulder = _point(row, "left_shoulder", threshold=visibility_threshold)
    right_shoulder = _point(row, "right_shoulder", threshold=visibility_threshold)
    left_elbow = _point(row, "left_elbow", threshold=visibility_threshold)
    right_elbow = _point(row, "right_elbow", threshold=visibility_threshold)
    left_wrist = _point(row, "left_wrist", threshold=visibility_threshold)
    right_wrist = _point(row, "right_wrist", threshold=visibility_threshold)
    left_hip = _point(row, "left_hip", threshold=visibility_threshold)
    right_hip = _point(row, "right_hip", threshold=visibility_threshold)

    # 2. Puntos derivados (centros de masa)
    shoulder_center = _midpoint(left_shoulder, right_shoulder)
    hip_center = _midpoint(left_hip, right_hip)

    # 3. Cálculos de distancias e inclinaciones
    shoulder_width = None
    if left_shoulder is not None and right_shoulder is not None:
        shoulder_width = math.hypot(
            right_shoulder[0] - left_shoulder[0],
            right_shoulder[1] - left_shoulder[1],
        )

    # Inclinación de hombros (debería ser 0°)
    shoulder_tilt_deg = None
    if left_shoulder is not None and right_shoulder is not None:
        shoulder_tilt_deg = _line_tilt_from_horizontal(left_shoulder, right_shoulder)

    # Diferencia de altura entre hombros (proporcional al ancho)
    shoulder_height_diff_ratio = None
    if left_shoulder is not None and right_shoulder is not None and shoulder_width and shoulder_width > 0:
        shoulder_height_diff_ratio = abs(left_shoulder[1] - right_shoulder[1]) / shoulder_width

    # Inclinación del tronco respecto a la vertical
    trunk_tilt_deg = None
    if shoulder_center is not None and hip_center is not None:
        trunk_tilt_deg = _line_tilt_from_vertical(shoulder_center, hip_center)

    # Desviación lateral de la cabeza
    head_lateral_offset_ratio = None
    if nose is not None and shoulder_center is not None and shoulder_width and shoulder_width > 0:
        head_lateral_offset_ratio = abs(nose[0] - shoulder_center[0]) / shoulder_width

    # Inclinación del cuello
    neck_tilt_deg = None
    if nose is not None and shoulder_center is not None:
        neck_tilt_deg = _line_tilt_from_vertical(shoulder_center, nose)

    # Ángulos de los codos (idealmente cercanos a 90°-100°)
    left_elbow_angle_deg = None
    if left_shoulder is not None and left_elbow is not None and left_wrist is not None:
        left_elbow_angle_deg = _joint_angle(left_shoulder, left_elbow, left_wrist)

    right_elbow_angle_deg = None
    if right_shoulder is not None and right_elbow is not None and right_wrist is not None:
        right_elbow_angle_deg = _joint_angle(right_shoulder, right_elbow, right_wrist)

    return {
        "shoulder_width": shoulder_width,
        "shoulder_tilt_deg": shoulder_tilt_deg,
        "shoulder_height_diff_ratio": shoulder_height_diff_ratio,
        "trunk_tilt_deg": trunk_tilt_deg,
        "head_lateral_offset_ratio": head_lateral_offset_ratio,
        "neck_tilt_deg": neck_tilt_deg,
        "left_elbow_angle_deg": left_elbow_angle_deg,
        "right_elbow_angle_deg": right_elbow_angle_deg,
    }


def evaluate_posture_metrics(metrics: dict) -> dict:
    """
    EVALUADOR ERGONÓMICO:
    Aplica los umbrales de salud laboral a las métricas calculadas.
    """
    # Evaluación de hombros
    shoulder_tilt_status = _severity_from_max(
        metrics.get("shoulder_tilt_deg"),
        adequate_max=5.0,
        improvable_max=10.0,
    )
    shoulder_height_status = _severity_from_max(
        metrics.get("shoulder_height_diff_ratio"),
        adequate_max=0.03,
        improvable_max=0.07,
    )
    # Se elige el peor estado entre los dos indicadores de hombros
    shoulder_status = max(
        [shoulder_tilt_status, shoulder_height_status],
        key=lambda item: SEVERITY_ORDER[item],
    )

    # Evaluación del tronco
    trunk_status = _severity_from_max(metrics.get("trunk_tilt_deg"), adequate_max=6.0, improvable_max=12.0)
    
    # Evaluación de la cabeza y cuello
    head_offset_status = _severity_from_max(
        metrics.get("head_lateral_offset_ratio"),
        adequate_max=0.08,
        improvable_max=0.16,
    )
    neck_tilt_status = _severity_from_max(
        metrics.get("neck_tilt_deg"),
        adequate_max=8.0,
        improvable_max=15.0,
    )
    head_status = max(
        [head_offset_status, neck_tilt_status],
        key=lambda item: SEVERITY_ORDER[item],
    )
    
    # En webcam frontal los antebrazos casi nunca entran completos en plano. Conservamos los ángulos como métrica
    # de depuración, pero no se usan para clasificar ni para generar feedback en el MVP frontal.
    left_elbow_status = "insufficient_data"
    right_elbow_status = "insufficient_data"

    statuses = {
        "shoulder_tilt_status": shoulder_tilt_status,
        "shoulder_height_status": shoulder_height_status,
        "shoulder_status": shoulder_status,
        "trunk_status": trunk_status,
        "head_offset_status": head_offset_status,
        "neck_tilt_status": neck_tilt_status,
        "head_status": head_status,
        "left_elbow_status": left_elbow_status,
        "right_elbow_status": right_elbow_status,
    }

    # CÁLCULO DEL ESTADO GLOBAL
    # Se basa en cabeza/cuello, hombros y tronco. Los codos quedan fuera por baja fiabilidad en webcam frontal.
    diagnostic_keys = {"shoulder_status", "trunk_status", "head_status"}
    available_statuses = [
        status
        for key, status in statuses.items()
        if key in diagnostic_keys and status != "insufficient_data"
    ]
    if len(available_statuses) < 2:
        overall_status = "insufficient_data"
    else:
        worst_status = max(available_statuses, key=lambda item: SEVERITY_ORDER[item])
        overall_status = worst_status

    # GENERACIÓN DE CONSEJOS (FEEDBACK)
    feedback = []
    if head_status in {"improvable", "risk"}:
        if head_offset_status in {"improvable", "risk"}:
            feedback.append("Recoloca la cabeza sobre el eje de los hombros.")
        if neck_tilt_status in {"improvable", "risk"}:
            feedback.append("Reduce la inclinacion lateral del cuello.")
    if shoulder_status in {"improvable", "risk"}:
        feedback.append("Intenta mantener los hombros mas nivelados y relajados.")
    if trunk_status in {"improvable", "risk"}:
        feedback.append("Recentra el tronco para evitar inclinaciones mantenidas.")

    return {
        **statuses,
        "overall_status": overall_status,
        "feedback": " ".join(feedback) if feedback else "Sin alertas principales en esta primera revision.",
    }

# --- FUNCIONES DE ALTO NIVEL ---

def analyze_pose_dataframe(
    pose_df: pd.DataFrame,
    *,
    visibility_threshold: float = 0.35,
) -> pd.DataFrame:
    """Procesa una tabla completa de detecciones y añade las columnas de análisis ergonómico"""
    if pose_df.empty:
        return pd.DataFrame()

    rows = []
    for pose_row in pose_df.to_dict(orient="records"):
        metrics = extract_posture_metrics(pose_row, visibility_threshold=visibility_threshold)
        evaluation = evaluate_posture_metrics(metrics)
        rows.append(
            {
                "image_path": pose_row.get("image_path"),
                "image_name": pose_row.get("image_name"),
                "group": pose_row.get("group"),
                "split": pose_row.get("split"),
                "pose_detected": pose_row.get("pose_detected"),
                **metrics,
                **evaluation,
            }
        )

    return pd.DataFrame(rows)


def analyze_pose_row(
    pose_row: dict | pd.Series,
    *,
    visibility_threshold: float = 0.35,
) -> dict:
    """Procesa una única fila de detección para devolver su diagnóstico"""
    metrics = extract_posture_metrics(pose_row, visibility_threshold=visibility_threshold)
    evaluation = evaluate_posture_metrics(metrics)
    return {
        "image_path": pose_row.get("image_path"),
        "image_name": pose_row.get("image_name"),
        "group": pose_row.get("group"),
        "split": pose_row.get("split"),
        "pose_detected": pose_row.get("pose_detected"),
        **metrics,
        **evaluation,
    }
