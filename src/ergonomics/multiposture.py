from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .paths import RAW_DATA_DIR


MULTIPOSTURE_DIR = RAW_DATA_DIR / "multiposture_zenodo_14230872"
MULTIPOSTURE_CSV_PATH = MULTIPOSTURE_DIR / "data.csv"

UPPERBODY_LABELS = {
    "TUP": "upright_trunk",
    "TLF": "trunk_leaning_forward",
    "TLB": "trunk_leaning_backward",
    "TLR": "trunk_leaning_right",
    "TLL": "trunk_leaning_left",
}


def load_multiposture_dataframe(csv_path: str | Path = MULTIPOSTURE_CSV_PATH) -> pd.DataFrame:
    """Carga el CSV de MultiPosture descargado desde Zenodo."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontro MultiPosture en {csv_path}. Descarga data.csv desde Zenodo 14230872."
        )
    return pd.read_csv(csv_path)


def _point_3d(row: pd.Series, landmark_name: str) -> tuple[float, float, float]:
    return (
        float(row[f"{landmark_name}_x"]),
        float(row[f"{landmark_name}_y"]),
        float(row[f"{landmark_name}_z"]),
    )


def _midpoint_3d(point_a: tuple[float, float, float], point_b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        (point_a[0] + point_b[0]) / 2.0,
        (point_a[1] + point_b[1]) / 2.0,
        (point_a[2] + point_b[2]) / 2.0,
    )


def _distance_3d(point_a: tuple[float, float, float], point_b: tuple[float, float, float]) -> float:
    return math.sqrt(
        (point_b[0] - point_a[0]) ** 2
        + (point_b[1] - point_a[1]) ** 2
        + (point_b[2] - point_a[2]) ** 2
    )


def _angle_from_vertical(delta_horizontal: float, delta_vertical: float) -> float:
    if delta_vertical == 0.0:
        return 90.0
    return math.degrees(math.atan2(abs(delta_horizontal), abs(delta_vertical)))


def extract_multiposture_metrics(row: pd.Series) -> dict:
    """
    Calcula variables 3D comparables con las reglas laterales.
    En este dataset, y es el eje vertical y z representa profundidad.
    """
    nose = _point_3d(row, "nose")
    left_shoulder = _point_3d(row, "left_shoulder")
    right_shoulder = _point_3d(row, "right_shoulder")
    left_hip = _point_3d(row, "left_hip")
    right_hip = _point_3d(row, "right_hip")

    shoulder_center = _midpoint_3d(left_shoulder, right_shoulder)
    hip_center = _midpoint_3d(left_hip, right_hip)
    torso_length_3d = _distance_3d(shoulder_center, hip_center)

    trunk_forward_tilt_3d_deg = _angle_from_vertical(
        shoulder_center[2] - hip_center[2],
        shoulder_center[1] - hip_center[1],
    )
    trunk_lateral_tilt_3d_deg = _angle_from_vertical(
        shoulder_center[0] - hip_center[0],
        shoulder_center[1] - hip_center[1],
    )
    shoulder_hip_depth_offset_ratio = (
        abs(shoulder_center[2] - hip_center[2]) / torso_length_3d if torso_length_3d else math.nan
    )
    head_forward_offset_3d_ratio = (
        abs(nose[2] - shoulder_center[2]) / torso_length_3d if torso_length_3d else math.nan
    )
    neck_forward_tilt_3d_deg = _angle_from_vertical(
        nose[2] - shoulder_center[2],
        nose[1] - shoulder_center[1],
    )

    return {
        "subject": row.get("subject"),
        "upperbody_label": row.get("upperbody_label"),
        "upperbody_label_name": UPPERBODY_LABELS.get(row.get("upperbody_label"), row.get("upperbody_label")),
        "lowerbody_label": row.get("lowerbody_label"),
        "torso_length_3d": torso_length_3d,
        "trunk_forward_tilt_3d_deg": trunk_forward_tilt_3d_deg,
        "trunk_lateral_tilt_3d_deg": trunk_lateral_tilt_3d_deg,
        "shoulder_hip_depth_offset_ratio": shoulder_hip_depth_offset_ratio,
        "head_forward_offset_3d_ratio": head_forward_offset_3d_ratio,
        "neck_forward_tilt_3d_deg": neck_forward_tilt_3d_deg,
    }


def build_multiposture_metric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte el CSV crudo en una tabla de metricas ergonomicas 3D."""
    if df.empty:
        return pd.DataFrame()
    return pd.DataFrame([extract_multiposture_metrics(row) for _, row in df.iterrows()])


def summarize_multiposture_metrics(
    metric_df: pd.DataFrame,
    *,
    metrics: list[str] | None = None,
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> pd.DataFrame:
    """Resumen por etiqueta experta para usarlo como apoyo de calibracion."""
    if metric_df.empty:
        return pd.DataFrame()

    metrics = metrics or [
        "trunk_forward_tilt_3d_deg",
        "trunk_lateral_tilt_3d_deg",
        "shoulder_hip_depth_offset_ratio",
        "head_forward_offset_3d_ratio",
        "neck_forward_tilt_3d_deg",
    ]
    rows: list[dict] = []
    for label, group_df in metric_df.groupby("upperbody_label", dropna=False):
        for metric in metrics:
            if metric not in group_df.columns:
                continue
            series = group_df[metric].dropna()
            if series.empty:
                continue
            row = {
                "upperbody_label": label,
                "upperbody_label_name": UPPERBODY_LABELS.get(label, label),
                "metric": metric,
                "count": int(series.shape[0]),
                "mean": float(series.mean()),
            }
            for quantile in quantiles:
                row[f"q{int(quantile * 100):02d}"] = float(series.quantile(quantile))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["metric", "upperbody_label"]).reset_index(drop=True)
