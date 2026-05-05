from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .paths import ERGONOMICS_RESULTS_DIR
from .pose_inference import MediaPipePoseConfig, MediaPipePoseEstimator
from .posture_rules import analyze_pose_row
from .lateral_rules import analyze_lateral_pose_row
from .yolo_pose_inference import YoloPoseConfig, YoloPoseEstimator
from .reporting import (
    build_group_status_summary,
    build_metric_summary_by_group,
    build_status_summary,
    save_dataframe,
)

# --- UTILIDADES DE INTERFAZ ---

def progress(iterable, **kwargs):
    """
    Envuelve un iterable con una barra de progreso (tqdm).
    Si la librería tqdm no está instalada, devuelve el iterable normal 
    para no romper la ejecución.
    """
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kwargs)
    except Exception:
        return iterable

# --- GESTIÓN DE RESULTADOS (ARTEFACTOS) ---

@dataclass(frozen=True)
class LongRunArtifacts:
    """
    Estructura de datos que rastrea todas las rutas de salida de una 'Gran Ejecución'.
    Centraliza dónde se guardan los landmarks, el análisis y los informes finales.
    """
    output_dir: Path
    manifest_path: Path
    pose_path: Path
    analysis_path: Path
    status_summary_path: Path
    group_status_summary_path: Path
    metric_summary_path: Path
    processed_images: int

# --- LÓGICA DE PERSISTENCIA INCREMENTAL ---
# Estas funciones permiten que el sistema no pierda datos si se corta la luz 
# o se cierra el programa, ya que escriben en el disco paso a paso.

def _append_rows(path: Path, rows: list[dict]) -> None:
    """
    Escribe filas en un CSV de forma incremental (modo 'append').
    Si el archivo no existe, escribe la cabecera; si existe, solo añade los datos.
    Esto evita cargar todo en RAM, permitiendo procesar datasets gigantescos.
    """
    if not rows:
        return
    frame = pd.DataFrame(rows)
    write_header = not path.exists()
    frame.to_csv(path, mode="a", header=write_header, index=False)


def _load_processed_image_paths(path: Path) -> set[str]:
    """
    SISTEMA DE CHECKPOINT (Punto de control):
    Lee el archivo de resultados actual para saber qué imágenes ya han sido procesadas.
    Devuelve un 'set' para una búsqueda ultra rápida.
    """
    if not path.exists():
        return set()
    try:
        # Solo leemos la columna de rutas para ahorrar memoria
        existing = pd.read_csv(path, usecols=["image_path"])
    except Exception:
        return set()
    return set(existing["image_path"].astype(str))


def _load_dataframe(path: Path) -> pd.DataFrame:
    """Carga de seguridad de un DataFrame desde CSV."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def run_incremental_long_pipeline(
    records_df: pd.DataFrame,
    *,
    run_label: str,                     # Nombre de la carpeta de resultados (ej: "test_inicial")
    pose_config: MediaPipePoseConfig,    # Configuración de la IA
    visibility_threshold: float = 0.35, # Confianza mínima para los puntos clave
    checkpoint_every: int = 100,        # Guardar en disco cada 100 imágenes
    resume: bool = True,                # Si es True, no repite lo que ya está hecho
    output_root: Path = ERGONOMICS_RESULTS_DIR,
) -> LongRunArtifacts:
    """
    Ejecuta el pipeline completo de detección y análisis ergonómico sobre un dataset.
    Está diseñado para ser eficiente en memoria y permitir la recuperación ante fallos.
    """
    # 1. PREPARACIÓN DE DIRECTORIOS Y MANIFIESTO
    output_dir = output_root / run_label
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "execution_manifest.csv"
    pose_path = output_dir / "pose_landmarks.csv"
    analysis_path = output_dir / "ergonomic_analysis.csv"
    status_summary_path = output_dir / "status_summary.csv"
    group_status_summary_path = output_dir / "group_status_summary.csv"
    metric_summary_path = output_dir / "metric_summary_by_group.csv"

    # Guardamos el registro de qué imágenes vamos a procesar originalmente
    records_df = records_df.copy()
    records_df["image_path"] = records_df["image_path"].astype(str)
    save_dataframe(records_df, manifest_path)

    # 2. LÓGICA DE REANUDACIÓN (RESUME)
    if resume:
        # Cargamos lo que ya se hizo para no repetirlo
        processed = _load_processed_image_paths(pose_path)
        pending_df = records_df[~records_df["image_path"].isin(processed)].copy()
    else:
        # Si no reanudamos, borramos archivos previos para empezar de cero
        pending_df = records_df.copy()
        for path in [pose_path, analysis_path, status_summary_path, group_status_summary_path, metric_summary_path]:
            if path.exists():
                path.unlink()

    pose_rows: list[dict] = []
    analysis_rows: list[dict] = []

    # 3. BUCLE PRINCIPAL DE PROCESAMIENTO
    # Usamos un context manager para asegurar que el modelo se cierra correctamente
    with MediaPipePoseEstimator(config=pose_config) as estimator:
        for item in progress(
            pending_df.to_dict(orient="records"),
            total=len(pending_df),
            desc="Long run ergonomico",
        ):
            # A. Inferencia: La IA busca el esqueleto
            pose_row = estimator.infer_image(
                item["image_path"],
                metadata={"group": item.get("group"), "split": item.get("split")},
            )
            
            # B. Análisis: Calculamos ángulos y estados ergonómicos
            analysis_row = analyze_pose_row(
                pose_row,
                visibility_threshold=visibility_threshold,
            )
            
            pose_rows.append(pose_row)
            analysis_rows.append(analysis_row)

            # 4. GUARDADO INCREMENTAL (CHECKPOINTING)
            # Si acumulamos suficientes filas, escribimos en disco y vaciamos la RAM
            if len(pose_rows) >= checkpoint_every:
                _append_rows(pose_path, pose_rows)
                _append_rows(analysis_path, analysis_rows)
                pose_rows.clear()
                analysis_rows.clear()

    # Escribimos los restos finales que quedaron después del último checkpoint
    _append_rows(pose_path, pose_rows)
    _append_rows(analysis_path, analysis_rows)

    # 5. GENERACIÓN DE INFORMES FINALES (REPORTING)
    # Una vez procesado todo, cargamos los resultados para generar estadísticas globales
    analysis_df = _load_dataframe(analysis_path)
    
    status_summary_df = build_status_summary(analysis_df)
    group_status_summary_df = build_group_status_summary(analysis_df)
    metric_summary_df = build_metric_summary_by_group(
        analysis_df,
        metrics=[
            "shoulder_tilt_deg",
            "shoulder_height_diff_ratio",
            "head_lateral_offset_ratio",
            "neck_tilt_deg",
            "trunk_tilt_deg",
            "left_elbow_angle_deg",
            "right_elbow_angle_deg",
        ],
    )

    # Guardamos los informes estadísticos
    save_dataframe(status_summary_df, status_summary_path)
    save_dataframe(group_status_summary_df, group_status_summary_path)
    save_dataframe(metric_summary_df, metric_summary_path)

    # Devolvemos un objeto con todas las rutas y el conteo final de éxito
    return LongRunArtifacts(
        output_dir=output_dir,
        manifest_path=manifest_path,
        pose_path=pose_path,
        analysis_path=analysis_path,
        status_summary_path=status_summary_path,
        group_status_summary_path=group_status_summary_path,
        metric_summary_path=metric_summary_path,
        processed_images=len(analysis_df),
    )


def run_incremental_lateral_yolo_pipeline(
    records_df: pd.DataFrame,
    *,
    run_label: str,
    yolo_config: YoloPoseConfig,
    visibility_threshold: float = 0.3,
    checkpoint_every: int = 100,
    resume: bool = True,
    output_root: Path = ERGONOMICS_RESULTS_DIR,
) -> LongRunArtifacts:
    """
    Ejecuta el pipeline lateral con YOLO Pose.
    Mantiene los mismos artefactos que la corrida frontal para facilitar comparaciones.
    """
    output_dir = output_root / run_label
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "execution_manifest.csv"
    pose_path = output_dir / "pose_landmarks.csv"
    analysis_path = output_dir / "ergonomic_analysis.csv"
    status_summary_path = output_dir / "status_summary.csv"
    group_status_summary_path = output_dir / "group_status_summary.csv"
    metric_summary_path = output_dir / "metric_summary_by_group.csv"

    records_df = records_df.copy()
    records_df["image_path"] = records_df["image_path"].astype(str)
    save_dataframe(records_df, manifest_path)

    if resume:
        processed = _load_processed_image_paths(pose_path)
        pending_df = records_df[~records_df["image_path"].isin(processed)].copy()
    else:
        pending_df = records_df.copy()
        for path in [pose_path, analysis_path, status_summary_path, group_status_summary_path, metric_summary_path]:
            if path.exists():
                path.unlink()

    pose_rows: list[dict] = []
    analysis_rows: list[dict] = []

    with YoloPoseEstimator(config=yolo_config) as estimator:
        for item in progress(
            pending_df.to_dict(orient="records"),
            total=len(pending_df),
            desc="Long run lateral YOLO",
        ):
            pose_row = estimator.infer_image(
                item["image_path"],
                metadata={"group": item.get("group"), "split": item.get("split")},
            )
            analysis_row = analyze_lateral_pose_row(
                pose_row,
                visibility_threshold=visibility_threshold,
            )

            pose_rows.append(pose_row)
            analysis_rows.append(analysis_row)

            if len(pose_rows) >= checkpoint_every:
                _append_rows(pose_path, pose_rows)
                _append_rows(analysis_path, analysis_rows)
                pose_rows.clear()
                analysis_rows.clear()

    _append_rows(pose_path, pose_rows)
    _append_rows(analysis_path, analysis_rows)

    analysis_df = _load_dataframe(analysis_path)
    status_summary_df = build_status_summary(analysis_df)
    group_status_summary_df = build_group_status_summary(analysis_df)
    metric_summary_df = build_metric_summary_by_group(
        analysis_df,
        metrics=[
            "head_forward_offset_ratio",
            "neck_forward_tilt_deg",
            "trunk_forward_tilt_deg",
            "shoulder_hip_offset_ratio",
            "lateral_elbow_angle_deg",
        ],
    )

    save_dataframe(status_summary_df, status_summary_path)
    save_dataframe(group_status_summary_df, group_status_summary_path)
    save_dataframe(metric_summary_df, metric_summary_path)

    return LongRunArtifacts(
        output_dir=output_dir,
        manifest_path=manifest_path,
        pose_path=pose_path,
        analysis_path=analysis_path,
        status_summary_path=status_summary_path,
        group_status_summary_path=group_status_summary_path,
        metric_summary_path=metric_summary_path,
        processed_images=len(analysis_df),
    )
