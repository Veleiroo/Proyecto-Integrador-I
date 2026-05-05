"""
PAQUETE ERGONOMICS
------------------
Este archivo centraliza el acceso a todos los submódulos de la librería, 
permitiendo una API limpia y organizada. 

Aquí se define qué funciones y clases son "públicas" para ser usadas 
en los notebooks y scripts de producción.
"""

# --- Módulo de Auditoría: Control de calidad y validación de datos ---
from .audit import (
    DEFAULT_AUDIT_LANDMARKS,
    DEFAULT_THRESHOLD_REFERENCE,
    RunAuditTables,
    build_component_status_summary,
    build_keypoint_visibility_summary,
    build_label_status_matrix,
    build_metric_quantile_summary,
    build_reference_threshold_candidates,
    load_run_audit_tables,
    merge_pose_analysis,
    select_audit_examples,
)

# --- Módulo de Benchmark: Comparativa y selección de modelos de IA ---
from .benchmark import (
    BenchmarkArtifacts,
    ModelDecision,
    choose_reference_models,
    load_benchmark_artifacts,
    plot_group_comparison,
    plot_model_overview,
    rank_models,
)

# --- Módulo de Datasets: Gestión del catálogo de imágenes y muestras ---
from .datasets import (
    DATASET_CATALOG,
    DatasetSpec,
    collect_image_records_df,
    plot_dataset_group_distribution,
    plot_dataset_volumes,
    summarize_available_datasets,
    summarize_dataset_groups,
)

# --- Módulo de Ejecución Larga (Long Run): Procesamiento por lotes incremental ---
from .long_run import LongRunArtifacts, run_incremental_long_pipeline
from .long_run import run_incremental_lateral_yolo_pipeline

# --- Módulo de Rutas: Gestión centralizada de directorios y archivos del sistema ---
from .paths import (
    ERGONOMICS_RESULTS_DIR,
    MEDIAPIPE_TASK_MODEL_PATH,
    POSE_BENCHMARK_RESULTS_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    YOLO_POSE_WEIGHTS_PATH,
)

# --- Módulo de Inferencia: El motor de detección de pose (MediaPipe) ---
from .pose_inference import (
    LANDMARK_IDS,
    MediaPipePoseConfig,
    MediaPipePoseEstimator,
    run_mediapipe_pose_batch,
)

# --- Módulo de Reglas Posturales: El cerebro ergonómico (Trigonometría y normas ROSA) ---
from .posture_rules import (
    analyze_pose_dataframe,
    analyze_pose_row,
    evaluate_posture_metrics,
    extract_posture_metrics,
)

# --- Módulo de Reglas Laterales: análisis de perfil con YOLO Pose ---
from .lateral_rules import (
    analyze_lateral_pose_dataframe,
    analyze_lateral_pose_row,
    evaluate_lateral_posture_metrics,
    extract_lateral_posture_metrics,
)

# --- Módulo de Reportes: Generación de estadísticas y guardado de resultados ---
from .reporting import (
    build_group_status_summary,
    build_metric_summary_by_group,
    build_status_summary,
    save_dataframe,
)

# --- Módulo de Muestreo: Lógica para balancear y seleccionar datos de prueba ---
from .sampling import select_balanced_sample, select_execution_records

# --- Módulo de Visualización: Herramientas gráficas y dibujos sobre imagen ---
from .visualization import (
    draw_pose_overlay,
    plot_keypoint_coverage_heatmap,
    plot_metric_by_group,
    plot_pose_gallery,
    plot_status_by_group,
    plot_status_distribution,
)

# --- Módulo de Inferencia YOLO: extracción de landmarks para vista lateral ---
from .yolo_pose_inference import (
    YOLO_LANDMARK_IDS,
    YoloPoseConfig,
    YoloPoseEstimator,
    run_yolo_pose_batch,
)

# --- EXPOSICIÓN DE LA API PÚBLICA ---
# La lista __all__ define qué elementos se exportan cuando alguien hace 
# 'from ergonomics import *'. Asegura la encapsulación del código.
__all__ = [
    "DEFAULT_AUDIT_LANDMARKS",
    "DEFAULT_THRESHOLD_REFERENCE",
    "BenchmarkArtifacts",
    "DATASET_CATALOG",
    "DatasetSpec",
    "ERGONOMICS_RESULTS_DIR",
    "LANDMARK_IDS",
    "LongRunArtifacts",
    "MEDIAPIPE_TASK_MODEL_PATH",
    "MediaPipePoseConfig",
    "MediaPipePoseEstimator",
    "ModelDecision",
    "POSE_BENCHMARK_RESULTS_DIR",
    "PROJECT_ROOT",
    "RAW_DATA_DIR",
    "RunAuditTables",
    "YOLO_LANDMARK_IDS",
    "YOLO_POSE_WEIGHTS_PATH",
    "YoloPoseConfig",
    "YoloPoseEstimator",
    "analyze_lateral_pose_dataframe",
    "analyze_lateral_pose_row",
    "analyze_pose_dataframe",
    "analyze_pose_row",
    "build_component_status_summary",
    "build_group_status_summary",
    "build_keypoint_visibility_summary",
    "build_label_status_matrix",
    "build_metric_quantile_summary",
    "build_metric_summary_by_group",
    "build_reference_threshold_candidates",
    "build_status_summary",
    "choose_reference_models",
    "collect_image_records_df",
    "draw_pose_overlay",
    "evaluate_posture_metrics",
    "evaluate_lateral_posture_metrics",
    "extract_lateral_posture_metrics",
    "extract_posture_metrics",
    "load_benchmark_artifacts",
    "plot_dataset_group_distribution",
    "plot_dataset_volumes",
    "plot_group_comparison",
    "plot_keypoint_coverage_heatmap",
    "plot_metric_by_group",
    "plot_status_by_group",
    "plot_model_overview",
    "plot_pose_gallery",
    "plot_status_distribution",
    "rank_models",
    "run_incremental_long_pipeline",
    "run_incremental_lateral_yolo_pipeline",
    "run_mediapipe_pose_batch",
    "run_yolo_pose_batch",
    "save_dataframe",
    "load_run_audit_tables",
    "merge_pose_analysis",
    "select_audit_examples",
    "select_balanced_sample",
    "select_execution_records",
    "summarize_available_datasets",
    "summarize_dataset_groups",
]
