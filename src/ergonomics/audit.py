from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .paths import ERGONOMICS_RESULTS_DIR

# Puntos clave (landmarks) que consideramos críticos para validar la calidad de la detección.
# Si estos puntos no son visibles, el análisis ergonómico no es fiable.
DEFAULT_AUDIT_LANDMARKS = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
]

# Diccionario de umbrales de referencia basados en el método ROSA / NTP 1173.
# Sirven para auditar si nuestras reglas actuales son demasiado estrictas o laxas.
DEFAULT_THRESHOLD_REFERENCE = {
    "shoulder_tilt_deg": {"adequate_max": 5.0, "improvable_max": 10.0},
    "shoulder_height_diff_ratio": {"adequate_max": 0.03, "improvable_max": 0.07},
    "head_lateral_offset_ratio": {"adequate_max": 0.08, "improvable_max": 0.16},
    "neck_tilt_deg": {"adequate_max": 8.0, "improvable_max": 15.0},
    "trunk_tilt_deg": {"adequate_max": 6.0, "improvable_max": 12.0},
}


@dataclass(frozen=True)
class RunAuditTables:
    """
    Contenedor inmutable para almacenar todos los DataFrames resultantes 
    de una ejecución (Run). Facilita el acceso centralizado a los resultados.
    """
    run_dir: Path           # Directorio donde residen los datos de la ejecución
    manifest_df: pd.DataFrame # Registro de qué imágenes se procesaron
    pose_df: pd.DataFrame     # Coordenadas y visibilidad de los landmarks
    analysis_df: pd.DataFrame # Métricas ergonómicas calculadas
    merged_df: pd.DataFrame   # Unión de pose + análisis para auditoría cruzada


def load_run_audit_tables(
    run_label: str,
    *,
    results_root: Path = ERGONOMICS_RESULTS_DIR,
) -> RunAuditTables:
    """
    Carga todos los archivos CSV de una ejecución específica y los 
    reconstruye en un objeto RunAuditTables.
    """
    run_dir = results_root / run_label
    # Carga de los 3 pilares de datos del proyecto
    manifest_df = pd.read_csv(run_dir / "execution_manifest.csv")
    pose_df = pd.read_csv(run_dir / "pose_landmarks.csv")
    analysis_df = pd.read_csv(run_dir / "ergonomic_analysis.csv")
    
    # Generamos la tabla maestra combinada
    merged_df = merge_pose_analysis(pose_df, analysis_df)

    return RunAuditTables(
        run_dir=run_dir,
        manifest_df=manifest_df,
        pose_df=pose_df,
        analysis_df=analysis_df,
        merged_df=merged_df,
    )


def merge_pose_analysis(pose_df: pd.DataFrame, analysis_df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza un 'join' inteligente entre los datos de la IA (pose) y la 
    lógica de negocio (análisis), evitando duplicar columnas de metadatos.
    """
    if pose_df.empty or analysis_df.empty:
        return pd.DataFrame()

    # Columnas que suelen repetirse y que queremos limpiar antes del merge
    duplicated_columns = {
        "image_name",
        "group",
        "split",
        "pose_detected",
    }
    
    # Eliminamos duplicados excepto la clave de unión (image_path)
    analysis_payload = analysis_df.drop(columns=[column for column in duplicated_columns if column in analysis_df.columns])
    
    # Unión 1 a 1 para asegurar integridad referencial
    merged_df = pose_df.merge(analysis_payload, on="image_path", how="inner", validate="one_to_one")
    return merged_df


def build_keypoint_visibility_summary(
    pose_df: pd.DataFrame,
    *,
    landmarks: list[str] | None = None,
    visibility_threshold: float = 0.35,
    group_col: str | None = "group",
) -> pd.DataFrame:
    """
    Métrica de calidad: Calcula qué porcentaje de imágenes muestran 
    realmente los puntos clave necesarios para el análisis.
    Fundamental para detectar si la cámara está mal situada o hay poca luz.
    """
    if pose_df.empty:
        base_columns = ["landmark", "visible_images", "total_images", "visibility_pct"]
        if group_col:
            base_columns.insert(0, group_col)
        return pd.DataFrame(columns=base_columns)

    landmarks = landmarks or DEFAULT_AUDIT_LANDMARKS
    rows: list[dict] = []

    # Agrupamos por tipo de postura (adecuada/riesgo) para ver dónde falla más la detección
    if group_col and group_col in pose_df.columns:
        grouped_items = pose_df.groupby(group_col, dropna=False)
    else:
        grouped_items = [("ALL", pose_df)]
        group_col = None

    for group_value, group_df in grouped_items:
        total_images = len(group_df)
        for landmark_name in landmarks:
            visibility_col = f"{landmark_name}_visibility"
            if visibility_col not in group_df.columns:
                continue
            
            # Contamos cuántas imágenes superan el umbral de confianza de MediaPipe
            visible_images = int((group_df[visibility_col].fillna(-1.0) >= visibility_threshold).sum())
            
            row = {
                "landmark": landmark_name,
                "visible_images": visible_images,
                "total_images": total_images,
                "visibility_pct": visible_images / total_images * 100.0 if total_images else 0.0,
            }
            if group_col:
                row[group_col] = group_value
            rows.append(row)

    summary_df = pd.DataFrame(rows)
    sort_columns = ["landmark"] if not group_col else [group_col, "landmark"]
    return summary_df.sort_values(sort_columns).reset_index(drop=True)

def build_component_status_summary(
    analysis_df: pd.DataFrame,
    *,
    component_columns: list[str] | None = None,
    group_col: str | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Genera un resumen estadístico de los estados (adequate, improvable, risk) 
    desglosado por componente corporal (hombros, cabeza, codos, etc.).
    
    Es importante para identificar qué reglas ergonómicas son las más críticas 
    o cuáles podrían estar necesitando una recalibración.
    """
    # Gestión de DataFrames vacíos para evitar errores en el pipeline
    if analysis_df.empty:
        base_columns = ["component", "status", "image_count", "share_pct"]
        if group_col:
            base_columns.insert(0, group_col)
        return pd.DataFrame(columns=base_columns)

    # Definimos los componentes por defecto a auditar si no se especifican otros.
    # Estos corresponden a las columnas de estado generadas por nuestro motor de reglas.
    component_columns = component_columns or [
        "shoulder_status",
        "head_status",
        "trunk_status",
        "left_elbow_status",
        "right_elbow_status",
        "overall_status",
    ]
    
    # Filtramos para asegurar que solo procesamos columnas que realmente existen en el DataFrame
    component_columns = [column for column in component_columns if column in analysis_df.columns]
    rows: list[dict] = []

    # Configuración de la agrupación: podemos analizar el total ("ALL") 
    # o desglosar por grupo (por ejemplo comparar imágenes etiquetadas como 'bien' vs 'mal')
    if group_col and group_col in analysis_df.columns:
        grouped_items = analysis_df.groupby(group_col, dropna=False)
    else:
        grouped_items = [("ALL", analysis_df)]
        group_col = None

    for group_value, group_df in grouped_items:
        for component_name in component_columns:
            # Contamos cuántas apariciones hay de cada estado 
            counts = group_df[component_name].value_counts(dropna=False)
            total = int(counts.sum())
            
            for status_name, image_count in counts.items():
                row = {
                    "component": component_name,
                    "status": status_name,
                    "image_count": int(image_count),
                    # Calculamos el porcentaje sobre el total si la normalización está activa
                    "share_pct": float(image_count) / total * 100.0 if normalize and total else None,
                }
                # Si hay agrupación, añadimos la etiqueta del grupo a la fila
                if group_col:
                    row[group_col] = group_value
                rows.append(row)

    # Convertimos la lista de resultados en un DataFrame final ordenado para su lectura
    summary_df = pd.DataFrame(rows)
    sort_columns = ["component", "status"] if not group_col else [group_col, "component", "status"]
    return summary_df.sort_values(sort_columns).reset_index(drop=True)


def build_label_status_matrix(
    analysis_df: pd.DataFrame,
    *,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Construye una matriz de contraste entre la etiqueta real del dataset (group) 
    y la predicción de nuestro sistema (overall_status).
    
    Es el equivalente a una Matriz de Confusión. Permite detectar si el sistema 
    está clasificando como 'riesgo' imágenes que el dataset considera 'adecuadas'.
    """
    if analysis_df.empty:
        return pd.DataFrame()

    # Agrupamos por grupo (etiqueta real) y estado calculado (predicción)
    matrix_df = (
        analysis_df.groupby(["group", "overall_status"], dropna=False)
        .size()                # Contamos ocurrencias
        .unstack(fill_value=0) # Pivotamos para crear la forma de matriz
        .sort_index()
    )
    
    # Si se activa la normalización, convertimos los valores en porcentajes por fila.
    # Esto facilita ver la tasa de acierto/error independientemente del volumen de datos.
    if normalize:
        # Dividimos cada celda por la suma de su fila (axis=1)
        matrix_df = matrix_df.div(matrix_df.sum(axis=1).replace(0, 1), axis=0) * 100.0
        
    return matrix_df.reset_index()



def build_metric_quantile_summary(
    analysis_df: pd.DataFrame,
    *,
    metrics: list[str],
    group_col: str = "group",
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> pd.DataFrame:
    """
    Realiza un análisis de distribución estadística (cuantiles) para métricas específicas.
    
    Sirve para 'calibrar' el sistema: si el 90% de las personas en el grupo 'adecuado' 
    tienen una inclinación de cuello de 7 grados, sabemos que nuestro umbral 
    debería estar cerca de ese valor.
    """
    if analysis_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    
    # Agrupamos por la columna de categoría (por defecto 'group')
    grouped_items = analysis_df.groupby(group_col, dropna=False) if group_col in analysis_df.columns else [("ALL", analysis_df)]
    
    for group_value, group_df in grouped_items:
        for metric_name in metrics:
            if metric_name not in group_df.columns:
                continue
            
            # Limpiamos nulos para asegurar cálculos precisos
            series = group_df[metric_name].dropna()
            if series.empty:
                continue
                
            # Calculamos estadísticas base: media y conteo de muestras
            row = {
                group_col: group_value,
                "metric": metric_name,
                "non_null_count": int(series.shape[0]),
                "mean": float(series.mean()),
            }
            
            # Calculamos cada cuantil solicitado (p10, p25, p50, etc.)
            # Esto ayuda a ver la dispersión de los ángulos ergonómicos.
            for quantile in quantiles:
                row[f"q{int(quantile * 100):02d}"] = float(series.quantile(quantile))
            
            rows.append(row)
            
    return pd.DataFrame(rows).sort_values([group_col, "metric"]).reset_index(drop=True)


def build_reference_threshold_candidates(
    analysis_df: pd.DataFrame,
    *,
    reference_group: str = "looks good",
    metric_thresholds: dict[str, dict[str, float]] | None = None,
    adequate_quantile: float = 0.75,
    improvable_quantile: float = 0.9,
) -> pd.DataFrame:
    """
    SISTEMA DE AUTOCALIBRACIÓN:
    Esta función sugiere nuevos umbrales (thresholds) basados en los datos reales.
    
    Toma como referencia al grupo que 'se ve bien' (looks good) y analiza sus ángulos.
    Si el 75% de la gente que está bien sentada tiene un ángulo de 6º, el sistema 
    sugiere que 6º debería ser el nuevo límite para 'Adecuado'.
    """
    if analysis_df.empty:
        return pd.DataFrame()

    # Usamos los umbrales actuales por defecto para comparar
    metric_thresholds = metric_thresholds or DEFAULT_THRESHOLD_REFERENCE
    
    # Filtramos solo las personas que el dataset etiqueta como "correctas"
    reference_df = analysis_df[analysis_df["group"] == reference_group].copy()
    if reference_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for metric_name, current_thresholds in metric_thresholds.items():
        if metric_name not in reference_df.columns:
            continue
        series = reference_df[metric_name].dropna()
        if series.empty:
            continue
            
        # PROPOSICIÓN DE CANDIDATOS:
        # Calculamos los cuantiles (por defecto el 75 y el 90) para proponer
        # límites que se ajusten a la realidad física captada por la cámara.
        rows.append(
            {
                "metric": metric_name,
                "reference_group": reference_group,
                "non_null_count": int(series.shape[0]),
                "current_adequate_max": current_thresholds.get("adequate_max"),
                "current_improvable_max": current_thresholds.get("improvable_max"),
                "candidate_adequate_max": float(series.quantile(adequate_quantile)),
                "candidate_improvable_max": float(series.quantile(improvable_quantile)),
                "reference_mean": float(series.mean()),
                "reference_median": float(series.median()),
            }
        )
    return pd.DataFrame(rows).sort_values("metric").reset_index(drop=True)


def select_audit_examples(
    merged_df: pd.DataFrame,
    *,
    filters: dict[str, object] | None = None,
    sort_by: str | list[str] | None = None,
    ascending: bool | list[bool] = False,
    limit: int = 9,
) -> pd.DataFrame:
    """
    BUSCADOR DE CASOS CRÍTICOS:
    Permite filtrar y extraer ejemplos específicos para inspección visual.
    
    Útil para encontrar, por ejemplo: "Las 9 imágenes con más inclinación 
    de hombros que el sistema ha marcado como Riesgo".
    """
    if merged_df.empty:
        return pd.DataFrame()

    selected_df = merged_df.copy()
    
    # Aplicación de filtros dinámicos (por grupo, por estado, por split, etc.)
    for column_name, expected_value in (filters or {}).items():
        if column_name not in selected_df.columns:
            continue
        if isinstance(expected_value, (list, tuple, set)):
            selected_df = selected_df[selected_df[column_name].isin(expected_value)]
        else:
            selected_df = selected_df[selected_df[column_name] == expected_value]

    # Ordenamos los resultados (ej: de mayor a menor riesgo)
    if sort_by is not None:
        selected_df = selected_df.sort_values(sort_by, ascending=ascending, na_position="last")

    # Devolvemos una muestra limitada para no saturar la visualización
    return selected_df.head(limit).reset_index(drop=True)
