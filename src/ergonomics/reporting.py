from __future__ import annotations

from pathlib import Path
import pandas as pd

# --- GENERACIÓN DE RESÚMENES ESTADÍSTICOS ---

def build_status_summary(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """
    RESUMEN GLOBAL:
    Cuenta cuántas imágenes hay en cada estado ergonómico (Adecuado, Riesgo, etc.)
    y calcula su porcentaje sobre el total.
    """
    if analysis_df.empty:
        return pd.DataFrame(columns=["overall_status", "image_count", "share_pct"])

    # Contamos las ocurrencias de cada estado (ej: 'adequate', 'risk')
    summary = analysis_df["overall_status"].value_counts(dropna=False).rename("image_count").reset_index()
    summary.columns = ["overall_status", "image_count"]
    
    # Calculamos el porcentaje relativo
    summary["share_pct"] = summary["image_count"] / summary["image_count"].sum() * 100.0
    return summary


def build_group_status_summary(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """
    RESUMEN POR GRUPOS:
    Desglosa el estado ergonómico según el grupo.
    Permite comparar qué grupos de trabajadores están en mayor riesgo.
    """
    if analysis_df.empty:
        return pd.DataFrame(columns=["group", "overall_status", "image_count", "share_pct"])

    # Agrupamos por grupo y estado para obtener los tamaños de cada segmento
    grouped = (
        analysis_df.groupby(["group", "overall_status"], dropna=False)
        .size()
        .rename("image_count")
        .reset_index()
    )
    
    # Calculamos el porcentaje dentro de cada grupo para una comparativa justa
    totals = grouped.groupby("group")["image_count"].transform("sum")
    grouped["share_pct"] = grouped["image_count"] / totals * 100.0
    
    return grouped.sort_values(["group", "overall_status"]).reset_index(drop=True)


def build_metric_summary_by_group(
    analysis_df: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """
    RESUMEN DE MÉTRICAS FÍSICAS:
    Calcula el valor promedio de los ángulos y distancias detectadas para cada grupo.
    Útil para saber, por ejemplo, cuál es la inclinación media del cuello en un dataset.
    """
    if analysis_df.empty:
        return pd.DataFrame()

    # Filtramos para usar solo las métricas que realmente existen en los datos
    available_metrics = [metric for metric in metrics if metric in analysis_df.columns]
    if not available_metrics:
        return pd.DataFrame()

    # Calculamos la media aritmética de cada métrica por grupo
    summary = analysis_df.groupby("group", dropna=False)[available_metrics].mean().reset_index()
    return summary.sort_values("group").reset_index(drop=True)


# --- UTILIDADES DE PERSISTENCIA ---

def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """
    GUARDADO SEGURO:
    Exporta los resultados a un archivo CSV.
    Se asegura de crear las carpetas necesarias si no existen (mkdir -p).
    """
    path = Path(path)
    # Creamos el árbol de directorios si es necesario para evitar errores de escritura
    path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(path, index=False)
    return path