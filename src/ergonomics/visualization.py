from __future__ import annotations

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import pandas as pd

from .pose_inference import LANDMARK_IDS, SKELETON_SEGMENTS

# --- CONFIGURACIÓN ESTÉTICA ---
# Definimos una paleta de colores coherente para cada estado ergonómico.
STATUS_COLORS = {
    "adequate": "#2a9d8f",         # Verde
    "improvable": "#f4a261",      # Naranja 
    "risk": "#e76f51",            # Rojo 
    "insufficient_data": "#8d99ae", # Gris 
}

# --- UTILIDADES DE IMAGEN ---

def load_rgb_image(image_path: str | Path):
    """
    Carga una imagen y la convierte a RGB.
    OpenCV lee en BGR por defecto, pero Matplotlib necesita RGB para mostrar colores reales
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --- DIBUJO DE POSES (OVERLAYS) ---

def draw_pose_overlay(image_path: str | Path, pose_row: pd.Series | dict, ax=None):
    """
    Dibuja el esqueleto detectado sobre la imagen original
    Muestra los puntos clave (nodos) y las conexiones (huesos)
    """
    image = load_rgb_image(image_path)
    height, width = image.shape[:2]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    ax.imshow(image)

    # 1. Dibujar los puntos (Landmarks)
    points = {}
    for landmark_name in LANDMARK_IDS:
        x = pose_row.get(f"{landmark_name}_x")
        y = pose_row.get(f"{landmark_name}_y")
        visibility = pose_row.get(f"{landmark_name}_visibility")
        
        # Solo dibujamos si el punto es válido y visible (umbral 0.35)
        if x is None or y is None or pd.isna(x) or pd.isna(y) or visibility is None or pd.isna(visibility):
            continue
        if float(visibility) < 0.35:
            continue
            
        # Convertimos coordenadas normalizadas (0-1) a píxeles reales
        pixel_point = (float(x) * width, float(y) * height)
        points[landmark_name] = pixel_point
        ax.scatter(pixel_point[0], pixel_point[1], s=36, color="#00b4d8")

    # 2. Dibujar las conexiones del esqueleto (Segments)
    for start_name, end_name in SKELETON_SEGMENTS:
        if start_name in points and end_name in points:
            ax.plot(
                [points[start_name][0], points[end_name][0]],
                [points[start_name][1], points[end_name][1]],
                color="#ffbe0b",
                linewidth=2.0,
            )

    ax.set_title(Path(image_path).name)
    ax.axis("off")
    fig.tight_layout()
    return fig, ax

# --- GRÁFICOS ESTADÍSTICOS (PLOTS) ---

def plot_status_distribution(analysis_df: pd.DataFrame, ax=None):
    """Genera un gráfico de barras con la distribución global de estados ergonómicos"""
    if analysis_df.empty:
        raise ValueError("No hay analisis ergonomico para representar.")

    # Contamos estados y reindexamos para mantener el orden de colores
    counts = analysis_df["overall_status"].value_counts().reindex(STATUS_COLORS.keys(), fill_value=0)
    counts = counts[counts > 0] # Solo mostramos estados que existan en el set actual

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
    else:
        fig = ax.figure

    colors = [STATUS_COLORS[label] for label in counts.index]
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title("Distribucion del estado ergonomico")
    ax.set_ylabel("Numero de imagenes")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_status_by_group(analysis_df: pd.DataFrame, *, normalize: bool = True, ax=None):
    """
    Crea un gráfico de barras apiladas para comparar el riesgo entre diferentes grupos
    Permite ver qué departamento o tipo de silla genera más posturas de riesgo
    """
    if analysis_df.empty:
        raise ValueError("No hay analisis ergonomico para representar.")

    # Tabulamos los datos para cruzarlos por grupo y estado
    grouped = (
        analysis_df.groupby(["group", "overall_status"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=STATUS_COLORS.keys(), fill_value=0)
        .sort_index()
    )
    grouped = grouped.loc[:, grouped.sum(axis=0) > 0] # Limpiamos columnas vacías
    
    # Si normalizamos, mostramos porcentajes (0-100%) en lugar de conteos
    if normalize:
        grouped = grouped.div(grouped.sum(axis=1).replace(0, 1), axis=0) * 100.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
    else:
        fig = ax.figure

    colors = [STATUS_COLORS[label] for label in grouped.columns]
    grouped.plot(kind="bar", stacked=True, ax=ax, color=colors)
    ax.set_title("Distribucion del estado ergonomico por grupo")
    ax.set_xlabel("Grupo")
    ax.set_ylabel("% de imagenes" if normalize else "Numero de imagenes")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_metric_by_group(analysis_df: pd.DataFrame, metric: str, title: str, ax=None):
    """Muestra el valor promedio de una métrica física (ej. inclinación del cuello) por grupo"""
    if analysis_df.empty:
        raise ValueError("No hay analisis ergonomico para representar.")
    if metric not in analysis_df.columns:
        raise KeyError(f"La metrica {metric} no existe en el analisis.")

    grouped = (
        analysis_df.groupby("group", dropna=False)[metric]
        .mean()
        .dropna()
        .sort_values(ascending=False)
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.2))
    else:
        fig = ax.figure

    ax.bar(grouped.index, grouped.values, color="#457b9d")
    ax.set_title(title)
    ax.set_xlabel("Grupo")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_keypoint_coverage_heatmap(coverage_df: pd.DataFrame, *, group_col: str = "group", ax=None):
    """
    Crea un mapa de calor que muestra la visibilidad de los puntos clave
    Ayuda a detectar si la cámara está mal colocada (ej. si nunca se ven las muñecas)
    """
    if coverage_df.empty:
        raise ValueError("No hay cobertura de keypoints para representar.")
    if group_col not in coverage_df.columns:
        raise KeyError(f"La columna {group_col} no existe en la cobertura.")

    pivot = (
        coverage_df.pivot(index="landmark", columns=group_col, values="visibility_pct")
        .fillna(0.0)
        .sort_index()
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
    else:
        fig = ax.figure

    image = ax.imshow(pivot.values, cmap="viridis", aspect="auto", vmin=0.0, vmax=100.0)
    ax.set_title("Cobertura de keypoints visibles (%)")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Añadimos el valor numérico dentro de cada celda para mayor claridad
    for row_index, landmark_name in enumerate(pivot.index):
        for col_index, group_name in enumerate(pivot.columns):
            value = pivot.loc[landmark_name, group_name]
            ax.text(col_index, row_index, f"{value:.1f}", ha="center", va="center", color="white", fontsize=8)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("% visible")
    fig.tight_layout()
    return fig, ax

# --- GALERÍA DE RESULTADOS ---

def plot_pose_gallery(gallery_df: pd.DataFrame, *, caption_fields: list[str] | None = None, title: str | None = None, ncols: int = 3):
    """
    Genera una cuadrícula de imágenes con sus esqueletos y datos ergonómicos
    Es el resumen visual definitivo para validar los resultados del sistema
    """
    if gallery_df.empty:
        raise ValueError("No hay casos para representar en la galeria.")

    caption_fields = caption_fields or []
    ncols = max(1, ncols)
    nrows = (len(gallery_df) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4.8, nrows * 5.8))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    # Iteramos sobre las imágenes seleccionadas y pintamos cada una
    for ax, (_, row) in zip(axes, gallery_df.iterrows(), strict=False):
        draw_pose_overlay(row["image_path"], row, ax=ax)
        
        # Añadimos subtítulos con métricas clave (ángulos, estado, etc.)
        caption_lines = []
        for field_name in caption_fields:
            if field_name not in row.index:
                continue
            value = row[field_name]
            if isinstance(value, float):
                caption_lines.append(f"{field_name}: {value:.3f}")
            else:
                caption_lines.append(f"{field_name}: {value}")
        if caption_lines:
            ax.set_xlabel("\n".join(caption_lines), fontsize=8)

    # Apagamos los ejes vacíos si la cuadrícula no está llena
    for ax in axes[len(gallery_df) :]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    return fig, axes