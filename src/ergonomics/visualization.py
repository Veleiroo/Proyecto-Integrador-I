from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from .pose_inference import LANDMARK_IDS, SKELETON_SEGMENTS

STATUS_COLORS = {
    "adequate": "#2a9d8f",
    "improvable": "#f4a261",
    "risk": "#e76f51",
    "insufficient_data": "#8d99ae",
}


def load_rgb_image(image_path: str | Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_pose_overlay(image_path: str | Path, pose_row: pd.Series | dict, ax=None):
    image = load_rgb_image(image_path)
    height, width = image.shape[:2]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    ax.imshow(image)

    points = {}
    for landmark_name in LANDMARK_IDS:
        x = pose_row.get(f"{landmark_name}_x")
        y = pose_row.get(f"{landmark_name}_y")
        visibility = pose_row.get(f"{landmark_name}_visibility")
        if x is None or y is None or pd.isna(x) or pd.isna(y) or visibility is None or pd.isna(visibility):
            continue
        if float(visibility) < 0.35:
            continue
        pixel_point = (float(x) * width, float(y) * height)
        points[landmark_name] = pixel_point
        ax.scatter(pixel_point[0], pixel_point[1], s=36, color="#00b4d8")

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


def plot_status_distribution(analysis_df: pd.DataFrame, ax=None):
    if analysis_df.empty:
        raise ValueError("No hay analisis ergonomico para representar.")

    counts = analysis_df["overall_status"].value_counts().reindex(STATUS_COLORS.keys(), fill_value=0)
    counts = counts[counts > 0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
    else:
        fig = ax.figure

    colors = [STATUS_COLORS[label] for label in counts.index]
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title("Distribucion del estado ergonomico")
    ax.set_xlabel("")
    ax.set_ylabel("Numero de imagenes")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_status_by_group(analysis_df: pd.DataFrame, *, normalize: bool = True, ax=None):
    if analysis_df.empty:
        raise ValueError("No hay analisis ergonomico para representar.")

    grouped = (
        analysis_df.groupby(["group", "overall_status"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=STATUS_COLORS.keys(), fill_value=0)
        .sort_index()
    )
    grouped = grouped.loc[:, grouped.sum(axis=0) > 0]
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


def plot_metric_by_group(
    analysis_df: pd.DataFrame,
    metric: str,
    title: str,
    ax=None,
):
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


def plot_keypoint_coverage_heatmap(
    coverage_df: pd.DataFrame,
    *,
    group_col: str = "group",
    ax=None,
):
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

    for row_index, landmark_name in enumerate(pivot.index):
        for col_index, group_name in enumerate(pivot.columns):
            value = pivot.loc[landmark_name, group_name]
            ax.text(col_index, row_index, f"{value:.1f}", ha="center", va="center", color="white", fontsize=8)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("% visible")
    fig.tight_layout()
    return fig, ax


def plot_pose_gallery(
    gallery_df: pd.DataFrame,
    *,
    caption_fields: list[str] | None = None,
    title: str | None = None,
    ncols: int = 3,
):
    if gallery_df.empty:
        raise ValueError("No hay casos para representar en la galeria.")

    caption_fields = caption_fields or []
    ncols = max(1, ncols)
    nrows = (len(gallery_df) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4.8, nrows * 5.8))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (_, row) in zip(axes, gallery_df.iterrows(), strict=False):
        draw_pose_overlay(row["image_path"], row, ax=ax)
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

    for ax in axes[len(gallery_df) :]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    return fig, axes
