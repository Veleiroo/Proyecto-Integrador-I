from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from .paths import RAW_DATA_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ORDER = ["train", "valid", "test", "unspecified"]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    notes: str
    local_dir_name: str
    source_format: str

    @property
    def local_dir(self) -> Path:
        return RAW_DATA_DIR / self.local_dir_name


DATASET_CATALOG: dict[str, DatasetSpec] = {
    "sitting_posture_4keypoint": DatasetSpec(
        key="sitting_posture_4keypoint",
        label="Sitting Posture 4 keypoint",
        notes="Mayormente lateral. Buen punto de partida para una validacion lateral.",
        local_dir_name="ikornproject_sitting-posture-rofqf_v4",
        source_format="coco",
    ),
    "sitting_posture_folder_v1": DatasetSpec(
        key="sitting_posture_folder_v1",
        label="Sitting Posture folder",
        notes="Version por carpetas con etiquetas de postura buenas y malas.",
        local_dir_name="pablos_sitting_posture_folder_v1",
        source_format="folder",
    ),
    "desk_posture_coco_v1": DatasetSpec(
        key="desk_posture_coco_v1",
        label="Desk Posture coco",
        notes="Dataset pequeno con pose etiquetada, util para contraste visual.",
        local_dir_name="pablos_desk_posture_coco_v1",
        source_format="coco",
    ),
    "posture_detection_folder_v1": DatasetSpec(
        key="posture_detection_folder_v1",
        label="Posture Detection folder",
        notes="Segundo contraste por carpetas con clases Good y Bad.",
        local_dir_name="pablos_posture_detection_folder_v1",
        source_format="folder",
    ),
    "posture_correction_v4_folder_v1": DatasetSpec(
        key="posture_correction_v4_folder_v1",
        label="Posture Correction v4 folder",
        notes="Webcam frontal. Es el dataset mas cercano al caso real del proyecto.",
        local_dir_name="pablos_posture_correction_v4_folder_v1",
        source_format="folder",
    ),
}


def get_dataset_spec(dataset_key: str) -> DatasetSpec:
    try:
        return DATASET_CATALOG[dataset_key]
    except KeyError as exc:
        available = ", ".join(sorted(DATASET_CATALOG))
        raise KeyError(f"Dataset desconocido: {dataset_key}. Disponibles: {available}") from exc


def looks_like_dataset_root(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False

    child_dirs = {child.name.lower() for child in path.iterdir() if child.is_dir()}
    child_files = {child.name.lower() for child in path.iterdir() if child.is_file()}
    markers = {"data.yaml", "readme.roboflow.txt", "readme.dataset.txt", "_annotations.coco.json"}

    return bool({"train", "valid"} <= child_dirs or "images" in child_dirs or markers & child_files)


def resolve_dataset_root(dataset_key: str) -> Path | None:
    base_dir = get_dataset_spec(dataset_key).local_dir
    if looks_like_dataset_root(base_dir):
        return base_dir

    for candidate in sorted(base_dir.rglob("*")):
        if candidate.is_dir() and looks_like_dataset_root(candidate):
            return candidate

    return None


def list_image_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def infer_group(image_path: Path, source_root: Path) -> str:
    relative = image_path.relative_to(source_root)
    parts = relative.parts
    split_names = {"train", "valid", "test"}

    if len(parts) >= 3 and parts[0].lower() in split_names and parts[1].lower() not in {"images", "labels"}:
        return parts[1]
    if len(parts) >= 2 and parts[0].lower() not in split_names:
        return parts[0]
    if parts and parts[0].lower() in split_names:
        return parts[0]
    return "unlabeled"


def infer_split(image_path: Path, source_root: Path) -> str | None:
    relative = image_path.relative_to(source_root)
    if relative.parts and relative.parts[0].lower() in {"train", "valid", "test"}:
        return relative.parts[0].lower()
    return None


def collect_image_records(dataset_key: str) -> list[dict]:
    dataset_root = resolve_dataset_root(dataset_key)
    if dataset_root is None:
        return []

    source_root = dataset_root / "images" if (dataset_root / "images").exists() else dataset_root
    records = []
    for image_path in list_image_files(source_root):
        records.append(
            {
                "image_path": image_path,
                "group": infer_group(image_path, source_root),
                "split": infer_split(image_path, source_root) or "unspecified",
            }
        )
    return records


def collect_image_records_df(dataset_key: str):
    import pandas as pd

    records = collect_image_records(dataset_key)
    if not records:
        return pd.DataFrame(columns=["image_path", "group", "split"])
    return pd.DataFrame(records)


def summarize_available_datasets():
    import pandas as pd

    rows = []
    for dataset_key, spec in DATASET_CATALOG.items():
        dataset_root = resolve_dataset_root(dataset_key)
        records = collect_image_records(dataset_key)
        groups = sorted({record["group"] for record in records})
        splits = sorted({record["split"] for record in records}, key=lambda value: SPLIT_ORDER.index(value) if value in SPLIT_ORDER else len(SPLIT_ORDER))
        rows.append(
            {
                "dataset_key": dataset_key,
                "label": spec.label,
                "format": spec.source_format,
                "is_available": dataset_root is not None,
                "dataset_root": str(dataset_root) if dataset_root else None,
                "total_images": len(records),
                "split_count": len(splits),
                "group_count": len(groups),
                "notes": spec.notes,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(["is_available", "total_images", "label"], ascending=[False, False, True]).reset_index(drop=True)


def summarize_dataset_groups(dataset_key: str):
    import pandas as pd

    records = collect_image_records(dataset_key)
    if not records:
        return pd.DataFrame(columns=["split", "group", "image_count"])

    df = pd.DataFrame(records)
    summary = (
        df.groupby(["split", "group"], dropna=False)
        .size()
        .rename("image_count")
        .reset_index()
    )
    return summary.sort_values(
        by=["split", "image_count", "group"],
        ascending=[True, False, True],
        key=lambda col: col.map(lambda value: SPLIT_ORDER.index(value) if value in SPLIT_ORDER else len(SPLIT_ORDER)) if col.name == "split" else col,
    ).reset_index(drop=True)


def plot_dataset_volumes(dataset_summary, ax=None):
    summary = dataset_summary.copy()
    if summary.empty:
        raise ValueError("No hay datasets para representar.")

    summary = summary.sort_values("total_images", ascending=True)
    colors = ["#2a9d8f" if available else "#b0b0b0" for available in summary["is_available"]]
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.8))
    else:
        fig = ax.figure

    ax.barh(summary["label"], summary["total_images"], color=colors)
    ax.set_title("Volumen de imagenes por dataset")
    ax.set_xlabel("Numero de imagenes")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)

    for patch, total_images in zip(ax.patches, summary["total_images"], strict=False):
        ax.text(total_images + max(summary["total_images"]) * 0.01, patch.get_y() + patch.get_height() / 2, str(total_images), va="center")

    fig.tight_layout()
    return fig, ax


def plot_dataset_group_distribution(group_summary, title: str, ax=None):
    import pandas as pd

    if group_summary.empty:
        raise ValueError("No hay informacion de grupos para representar.")

    plot_df = group_summary.copy()
    plot_df["split"] = pd.Categorical(plot_df["split"], categories=SPLIT_ORDER, ordered=True)
    pivot = (
        plot_df.pivot(index="group", columns="split", values="image_count")
        .fillna(0)
        .astype(int)
        .sort_index()
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    pivot.plot(kind="bar", ax=ax, color=["#4c78a8", "#f58518", "#54a24b", "#bab0ac"][: len(pivot.columns)])
    ax.set_title(title)
    ax.set_xlabel("Grupo")
    ax.set_ylabel("Numero de imagenes")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, ax
