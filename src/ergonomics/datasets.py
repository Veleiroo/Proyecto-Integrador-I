from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt

from .paths import RAW_DATA_DIR


# Extensiones de imagen soportadas por nuestras herramientas de procesamiento
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Orden estándar para la división de datos en Machine Learning
SPLIT_ORDER = ["train", "valid", "test", "unspecified"]

# --- GESTIÓN DE METADATOS DE DATASETS ---

@dataclass(frozen=True)
class DatasetSpec:
    """
    Especificación técnica de un dataset.
    Centraliza la información sobre el formato original y la ubicación local.
    """
    key: str              # Identificador corto (ej: 'frontal_v1')
    label: str            # Nombre legible para gráficas e informes
    notes: str            # Descripción del contenido (ej: 'Vista lateral')
    local_dir_name: str   # Nombre de la carpeta física en data/raw/
    source_format: str    # Formato de origen: 'coco' (JSON) o 'folder' (carpetas por clase)

    @property
    def local_dir(self) -> Path:
        """Devuelve la ruta absoluta a la carpeta del dataset."""
        return RAW_DATA_DIR / self.local_dir_name


# --- CATÁLOGO DE DATASETS ---
# Registro centralizado de todas las fuentes de datos disponibles. 
# Esto permite que el resto del código se refiera a los datos por su 'key'
# sin preocuparse de rutas físicas o nombres de carpetas complejos.
DATASET_CATALOG: dict[str, DatasetSpec] = {
    "sitting_posture_4keypoint": DatasetSpec(
        key="sitting_posture_4keypoint",
        label="Sitting Posture 4 keypoint",
        notes="Mayormente lateral. Buen punto de partida para una validación lateral.",
        local_dir_name="ikornproject_sitting-posture-rofqf_v4",
        source_format="coco",
    ),
    "sitting_posture_folder_v1": DatasetSpec(
        key="sitting_posture_folder_v1",
        label="Sitting Posture folder",
        notes="Versión organizada por carpetas con etiquetas de postura (buena/mala).",
        local_dir_name="pablos_sitting_posture_folder_v1",
        source_format="folder",
    ),
    "desk_posture_coco_v1": DatasetSpec(
        key="desk_posture_coco_v1",
        label="Desk Posture coco",
        notes="Dataset pequeño con pose etiquetada, útil para contraste visual.",
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
        notes="Webcam frontal. Es el dataset más cercano al caso real del proyecto.",
        local_dir_name="pablos_posture_correction_v4_folder_v1",
        source_format="folder",
    ),
    "multiposture_zenodo_14230872": DatasetSpec(
        key="multiposture_zenodo_14230872",
        label="MultiPosture keypoints",
        notes="Keypoints 3D de postura sentada con etiquetas de tronco validadas por expertos. No incluye imagenes.",
        local_dir_name="multiposture_zenodo_14230872",
        source_format="keypoints_csv",
    ),
}

# --- LÓGICA DE RESOLUCIÓN DE RUTAS ---

def get_dataset_spec(dataset_key: str) -> DatasetSpec:
    """Busca la configuración de un dataset por su clave identificadora."""
    try:
        return DATASET_CATALOG[dataset_key]
    except KeyError as exc:
        available = ", ".join(sorted(DATASET_CATALOG))
        raise KeyError(f"Dataset desconocido: {dataset_key}. Disponibles: {available}") from exc


def looks_like_dataset_root(path: Path) -> bool:
    """
    Heurística para verificar si una carpeta es la raíz de un dataset.
    Busca archivos marcadores típicos de Roboflow o estructuras de ML estándar.
    """
    if not path.exists() or not path.is_dir():
        return False

    child_dirs = {child.name.lower() for child in path.iterdir() if child.is_dir()}
    child_files = {child.name.lower() for child in path.iterdir() if child.is_file()}
    # Archivos comunes que indican que estamos en la carpeta correcta
    markers = {"data.yaml", "readme.roboflow.txt", "readme.dataset.txt", "_annotations.coco.json"}

    return bool({"train", "valid"} <= child_dirs or "images" in child_dirs or markers & child_files)


def resolve_dataset_root(dataset_key: str) -> Path | None:
    """
    Localiza la carpeta raíz 'real' de un dataset.
    Resuelve el problema de las subcarpetas anidadas que a veces generan los unzip.
    """
    spec = get_dataset_spec(dataset_key)
    base_dir = spec.local_dir
    if spec.source_format == "keypoints_csv":
        return base_dir if (base_dir / "data.csv").exists() else None
    
    # Si la base ya parece ser la raíz, la devolvemos
    if looks_like_dataset_root(base_dir):
        return base_dir

    # Si no, buscamos recursivamente una carpeta que contenga los marcadores de dataset
    for candidate in sorted(base_dir.rglob("*")):
        if candidate.is_dir() and looks_like_dataset_root(candidate):
            return candidate

    return None

# --- UTILIDADES DE EXPLORACIÓN DE ARCHIVOS ---

def list_image_files(root: Path) -> list[Path]:
    """
    Busca recursivamente todos los archivos de imagen dentro de una ruta.
    Filtra por las extensiones permitidas (jpg, png, etc.) definidas globalmente.
    """
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_coco_group_lookup(dataset_root: Path) -> dict[str, str]:
    """
    Lee las etiquetas de los datasets COCO descargados por Roboflow.
    Devuelve un mapa filename -> categoria para no perder las clases Good/Bad
    cuando las imagenes estan directamente dentro de train/valid/test.
    """
    lookup: dict[str, str] = {}
    for annotation_path in sorted(dataset_root.rglob("_annotations.coco.json")):
        try:
            payload = json.loads(annotation_path.read_text())
        except Exception:
            continue

        categories = {
            int(category["id"]): str(category.get("name", category["id"]))
            for category in payload.get("categories", [])
            if "id" in category
        }
        images = {
            int(image["id"]): str(image.get("file_name", ""))
            for image in payload.get("images", [])
            if "id" in image
        }

        for annotation in payload.get("annotations", []):
            image_name = images.get(int(annotation.get("image_id", -1)))
            category_name = categories.get(int(annotation.get("category_id", -1)))
            if image_name and category_name:
                lookup[Path(image_name).name] = category_name
    return lookup

# --- INFERENCIA AUTOMÁTICA DE METADATOS ---
# Estas funciones evitan tener que escribir archivos de etiquetas a mano; 
# infieren la etiqueta mirando la estructura de carpetas.

def infer_group(image_path: Path, source_root: Path, coco_group_lookup: dict[str, str] | None = None) -> str:
    """
    Determina a qué grupo de postura pertenece la imagen (ej: 'adecuada', 'riesgo')
    analizando la ruta relativa del archivo. Soporta estructuras tipo Roboflow.
    """
    relative = image_path.relative_to(source_root)
    parts = relative.parts
    split_names = {"train", "valid", "test"}

    if coco_group_lookup and image_path.name in coco_group_lookup:
        return coco_group_lookup[image_path.name]

    # Caso 1: Estructura Roboflow (train/clase/imagen.jpg)
    if len(parts) >= 3 and parts[0].lower() in split_names and parts[1].lower() not in {"images", "labels"}:
        return parts[1]
    # Caso 2: Carpetas directas por clase (clase/imagen.jpg)
    if len(parts) >= 2 and parts[0].lower() not in split_names:
        return parts[0]
    # Caso 3: Sin clases, solo carpetas de split
    if parts and parts[0].lower() in split_names:
        return parts[0]
    return "unlabeled"


def infer_split(image_path: Path, source_root: Path) -> str | None:
    """
    Identifica si la imagen es para Entrenamiento, Validación o Test 
    mirando si está dentro de una carpeta con esos nombres.
    """
    relative = image_path.relative_to(source_root)
    if relative.parts and relative.parts[0].lower() in {"train", "valid", "test"}:
        return relative.parts[0].lower()
    return None

# --- RECOLECCIÓN Y CONSOLIDACIÓN DE DATOS ---

def collect_image_records(dataset_key: str) -> list[dict]:
    """
    Genera una lista de diccionarios con la información completa de cada imagen 
    encontrada en un dataset específico.
    """
    dataset_root = resolve_dataset_root(dataset_key)
    if dataset_root is None:
        return []
    if get_dataset_spec(dataset_key).source_format == "keypoints_csv":
        return []

    # Ajuste para datasets que vienen dentro de una subcarpeta 'images'
    source_root = dataset_root / "images" if (dataset_root / "images").exists() else dataset_root
    coco_group_lookup = load_coco_group_lookup(dataset_root)
    records = []
    for image_path in list_image_files(source_root):
        records.append(
            {
                "image_path": image_path,
                "group": infer_group(image_path, source_root, coco_group_lookup),
                "split": infer_split(image_path, source_root) or "unspecified",
            }
        )
    return records


def collect_image_records_df(dataset_key: str):
    """Encapsula los registros en un DataFrame de Pandas para análisis rápido."""
    import pandas as pd

    records = collect_image_records(dataset_key)
    if not records:
        return pd.DataFrame(columns=["image_path", "group", "split"])
    return pd.DataFrame(records)

# --- FUNCIONES DE RESUMEN (REPORTING) ---

def summarize_available_datasets():
    """
    Crea un resumen de todos los datasets registrados.
    Permite saber de un vistazo cuántas imágenes tenemos disponibles en total.
    """
    import pandas as pd

    rows = []
    for dataset_key, spec in DATASET_CATALOG.items():
        dataset_root = resolve_dataset_root(dataset_key)
        records = collect_image_records(dataset_key)

        keypoint_rows = 0
        if spec.source_format == "keypoints_csv":
            csv_path = spec.local_dir / "data.csv"
            if csv_path.exists():
                try:
                    keypoint_rows = len(pd.read_csv(csv_path, usecols=["upperbody_label"]))
                except Exception:
                    keypoint_rows = 0
        
        # Extraemos grupos y splits únicos presentes en los archivos
        groups = sorted({record["group"] for record in records})
        splits = sorted({record["split"] for record in records}, key=lambda value: SPLIT_ORDER.index(value) if value in SPLIT_ORDER else len(SPLIT_ORDER))
        
        rows.append(
            {
                "dataset_key": dataset_key,
                "label": spec.label,
                "format": spec.source_format,
                "is_available": dataset_root is not None or keypoint_rows > 0,
                "dataset_root": str(dataset_root) if dataset_root else None,
                "total_images": len(records) if spec.source_format != "keypoints_csv" else keypoint_rows,
                "split_count": len(splits),
                "group_count": len(groups),
                "notes": spec.notes,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    # Ordenamos: primero los disponibles y luego los más grandes
    return summary.sort_values(["is_available", "total_images", "label"], ascending=[False, False, True]).reset_index(drop=True)


def summarize_dataset_groups(dataset_key: str):
    """
    Genera un desglose detallado de cuántas imágenes hay por cada postura y split.
    Vital para comprobar si el dataset está balanceado.
    """
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
        # Ordenamos según el SPLIT_ORDER estándar (train -> valid -> test)
        key=lambda col: col.map(lambda value: SPLIT_ORDER.index(value) if value in SPLIT_ORDER else len(SPLIT_ORDER)) if col.name == "split" else col,
    ).reset_index(drop=True)


# --- FUNCIONES DE VISUALIZACIÓN DE DATOS (DATA AUDIT VISUALS) ---

def plot_dataset_volumes(dataset_summary, ax=None):
    """
    Genera un gráfico de barras horizontales que muestra el tamaño de cada dataset.
    
    CÓDIGO DE COLORES:
    - Verde (#2a9d8f): Dataset descargado y listo para usar.
    - Gris (#b0b0b0): Dataset registrado en el catálogo pero no encontrado en local.
    """
    summary = dataset_summary.copy()
    if summary.empty:
        raise ValueError("No hay datasets para representar.")

    # Ordenamos por volumen para que la gráfica sea más legible (de menor a mayor)
    summary = summary.sort_values("total_images", ascending=True)
    
    # Asignamos colores según disponibilidad: ayuda a detectar fallos de descarga en el equipo
    colors = ["#2a9d8f" if available else "#b0b0b0" for available in summary["is_available"]]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.8))
    else:
        fig = ax.figure

    # Dibujamos las barras horizontales
    ax.barh(summary["label"], summary["total_images"], color=colors)
    ax.set_title("Volumen de imagenes por dataset")
    ax.set_xlabel("Numero de imagenes")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)

    # Añadimos etiquetas de texto con el número exacto al final de cada barra
    for patch, total_images in zip(ax.patches, summary["total_images"], strict=False):
        ax.text(total_images + max(summary["total_images"]) * 0.01, 
                patch.get_y() + patch.get_height() / 2, 
                str(total_images), 
                va="center")

    fig.tight_layout()
    return fig, ax


def plot_dataset_group_distribution(group_summary, title: str, ax=None):
    """
    Genera una gráfica de barras agrupadas para analizar la distribución 
    de las posturas (grupos) según el split (Entrenamiento, Validación, Test).
    
    Es CRÍTICO para detectar si una clase (ej: 'riesgo') está infra-representada 
    en el conjunto de pruebas.
    """
    import pandas as pd

    if group_summary.empty:
        raise ValueError("No hay informacion de grupos para representar.")

    plot_df = group_summary.copy()
    # Aseguramos que los splits sigan el orden lógico: Train -> Valid -> Test
    plot_df["split"] = pd.Categorical(plot_df["split"], categories=SPLIT_ORDER, ordered=True)
    
    # Pivotamos la tabla: Eje X = Grupos de postura | Colores = Splits
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

    # Dibujamos las barras agrupadas con una paleta de colores 
    pivot.plot(kind="bar", ax=ax, color=["#4c78a8", "#f58518", "#54a24b", "#bab0ac"][: len(pivot.columns)])
    
    ax.set_title(title)
    ax.set_xlabel("Grupo")
    ax.set_ylabel("Numero de imagenes")
    ax.tick_params(axis="x", rotation=20) # Rotamos etiquetas para evitar que se pisen
    ax.grid(axis="y", alpha=0.25)
    
    fig.tight_layout()
    return fig, ax
