# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Benchmark inicial de modelos de pose para ergonomia
#
# Objetivo:
# - elegir un dataset de arranque
# - preparar un subset pequeno y reproducible
# - comparar `MediaPipe Pose`, `YOLO Pose` y `MoveNet` sin reentrenar
#
# En esta fase solo buscamos una primera senal tecnica: si los keypoints son estables y utiles para reglas ergonomicas.

# %%
from __future__ import annotations

import random
import shutil
import time
from pathlib import Path

SEED = 7
random.seed(SEED)


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
    return current


PROJECT_ROOT = find_project_root()
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "notebooks" / "pose_benchmark" / "results"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"RAW_DATA_DIR: {RAW_DATA_DIR}")
print(f"RESULTS_DIR: {RESULTS_DIR}")


# %% [markdown]
# ## 1. Datasets disponibles
#
# Los cinco recursos quedan registrados aqui con:
# - una descripcion corta
# - la ruta local donde queremos guardarlos
# - la configuracion de descarga de Roboflow
#
# El dataset activo se cambia con una sola variable: `ACTIVE_DATASET_KEY`.

# %%
DATASETS = {
    "sitting_posture_4keypoint": {
        "label": "Sitting Posture 4 keypoint",
        "notes": "Mayormente lateral. Buen punto de partida para el benchmark base.",
        "local_dir": RAW_DATA_DIR / "ikornproject_sitting-posture-rofqf_v4",
        "download": {
            "workspace": "ikornproject",
            "project": "sitting-posture-rofqf",
            "version": 4,
            "format": "coco",
        },
    },
    "sitting_posture_folder_v1": {
        "label": "Sitting Posture folder",
        "notes": "Alternativa simple por carpetas.",
        "local_dir": RAW_DATA_DIR / "pablos_sitting_posture_folder_v1",
        "download": {
            "workspace": "pablos-workspace-bcimu",
            "project": "Sitting Posture",
            "version": 1,
            "format": "folder",
        },
    },
    "desk_posture_coco_v1": {
        "label": "Desk Posture coco",
        "notes": "Util para validacion visual porque ya trae puntos marcados.",
        "local_dir": RAW_DATA_DIR / "pablos_desk_posture_coco_v1",
        "download": {
            "workspace": "pablos-workspace-bcimu",
            "project": "Desk Posture",
            "version": 1,
            "format": "coco",
        },
    },
    "posture_detection_folder_v1": {
        "label": "Posture Detection folder",
        "notes": "Segundo contraste con estructura por carpetas.",
        "local_dir": RAW_DATA_DIR / "pablos_posture_detection_folder_v1",
        "download": {
            "workspace": "pablos-workspace-bcimu",
            "project": "Posture_Detection",
            "version": 1,
            "format": "folder",
        },
    },
    "posture_correction_v4_folder_v1": {
        "label": "Posture Correction v4 folder",
        "notes": "Webcam frontal. Se parece mas al caso final del proyecto.",
        "local_dir": RAW_DATA_DIR / "pablos_posture_correction_v4_folder_v1",
        "download": {
            "workspace": "pablos-workspace-bcimu",
            "project": "posture_correction_v4",
            "version": 1,
            "format": "folder",
        },
    },
}

ACTIVE_DATASET_KEY = "sitting_posture_4keypoint"
ACTIVE_DATASET = DATASETS[ACTIVE_DATASET_KEY]
RAW_DATASET_OVERRIDE = None

SUBSET_DIR = PROJECT_ROOT / "data" / "pose_subset" / ACTIVE_DATASET_KEY
(SUBSET_DIR / "images").mkdir(parents=True, exist_ok=True)

for dataset in DATASETS.values():
    dataset["local_dir"].mkdir(parents=True, exist_ok=True)

print("Dataset activo:", ACTIVE_DATASET_KEY)
print("Descripcion:", ACTIVE_DATASET["notes"])
print("Ruta esperada:", ACTIVE_DATASET["local_dir"])


# %% [markdown]
# ## 2. Descarga opcional con Roboflow
#
# Si ya teneis los datasets descargados, podeis saltar esta parte.
# Si no, este helper os permite bajarlos a una ruta fija dentro de `data/raw/`.
#
# Requisito:
# - definir `ROBOFLOW_API_KEY` en el entorno antes de ejecutar la descarga

# %%
# %pip install roboflow


def get_roboflow_client():
    import os
    from roboflow import Roboflow

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("Falta ROBOFLOW_API_KEY en el entorno.")
    return Roboflow(api_key=api_key)


def download_dataset(dataset_key: str):
    cfg = DATASETS[dataset_key]
    target_dir = cfg["local_dir"]
    download_cfg = cfg["download"]

    client = get_roboflow_client()
    project = client.workspace(download_cfg["workspace"]).project(download_cfg["project"])
    version = project.version(download_cfg["version"])
    downloaded = version.download(download_cfg["format"], location=str(target_dir))

    print(f"Descargado: {dataset_key}")
    print(f"Ruta local: {downloaded.location}")
    return downloaded


list(DATASETS)


# %%
# Ejecuta solo las lineas que necesites.

# download_dataset("sitting_posture_4keypoint")
# download_dataset("sitting_posture_folder_v1")
# download_dataset("desk_posture_coco_v1")
# download_dataset("posture_detection_folder_v1")
# download_dataset("posture_correction_v4_folder_v1")


# %% [markdown]
# ## 3. Preparar el subset
#
# Para la comparativa inicial no hace falta usar todo el dataset.
# Nos quedamos con un subset pequeno y reproducible, pensado para inspeccion manual y benchmark rapido.

# %%
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def looks_like_dataset_root(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False

    child_dirs = {child.name.lower() for child in path.iterdir() if child.is_dir()}
    child_files = {child.name.lower() for child in path.iterdir() if child.is_file()}

    return bool(
        {"train", "valid"} <= child_dirs
        or "images" in child_dirs
        or {"data.yaml", "readme.roboflow.txt", "readme.dataset.txt"} & child_files
    )


def resolve_dataset_root(dataset_key: str, override: Path | None = None) -> Path | None:
    if override is not None:
        override = Path(override)
        if override.exists():
            return override

    base_dir = DATASETS[dataset_key]["local_dir"]
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
    return relative.parts[0] if len(relative.parts) > 1 else "unlabeled"


def collect_image_records(source_root: Path) -> list[dict]:
    records = []
    for image_path in list_image_files(source_root):
        records.append(
            {
                "image_path": image_path,
                "group": infer_group(image_path, source_root),
            }
        )
    return records


def build_subset(
    records: list[dict],
    target_dir: Path,
    max_per_group: int = 4,
    max_total: int = 18,
    seed: int = SEED,
) -> list[dict]:
    rng = random.Random(seed)
    images_dir = target_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for existing_file in images_dir.glob("*"):
        if existing_file.is_file():
            existing_file.unlink()

    grouped_records: dict[str, list[dict]] = {}
    for record in records:
        grouped_records.setdefault(record["group"], []).append(record)

    selected_records = []
    for group, group_records in sorted(grouped_records.items()):
        shuffled_records = group_records[:]
        rng.shuffle(shuffled_records)
        selected_records.extend(shuffled_records[:max_per_group])

    rng.shuffle(selected_records)
    selected_records = selected_records[:max_total]

    manifest = []
    for index, record in enumerate(selected_records, start=1):
        source_path = record["image_path"]
        safe_group = record["group"].replace(" ", "_").replace("/", "_")
        target_path = images_dir / f"{index:03d}_{safe_group}_{source_path.name}"
        shutil.copy2(source_path, target_path)
        manifest.append(
            {
                "index": index,
                "group": record["group"],
                "source_path": str(source_path),
                "subset_path": str(target_path),
            }
        )

    return manifest


dataset_root = resolve_dataset_root(ACTIVE_DATASET_KEY, RAW_DATASET_OVERRIDE)
subset_manifest = []

if dataset_root is None:
    print("Dataset no encontrado.")
    print("Ruta esperada:", ACTIVE_DATASET["local_dir"])
else:
    source_root = dataset_root / "images" if (dataset_root / "images").exists() else dataset_root
    records = collect_image_records(source_root)
    subset_manifest = build_subset(records, SUBSET_DIR)

    print("Dataset detectado en:", dataset_root)
    print("Imagenes encontradas:", len(records))
    print("Subset creado en:", SUBSET_DIR / "images")
    print("Tamano del subset:", len(subset_manifest))


# %% [markdown]
# ## 4. Dependencias del benchmark

# %%
# %pip install opencv-python mediapipe ultralytics tensorflow tensorflow-hub pandas


def check_optional_dependencies() -> dict[str, bool]:
    package_checks = {
        "opencv-python": "cv2",
        "mediapipe": "mediapipe",
        "ultralytics": "ultralytics",
        "tensorflow": "tensorflow",
        "tensorflow-hub": "tensorflow_hub",
        "pandas": "pandas",
    }

    status = {}
    for label, import_name in package_checks.items():
        try:
            __import__(import_name)
            status[label] = True
        except ImportError:
            status[label] = False
    return status


dependency_status = check_optional_dependencies()
dependency_status


# %% [markdown]
# ## 5. Modelos y benchmark

# %%
MEDIAPIPE_REQUIRED_IDS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_hip": 23,
    "right_hip": 24,
}

COCO_REQUIRED_IDS = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_hip": 11,
    "right_hip": 12,
}


def compute_core_support(scores: list[float], threshold: float = 0.3) -> tuple[int, bool]:
    available = sum(score >= threshold for score in scores)
    return available, available == len(scores)


def load_mediapipe_pose():
    import mediapipe as mp

    return mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    )


def load_yolo_pose(weights: str = "yolov8n-pose.pt"):
    from ultralytics import YOLO

    return YOLO(weights)


def load_movenet():
    import tensorflow_hub as hub

    module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    return module.signatures["serving_default"]


def run_mediapipe_pose(model, image_path: Path, min_visibility: float = 0.3) -> dict:
    import cv2

    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    start = time.perf_counter()
    result = model.process(image_rgb)
    runtime_ms = (time.perf_counter() - start) * 1000

    if not result.pose_landmarks:
        return {
            "model": "mediapipe_pose",
            "runtime_ms": runtime_ms,
            "total_keypoints": 0,
            "confident_keypoints": 0,
            "required_keypoints_present": 0,
            "can_measure_core_angles": False,
        }

    scores = [float(landmark.visibility) for landmark in result.pose_landmarks.landmark]
    required_scores = [scores[index] for index in MEDIAPIPE_REQUIRED_IDS.values()]
    required_count, can_measure = compute_core_support(required_scores, threshold=min_visibility)

    return {
        "model": "mediapipe_pose",
        "runtime_ms": runtime_ms,
        "total_keypoints": len(scores),
        "confident_keypoints": int(sum(score >= min_visibility for score in scores)),
        "required_keypoints_present": required_count,
        "can_measure_core_angles": can_measure,
    }


def run_yolo_pose(model, image_path: Path, min_confidence: float = 0.3) -> dict:
    import numpy as np

    start = time.perf_counter()
    result = model.predict(source=str(image_path), verbose=False)[0]
    runtime_ms = (time.perf_counter() - start) * 1000

    if result.keypoints is None:
        return {
            "model": "yolo_pose",
            "runtime_ms": runtime_ms,
            "total_keypoints": 0,
            "confident_keypoints": 0,
            "required_keypoints_present": 0,
            "can_measure_core_angles": False,
        }

    keypoint_data = getattr(result.keypoints, "data", None)
    if keypoint_data is None or len(keypoint_data) == 0:
        return {
            "model": "yolo_pose",
            "runtime_ms": runtime_ms,
            "total_keypoints": 0,
            "confident_keypoints": 0,
            "required_keypoints_present": 0,
            "can_measure_core_angles": False,
        }

    arr = keypoint_data[0].cpu().numpy()
    if arr.ndim != 2:
        arr = np.asarray(arr).reshape(-1, arr.shape[-1])

    scores = arr[:, 2].astype(float).tolist() if arr.shape[1] >= 3 else [1.0] * arr.shape[0]
    required_scores = [scores[index] for index in COCO_REQUIRED_IDS.values()]
    required_count, can_measure = compute_core_support(required_scores, threshold=min_confidence)

    return {
        "model": "yolo_pose",
        "runtime_ms": runtime_ms,
        "total_keypoints": len(scores),
        "confident_keypoints": int(sum(score >= min_confidence for score in scores)),
        "required_keypoints_present": required_count,
        "can_measure_core_angles": can_measure,
    }


def run_movenet(model, image_path: Path, input_size: int = 192, min_confidence: float = 0.3) -> dict:
    import cv2
    import numpy as np
    import tensorflow as tf

    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize_with_pad(np.expand_dims(image_rgb, axis=0), input_size, input_size)
    input_tensor = tf.cast(resized, dtype=tf.int32)

    start = time.perf_counter()
    outputs = model(input_tensor)
    runtime_ms = (time.perf_counter() - start) * 1000

    keypoints = outputs["output_0"].numpy()[0, 0, :, :]
    scores = keypoints[:, 2].astype(float).tolist()
    required_scores = [scores[index] for index in COCO_REQUIRED_IDS.values()]
    required_count, can_measure = compute_core_support(required_scores, threshold=min_confidence)

    return {
        "model": "movenet",
        "runtime_ms": runtime_ms,
        "total_keypoints": len(scores),
        "confident_keypoints": int(sum(score >= min_confidence for score in scores)),
        "required_keypoints_present": required_count,
        "can_measure_core_angles": can_measure,
    }


MODEL_LOADERS = {
    "mediapipe_pose": load_mediapipe_pose,
    "yolo_pose": load_yolo_pose,
    "movenet": load_movenet,
}

MODEL_RUNNERS = {
    "mediapipe_pose": run_mediapipe_pose,
    "yolo_pose": run_yolo_pose,
    "movenet": run_movenet,
}


def load_selected_models(model_names: tuple[str, ...]) -> dict[str, object]:
    loaded_models = {}
    for model_name in model_names:
        try:
            loaded_models[model_name] = MODEL_LOADERS[model_name]()
            print(f"Modelo cargado correctamente: {model_name}")
        except Exception as exc:
            print(f"No se pudo cargar {model_name}: {exc}")
    return loaded_models


def run_benchmark(
    image_paths: list[Path],
    model_names: tuple[str, ...] = ("mediapipe_pose", "yolo_pose", "movenet"),
    max_images: int = 6,
) -> list[dict]:
    loaded_models = load_selected_models(model_names)
    results = []

    for image_path in image_paths[:max_images]:
        for model_name, model in loaded_models.items():
            try:
                metrics = MODEL_RUNNERS[model_name](model, image_path)
                metrics["image_name"] = image_path.name
                metrics["error"] = None
            except Exception as exc:
                metrics = {
                    "model": model_name,
                    "image_name": image_path.name,
                    "runtime_ms": None,
                    "total_keypoints": None,
                    "confident_keypoints": None,
                    "required_keypoints_present": None,
                    "can_measure_core_angles": False,
                    "error": str(exc),
                }
            results.append(metrics)

    return results


subset_images = sorted((SUBSET_DIR / "images").glob("*"))
benchmark_results = []

if not subset_images:
    print("No hay imagenes en el subset. Ejecuta antes la preparacion del dataset.")
else:
    benchmark_results = run_benchmark(subset_images, max_images=min(6, len(subset_images)))
    benchmark_results[:5]


# %% [markdown]
# ## 6. Resumen

# %%
def summarize_results(records: list[dict]):
    if not records:
        print("No hay resultados todavia.")
        return None, None

    import pandas as pd

    df = pd.DataFrame(records)
    summary = (
        df.groupby("model", dropna=False)[
            ["runtime_ms", "total_keypoints", "confident_keypoints", "required_keypoints_present"]
        ]
        .mean(numeric_only=True)
        .sort_values(by=["required_keypoints_present", "runtime_ms"], ascending=[False, True])
    )

    return df, summary


if benchmark_results:
    df_results, df_summary = summarize_results(benchmark_results)
    display(df_results)
    display(df_summary)

    df_results.to_csv(RESULTS_DIR / "pose_benchmark_results.csv", index=False)
    df_summary.to_csv(RESULTS_DIR / "pose_benchmark_summary.csv")
    print(f"Resultados guardados en: {RESULTS_DIR}")
else:
    print("Todavia no hay resultados para resumir.")


# %% [markdown]
# ## 7. Siguiente paso recomendado
#
# Cuando esta primera comparativa funcione:
# - revisar 5-10 fallos visuales a mano
# - elegir un modelo principal y uno de respaldo
# - anadir un dataset mas frontal o mixto
# - pasar de keypoints a metricas ergonomicas
