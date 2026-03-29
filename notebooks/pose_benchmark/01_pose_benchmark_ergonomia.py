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
# Este notebook arranca la fase tecnica del proyecto con un objetivo muy concreto: comparar tres modelos preentrenados de pose estimation en un escenario de ergonomia en puesto sedentario.
#
# Objetivos de esta primera version:
# - Elegir un dataset inicial para el benchmark.
# - Preparar un subset pequeno y reproducible de imagenes.
# - Probar `MediaPipe Pose`, `YOLO Pose` y `MoveNet` sin reentrenar.
# - Medir una primera senal de viabilidad para el producto final.
#
# Fuera de alcance por ahora:
# - Reentrenar modelos.
# - Construir el motor completo de reglas ergonomicas.
# - Desarrollar la aplicacion final.
#

# %% [markdown]
# ## Pregunta de trabajo
#
# Que modelo preentrenado parece mas adecuado para una primera PoC de analisis ergonomico basada en camara?
#
# Criterios iniciales de exito:
# - Detectar keypoints relevantes para cuello, hombros, codos y tronco.
# - Mantener una latencia razonable en CPU.
# - Ser robusto en imagenes reales de personas sentadas frente al ordenador.
# - Permitir una capa posterior de reglas interpretables inspiradas en ROSA / ISO 9241-5.
#

# %%
# Setup reproducible y rutas del experimento.
# Usamos solo librerias estandar en esta celda para que el notebook arranque incluso
# antes de instalar dependencias pesadas como TensorFlow o MediaPipe.

from __future__ import annotations

import json
import random
import shutil
import time
import zipfile
from pathlib import Path

SEED = 7
random.seed(SEED)


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / '.git').exists():
            return candidate
    return current


PROJECT_ROOT = find_project_root()
NOTEBOOK_DIR = PROJECT_ROOT / 'notebooks' / 'pose_benchmark'
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
SUBSET_DIR = PROJECT_ROOT / 'data' / 'pose_subset' / 'sitting_posture_initial'
RESULTS_DIR = NOTEBOOK_DIR / 'results'

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
SUBSET_DIR.mkdir(parents=True, exist_ok=True)
(SUBSET_DIR / 'images').mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f'PROJECT_ROOT: {PROJECT_ROOT}')
print(f'RAW_DATA_DIR: {RAW_DATA_DIR}')
print(f'SUBSET_DIR: {SUBSET_DIR}')
print(f'RESULTS_DIR: {RESULTS_DIR}')


# %% [markdown]
# ## Plan del experimento
#
# Hipotesis iniciales:
# - `MediaPipe Pose` y `MoveNet` deberian ser mas ligeros en CPU.
# - `YOLO Pose` podria ofrecer keypoints 2D mas estables en imagenes complejas, pero con peor latencia.
# - No necesitamos entrenar todavia; primero hay que validar si la salida de pose sirve para extraer variables ergonomicas utiles.
#
# Metricas a registrar en esta fase:
# - Tiempo de inferencia por imagen.
# - Numero total de keypoints detectados.
# - Numero de keypoints con confianza suficiente.
# - Disponibilidad de keypoints esenciales para ergonomia: nariz, hombros, codos y caderas.
# - Fallos observados en perspectiva, iluminacion u oclusiones.
#

# %% [markdown]
# ## 1. Eleccion del dataset inicial
#
# Datasets / modelos sugeridos en la documentacion de DataLife:
#
# | Recurso | Link | Motivo para considerarlo | Riesgo inicial |
# | --- | --- | --- | --- |
# | Sitting Posture Computer Vision Dataset | https://universe.roboflow.com/ikornproject/sitting-posture-rofqf | Muy alineado con el caso de personas sentadas | Predomina la vista lateral; no cubre aun el caso final completo |
# | Sitting Posture Computer Vision Model | https://universe.roboflow.com/dataset-sqm0h/sitting-posture-ezkda | Puede servir como referencia de etiquetado | No esta claro si es mejor fuente de imagenes que el dataset anterior |
# | posture_correction_v4 Computer Vision Model | https://universe.roboflow.com/posturecorrection/posture_correction_v4 | Relacionado con correccion postural | Puede estar mas orientado a clasificacion que a benchmark visual inicial |
# | Posture_Detection Computer Vision Dataset | https://universe.roboflow.com/posture-magj9/posture_detection-5huqr | Candidato alternativo si el primero falla | Puede ser menos especifico para contexto de oficina |
# | Desk Posture Computer Vision Model | https://universe.roboflow.com/roboflow-test-o7w8z/desk-posture | Cercano al escenario de escritorio | No sabemos aun balance de clases ni calidad real |
#
# Decision de arranque:
# - Empezar con **Sitting Posture Computer Vision Dataset**.
# - Motivo: aunque es mayoritariamente lateral, sirve como primer banco de pruebas para validar si la pipeline de pose estimation detecta cuello, hombros, codos y tronco con estabilidad.
# - La decision es provisional. Si el benchmark sale bien, el siguiente paso natural es anadir un dataset mas frontal o mixto.
#

# %%
# Registro de datasets candidatos y seleccion del primero.
# Anotamos tambien el workspace, el project slug y la version de Roboflow,
# porque la descarga en Universe se hace siempre sobre una version concreta.

DATASET_CANDIDATES = [
    {
        'name': 'Sitting Posture Computer Vision Dataset',
        'workspace': 'ikornproject',
        'project': 'sitting-posture-rofqf',
        'version': 4,
        'task': 'keypoint-detection',
        'page_url': 'https://universe.roboflow.com/ikornproject/sitting-posture-rofqf',
        'dataset_version_url': 'https://universe.roboflow.com/ikornproject/sitting-posture-rofqf/dataset/4',
        'why': 'Buen punto de partida para validar un benchmark lateral sin reentrenamiento.',
        'limitations': 'Predomina la vista lateral; no representa todavia el escenario final frontal/mixto.',
    },
    {
        'name': 'Posture_Detection Computer Vision Dataset',
        'workspace': 'posture-magj9',
        'project': 'posture_detection-5huqr',
        'version': 1,
        'task': 'unknown',
        'page_url': 'https://universe.roboflow.com/posture-magj9/posture_detection-5huqr',
        'dataset_version_url': 'https://universe.roboflow.com/posture-magj9/posture_detection-5huqr',
        'why': 'Alternativa si el dataset principal tiene poca variedad o mal etiquetado.',
        'limitations': 'No esta claro si se ajusta tan bien al contexto de escritorio.',
    },
    {
        'name': 'Desk Posture Computer Vision Model',
        'workspace': 'roboflow-test-o7w8z',
        'project': 'desk-posture',
        'version': 1,
        'task': 'unknown',
        'page_url': 'https://universe.roboflow.com/roboflow-test-o7w8z/desk-posture',
        'dataset_version_url': 'https://universe.roboflow.com/roboflow-test-o7w8z/desk-posture',
        'why': 'Reserva por su cercania al contexto de escritorio.',
        'limitations': 'No sabemos aun si el recurso es mejor como dataset o como referencia.',
    },
]

SELECTED_DATASET = DATASET_CANDIDATES[0]
DOWNLOAD_REFERENCE = (
    f"{SELECTED_DATASET['workspace']}/{SELECTED_DATASET['project']}/{SELECTED_DATASET['version']}"
)
EXPECTED_RAW_DIR = RAW_DATA_DIR / (
    f"{SELECTED_DATASET['workspace']}_{SELECTED_DATASET['project']}_v{SELECTED_DATASET['version']}"
)

# Si Roboflow extrae la carpeta con otro nombre o la guardais en otra ruta,
# podeis apuntarla manualmente aqui.
RAW_DATASET_OVERRIDE = None

print('Dataset inicial seleccionado:')
print(json.dumps(SELECTED_DATASET, indent=2))
print(f'Download reference: {DOWNLOAD_REFERENCE}')
print(f'Ruta recomendada para dejarlo extraido: {EXPECTED_RAW_DIR}')


# %% [markdown]
# ## 2. Como descargarlo desde Roboflow Universe
#
# La parte importante es que Roboflow descarga **versiones concretas**, no solo el proyecto generico.
#
# Ruta recomendada para este notebook:
# - Abrir `SELECTED_DATASET['dataset_version_url']`.
# - Ir a `Download Dataset`.
# - Elegir un formato compatible con la tarea que os muestre la propia UI.
# - Bajar el ZIP y extraerlo en `EXPECTED_RAW_DIR`, o bien poner la ruta real en `RAW_DATASET_OVERRIDE`.
#
# Notas practicas:
# - Para este experimento, como solo necesitamos imagenes y una estructura reproducible, no hace falta obsesionarse con el formato exacto de anotacion.
# - Si quereis automatizar la descarga con codigo, Roboflow suele ofrecer `Show Download Code`, pero esa via normalmente requiere API key.
# - La API directa de exportacion tambien requiere API key, asi que la opcion mas simple al principio es bajar el ZIP desde la web.
#

# %% [markdown]
# ### 2.1 Snippets de descarga automatizada
#
# Los siguientes bloques vienen de los snippets de Roboflow que vais usando para los distintos forks. **No conviene dejar la API key hardcodeada en el notebook**; se recomienda leerla desde la variable de entorno `ROBOFLOW_API_KEY`.
#
# Antes de ejecutarlo en Jupyter, podeis definir la clave en una celda aparte:
#
# ```python
# import os
# os.environ['ROBOFLOW_API_KEY'] = 'TU_API_KEY'
# ```
#
# Mejor aun: definirla fuera del notebook en el entorno del sistema para no exponerla en la repo.
#
# Rutas objetivo recomendadas dentro de la repo:
# - `sitting_posture_4keypoint` -> `EXPECTED_RAW_DIR`
# - `sitting_posture_folder_v1` -> `data/raw/pablos_sitting_posture_folder_v1/`
# - `desk_posture_coco_v1` -> `data/raw/pablos_desk_posture_coco_v1/`
# - `posture_detection_folder_v1` -> `data/raw/pablos_posture_detection_folder_v1/`
# - `posture_correction_v4_folder_v1` -> `data/raw/pablos_posture_correction_v4_folder_v1/`
#

# %%
# Setup comun para los snippets de descarga.
# Centralizamos la conexion a Roboflow y fijamos las rutas destino para que la repo
# mantenga siempre la misma estructura independientemente de como llame Roboflow a las carpetas.

# # %pip install roboflow

import os

from roboflow import Roboflow

ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
if not ROBOFLOW_API_KEY:
    raise ValueError(
        'Falta ROBOFLOW_API_KEY. Define la variable de entorno antes de ejecutar este snippet.'
    )

rf = Roboflow(api_key=ROBOFLOW_API_KEY)

DOWNLOAD_TARGETS = {
    'sitting_posture_4keypoint': EXPECTED_RAW_DIR,
    'sitting_posture_folder_v1': RAW_DATA_DIR / 'pablos_sitting_posture_folder_v1',
    'desk_posture_coco_v1': RAW_DATA_DIR / 'pablos_desk_posture_coco_v1',
    'posture_detection_folder_v1': RAW_DATA_DIR / 'pablos_posture_detection_folder_v1',
    'posture_correction_v4_folder_v1': RAW_DATA_DIR / 'pablos_posture_correction_v4_folder_v1',
}

for target_dir in DOWNLOAD_TARGETS.values():
    target_dir.mkdir(parents=True, exist_ok=True)


def download_roboflow_dataset(
    *,
    workspace: str,
    project_name: str,
    version: int,
    model_format: str,
    target_dir: Path,
):
    project = rf.workspace(workspace).project(project_name)
    version_obj = project.version(version)
    downloaded = version_obj.download(model_format=model_format, location=str(target_dir))
    print(f'Descargado: {workspace}/{project_name}/v{version} [{model_format}]')
    print(f'Ruta local: {downloaded.location}')
    return downloaded


DOWNLOAD_TARGETS


# %%
# Snippet 1: sitting_posture_4keypoint
# Formato: COCO. Es el dataset inicial del benchmark y descarga en EXPECTED_RAW_DIR.

downloaded_sitting_posture_4keypoint = download_roboflow_dataset(
    workspace='ikornproject',
    project_name='sitting-posture-rofqf',
    version=4,
    model_format='coco',
    target_dir=DOWNLOAD_TARGETS['sitting_posture_4keypoint'],
)


# %%
# Snippet 2: Sitting Posture Computer Vision Model
# Formato: folder. Puede servir como apoyo visual y alternativa con estructura simple por carpetas.

downloaded_sitting_posture_folder = download_roboflow_dataset(
    workspace='pablos-workspace-bcimu',
    project_name='Sitting Posture',
    version=1,
    model_format='folder',
    target_dir=DOWNLOAD_TARGETS['sitting_posture_folder_v1'],
)


# %%
# Snippet 3: Desk Posture
# Formato: COCO. Como ya tiene puntos marcados en las imagenes, puede ser util para validacion visual.

downloaded_desk_posture_coco = download_roboflow_dataset(
    workspace='pablos-workspace-bcimu',
    project_name='Desk Posture',
    version=1,
    model_format='coco',
    target_dir=DOWNLOAD_TARGETS['desk_posture_coco_v1'],
)


# %%
# Snippet 4: posture_detection
# Formato: folder. Incluye algunas anotaciones sobre postura y puede servir como segundo contraste.

downloaded_posture_detection_folder = download_roboflow_dataset(
    workspace='pablos-workspace-bcimu',
    project_name='Posture_Detection',
    version=1,
    model_format='folder',
    target_dir=DOWNLOAD_TARGETS['posture_detection_folder_v1'],
)


# %%
# Snippet 5: posture_correction_v4
# Formato: folder. Es interesante porque trae webcam frontal, que se parece mas al caso final.

downloaded_posture_correction_v4_folder = download_roboflow_dataset(
    workspace='pablos-workspace-bcimu',
    project_name='posture_correction_v4',
    version=1,
    model_format='folder',
    target_dir=DOWNLOAD_TARGETS['posture_correction_v4_folder_v1'],
)


# %% [markdown]
# ## 3. Preparacion del subset
#
# En esta fase no necesitamos miles de imagenes. Para una primera comparativa es mejor un subset pequeno, equilibrado y facil de inspeccionar manualmente.
#
# Propuesta de subset inicial:
# - Entre 12 y 18 imagenes.
# - Varias posturas y angulos de camara si el dataset los incluye.
# - Mantener etiquetas si existen, pero el benchmark inicial puede funcionar incluso solo con imagenes.
#
# Notas sobre la resolucion de rutas:
# - El notebook intenta encontrar el dataset aunque el ZIP se extraiga con un nombre distinto.
# - Si aun asi no lo detecta, usad `RAW_DATASET_OVERRIDE = Path('/ruta/real/al/dataset')` en la celda anterior.
#

# %%
# Helpers para localizar imagenes y construir un subset pequeno.
# El objetivo aqui es evitar suposiciones fragiles sobre el nombre final de la carpeta
# que genera Roboflow al descargar y descomprimir el dataset.

import re

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def normalize_token(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', value.lower()).strip('-')


def build_dataset_tokens(dataset: dict) -> set[str]:
    raw_values = [
        dataset['name'],
        dataset['workspace'],
        dataset['project'],
        f"v{dataset['version']}",
        str(dataset['version']),
    ]

    tokens = set()
    for value in raw_values:
        normalized = normalize_token(value)
        tokens.update(token for token in normalized.split('-') if len(token) >= 2)
    return tokens


def list_image_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob('*') if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def infer_group(image_path: Path, source_root: Path) -> str:
    relative = image_path.relative_to(source_root)
    return relative.parts[0] if len(relative.parts) > 1 else 'unlabeled'


def collect_image_records(source_root: Path) -> list[dict]:
    records = []
    for image_path in list_image_files(source_root):
        records.append(
            {
                'image_path': image_path,
                'group': infer_group(image_path, source_root),
            }
        )
    return records


def looks_like_dataset_root(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False

    child_dirs = {child.name.lower() for child in path.iterdir() if child.is_dir()}
    child_files = {child.name.lower() for child in path.iterdir() if child.is_file()}

    if {'train', 'valid'} <= child_dirs:
        return True
    if 'images' in child_dirs:
        return True
    if {'data.yaml', 'readme.roboflow.txt', 'readme.dataset.txt'} & child_files:
        return True

    return False


def score_candidate_root(path: Path, tokens: set[str]) -> int:
    normalized_path = normalize_token(str(path))
    token_score = sum(token in normalized_path for token in tokens)
    marker_score = 0

    child_names = {child.name.lower() for child in path.iterdir()} if path.exists() else set()
    if {'train', 'valid'} <= child_names:
        marker_score += 3
    if 'images' in child_names:
        marker_score += 2
    if {'data.yaml', 'readme.roboflow.txt', 'readme.dataset.txt'} & child_names:
        marker_score += 2

    return token_score + marker_score


def extract_matching_archives(raw_data_dir: Path, expected_dir: Path, tokens: set[str]) -> list[Path]:
    extracted_archives = []
    for archive_path in sorted(raw_data_dir.glob('*.zip')):
        normalized_name = normalize_token(archive_path.stem)
        token_hits = sum(token in normalized_name for token in tokens)
        if token_hits < 2:
            continue

        expected_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as zip_file:
            zip_file.extractall(expected_dir)
        extracted_archives.append(archive_path)
        break

    return extracted_archives


def resolve_dataset_root(raw_data_dir: Path, dataset: dict, override: Path | None = None) -> Path | None:
    if override is not None:
        override = Path(override)
        if override.exists():
            return override

    direct_candidates = [
        EXPECTED_RAW_DIR,
        EXPECTED_RAW_DIR / 'images',
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate.parent if candidate.name == 'images' else candidate

    tokens = build_dataset_tokens(dataset)
    extract_matching_archives(raw_data_dir, EXPECTED_RAW_DIR, tokens)

    candidate_roots = []
    search_space = [raw_data_dir, *raw_data_dir.rglob('*')]
    for candidate in search_space:
        if not candidate.is_dir():
            continue
        if not looks_like_dataset_root(candidate):
            continue
        candidate_roots.append((score_candidate_root(candidate, tokens), candidate))

    if not candidate_roots:
        return None

    candidate_roots.sort(key=lambda item: (-item[0], len(item[1].parts)))
    return candidate_roots[0][1]


def build_subset(
    records: list[dict],
    target_dir: Path,
    max_per_group: int = 4,
    max_total: int = 18,
    seed: int = SEED,
) -> list[dict]:
    rng = random.Random(seed)
    images_dir = target_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    for existing_file in images_dir.glob('*'):
        if existing_file.is_file():
            existing_file.unlink()

    grouped_records: dict[str, list[dict]] = {}
    for record in records:
        grouped_records.setdefault(record['group'], []).append(record)

    selected_records = []
    for group, group_records in sorted(grouped_records.items()):
        shuffled_records = group_records[:]
        rng.shuffle(shuffled_records)
        selected_records.extend(shuffled_records[:max_per_group])

    rng.shuffle(selected_records)
    selected_records = selected_records[:max_total]

    manifest = []
    for index, record in enumerate(selected_records, start=1):
        source_path = record['image_path']
        safe_group = record['group'].replace(' ', '_').replace('/', '_')
        target_name = f"{index:03d}_{safe_group}_{source_path.name}"
        target_path = images_dir / target_name
        shutil.copy2(source_path, target_path)
        manifest.append(
            {
                'index': index,
                'group': record['group'],
                'source_path': str(source_path),
                'subset_path': str(target_path),
            }
        )

    manifest_path = target_dir / 'subset_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


dataset_root = resolve_dataset_root(RAW_DATA_DIR, SELECTED_DATASET, RAW_DATASET_OVERRIDE)
subset_manifest = []

if dataset_root is None:
    print('Dataset local no encontrado todavia.')
    print(f'Abre esta URL: {SELECTED_DATASET["dataset_version_url"]}')
    print(f'Extrae el ZIP en: {EXPECTED_RAW_DIR}')
    if any(RAW_DATA_DIR.iterdir()):
        print('Contenido actual de data/raw:')
        for child in sorted(RAW_DATA_DIR.iterdir()):
            print(f'- {child}')
else:
    source_root = dataset_root / 'images' if (dataset_root / 'images').exists() else dataset_root
    records = collect_image_records(source_root)
    print(f'Dataset detectado en: {dataset_root}')
    print(f'Se han encontrado {len(records)} imagenes en bruto.')
    subset_manifest = build_subset(records, SUBSET_DIR)
    print(f'Subset creado con {len(subset_manifest)} imagenes en {(SUBSET_DIR / "images")}')
    subset_manifest[:3]


# %% [markdown]
# ## 4. Carga de modelos preentrenados
#
# En esta fase queremos evaluar modelos tal como vienen entrenados por defecto. La pregunta no es todavia si podemos mejorar su precision con fine-tuning, sino si la senal base ya es util para construir el producto.
#
# Dependencias esperadas:
# - `opencv-python`
# - `mediapipe`
# - `ultralytics`
# - `tensorflow`
# - `tensorflow-hub`
# - `pandas` para la tabla resumen final
#
# Si hace falta instalar todo desde el notebook, se puede descomentar la linea `%pip` de la siguiente celda.
#

# %%
# Ejecutar solo la primera vez en el entorno.
# # %pip install opencv-python mediapipe ultralytics tensorflow tensorflow-hub pandas


def check_optional_dependencies() -> dict[str, bool]:
    package_checks = {
        'opencv-python': 'cv2',
        'mediapipe': 'mediapipe',
        'ultralytics': 'ultralytics',
        'tensorflow': 'tensorflow',
        'tensorflow-hub': 'tensorflow_hub',
        'pandas': 'pandas',
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


# %%
# Wrappers de inferencia para los tres modelos.
# Todos devuelven una estructura similar para facilitar la comparacion posterior.

MEDIAPIPE_REQUIRED_IDS = {
    'nose': 0,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_hip': 23,
    'right_hip': 24,
}

COCO_REQUIRED_IDS = {
    'nose': 0,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_hip': 11,
    'right_hip': 12,
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


def load_yolo_pose(weights: str = 'yolov8n-pose.pt'):
    from ultralytics import YOLO

    return YOLO(weights)


def load_movenet():
    import tensorflow_hub as hub

    module = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
    return module.signatures['serving_default']


def run_mediapipe_pose(model, image_path: Path, min_visibility: float = 0.3) -> dict:
    import cv2

    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    start = time.perf_counter()
    result = model.process(image_rgb)
    runtime_ms = (time.perf_counter() - start) * 1000

    if not result.pose_landmarks:
        return {
            'model': 'mediapipe_pose',
            'runtime_ms': runtime_ms,
            'total_keypoints': 0,
            'confident_keypoints': 0,
            'required_keypoints_present': 0,
            'can_measure_core_angles': False,
        }

    scores = [float(landmark.visibility) for landmark in result.pose_landmarks.landmark]
    required_scores = [scores[index] for index in MEDIAPIPE_REQUIRED_IDS.values()]
    required_count, can_measure = compute_core_support(required_scores, threshold=min_visibility)

    return {
        'model': 'mediapipe_pose',
        'runtime_ms': runtime_ms,
        'total_keypoints': len(scores),
        'confident_keypoints': int(sum(score >= min_visibility for score in scores)),
        'required_keypoints_present': required_count,
        'can_measure_core_angles': can_measure,
    }


def run_yolo_pose(model, image_path: Path, min_confidence: float = 0.3) -> dict:
    import numpy as np

    start = time.perf_counter()
    result = model.predict(source=str(image_path), verbose=False)[0]
    runtime_ms = (time.perf_counter() - start) * 1000

    if result.keypoints is None:
        return {
            'model': 'yolo_pose',
            'runtime_ms': runtime_ms,
            'total_keypoints': 0,
            'confident_keypoints': 0,
            'required_keypoints_present': 0,
            'can_measure_core_angles': False,
        }

    keypoint_data = getattr(result.keypoints, 'data', None)
    if keypoint_data is None or len(keypoint_data) == 0:
        return {
            'model': 'yolo_pose',
            'runtime_ms': runtime_ms,
            'total_keypoints': 0,
            'confident_keypoints': 0,
            'required_keypoints_present': 0,
            'can_measure_core_angles': False,
        }

    arr = keypoint_data[0].cpu().numpy()
    if arr.ndim != 2:
        arr = np.asarray(arr).reshape(-1, arr.shape[-1])

    if arr.shape[1] >= 3:
        scores = arr[:, 2].astype(float).tolist()
    else:
        scores = [1.0] * arr.shape[0]

    required_scores = [scores[index] for index in COCO_REQUIRED_IDS.values()]
    required_count, can_measure = compute_core_support(required_scores, threshold=min_confidence)

    return {
        'model': 'yolo_pose',
        'runtime_ms': runtime_ms,
        'total_keypoints': len(scores),
        'confident_keypoints': int(sum(score >= min_confidence for score in scores)),
        'required_keypoints_present': required_count,
        'can_measure_core_angles': can_measure,
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

    keypoints = outputs['output_0'].numpy()[0, 0, :, :]
    scores = keypoints[:, 2].astype(float).tolist()
    required_scores = [scores[index] for index in COCO_REQUIRED_IDS.values()]
    required_count, can_measure = compute_core_support(required_scores, threshold=min_confidence)

    return {
        'model': 'movenet',
        'runtime_ms': runtime_ms,
        'total_keypoints': len(scores),
        'confident_keypoints': int(sum(score >= min_confidence for score in scores)),
        'required_keypoints_present': required_count,
        'can_measure_core_angles': can_measure,
    }



# %%
# Bucle principal del benchmark.
# Si falta alguna dependencia o un modelo no carga, no abortamos toda la prueba:
# registramos el error y seguimos con el resto.

MODEL_LOADERS = {
    'mediapipe_pose': load_mediapipe_pose,
    'yolo_pose': load_yolo_pose,
    'movenet': load_movenet,
}

MODEL_RUNNERS = {
    'mediapipe_pose': run_mediapipe_pose,
    'yolo_pose': run_yolo_pose,
    'movenet': run_movenet,
}


def load_selected_models(model_names: tuple[str, ...]) -> dict[str, object]:
    loaded_models = {}
    for model_name in model_names:
        try:
            loaded_models[model_name] = MODEL_LOADERS[model_name]()
            print(f'Modelo cargado correctamente: {model_name}')
        except Exception as exc:
            print(f'No se pudo cargar {model_name}: {exc}')
    return loaded_models


def run_benchmark(
    image_paths: list[Path],
    model_names: tuple[str, ...] = ('mediapipe_pose', 'yolo_pose', 'movenet'),
    max_images: int = 6,
) -> list[dict]:
    loaded_models = load_selected_models(model_names)
    results = []

    for image_path in image_paths[:max_images]:
        for model_name, model in loaded_models.items():
            try:
                metrics = MODEL_RUNNERS[model_name](model, image_path)
                metrics['image_name'] = image_path.name
                metrics['error'] = None
            except Exception as exc:
                metrics = {
                    'model': model_name,
                    'image_name': image_path.name,
                    'runtime_ms': None,
                    'total_keypoints': None,
                    'confident_keypoints': None,
                    'required_keypoints_present': None,
                    'can_measure_core_angles': False,
                    'error': str(exc),
                }

            results.append(metrics)

    return results


subset_images = sorted((SUBSET_DIR / 'images').glob('*'))
benchmark_results = []

if not subset_images:
    print('Aun no hay imagenes en el subset. Ejecuta primero la preparacion del dataset.')
else:
    benchmark_results = run_benchmark(
        image_paths=subset_images,
        max_images=min(6, len(subset_images)),
    )
    benchmark_results[:5]


# %% [markdown]
# ## 5. Resumen de metricas
#
# En esta fase todavia no hablamos de accuracy contra ground truth ergonomico. Lo que buscamos es una comparativa rapida de viabilidad tecnica:
# - Que modelo detecta mas keypoints utiles?
# - Cual permite medir mejor los angulos base de cuello / hombros / tronco?
# - Cual tiene una latencia razonable para una futura version en tiempo real?
#

# %%
# Construimos una tabla resumen y la guardamos para poder reutilizarla en la memoria
# o en futuras diapositivas comparativas.


def summarize_results(records: list[dict]):
    if not records:
        print('No hay resultados todavia.')
        return None, None

    import pandas as pd

    df = pd.DataFrame(records)
    summary = (
        df.groupby('model', dropna=False)[
            ['runtime_ms', 'total_keypoints', 'confident_keypoints', 'required_keypoints_present']
        ]
        .mean(numeric_only=True)
        .sort_values(by=['required_keypoints_present', 'runtime_ms'], ascending=[False, True])
    )

    return df, summary


if benchmark_results:
    df_results, df_summary = summarize_results(benchmark_results)
    display(df_results)
    display(df_summary)

    df_results.to_csv(RESULTS_DIR / 'pose_benchmark_results.csv', index=False)
    df_summary.to_csv(RESULTS_DIR / 'pose_benchmark_summary.csv')
    print(f'Resultados guardados en: {RESULTS_DIR}')
else:
    print('Todavia no hay resultados para resumir.')


# %% [markdown]
# ## 6. Conclusiones y siguientes pasos
#
# Cuando este notebook tenga resultados, las siguientes decisiones recomendadas son:
# - Verificar manualmente 5-10 casos donde un modelo falle claramente.
# - Elegir un candidato principal y uno de respaldo.
# - Pasar de benchmark visual a metricas ergonomicas: angulo cervical, inclinacion del tronco, simetria de hombros y angulo de codo.
# - Anadir un segundo dataset mas frontal o mixto para acercarnos mejor al caso final del proyecto.
#
# Idea de criterio de seleccion final:
# - Si un modelo es muy rapido pero pierde hombros / codos / caderas con frecuencia, no sirve para el sistema final.
# - Si un modelo es algo mas lento pero ofrece keypoints mas estables y utiles para reglas ergonomicas, seguramente sea mejor eleccion.
#
