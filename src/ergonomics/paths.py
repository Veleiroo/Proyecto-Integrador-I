from __future__ import annotations

from pathlib import Path

# --- LÓGICA DE LOCALIZACIÓN DEL PROYECTO ---

def find_project_root(start: Path | None = None) -> Path:
    """
    Busca la carpeta raíz del proyecto subiendo por el árbol de directorios.
    
    Sube por el árbol de carpetas hasta encontrar el directorio que contiene '.git'.
    Esto garantiza que las rutas funcionen igual en el ordenador de cualquier 
    miembro del equipo.
    """
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
        
    for candidate in [current, *current.parents]:
        # El archivo .root indica que estamos en la raíz del repositorio
        if (candidate / ".root").exists():
            return candidate
            
    # Si no lo encontramos, subimos 2 niveles desde src/ergonomics/
    return current.parents[1] if len(current.parents) > 1 else current

# --- DEFINICIÓN DE RUTAS GLOBALES ---

# Punto de referencia principal: la raíz del proyecto
PROJECT_ROOT = find_project_root(Path(__file__).resolve())

# Carpetas de datos
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Carpetas de modelos
MODELS_DIR = PROJECT_ROOT / "models"
MEDIAPIPE_MODELS_DIR = MODELS_DIR / "mediapipe"
MEDIAPIPE_TASK_MODEL_PATH = MEDIAPIPE_MODELS_DIR / "pose_landmarker_lite.task"
YOLO_MODELS_DIR = MODELS_DIR / "yolo"
YOLO_POSE_WEIGHTS_PATH = YOLO_MODELS_DIR / "yolo11s-pose.pt"

# Carpetas de resultados
POSE_BENCHMARK_RESULTS_DIR = PROJECT_ROOT / "notebooks" / "pose_benchmark" / "results"
ERGONOMICS_RESULTS_DIR = PROJECT_ROOT / "notebooks" / "ergonomics" / "results"
