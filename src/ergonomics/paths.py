from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
    return current


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
MEDIAPIPE_MODELS_DIR = MODELS_DIR / "mediapipe"
MEDIAPIPE_TASK_MODEL_PATH = MEDIAPIPE_MODELS_DIR / "pose_landmarker_lite.task"
POSE_BENCHMARK_RESULTS_DIR = PROJECT_ROOT / "notebooks" / "pose_benchmark" / "results"
ERGONOMICS_RESULTS_DIR = PROJECT_ROOT / "notebooks" / "ergonomics" / "results"
