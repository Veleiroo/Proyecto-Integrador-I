from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import venv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = PROJECT_ROOT / ".venv"
MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _run(command: list[str | Path], *, cwd: Path = PROJECT_ROOT) -> None:
    subprocess.run([str(item) for item in command], cwd=cwd, check=True)


def _ensure_venv() -> Path:
    if not _venv_python().exists():
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    return _venv_python()


def _install_python_dependencies(python: Path) -> None:
    _run([python, "-m", "pip", "install", "--upgrade", "pip"])
    _run([python, "-m", "pip", "install", "-r", "requirements-api.txt"])


def _download_mediapipe_model() -> None:
    target = PROJECT_ROOT / "models" / "mediapipe" / "pose_landmarker_lite.task"
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, target)


def _download_yolo_model(python: Path) -> None:
    target = PROJECT_ROOT / "models" / "yolo" / "yolo11s-pose.pt"
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    code = """
from pathlib import Path
from ultralytics import YOLO

target = Path("models/yolo/yolo11s-pose.pt")
YOLO("yolo11s-pose.pt")
Path("yolo11s-pose.pt").replace(target)
"""
    _run([python, "-c", code])


def _install_web_dependencies(*, build: bool) -> None:
    if shutil.which("npm") is None:
        raise RuntimeError("npm no esta disponible. Instala Node.js 20+ o ejecuta con --skip-web.")
    web_dir = PROJECT_ROOT / "apps" / "web"
    _run(["npm", "install"], cwd=web_dir)
    if build:
        _run(["npm", "run", "build"], cwd=web_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the local PostureOS runtime.")
    parser.add_argument("--skip-web", action="store_true", help="Do not install or build the React frontend.")
    parser.add_argument("--skip-web-build", action="store_true", help="Install frontend dependencies without building.")
    args = parser.parse_args()

    python = _ensure_venv()
    _install_python_dependencies(python)
    _download_mediapipe_model()
    _download_yolo_model(python)
    if not args.skip_web:
        _install_web_dependencies(build=not args.skip_web_build)

    print("Local runtime ready.")
    print(f"Python: {python}")
    print(f"Backend: PYTHONPATH=src {python} -m uvicorn ergonomics.api:app --host 0.0.0.0 --port 8000")
    print("Frontend: cd apps/web && npm run dev")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
