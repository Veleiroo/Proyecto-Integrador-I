from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = PROJECT_ROOT / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
REQUIRED_PATHS = [
    PROJECT_ROOT / "models" / "mediapipe" / "pose_landmarker_lite.task",
    PROJECT_ROOT / "models" / "yolo" / "yolo11s-pose.pt",
    PROJECT_ROOT / "apps" / "web" / "dist" / "index.html",
]


def _run(command: list[str | Path]) -> None:
    subprocess.run([str(item) for item in command], cwd=PROJECT_ROOT, check=True)


def _launcher_python() -> str:
    return sys.executable or "python"


def _runtime_is_ready() -> bool:
    return VENV_PYTHON.exists() and all(path.exists() for path in REQUIRED_PATHS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Install missing runtime pieces and start PostureOS.")
    parser.add_argument("--force-bootstrap", action="store_true", help="Reinstall dependencies and rebuild the frontend before starting.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically.")
    parser.add_argument("--dev-web", action="store_true", help="Start Vite in development mode.")
    args = parser.parse_args()

    if args.force_bootstrap or not _runtime_is_ready():
        _run([_launcher_python(), "scripts/bootstrap_local.py", "--skip-web"])

    start_command: list[str | Path] = [VENV_PYTHON, "scripts/start_local.py"]
    if args.no_browser:
        start_command.append("--no-browser")
    if args.dev_web:
        start_command.append("--dev-web")
    _run(start_command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
