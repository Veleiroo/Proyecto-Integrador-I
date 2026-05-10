from __future__ import annotations

import os
import argparse
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = PROJECT_ROOT / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")


def _start(command: list[str | Path], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.Popen:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.Popen([str(item) for item in command], cwd=cwd, env=merged_env)


def main() -> int:
    parser = argparse.ArgumentParser(description="Start PostureOS locally.")
    parser.add_argument("--dev-web", action="store_true", help="Run the Vite dev server instead of serving the built web app.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically.")
    args = parser.parse_args()

    if not VENV_PYTHON.exists():
        print("No existe .venv. Ejecuta primero: python scripts/bootstrap_local.py", file=sys.stderr)
        return 1

    backend_env = {"PYTHONPATH": "src"}
    backend = _start(
        [VENV_PYTHON, "-m", "uvicorn", "ergonomics.api:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=PROJECT_ROOT,
        env=backend_env,
    )
    processes = [backend]
    frontend = None
    if args.dev_web:
        frontend = _start(["npm", "run", "dev"], cwd=PROJECT_ROOT / "apps" / "web")
        processes.append(frontend)

    def stop(_signum: int | None = None, _frame: object | None = None) -> None:
        for process in processes:
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    print("Backend: http://localhost:8000")
    print("Frontend: http://localhost:5173" if args.dev_web else "App: http://localhost:8000")
    if not args.no_browser:
        webbrowser.open("http://localhost:5173" if args.dev_web else "http://localhost:8000")

    try:
        while True:
            for process in processes:
                code = process.poll()
                if code is not None:
                    stop()
                    return code
            time.sleep(1)
    except KeyboardInterrupt:
        stop()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
