#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if command -v python3 >/dev/null 2>&1; then
  exec python3 scripts/launch_local.py
fi

if command -v python >/dev/null 2>&1; then
  exec python scripts/launch_local.py
fi

echo "Python 3 no esta instalado o no esta en PATH." >&2
read -r -p "Pulsa Enter para cerrar..."
exit 1
