# Setup

Guía corta para levantar el proyecto en local.

## 1. Entorno Python

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Backend local:

```bash
python -m pip install -r requirements-api.txt
PYTHONPATH=src uvicorn ergonomics.api:app --host 0.0.0.0 --port 8000
```

Notebooks y benchmark completos:

```bash
python -m pip install -r requirements.txt
```

## 2. Variables de entorno

Para descargar datasets desde Roboflow crea un archivo `.env` en la raíz:

```bash
ROBOFLOW_API_KEY=tu_api_key
```

Variables opcionales de la aplicación:

```bash
ERGONOMICS_DB_PATH=data/app/postureos.sqlite3
ERGONOMICS_REQUIRE_AUTH=true
ERGONOMICS_SEED_DEFAULT_USERS=true
ERGONOMICS_MAX_UPLOAD_MB=8
YOLO_DEVICE=auto
```

## 3. Frontend

```bash
cd apps/web
npm install
npm run dev
```

La web arranca en `http://localhost:5173` y usa `http://localhost:8000` como backend por defecto.

## 4. Verificación

```bash
python -m compileall -q src notebooks/pose_benchmark/run_pose_batch.py
cd apps/web
npm run typecheck
npm run build
```

Para Playwright, deja `npm run dev` activo en otra terminal y ejecuta:

```bash
npm test
```
