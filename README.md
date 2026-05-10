# Proyecto Integrador I

Prueba de concepto de análisis ergonómico mediante visión por computador para puestos de trabajo sedentarios.

El objetivo es detectar desviaciones posturales a partir de cámara o imagen, convertirlas en métricas interpretables y ofrecer feedback preventivo. No está planteado como herramienta de supervisión laboral ni como sistema de identificación.

## Estado Actual

El repositorio contiene tres bloques:

1. Benchmark de modelos de pose para elegir el detector más útil en este caso.
2. Pipeline ergonómico modular para webcam frontal.
3. Extensión lateral y aplicación local con backend FastAPI y frontend React/Vite.

Decisiones técnicas principales:

- MediaPipe Pose es el modelo principal para la vista frontal.
- YOLO11s Pose se usa en vista lateral porque ofrece mejor cobertura de perfil.
- MoveNet queda como modelo comparativo en el benchmark, no como opción principal.
- El MVP frontal se centra en cabeza/cuello, hombros y tronco.
- Los codos se conservan como métrica, pero no elevan por sí solos el diagnóstico porque suelen quedar fuera de plano.

## Estructura

```text
docs/                         Documentación del proyecto en PDF
data/raw/                     Datasets locales descargados manualmente
data/pose_subset/             Subsets generados para pruebas
models/                       Modelos descargados localmente
notebooks/pose_benchmark/     Benchmark de modelos de pose
notebooks/ergonomics/         Pipeline, auditoría y calibración
src/ergonomics/               Código reutilizable del pipeline y backend
apps/web/                     Interfaz React/Vite
```

Los datos, modelos y resultados generados están ignorados por Git. El repositorio versiona código, notebooks y documentación, no artefactos pesados o regenerables.

## Datasets

Los datasets se registran en `src/ergonomics/datasets.py`.

- `posture_correction_v4_folder_v1`: dataset frontal principal del MVP.
- `sitting_posture_4keypoint`: dataset mayormente lateral para validar perfil.
- `multiposture_zenodo_14230872`: keypoints 3D con etiquetas expertas de postura sentada, útil para calibrar tronco.
- `desk_posture_coco_v1`: dataset pequeño para contraste lateral o dorsal.
- `posture_detection_folder_v1` y `sitting_posture_folder_v1`: datasets auxiliares con etiquetas más ruidosas.

## Pipeline

Módulos principales:

- `paths.py`: rutas del proyecto.
- `datasets.py`: catálogo y lectura de datasets.
- `sampling.py`: selección de muestras.
- `pose_inference.py`: inferencia frontal con MediaPipe.
- `yolo_pose_inference.py`: inferencia lateral con YOLO Pose.
- `posture_rules.py`: reglas frontales.
- `lateral_rules.py`: reglas laterales.
- `long_run.py`: ejecución incremental sobre datasets.
- `audit.py`: auditoría de resultados y calibración.
- `reporting.py` y `visualization.py`: tablas y gráficas.
- `app_service.py`, `api.py`, `app_storage.py`, `app_security.py`, `app_config.py`: capa de aplicación local.

### Métricas Frontales

- `shoulder_tilt_deg`
- `shoulder_height_diff_ratio`
- `head_lateral_offset_ratio`
- `neck_tilt_deg`
- `trunk_tilt_deg`
- `left_elbow_angle_deg`
- `right_elbow_angle_deg`

### Métricas Laterales

- `head_forward_offset_ratio`
- `neck_forward_tilt_deg`
- `trunk_forward_tilt_deg`
- `shoulder_hip_offset_ratio`
- `lateral_elbow_angle_deg`

Los estados posibles son `adequate`, `improvable`, `risk` e `insufficient_data`.

## Notebooks

Orden recomendado:

1. `notebooks/pose_benchmark/01_pose_benchmark_ergonomia.ipynb`
2. `notebooks/ergonomics/02_pipeline_ergonomico_base.ipynb`
3. `notebooks/ergonomics/03_pipeline_ergonomico_long_run.ipynb`
4. `notebooks/ergonomics/04_auditoria_frontal_y_calibracion.ipynb`
5. `notebooks/ergonomics/05_pipeline_lateral_yolo.ipynb`
6. `notebooks/ergonomics/06_calibracion_lateral_multiposture.ipynb`

## Aplicación Local

La app se ejecuta en local:

- Backend FastAPI: `src/ergonomics/api.py`
- Frontend React/Vite: `apps/web/`
- Base de datos SQLite local cifrada.
- Las imágenes se procesan como temporales y se eliminan al terminar.
- El perfil técnico (`role=dev`) tiene una pantalla de depuración visual que guarda capturas en `data/app/dev_captures/`.

Endpoints principales:

- `POST /api/auth/login`
- `GET /api/auth/me`
- `GET /api/summary`
- `GET /api/analyses`
- `DELETE /api/analyses`
- `POST /api/analyze/front`
- `POST /api/analyze/lateral`
- `POST /api/analyze/combined`
- `POST /api/dev/analyze-image` solo para perfil técnico.

## Instalación Rápida

Backend:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-api.txt
PYTHONPATH=src uvicorn ergonomics.api:app --host 0.0.0.0 --port 8000
```

Instalacion automatizada:

```bash
python scripts/bootstrap_local.py
python scripts/start_local.py
```

Lanzadores de escritorio:

```text
launchers/windows/PostureOS.bat
launchers/windows/PostureOS.ps1
launchers/linux-macos/postureos.sh
launchers/linux-macos/PostureOS.command
```

Estos lanzadores ejecutan `scripts/launch_local.py`: si falta `.venv`, modelos o `apps/web/dist`, preparan el entorno y luego abren la aplicación en `http://localhost:8000`.

Modelos locales:

```bash
mkdir -p models/mediapipe models/yolo
curl -L \
  -o models/mediapipe/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
python - <<'PY'
from pathlib import Path
from ultralytics import YOLO

target = Path("models/yolo/yolo11s-pose.pt")
if not target.exists():
    YOLO("yolo11s-pose.pt")
    Path("yolo11s-pose.pt").replace(target)
PY
```

Frontend:

```bash
cd apps/web
npm install
npm run dev
```

La web usa `VITE_API_BASE_URL`; si no se define, conecta a `http://localhost:8000`.

## Variables Útiles

```bash
ERGONOMICS_DB_PATH=/ruta/local/postureos.sqlite3
ERGONOMICS_REQUIRE_AUTH=true
ERGONOMICS_SEED_DEFAULT_USERS=true
ERGONOMICS_MAX_UPLOAD_MB=8
YOLO_DEVICE=auto
YOLO_POSE_WEIGHTS_PATH=models/yolo/yolo11s-pose.pt
```

Si no se define `ERGONOMICS_SECRET_KEY`, el backend genera una clave local junto a la base de datos. En modo local se crean usuarios iniciales `admin`/`admin` y `Pablo`/`1234`, salvo que se desactive con `ERGONOMICS_SEED_DEFAULT_USERS=false`.

Los usuarios iniciales solo se crean si no existen, por lo que un cambio manual de contraseña no se sobrescribe en reinicios posteriores.

## Empaquetado Local

El proyecto puede distribuirse como una carpeta ejecutable con:

- `scripts/bootstrap_local.py`: crea `.venv`, instala dependencias, descarga MediaPipe y YOLO11s, e instala/construye el frontend.
- `scripts/start_local.py`: arranca el servidor local y sirve la web compilada desde FastAPI.
- `scripts/launch_local.py`: lanzador idempotente; instala lo que falte y arranca la app.
- `launchers/`: accesos directos por sistema operativo.

Para un `.exe` nativo de Windows, la ruta recomendable es envolver `scripts/launch_local.py` con PyInstaller o un instalador ligero. No conviene meter los pesos ni `node_modules` en Git; el bootstrap los descarga y regenera.

Para los notebooks completos:

```bash
python -m pip install -r requirements.txt
```

## Pruebas

Backend y módulos Python:

```bash
python -m compileall -q src notebooks/pose_benchmark/run_pose_batch.py
```

Frontend:

```bash
cd apps/web
npm run typecheck
npm run build
npm test
```

Las pruebas de Playwright esperan que la web esté disponible en `http://localhost:5173`, salvo que se defina `PLAYWRIGHT_BASE_URL`.

## Documentación

La documentación de referencia está en `docs/`:

- `AVP1 (Plan de proyecto).pdf`
- `Caso_de_uso_Salud_DATOS.pdf`
