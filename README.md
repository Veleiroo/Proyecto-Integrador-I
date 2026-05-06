# Proyecto Integrador I

Prueba de concepto de analisis ergonomico mediante vision por computador para puestos de trabajo sedentarios.

## Objetivo

El proyecto busca construir una base tecnica para un futuro sistema de feedback ergonomico preventivo a partir de imagenes de webcam. La idea no es hacer supervision laboral ni reconocimiento de identidad, sino detectar desviaciones posturales y traducirlas a reglas interpretables y recomendaciones simples.

La documentacion de referencia esta en `Documentación/`:

- `AVP1 (Plan de proyecto).pdf`
- `Caso_de_uso_Salud_DATOS.pdf`

## Estado actual

La repo ya no esta solo en fase de benchmark. Ahora mismo hay tres bloques de trabajo:

1. Benchmark de modelos de pose
2. Pipeline ergonomico modular sobre el caso de webcam frontal
3. Extension lateral para medir cabeza adelantada y tronco en perfil

### Decisiones tecnicas tomadas

- `MediaPipe Pose` es el modelo principal para el caso frontal actual.
- `YOLO Pose` es el modelo elegido para vista lateral porque el benchmark de perfil da mejor cobertura de keypoints y mas imagenes listas para analisis lateral.
- `MoveNet` queda descartado para lateral por baja disponibilidad de keypoints utiles.
- El MVP actual se centra en tren superior frontal: cabeza, cuello y simetria de hombros.
- El analisis de cabeza adelantada y tronco queda planteado como extension lateral, no como requisito del MVP frontal.

## Estructura

```text
Documentación/                PDFs del planning y del caso de uso
data/raw/                     Datasets descargados localmente
data/pose_subset/             Subsets generados para pruebas
models/                       Assets descargados por los notebooks
notebooks/pose_benchmark/     Benchmark inicial de modelos de pose
notebooks/ergonomics/         Notebooks de pipeline, ejecucion larga y auditoria
src/ergonomics/               Modulos reutilizables del pipeline ergonomico
```

## Datasets usados

Los datasets se resuelven desde `src/ergonomics/datasets.py`.

- `posture_correction_v4_folder_v1`
  Dataset principal del MVP. Webcam frontal. Es el mas cercano al caso real del proyecto.
- `sitting_posture_4keypoint`
  Dataset mas lateral. Buen candidato para validar una futura extension de perfil.
- `multiposture_zenodo_14230872`
  Dataset de keypoints 3D, sin imagenes, con etiquetas expertas de postura sentada. Es mejor candidato para calibrar reglas laterales de tronco que los datasets visuales `Good/Bad`.
- `desk_posture_coco_v1`
  Dataset pequeno, tambien util para contraste lateral o dorsal.
- `posture_detection_folder_v1`
  Dataset auxiliar por carpetas con bastante ruido en etiquetas.
- `sitting_posture_folder_v1`
  Dataset auxiliar por carpetas con mezcla de clases y algunos casos ruidosos.

## Benchmark de pose

Notebook principal:

- `notebooks/pose_benchmark/01_pose_benchmark_ergonomia.ipynb`

Script asociado:

- `notebooks/pose_benchmark/run_pose_batch.py`

Que hace:

1. Selecciona un dataset activo
2. Construye un subset equilibrado
3. Ejecuta `YOLO Pose`, `MoveNet` y `MediaPipe Pose`
4. Genera tablas y graficas comparativas
5. Guarda resultados en `notebooks/pose_benchmark/results/`

Resultado practico de esta fase:

- `MediaPipe Pose` sale como opcion principal para el dataset frontal base
- `YOLO Pose` no compensa para el caso frontal, pero si gana en la comparativa lateral
- `MoveNet` es util como contraste, pero queda por detras en cobertura ergonomica de tren superior y lateral

## Pipeline ergonomico

El pipeline modular esta en `src/ergonomics/`:

- `paths.py`
- `datasets.py`
- `sampling.py`
- `pose_inference.py`
- `yolo_pose_inference.py`
- `posture_rules.py`
- `lateral_rules.py`
- `reporting.py`
- `visualization.py`
- `long_run.py`
- `audit.py`

### Notebooks de ergonomia

- `notebooks/ergonomics/02_pipeline_ergonomico_base.ipynb`
  Primera validacion del flujo `imagen -> pose -> variables -> reglas -> feedback`.
- `notebooks/ergonomics/03_pipeline_ergonomico_long_run.ipynb`
  Corrida larga y reanudable sobre un lote grande o el dataset completo.
- `notebooks/ergonomics/04_auditoria_frontal_y_calibracion.ipynb`
  Auditoria de etiquetas, falsos positivos y candidatos de recalibracion para el caso frontal.
- `notebooks/ergonomics/05_pipeline_lateral_yolo.ipynb`
  Primera validacion del flujo lateral `imagen de perfil -> YOLO Pose -> variables laterales -> reglas -> feedback`.
- `notebooks/ergonomics/06_calibracion_lateral_multiposture.ipynb`
  Calibracion de variables laterales usando keypoints 3D y etiquetas expertas de MultiPosture.

### Variables que ya se calculan

En el estado actual, el motor de reglas mide:

- `shoulder_tilt_deg`
- `shoulder_height_diff_ratio`
- `head_lateral_offset_ratio`
- `neck_tilt_deg`
- `trunk_tilt_deg`
- `left_elbow_angle_deg`
- `right_elbow_angle_deg`

Estas variables se traducen a estados:

- `adequate`
- `improvable`
- `risk`
- `insufficient_data`

### Variables laterales iniciales

Para la vista de perfil, el motor lateral mide:

- `head_forward_offset_ratio`
- `neck_forward_tilt_deg`
- `trunk_forward_tilt_deg`
- `shoulder_hip_offset_ratio`
- `lateral_elbow_angle_deg`

La vista lateral no exige ver ambos lados del cuerpo. El sistema escoge el lado con mejor cadena visible: nariz, hombro, codo y cadera.

## Hallazgos actuales

La corrida larga ya ejecutada sobre `posture_correction_v4_folder_v1` confirma varias cosas:

- El encuadre frontal da muy buena señal para nariz y hombros.
- Caderas y muñecas casi nunca aparecen, asi que no tiene sentido dar mucho peso a tronco y codo completo en el MVP frontal.
- El grupo `looks good` esta cayendo demasiado en `risk`, lo que indica que hay umbrales por recalibrar.
- La auditoria actual apunta a que `shoulder_height_diff_ratio` esta siendo demasiado estricto para este dataset.
- La medicion de cabeza adelantada no queda bien resuelta con este encuadre frontal y debe tratarse como linea lateral.
- El dataset `sitting_posture_4keypoint` es el candidato principal para perfil porque contiene imagenes laterales de escritorio con clases `Good` y `Bad`.
- Las etiquetas `Good` y `Bad` de los datasets laterales visuales no deben tomarse como verdad ROSA. MultiPosture aporta una referencia mas fiable para calibrar tronco.
- Tras recalibrar tronco con MultiPosture, la corrida lateral queda mas informativa: `Bad` concentra mas riesgo que `Good`, mientras cabeza/cuello se mantiene como señal auxiliar pendiente de un dataset especifico.

## Como trabajar con la repo

Flujo recomendado:

1. Descargar manualmente los datasets en `data/raw/`
2. Activar el entorno de trabajo
3. Ejecutar el notebook que corresponda a la fase que quieras revisar

Orden sugerido:

1. `notebooks/pose_benchmark/01_pose_benchmark_ergonomia.ipynb`
2. `notebooks/ergonomics/02_pipeline_ergonomico_base.ipynb`
3. `notebooks/ergonomics/03_pipeline_ergonomico_long_run.ipynb`
4. `notebooks/ergonomics/04_auditoria_frontal_y_calibracion.ipynb`
5. `notebooks/ergonomics/05_pipeline_lateral_yolo.ipynb`
6. `notebooks/ergonomics/06_calibracion_lateral_multiposture.ipynb`

Dependencias principales:

- `opencv-python`
- `mediapipe`
- `tensorflow`
- `tensorflow-hub`
- `ultralytics`
- `pandas`

## Aplicacion web

La primera version de aplicacion esta separada en dos piezas:

- Backend FastAPI en `src/ergonomics/api.py`
- Frontend React/Vite en `apps/web/`

La separacion de codigo queda asi:

- `src/ergonomics/pose_inference.py`, `posture_rules.py`, `yolo_pose_inference.py`, `lateral_rules.py` y modulos cercanos: nucleo cientifico usado por notebooks y app.
- `src/ergonomics/app_service.py`: adaptador de inferencia y reglas para producto.
- `src/ergonomics/api.py`, `app_config.py`, `app_security.py`, `app_storage.py`: capa de aplicacion, API, autenticacion y persistencia.
- `notebooks/`: justificacion experimental y trazabilidad de decisiones.

La web ya incluye una pantalla principal de camara para comprobar encuadre e iniciar una sesion de seguimiento, mas una pantalla de revision por imagen como respaldo si la camara no esta disponible. Tambien incorpora modo claro/oscuro persistente para presentacion y uso real.

La aplicacion queda planteada como local-first: el backend corre en el ordenador del usuario, usa SQLite local, separa usuarios con login, guarda contrasenas hasheadas y cifra resultados. Las imagenes no se guardan por defecto: el API las procesa como temporales y las borra al terminar.

Para levantar el backend:

```bash
python -m pip install -r requirements-api.txt
PYTHONPATH=src uvicorn ergonomics.api:app --host 0.0.0.0 --port 8000
```

Variables recomendadas en despliegue:

```bash
ERGONOMICS_DB_PATH=/ruta/local/postureos.sqlite3
ERGONOMICS_REQUIRE_AUTH=true
ERGONOMICS_SEED_DEFAULT_USERS=true
```

Si no se define `ERGONOMICS_SECRET_KEY`, el backend genera una clave local en un archivo `.key` junto a la base de datos. En modo local se crean usuarios iniciales `admin`/`admin` y `Pablo`/`1234`, salvo que se desactive con `ERGONOMICS_SEED_DEFAULT_USERS=false`.

Para levantar la web:

```bash
cd apps/web
npm install
npm run dev
```

En el modo objetivo, backend y web se ejecutan localmente. El modo centralizado queda como extension opcional para empresas que ya dispongan de infraestructura propia.
- `matplotlib`
- `tqdm`
- `jupytext`
- `nbclient`

## Artefactos generados

La repo esta configurada para no versionar:

- datasets descargados
- subsets generados
- modelos descargados
- resultados del benchmark
- resultados de `notebooks/ergonomics/results/`
- caches de Python

Esto permite dejar en Git el codigo, los notebooks y la documentacion, pero no los artefactos pesados o regenerables.

## Siguiente paso recomendado

El siguiente paso tecnico con mas sentido es ejecutar y auditar la extension lateral:

- ejecutar `05_pipeline_lateral_yolo.ipynb` sobre el dataset lateral completo
- revisar visualmente casos `Good` y `Bad` clasificados como `risk` o `improvable`
- usar `06_calibracion_lateral_multiposture.ipynb` para recalibrar umbrales laterales de tronco
- integrar despues la decision frontal + lateral en una lectura ergonomica conjunta
