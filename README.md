# Proyecto Integrador I

Prueba de concepto de analisis ergonomico mediante vision por computador.

## Objetivo

El proyecto busca evaluar posturas de trabajo frente a una webcam y extraer una base tecnica para un futuro sistema de feedback ergonomico. En esta fase no hay producto final cerrado ni entrenamiento propio del modelo: el foco esta en comparar modelos de pose preentrenados y medir si sus keypoints son utiles para reglas ergonomicas.

## Estado actual

Lo que ya existe en la repo:

- documentacion base del proyecto en `Documentación/`
- benchmark inicial de modelos de pose en `notebooks/pose_benchmark/`
- comparativa de `YOLO Pose`, `MoveNet` y `MediaPipe Pose`
- dataset principal de arranque: `posture_correction_v4`

Decisiones de esta primera iteracion:

- `YOLO Pose` usa GPU si `torch` la detecta
- `MoveNet` se ejecuta en CPU por estabilidad
- `MediaPipe Pose` se ejecuta en CPU
- la comparativa actual se centra en tren superior, porque el dataset principal no siempre muestra bien caderas y tronco inferior

## Estructura

```text
Documentación/                PDFs del planning y del caso de uso
data/raw/                     Datasets descargados localmente
data/pose_subset/             Subsets generados para pruebas
models/                       Assets descargados por los notebooks
notebooks/pose_benchmark/     Notebook principal y utilidades del benchmark
```

## Notebook principal

Archivo principal:

- `notebooks/pose_benchmark/01_pose_benchmark_ergonomia.ipynb`

Que hace:

1. Selecciona el dataset activo
2. Escanea las imagenes y prepara un subset equilibrado
3. Ejecuta los tres modelos de pose
4. Genera tablas y graficas comparativas
5. Guarda resultados en `notebooks/pose_benchmark/results/`

## Como ejecutarlo

El flujo esperado es:

1. Descargar los datasets manualmente en `data/raw/`
2. Activar el entorno de trabajo
3. Abrir el notebook y ejecutarlo de arriba a abajo

Dependencias principales:

- `opencv-python`
- `mediapipe`
- `ultralytics`
- `tensorflow`
- `tensorflow-hub`
- `pandas`
- `tqdm`

## Datos y archivos generados

La repo esta configurada para no versionar:

- datasets descargados
- subsets generados
- pesos de modelos descargados
- resultados del benchmark
- caches de Python

Esto permite mantener Git limpio y dejar solo codigo, notebooks y documentacion.

## Siguiente paso

Tras esta comparativa inicial, los siguientes pasos serán:

- repetir el benchmark sobre un segundo dataset de validacion
- revisar fallos visuales por clase
- elegir modelo principal y modelo de respaldo
- empezar a convertir keypoints en reglas ergonomicas interpretables
