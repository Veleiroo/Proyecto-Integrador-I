# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: vpc2
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fase 2: ejecucion larga del pipeline ergonomico
#
# Esta libreta esta pensada para una corrida larga y reanudable del pipeline actual.
#
# Objetivo:
# - ejecutar `MediaPipe Pose` sobre un lote grande o el dataset completo
# - guardar checkpoints incrementales para no perder trabajo
# - dejar artefactos estables para revisar resultados sin repetir toda la corrida
#
# Salidas esperadas por corrida:
# - `execution_manifest.csv`
# - `pose_landmarks.csv`
# - `ergonomic_analysis.csv`
# - `status_summary.csv`
# - `group_status_summary.csv`
# - `metric_summary_by_group.csv`
#
#

# %%
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
    return current


PROJECT_ROOT = find_project_root()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

plt.style.use("seaborn-v0_8-whitegrid")
pd.options.display.max_columns = 60
pd.options.display.float_format = "{:,.3f}".format

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"SRC_DIR: {SRC_DIR}")



# %%
from ergonomics import (
    ERGONOMICS_RESULTS_DIR,
    MEDIAPIPE_TASK_MODEL_PATH,
    MediaPipePoseConfig,
    build_group_status_summary,
    build_metric_summary_by_group,
    build_status_summary,
    draw_pose_overlay,
    plot_metric_by_group,
    plot_status_by_group,
    plot_status_distribution,
    run_incremental_long_pipeline,
    select_execution_records,
)



# %% [markdown]
# ## 1. Configuracion
#
# Esta es la celda clave.
# Para una corrida larga de verdad, deja `MAX_IMAGES = None`.
# Si solo quieres probar, puedes limitarla con un entero pequeno o con la variable de entorno `ERGO_MAX_IMAGES`.
#
#

# %%
DATASET_KEY = "posture_correction_v4_folder_v1"
EXECUTION_STRATEGY = "full_dataset"
SPLITS = ("train", "valid", "test")
MAX_IMAGES = None
BALANCED_PER_GROUP = 300
SEED = 11

VISIBILITY_THRESHOLD = 0.35
CHECKPOINT_EVERY = 100
RESUME_RUN = True

DEFAULT_RUN_LABEL = f"{DATASET_KEY}_{EXECUTION_STRATEGY}"
RUN_LABEL = os.getenv("ERGO_RUN_LABEL", DEFAULT_RUN_LABEL)

if os.getenv("ERGO_MAX_IMAGES"):
    MAX_IMAGES = int(os.environ["ERGO_MAX_IMAGES"])

RUN_OUTPUT_DIR = ERGONOMICS_RESULTS_DIR / RUN_LABEL
RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

display(
    Markdown(
        f"""
**Configuracion activa**

- Dataset: `{DATASET_KEY}`
- Estrategia: `{EXECUTION_STRATEGY}`
- Splits: `{SPLITS}`
- Max images: `{MAX_IMAGES}`
- Balanced per group: `{BALANCED_PER_GROUP}`
- Threshold de visibilidad: `{VISIBILITY_THRESHOLD}`
- Checkpoint cada: `{CHECKPOINT_EVERY}` imagenes
- Resume: `{RESUME_RUN}`
- Carpeta de salida: `{RUN_OUTPUT_DIR}`
- Modelo MediaPipe: `{MEDIAPIPE_TASK_MODEL_PATH}`
"""
    )
)



# %% [markdown]
# ## 2. Manifest de ejecucion
#
# Aqui se define la lista efectiva de imagenes a procesar.
# En la estrategia `full_dataset`, esto normalmente significa todo el dataset base salvo que limites `MAX_IMAGES`.
#
#

# %%
records_df = select_execution_records(
    DATASET_KEY,
    strategy=EXECUTION_STRATEGY,
    per_group=BALANCED_PER_GROUP,
    max_images=MAX_IMAGES,
    splits=SPLITS,
    seed=SEED,
)

display(records_df.head(12))
display(records_df.groupby(["split", "group"]).size().rename("image_count").reset_index())
print(f"Total de imagenes programadas: {len(records_df)}")



# %% [markdown]
# ## 3. Corrida larga y reanudable
#
# Esta celda puede tardar.
# Si la ejecucion se interrumpe, vuelve a lanzar la libreta con `RESUME_RUN = True`.
# El pipeline detectara las imagenes ya procesadas y seguira desde ahi.
#
#

# %%
pose_config = MediaPipePoseConfig(
    model_path=MEDIAPIPE_TASK_MODEL_PATH,
    min_visibility=VISIBILITY_THRESHOLD,
)

run_started_at = datetime.now()
artifacts = run_incremental_long_pipeline(
    records_df,
    run_label=RUN_LABEL,
    pose_config=pose_config,
    visibility_threshold=VISIBILITY_THRESHOLD,
    checkpoint_every=CHECKPOINT_EVERY,
    resume=RESUME_RUN,
)
run_finished_at = datetime.now()

display(
    pd.DataFrame(
        [
            {"item": "Run label", "value": RUN_LABEL},
            {"item": "Output dir", "value": str(artifacts.output_dir)},
            {"item": "Processed images", "value": artifacts.processed_images},
            {"item": "Started at", "value": str(run_started_at)},
            {"item": "Finished at", "value": str(run_finished_at)},
            {"item": "Duration", "value": str(run_finished_at - run_started_at)},
        ]
    )
)



# %% [markdown]
# ## 4. Cargar artefactos guardados
#
# A partir de aqui todo sale de los CSV guardados en disco.
# Eso permite revisar resultados sin rehacer la corrida completa.
#
#

# %%
manifest_df = pd.read_csv(artifacts.manifest_path)
pose_df = pd.read_csv(artifacts.pose_path)
analysis_df = pd.read_csv(artifacts.analysis_path)

status_summary_df = build_status_summary(analysis_df)
group_status_summary_df = build_group_status_summary(analysis_df)
metric_summary_df = build_metric_summary_by_group(
    analysis_df,
    metrics=[
        "shoulder_tilt_deg",
        "shoulder_height_diff_ratio",
        "head_lateral_offset_ratio",
        "neck_tilt_deg",
        "trunk_tilt_deg",
        "left_elbow_angle_deg",
        "right_elbow_angle_deg",
    ],
)

display(status_summary_df)
display(group_status_summary_df.head(12))
display(metric_summary_df)



# %% [markdown]
# ## 5. Lectura rapida
#
# Estas graficas son la primera capa de analisis para decidir si el pipeline esta listo para una fase mas estable
# o si todavia hace falta refinar reglas y umbrales.
#
#

# %%
fig, ax = plot_status_distribution(analysis_df)
plt.show()

fig, ax = plot_status_by_group(analysis_df, normalize=True)
plt.show()

fig, ax = plot_metric_by_group(
    analysis_df,
    metric="neck_tilt_deg",
    title="Inclinacion lateral media del cuello por grupo",
)
plt.show()

fig, ax = plot_metric_by_group(
    analysis_df,
    metric="shoulder_height_diff_ratio",
    title="Asimetria escapular media por grupo",
)
plt.show()



# %% [markdown]
# ## 6. Casos de ejemplo
#
# Esta seccion ayuda a revisar ejemplos representativos sin abrir manualmente decenas o cientos de imagenes.
#
#

# %%
examples_df = (
    analysis_df.dropna(subset=["overall_status"])
    .sort_values(
        by=["overall_status", "group", "head_lateral_offset_ratio", "shoulder_height_diff_ratio"],
        ascending=[True, True, False, False],
    )
    .groupby("overall_status", as_index=False)
    .head(2)
    .reset_index(drop=True)
)

display(
    examples_df[
        [
            "image_name",
            "group",
            "overall_status",
            "shoulder_tilt_deg",
            "shoulder_height_diff_ratio",
            "head_lateral_offset_ratio",
            "neck_tilt_deg",
            "feedback",
        ]
    ]
)



# %%
pose_lookup = pose_df.set_index("image_path")

fig, axes = plt.subplots(1, len(examples_df), figsize=(5.5 * len(examples_df), 5.5))
if len(examples_df) == 1:
    axes = [axes]

for ax, (_, row) in zip(axes, examples_df.iterrows(), strict=False):
    pose_row = pose_lookup.loc[row["image_path"]]
    draw_pose_overlay(row["image_path"], pose_row, ax=ax)
    ax.set_title(f"{row['overall_status']}\n{row['group']}")

plt.show()



# %% [markdown]
# ## 7. Resumen final
#
# Si esta corrida deja una salida razonable, el siguiente paso ya deberia ser valorar resultados y ajustar reglas,
# no rehacer la infraestructura.
#
#

# %%
final_snapshot_df = pd.DataFrame(
    [
        {"item": "Run label", "value": RUN_LABEL},
        {"item": "Manifest images", "value": len(manifest_df)},
        {"item": "Pose rows", "value": len(pose_df)},
        {"item": "Analysis rows", "value": len(analysis_df)},
        {"item": "Adequate", "value": int((analysis_df["overall_status"] == "adequate").sum())},
        {"item": "Improvable", "value": int((analysis_df["overall_status"] == "improvable").sum())},
        {"item": "Risk", "value": int((analysis_df["overall_status"] == "risk").sum())},
        {"item": "Results dir", "value": str(artifacts.output_dir)},
    ]
)
display(final_snapshot_df)

