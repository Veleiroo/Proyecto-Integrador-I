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
# # Fase 2: pipeline ergonomico inicial
#
# Objetivo de esta libreta:
# - dejar fijado el contexto tecnico con el benchmark ya hecho
# - reutilizar modulos `.py` en vez de cargar el notebook con logica larga
# - ejecutar una primera demo funcional del pipeline ergonomico sobre imagenes reales
#
# Alcance de esta iteracion:
# - caso de uso de webcam frontal
# - analisis interpretable de tren superior
# - primera capa de variables y reglas ergonomicas
#
# Criterio de cierre:
# - seleccionar una muestra equilibrada del dataset base
# - ejecutar `MediaPipe Pose` sobre esa muestra
# - transformar landmarks en variables posturales legibles
# - clasificar cada imagen en `adequate`, `improvable`, `risk` o `insufficient_data`
#
#
#
#

# %%
from __future__ import annotations

import sys
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
pd.options.display.max_columns = 40
pd.options.display.float_format = "{:,.3f}".format

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"SRC_DIR: {SRC_DIR}")





# %%
from ergonomics import (
    MEDIAPIPE_TASK_MODEL_PATH,
    MediaPipePoseConfig,
    analyze_pose_dataframe,
    choose_reference_models,
    draw_pose_overlay,
    load_benchmark_artifacts,
    plot_dataset_group_distribution,
    plot_dataset_volumes,
    plot_group_comparison,
    plot_metric_by_group,
    plot_model_overview,
    plot_status_by_group,
    plot_status_distribution,
    rank_models,
    run_mediapipe_pose_batch,
    select_balanced_sample,
    summarize_available_datasets,
    summarize_dataset_groups,
)





# %% [markdown]
# ## 1. Punto de partida del proyecto
#
# Antes de movernos a reglas ergonomicas, conviene dejar documentado el suelo tecnico:
# - que datasets tenemos disponibles ya en local
# - que benchmark previo tenemos guardado
# - cual es el modelo de referencia para esta fase
#
# Esta seccion no reejecuta el benchmark. Solo recupera y resume lo que ya existe para no perder contexto entre iteraciones.
#
#
#
#

# %%
dataset_summary = summarize_available_datasets()
display(
    dataset_summary[
        ["label", "format", "is_available", "total_images", "split_count", "group_count", "notes"]
    ]
)

fig, ax = plot_dataset_volumes(dataset_summary)
plt.show()





# %%
artifacts = load_benchmark_artifacts()
ranked_models = rank_models(artifacts.summary)
decision = choose_reference_models(artifacts.summary)

display(
    ranked_models[
        [
            "model",
            "backend",
            "phase_fit_score",
            "upper_body_ready_pct",
            "required_keypoint_rate",
            "runtime_ms_median",
            "full_ergonomic_ready_pct",
        ]
    ]
)

display(
    pd.DataFrame(
        [
            {
                "primary_model": decision.primary_model,
                "backup_model": decision.backup_model,
                "current_scope": decision.current_scope,
                "rationale": decision.rationale,
            }
        ]
    )
)

fig, axes = plot_model_overview(artifacts.summary)
plt.show()

fig, ax = plot_group_comparison(
    artifacts.group_summary,
    metric="upper_body_ready_pct",
    title="Preparacion del tren superior por grupo y modelo",
)
plt.show()





# %% [markdown]
# ## 2. Dataset base y muestra de trabajo
#
# En vez de lanzar toda la fase 2 sobre miles de imagenes desde el principio, aqui usamos una muestra pequena y equilibrada.
# Esto permite:
# - validar rapidamente que el pipeline nuevo funciona
# - inspeccionar manualmente casos de ejemplo
# - iterar sobre reglas y umbrales sin esperar una ejecucion larga
#
# Si esta demo se comporta bien, el siguiente paso sera escalar la ejecucion a mas imagenes del mismo dataset.
#
#
#
#

# %%
TARGET_DATASET_KEY = "posture_correction_v4_folder_v1"
SAMPLE_PER_GROUP = 4
VISIBILITY_THRESHOLD = 0.35

target_group_summary = summarize_dataset_groups(TARGET_DATASET_KEY)
display(target_group_summary)

fig, ax = plot_dataset_group_distribution(
    target_group_summary,
    title="Distribucion del dataset base por split y grupo",
)
plt.show()

sample_df = select_balanced_sample(
    TARGET_DATASET_KEY,
    per_group=SAMPLE_PER_GROUP,
    splits=("train", "valid", "test"),
    seed=7,
)
display(sample_df)

display(
    Markdown(
        f"""
**Configuracion de la demo**

- Dataset base: `{TARGET_DATASET_KEY}`
- Muestra por grupo: `{SAMPLE_PER_GROUP}`
- Total de imagenes en la muestra: `{len(sample_df)}`
- Modelo de landmarks: `{decision.primary_model}`
- Asset local de MediaPipe: `{MEDIAPIPE_TASK_MODEL_PATH}`
"""
    )
)





# %% [markdown]
# ## 3. Inferencia de pose sobre la muestra
#
# Aqui ya dejamos el simple contexto y entramos en el siguiente paso real del proyecto.
# La idea es:
# 1. correr `MediaPipe Pose` sobre una muestra pequena del dataset base
# 2. guardar landmarks normalizados y visibilidades
# 3. reutilizar esos landmarks como entrada del modulo ergonomico
#
# En esta iteracion no hacemos feedback en tiempo real ni video. Solo validamos el flujo sobre imagen estatica.
#
#
#
#

# %%
pose_config = MediaPipePoseConfig(
    model_path=MEDIAPIPE_TASK_MODEL_PATH,
    min_visibility=VISIBILITY_THRESHOLD,
)
pose_demo_df = run_mediapipe_pose_batch(sample_df, config=pose_config)

display(
    pose_demo_df[
        [
            "image_name",
            "group",
            "split",
            "pose_detected",
            "visible_landmarks_count",
            "pose_landmarks_count",
        ]
    ]
)





# %% [markdown]
# ## 4. Variables posturales y reglas iniciales
#
# En esta primera version no intentamos resolver un ROSA completo.
# Nos centramos en variables que si tienen sentido con webcam frontal y tren superior:
# - inclinacion de hombros
# - simetria escapular a partir de la diferencia de altura entre hombros
# - desplazamiento lateral de la cabeza respecto al eje de hombros
# - inclinacion lateral del cuello
# - inclinacion del tronco visible
# - angulos de codo izquierdo y derecho
#
# Las reglas son deliberadamente simples e interpretables. Si funcionan y son utiles, luego se refinan.
#
#
#
#

# %%
analysis_df = analyze_pose_dataframe(
    pose_demo_df,
    visibility_threshold=VISIBILITY_THRESHOLD,
)

demo_results_df = pose_demo_df[
    [
        "image_path",
        "visible_landmarks_count",
        "pose_landmarks_count",
        "pose_detected",
    ]
].merge(
    analysis_df,
    on=["image_path", "pose_detected"],
    how="inner",
)

display(
    demo_results_df[
        [
            "image_name",
            "group",
            "overall_status",
            "shoulder_tilt_deg",
            "shoulder_height_diff_ratio",
            "head_lateral_offset_ratio",
            "neck_tilt_deg",
            "trunk_tilt_deg",
            "left_elbow_angle_deg",
            "right_elbow_angle_deg",
            "feedback",
        ]
    ]
)





# %%
fig, ax = plot_status_distribution(demo_results_df)
plt.show()

fig, ax = plot_metric_by_group(
    demo_results_df,
    metric="shoulder_tilt_deg",
    title="Media de inclinacion de hombros por grupo",
)
plt.show()

fig, ax = plot_metric_by_group(
    demo_results_df,
    metric="head_lateral_offset_ratio",
    title="Desviacion lateral media de la cabeza por grupo",
)
plt.show()

fig, ax = plot_metric_by_group(
    demo_results_df,
    metric="neck_tilt_deg",
    title="Inclinacion lateral media del cuello por grupo",
)
plt.show()





# %% [markdown]
# ## 5. Inspeccion visual de casos de ejemplo
#
# Esta parte es importante para la siguiente iteracion:
# no basta con obtener una etiqueta final; necesitamos ver si los landmarks y las reglas tienen sentido visualmente.
# Por eso mostramos algunos ejemplos de la muestra con overlays sencillos de pose.
#
#
#
#

# %%
examples_df = (
    demo_results_df.sort_values(
        by=["overall_status", "group", "visible_landmarks_count"],
        ascending=[False, True, False],
    )
    .groupby("group", as_index=False)
    .head(1)
    .reset_index(drop=True)
)

fig, axes = plt.subplots(1, len(examples_df), figsize=(5.5 * len(examples_df), 5.5))
if len(examples_df) == 1:
    axes = [axes]

pose_lookup = pose_demo_df.set_index("image_path")

for ax, (_, example_row) in zip(axes, examples_df.iterrows(), strict=False):
    pose_row = pose_lookup.loc[example_row["image_path"]]
    draw_pose_overlay(example_row["image_path"], pose_row, ax=ax)
    ax.set_title(f"{example_row['group']}\n{example_row['overall_status']}")

plt.show()





# %%
display(
    examples_df[
        [
            "image_name",
            "group",
            "overall_status",
            "feedback",
        ]
    ]
)





# %% [markdown]
# ## 6. Lectura de esta iteracion
#
# Lo importante de esta libreta no es solo que ejecute, sino lo que deja listo para la siguiente:
# - ya tenemos un flujo modular `muestra -> landmarks -> variables -> reglas -> feedback`
# - el notebook ya no contiene la logica pesada
# - los umbrales y las reglas quedan en modulos que podremos ajustar sin reescribir la libreta
#
# Siguiente paso recomendado:
# - correr esta misma tuberia sobre mas imagenes del dataset base
# - revisar ejemplos falsos positivos y falsos negativos para ajustar reglas
# - decidir que reglas pasan a la siguiente fase estable del proyecto
#
#
#
#

# %%
phase_snapshot = pd.DataFrame(
    [
        {"item": "Modelo principal", "value": decision.primary_model},
        {"item": "Modelo de respaldo", "value": decision.backup_model},
        {"item": "Dataset de arranque", "value": TARGET_DATASET_KEY},
        {"item": "Total de imagenes de la demo", "value": len(sample_df)},
        {"item": "Threshold de visibilidad", "value": VISIBILITY_THRESHOLD},
        {"item": "Siguiente modulo esperado", "value": "mejora de variables y ajuste de reglas"},
    ]
)
display(phase_snapshot)




# %% [markdown]
# ## 7. Escalado controlado del pipeline
#
# Una demo de 12 imagenes sirve para comprobar que la tuberia funciona, pero sigue siendo pequena para leer patrones.
# En esta ultima seccion ampliamos la muestra de forma moderada para responder a una pregunta mas util:
#
# - que distribucion de estados sale por grupo cuando analizamos un lote algo mayor
#
# No buscamos precision final todavia. Solo comprobar que la fase 2 ya es escalable sin cambiar de estructura.
#
#

# %%
EXPANDED_SAMPLE_PER_GROUP = 12

expanded_sample_df = select_balanced_sample(
    TARGET_DATASET_KEY,
    per_group=EXPANDED_SAMPLE_PER_GROUP,
    splits=("train", "valid", "test"),
    seed=11,
)
expanded_pose_df = run_mediapipe_pose_batch(expanded_sample_df, config=pose_config)
expanded_analysis_df = analyze_pose_dataframe(
    expanded_pose_df,
    visibility_threshold=VISIBILITY_THRESHOLD,
)

expanded_summary_df = (
    expanded_analysis_df.groupby(["group", "overall_status"], dropna=False)
    .size()
    .rename("image_count")
    .reset_index()
    .sort_values(["group", "overall_status"])
)

display(expanded_summary_df)
display(expanded_analysis_df["overall_status"].value_counts().rename("image_count"))



# %%
fig, ax = plot_status_distribution(expanded_analysis_df)
plt.show()

fig, ax = plot_status_by_group(expanded_analysis_df, normalize=True)
plt.show()

fig, ax = plot_metric_by_group(
    expanded_analysis_df,
    metric="neck_tilt_deg",
    title="Inclinacion lateral media del cuello por grupo en el lote ampliado",
)
plt.show()



# %%
expanded_phase_snapshot = pd.DataFrame(
    [
        {"item": "Imagenes del lote ampliado", "value": len(expanded_analysis_df)},
        {"item": "Adequate", "value": int((expanded_analysis_df["overall_status"] == "adequate").sum())},
        {"item": "Improvable", "value": int((expanded_analysis_df["overall_status"] == "improvable").sum())},
        {"item": "Risk", "value": int((expanded_analysis_df["overall_status"] == "risk").sum())},
    ]
)
display(expanded_phase_snapshot)

