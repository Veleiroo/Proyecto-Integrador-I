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
# # Fase 2: auditoria frontal y calibracion ergonomica
#
# Esta libreta no repite la inferencia de pose. Trabaja sobre la corrida larga ya guardada
# para responder tres preguntas:
#
# - que partes del cuerpo estan realmente visibles en el encuadre frontal de webcam
# - que contradicciones hay entre las etiquetas del dataset y la salida del motor de reglas
# - que umbrales conviene revisar antes de pasar al siguiente paso del proyecto
#
# El foco es el caso de uso realista de webcam frontal. La linea lateral queda recogida
# como extension futura o validacion complementaria, no como bloqueo del MVP frontal.

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
pd.options.display.max_columns = 80
pd.options.display.float_format = "{:,.3f}".format

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"SRC_DIR: {SRC_DIR}")


# %%
from ergonomics import (
    DATASET_CATALOG,
    DEFAULT_AUDIT_LANDMARKS,
    build_component_status_summary,
    build_keypoint_visibility_summary,
    build_label_status_matrix,
    build_reference_threshold_candidates,
    load_run_audit_tables,
    plot_keypoint_coverage_heatmap,
    plot_pose_gallery,
    plot_status_by_group,
    plot_status_distribution,
    select_audit_examples,
    summarize_available_datasets,
)


# %% [markdown]
# ## 1. Configuracion de auditoria
#
# Partimos de la corrida larga ya completada sobre el dataset frontal base.
# Todo el analisis posterior sale de los CSV guardados en disco.

# %%
RUN_LABEL = "posture_correction_v4_full_dataset_v1"
VISIBILITY_THRESHOLD = 0.35
REFERENCE_GROUP = "looks good"
TARGET_DATASET_KEY = "posture_correction_v4_folder_v1"

target_spec = DATASET_CATALOG[TARGET_DATASET_KEY]
display(
    Markdown(
        f"""
**Contexto de auditoria**

- Run label: `{RUN_LABEL}`
- Dataset base: `{TARGET_DATASET_KEY}`
- Descripcion: `{target_spec.notes}`
- Threshold de visibilidad: `{VISIBILITY_THRESHOLD}`
- Grupo de referencia para calibracion: `{REFERENCE_GROUP}`
"""
    )
)


# %% [markdown]
# ## 2. Carga de artefactos
#
# Cargamos la manifest, los landmarks estimados y el analisis ergonomico ya calculado.
# La tabla fusionada sirve para auditar casos concretos con overlay y metricas.

# %%
run_tables = load_run_audit_tables(RUN_LABEL)
manifest_df = run_tables.manifest_df
pose_df = run_tables.pose_df
analysis_df = run_tables.analysis_df
audit_df = run_tables.merged_df

display(
    pd.DataFrame(
        [
            {"item": "Run dir", "value": str(run_tables.run_dir)},
            {"item": "Imagenes en manifest", "value": len(manifest_df)},
            {"item": "Filas pose", "value": len(pose_df)},
            {"item": "Filas analisis", "value": len(analysis_df)},
            {"item": "Filas fusionadas", "value": len(audit_df)},
        ]
    )
)

display(analysis_df.head(5))


# %% [markdown]
# ## 3. Lectura ejecutiva
#
# Antes de entrar al detalle visual, conviene dejar clara la situacion global.
# Aqui vemos la distribucion del estado ergonomico y donde chocan mas las etiquetas del dataset
# con la clasificacion actual del motor de reglas.

# %%
status_summary_df = (
    analysis_df["overall_status"]
    .value_counts()
    .rename_axis("overall_status")
    .reset_index(name="image_count")
)
status_summary_df["share_pct"] = status_summary_df["image_count"] / status_summary_df["image_count"].sum() * 100.0

label_status_matrix_df = build_label_status_matrix(analysis_df, normalize=True)

display(status_summary_df)
display(label_status_matrix_df)

fig, ax = plot_status_distribution(analysis_df)
plt.show()

fig, ax = plot_status_by_group(analysis_df, normalize=True)
plt.show()


# %% [markdown]
# ## 4. Limites reales del encuadre frontal
#
# Esta es la primera comprobacion importante del notebook.
# Si determinadas partes del cuerpo casi nunca aparecen, no tiene sentido darles mucho peso
# en la clasificacion frontal.

# %%
coverage_by_group_df = build_keypoint_visibility_summary(
    pose_df,
    landmarks=DEFAULT_AUDIT_LANDMARKS,
    visibility_threshold=VISIBILITY_THRESHOLD,
    group_col="group",
)

coverage_overall_df = build_keypoint_visibility_summary(
    pose_df,
    landmarks=DEFAULT_AUDIT_LANDMARKS,
    visibility_threshold=VISIBILITY_THRESHOLD,
    group_col=None,
)

coverage_pivot_df = (
    coverage_by_group_df.pivot(index="landmark", columns="group", values="visibility_pct")
    .sort_index()
)

display(coverage_overall_df.sort_values("visibility_pct", ascending=False))
display(coverage_pivot_df)

fig, ax = plot_keypoint_coverage_heatmap(coverage_by_group_df, group_col="group")
plt.show()


# %% [markdown]
# ## 5. Que componentes estan dominando el error
#
# El objetivo aqui es ver si el problema viene sobre todo de cabeza, hombros, tronco o brazos.
# Con eso podemos decidir que parte del motor de reglas merece recalibracion inmediata y que parte
# debe quedar como secundaria o directamente fuera del alcance frontal.

# %%
component_summary_df = build_component_status_summary(
    analysis_df,
    group_col="group",
    normalize=True,
)

component_risk_view_df = (
    component_summary_df[component_summary_df["status"].isin(["risk", "improvable", "insufficient_data"])]
    .pivot(index="component", columns=["group", "status"], values="share_pct")
    .fillna(0.0)
    .sort_index()
)

display(component_summary_df.head(24))
display(component_risk_view_df)


# %% [markdown]
# ## 6. Calibracion preliminar de umbrales con el grupo `looks good`
#
# Aqui comparamos los umbrales actuales con percentiles del grupo que el dataset considera
# visualmente aceptable. No se trata de cambiar reglas a ciegas, sino de localizar donde
# el motor esta claramente desalineado con la semantica del dataset frontal.

# %%
threshold_candidates_df = build_reference_threshold_candidates(
    analysis_df,
    reference_group=REFERENCE_GROUP,
    adequate_quantile=0.75,
    improvable_quantile=0.90,
)

threshold_review_df = threshold_candidates_df.copy()
threshold_review_df["observacion"] = [
    "alineado" if metric == "shoulder_tilt_deg"
    else "demasiado estricto en frontal" if metric == "shoulder_height_diff_ratio"
    else "demasiado permisivo en frontal" if metric in {"head_lateral_offset_ratio", "neck_tilt_deg"}
    else "sin soporte suficiente en frontal"
    for metric in threshold_review_df["metric"]
]

display(threshold_review_df)


# %% [markdown]
# ## 7. Casos a revisar visualmente
#
# Esta seccion es la mas util para discutir con criterio si el problema esta en la etiqueta,
# en la geometria del encuadre o en el umbral actual.

# %%
looks_good_risk_df = select_audit_examples(
    audit_df,
    filters={"group": "looks good", "overall_status": "risk"},
    sort_by=["shoulder_height_diff_ratio", "head_lateral_offset_ratio"],
    ascending=[False, False],
    limit=6,
)

display(
    looks_good_risk_df[
        [
            "image_name",
            "group",
            "overall_status",
            "shoulder_height_diff_ratio",
            "shoulder_tilt_deg",
            "head_lateral_offset_ratio",
            "neck_tilt_deg",
        ]
    ]
)

fig, axes = plot_pose_gallery(
    looks_good_risk_df,
    caption_fields=[
        "overall_status",
        "shoulder_height_diff_ratio",
        "shoulder_tilt_deg",
        "head_lateral_offset_ratio",
        "neck_tilt_deg",
    ],
    title="Casos 'looks good' clasificados como risk",
    ncols=3,
)
plt.show()


# %%
straighten_head_adequate_df = select_audit_examples(
    audit_df,
    filters={"group": "straighten head", "overall_status": "adequate"},
    sort_by=["head_lateral_offset_ratio", "neck_tilt_deg"],
    ascending=[True, True],
    limit=6,
)

display(
    straighten_head_adequate_df[
        [
            "image_name",
            "group",
            "overall_status",
            "shoulder_height_diff_ratio",
            "head_lateral_offset_ratio",
            "neck_tilt_deg",
        ]
    ]
)

fig, axes = plot_pose_gallery(
    straighten_head_adequate_df,
    caption_fields=[
        "overall_status",
        "head_lateral_offset_ratio",
        "neck_tilt_deg",
        "shoulder_height_diff_ratio",
    ],
    title="Casos 'straighten head' clasificados como adequate",
    ncols=3,
)
plt.show()


# %%
sit_up_straight_audit_df = pd.concat(
    [
        select_audit_examples(
            audit_df,
            filters={"group": "sit up straight", "overall_status": status_name},
            sort_by=["shoulder_height_diff_ratio", "head_lateral_offset_ratio"],
            ascending=[status_name == "adequate", status_name == "adequate"],
            limit=2,
        )
        for status_name in ["adequate", "improvable", "risk"]
    ],
    ignore_index=True,
)

display(
    sit_up_straight_audit_df[
        [
            "image_name",
            "group",
            "overall_status",
            "shoulder_height_diff_ratio",
            "head_lateral_offset_ratio",
            "neck_tilt_deg",
            "left_wrist_visibility",
            "right_wrist_visibility",
        ]
    ]
)

fig, axes = plot_pose_gallery(
    sit_up_straight_audit_df,
    caption_fields=[
        "overall_status",
        "shoulder_height_diff_ratio",
        "head_lateral_offset_ratio",
        "neck_tilt_deg",
    ],
    title="Muestra de 'sit up straight' en distintos estados",
    ncols=3,
)
plt.show()


# %% [markdown]
# ## 8. Linea lateral como extension del proyecto
#
# El frontal sigue siendo el MVP mas realista para oficina y teletrabajo. Aun asi, la documentacion
# del proyecto ya planteaba una camara lateral como opcion para medir mejor la espalda y la cabeza adelantada.
# Aqui dejamos identificados los datasets que mejor encajan con esa extension.

# %%
available_datasets_df = summarize_available_datasets()
lateral_candidates_df = available_datasets_df[
    available_datasets_df["dataset_key"].isin(["sitting_posture_4keypoint", "desk_posture_coco_v1"])
].copy()

display(
    lateral_candidates_df[
        ["dataset_key", "label", "format", "total_images", "group_count", "notes"]
    ]
)


# %% [markdown]
# ## 9. Decision operativa
#
# Esta libreta deja una conclusion clara para la siguiente iteracion:
#
# - el MVP frontal tiene sentido para cabeza lateral, cuello lateral y simetria de hombros
# - `trunk_status` no debe pesar en frontal porque casi nunca hay caderas visibles
# - el angulo completo de codo tampoco debe pesar demasiado mientras muñecas y antebrazos queden fuera
# - `shoulder_height_diff_ratio` esta castigando demasiado al grupo `looks good`
# - `head_lateral_offset_ratio` y `neck_tilt_deg` parecen mas permisivos de lo que sugiere el dataset
# - la medicion de cabeza adelantada debe quedar para la linea lateral

# %%
insights_md = f"""
**Lectura final de la auditoria**

- El dataset frontal base tiene `{len(analysis_df)}` imagenes auditadas.
- La clase `looks good` sigue cayendo en `risk` un `{label_status_matrix_df.loc[label_status_matrix_df['group'] == 'looks good', 'risk'].iloc[0]:.1f}%`.
- Las caderas solo son visibles en torno al `{coverage_overall_df.loc[coverage_overall_df['landmark'] == 'left_hip', 'visibility_pct'].iloc[0]:.1f}%` y `{coverage_overall_df.loc[coverage_overall_df['landmark'] == 'right_hip', 'visibility_pct'].iloc[0]:.1f}%` de las imagenes.
- Las muñecas aparecen en torno al `{coverage_overall_df.loc[coverage_overall_df['landmark'] == 'left_wrist', 'visibility_pct'].iloc[0]:.1f}%` y `{coverage_overall_df.loc[coverage_overall_df['landmark'] == 'right_wrist', 'visibility_pct'].iloc[0]:.1f}%`.
- El mejor siguiente paso ya no es comparar mas modelos, sino recalibrar el motor de reglas para el caso frontal.
"""

display(Markdown(insights_md))
