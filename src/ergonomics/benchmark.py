from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .paths import POSE_BENCHMARK_RESULTS_DIR


#Estructuras de datos (artefactos)

@dataclass(frozen=True)
class BenchmarkArtifacts:
    """
    Contenedor de resultados del benchmark. 
    Almacena los datos crudos, los resúmenes globales y los resúmenes 
    desglosados por tipo de postura (grupos).
    """
    results: pd.DataFrame
    summary: pd.DataFrame
    group_summary: pd.DataFrame


@dataclass(frozen=True)
class ModelDecision:
    """
    Representa la resolución final del proceso de selección.
    Incluye el modelo ganador, el de respaldo y la justificación técnica (rationale).
    """
    primary_model: str
    backup_model: str | None
    current_scope: str
    rationale: str

# Carga y limpieza de datos

def load_benchmark_artifacts(results_dir: Path = POSE_BENCHMARK_RESULTS_DIR) -> BenchmarkArtifacts:
    """
    Carga los archivos CSV generados durante la fase de experimentación.
    Asegura que las columnas de rendimiento y precisión se traten como números (float/int).
    """
    results_path = results_dir / "pose_benchmark_results.csv"
    summary_path = results_dir / "pose_benchmark_summary.csv"
    group_summary_path = results_dir / "pose_benchmark_group_summary.csv"

    # Validación de existencia: el benchmark debe ejecutarse antes que el pipeline
    if not results_path.exists() or not summary_path.exists() or not group_summary_path.exists():
        raise FileNotFoundError(
            "Faltan artefactos del benchmark. Ejecuta primero el benchmark de pose en notebooks/pose_benchmark/."
        )

    results = pd.read_csv(results_path)
    summary = pd.read_csv(summary_path)
    group_summary = pd.read_csv(group_summary_path)

    # Normalización de tipos de datos: convertimos porcentajes y milisegundos a tipos numéricos
    numeric_candidates = {
        "runtime_ms_mean",
        "runtime_ms_median",
        "required_keypoint_rate",
        "error_rate",
    }
    for frame in (results, summary, group_summary):
        for column in frame.columns:
            # Detectamos columnas de porcentaje o tiempo por su sufijo
            if column.endswith("_pct") or column.endswith("_ms") or column in numeric_candidates:
                try:
                    frame[column] = pd.to_numeric(frame[column])
                except (TypeError, ValueError):
                    continue

    return BenchmarkArtifacts(results=results, summary=summary, group_summary=group_summary)


# Sistema de ranking y puntuación

def rank_models(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Algoritmo de puntuación para jerarquizar los modelos de pose analizados.
    
    No solo mira la precisión, sino que equilibra tres factores:
    1. Cobertura del tren superior (Vital para el caso de uso frontal).
    2. Disponibilidad de puntos clave (Landmarks necesarios para ROSA).
    3. Eficiencia temporal (Velocidad de proceso).
    """
    if summary_df.empty:
        raise ValueError("El resumen del benchmark esta vacio.")

    ranked = summary_df.copy()
    
    # Cálculo de eficiencia de tiempo:
    # El modelo más lento recibe 0% y el más rápido 100% en esta sub-métrica.
    runtime = ranked["runtime_ms_median"].astype(float)
    if runtime.nunique() <= 1:
        ranked["runtime_efficiency_pct"] = 100.0
    else:
        ranked["runtime_efficiency_pct"] = 100.0 * (runtime.max() - runtime) / (runtime.max() - runtime.min())

    # --- FÓRMULA DE AJUSTE A LA FASE (PHASE FIT SCORE) ---
    # Pesos definidos para el alcance frontal de tren superior:
    # - 55%: Capacidad de ver el tren superior (Hombros, cuello, cabeza).
    # - 30%: Estabilidad de los puntos clave obligatorios.
    # - 15%: Velocidad (para cumplir con la 'No Interferencia' en el PC del usuario).
    ranked["phase_fit_score"] = (
        ranked["upper_body_ready_pct"].astype(float) * 0.55
        + ranked["required_keypoint_rate"].astype(float) * 0.30
        + ranked["runtime_efficiency_pct"].astype(float) * 0.15
    )

    # Ordenamos por la nota final: el mejor modelo queda arriba (índice 0).
    return ranked.sort_values(
        by=["phase_fit_score", "upper_body_ready_pct", "required_keypoint_rate", "runtime_ms_median"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def choose_reference_models(summary_df: pd.DataFrame) -> ModelDecision:
    """
    Selecciona el modelo principal y el de respaldo a partir del ranking calculado.
    """
    # Obtenemos el ranking basado en los pesos definidos (Precisión vs Velocidad)
    ranked = rank_models(summary_df)
    
    # El primer modelo de la lista es el ganador absoluto (Primary)
    primary = ranked.iloc[0]
    # Si hay más de un modelo, el segundo queda como respaldo (Backup)
    backup = ranked.iloc[1]["model"] if len(ranked) > 1 else None

    rationale = (
        f"Se prioriza {primary['model']} porque lidera la fase actual en tren superior "
        f"({primary['upper_body_ready_pct']:.1f}% de imagenes listas) y en cobertura base "
        f"({primary['required_keypoint_rate']:.1f}%)."
    )

    return ModelDecision(
        primary_model=str(primary["model"]),
        backup_model=str(backup) if backup is not None else None,
        current_scope="upper_body_frontal_webcam", # Definimos que el foco actual es frontal/superior
        rationale=rationale,
    )


def plot_model_overview(summary_df: pd.DataFrame):
    """
    Genera una comparativa visual de 'Los Tres Pilares': 
    latencia, cobertura de keypoints y estabilidad de puntos requeridos.
    """
    if summary_df.empty:
        raise ValueError("El resumen del benchmark esta vacio.")

    plot_df = rank_models(summary_df)
    
    # Creamos una figura con 3 sub-gráficas para comparar de un vistazo
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"] # Colores corporativos/estándar

    # 1. Gráfica de Latencia: ¿Quién es el más ligero? (Vital para la 'No Interferencia')
    plot_df.plot(x="model", y="runtime_ms_median", kind="bar", legend=False, ax=axes[0], color=colors[: len(plot_df)])
    axes[0].set_title("Latencia mediana")
    axes[0].set_ylabel("ms por imagen")
    axes[0].tick_params(axis="x", rotation=20)

    # 2. Cobertura Base: ¿Quién ve mejor los puntos clave del cuerpo?
    plot_df.plot(x="model", y="required_keypoint_rate", kind="bar", legend=False, ax=axes[1], color=colors[: len(plot_df)])
    axes[1].set_title("Cobertura de keypoints base")
    axes[1].set_ylabel("% de keypoints requeridos")
    axes[1].tick_params(axis="x", rotation=20)

    # 3. Imágenes Listas: ¿Quién permite hacer el análisis ROSA en más fotos?
    plot_df.plot(x="model", y="upper_body_ready_pct", kind="bar", legend=False, ax=axes[2], color=colors[: len(plot_df)])
    axes[2].set_title("Imagenes listas para tren superior")
    axes[2].set_ylabel("% de imagenes validas")
    axes[2].tick_params(axis="x", rotation=20)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    return fig, axes


def plot_group_comparison(group_summary_df: pd.DataFrame, metric: str, title: str):
    """
    Compara el rendimiento de los modelos según la postura (adecuada/riesgo).
    Ayuda a detectar si un modelo es muy bueno en posturas fáciles pero 
    falla estrepitosamente en posturas forzadas.
    """
    if group_summary_df.empty:
        raise ValueError("El resumen por grupos esta vacio.")
    if metric not in group_summary_df.columns:
        raise KeyError(f"La metrica {metric} no existe en el resumen por grupos.")

    # Pivotamos los datos para crear una gráfica de barras agrupadas por modelo
    pivot = (
        group_summary_df.pivot(index="group", columns="model", values=metric)
        .fillna(0)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(pivot.columns)])
    ax.set_title(title)
    ax.set_xlabel("Grupo")
    ax.set_ylabel("%")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, ax
