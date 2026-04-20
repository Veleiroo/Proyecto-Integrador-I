from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .paths import POSE_BENCHMARK_RESULTS_DIR


@dataclass(frozen=True)
class BenchmarkArtifacts:
    results: pd.DataFrame
    summary: pd.DataFrame
    group_summary: pd.DataFrame


@dataclass(frozen=True)
class ModelDecision:
    primary_model: str
    backup_model: str | None
    current_scope: str
    rationale: str


def load_benchmark_artifacts(results_dir: Path = POSE_BENCHMARK_RESULTS_DIR) -> BenchmarkArtifacts:
    results_path = results_dir / "pose_benchmark_results.csv"
    summary_path = results_dir / "pose_benchmark_summary.csv"
    group_summary_path = results_dir / "pose_benchmark_group_summary.csv"

    if not results_path.exists() or not summary_path.exists() or not group_summary_path.exists():
        raise FileNotFoundError(
            "Faltan artefactos del benchmark. Ejecuta primero el benchmark de pose en notebooks/pose_benchmark/."
        )

    results = pd.read_csv(results_path)
    summary = pd.read_csv(summary_path)
    group_summary = pd.read_csv(group_summary_path)

    numeric_candidates = {
        "runtime_ms_mean",
        "runtime_ms_median",
        "required_keypoint_rate",
        "error_rate",
    }
    for frame in (results, summary, group_summary):
        for column in frame.columns:
            if column.endswith("_pct") or column.endswith("_ms") or column in numeric_candidates:
                try:
                    frame[column] = pd.to_numeric(frame[column])
                except (TypeError, ValueError):
                    continue

    return BenchmarkArtifacts(results=results, summary=summary, group_summary=group_summary)


def rank_models(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        raise ValueError("El resumen del benchmark esta vacio.")

    ranked = summary_df.copy()
    runtime = ranked["runtime_ms_median"].astype(float)
    if runtime.nunique() <= 1:
        ranked["runtime_efficiency_pct"] = 100.0
    else:
        ranked["runtime_efficiency_pct"] = 100.0 * (runtime.max() - runtime) / (runtime.max() - runtime.min())

    ranked["phase_fit_score"] = (
        ranked["upper_body_ready_pct"].astype(float) * 0.55
        + ranked["required_keypoint_rate"].astype(float) * 0.30
        + ranked["runtime_efficiency_pct"].astype(float) * 0.15
    )

    return ranked.sort_values(
        by=["phase_fit_score", "upper_body_ready_pct", "required_keypoint_rate", "runtime_ms_median"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def choose_reference_models(summary_df: pd.DataFrame) -> ModelDecision:
    ranked = rank_models(summary_df)
    primary = ranked.iloc[0]
    backup = ranked.iloc[1]["model"] if len(ranked) > 1 else None

    rationale = (
        f"Se prioriza {primary['model']} porque lidera la fase actual en tren superior "
        f"({primary['upper_body_ready_pct']:.1f}% de imagenes listas) y en cobertura base "
        f"({primary['required_keypoint_rate']:.1f}%)."
    )

    return ModelDecision(
        primary_model=str(primary["model"]),
        backup_model=str(backup) if backup is not None else None,
        current_scope="upper_body_frontal_webcam",
        rationale=rationale,
    )


def plot_model_overview(summary_df: pd.DataFrame):
    if summary_df.empty:
        raise ValueError("El resumen del benchmark esta vacio.")

    plot_df = rank_models(summary_df)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    plot_df.plot(x="model", y="runtime_ms_median", kind="bar", legend=False, ax=axes[0], color=colors[: len(plot_df)])
    axes[0].set_title("Latencia mediana")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("ms por imagen")
    axes[0].tick_params(axis="x", rotation=20)

    plot_df.plot(x="model", y="required_keypoint_rate", kind="bar", legend=False, ax=axes[1], color=colors[: len(plot_df)])
    axes[1].set_title("Cobertura de keypoints base")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("% de keypoints requeridos")
    axes[1].tick_params(axis="x", rotation=20)

    plot_df.plot(x="model", y="upper_body_ready_pct", kind="bar", legend=False, ax=axes[2], color=colors[: len(plot_df)])
    axes[2].set_title("Imagenes listas para tren superior")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("% de imagenes validas")
    axes[2].tick_params(axis="x", rotation=20)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    return fig, axes


def plot_group_comparison(group_summary_df: pd.DataFrame, metric: str, title: str):
    if group_summary_df.empty:
        raise ValueError("El resumen por grupos esta vacio.")
    if metric not in group_summary_df.columns:
        raise KeyError(f"La metrica {metric} no existe en el resumen por grupos.")

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
