from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_status_summary(analysis_df: pd.DataFrame) -> pd.DataFrame:
    if analysis_df.empty:
        return pd.DataFrame(columns=["overall_status", "image_count", "share_pct"])

    summary = analysis_df["overall_status"].value_counts(dropna=False).rename("image_count").reset_index()
    summary.columns = ["overall_status", "image_count"]
    summary["share_pct"] = summary["image_count"] / summary["image_count"].sum() * 100.0
    return summary


def build_group_status_summary(analysis_df: pd.DataFrame) -> pd.DataFrame:
    if analysis_df.empty:
        return pd.DataFrame(columns=["group", "overall_status", "image_count", "share_pct"])

    grouped = (
        analysis_df.groupby(["group", "overall_status"], dropna=False)
        .size()
        .rename("image_count")
        .reset_index()
    )
    totals = grouped.groupby("group")["image_count"].transform("sum")
    grouped["share_pct"] = grouped["image_count"] / totals * 100.0
    return grouped.sort_values(["group", "overall_status"]).reset_index(drop=True)


def build_metric_summary_by_group(
    analysis_df: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    if analysis_df.empty:
        return pd.DataFrame()

    available_metrics = [metric for metric in metrics if metric in analysis_df.columns]
    if not available_metrics:
        return pd.DataFrame()

    summary = analysis_df.groupby("group", dropna=False)[available_metrics].mean().reset_index()
    return summary.sort_values("group").reset_index(drop=True)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
