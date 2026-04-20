from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .paths import ERGONOMICS_RESULTS_DIR

DEFAULT_AUDIT_LANDMARKS = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
]

DEFAULT_THRESHOLD_REFERENCE = {
    "shoulder_tilt_deg": {"adequate_max": 5.0, "improvable_max": 10.0},
    "shoulder_height_diff_ratio": {"adequate_max": 0.03, "improvable_max": 0.07},
    "head_lateral_offset_ratio": {"adequate_max": 0.08, "improvable_max": 0.16},
    "neck_tilt_deg": {"adequate_max": 8.0, "improvable_max": 15.0},
    "trunk_tilt_deg": {"adequate_max": 6.0, "improvable_max": 12.0},
}


@dataclass(frozen=True)
class RunAuditTables:
    run_dir: Path
    manifest_df: pd.DataFrame
    pose_df: pd.DataFrame
    analysis_df: pd.DataFrame
    merged_df: pd.DataFrame


def load_run_audit_tables(
    run_label: str,
    *,
    results_root: Path = ERGONOMICS_RESULTS_DIR,
) -> RunAuditTables:
    run_dir = results_root / run_label
    manifest_df = pd.read_csv(run_dir / "execution_manifest.csv")
    pose_df = pd.read_csv(run_dir / "pose_landmarks.csv")
    analysis_df = pd.read_csv(run_dir / "ergonomic_analysis.csv")
    merged_df = merge_pose_analysis(pose_df, analysis_df)
    return RunAuditTables(
        run_dir=run_dir,
        manifest_df=manifest_df,
        pose_df=pose_df,
        analysis_df=analysis_df,
        merged_df=merged_df,
    )


def merge_pose_analysis(pose_df: pd.DataFrame, analysis_df: pd.DataFrame) -> pd.DataFrame:
    if pose_df.empty or analysis_df.empty:
        return pd.DataFrame()

    duplicated_columns = {
        "image_name",
        "group",
        "split",
        "pose_detected",
    }
    analysis_payload = analysis_df.drop(columns=[column for column in duplicated_columns if column in analysis_df.columns])
    merged_df = pose_df.merge(analysis_payload, on="image_path", how="inner", validate="one_to_one")
    return merged_df


def build_keypoint_visibility_summary(
    pose_df: pd.DataFrame,
    *,
    landmarks: list[str] | None = None,
    visibility_threshold: float = 0.35,
    group_col: str | None = "group",
) -> pd.DataFrame:
    if pose_df.empty:
        base_columns = ["landmark", "visible_images", "total_images", "visibility_pct"]
        if group_col:
            base_columns.insert(0, group_col)
        return pd.DataFrame(columns=base_columns)

    landmarks = landmarks or DEFAULT_AUDIT_LANDMARKS
    rows: list[dict] = []

    if group_col and group_col in pose_df.columns:
        grouped_items = pose_df.groupby(group_col, dropna=False)
    else:
        grouped_items = [("ALL", pose_df)]
        group_col = None

    for group_value, group_df in grouped_items:
        total_images = len(group_df)
        for landmark_name in landmarks:
            visibility_col = f"{landmark_name}_visibility"
            if visibility_col not in group_df.columns:
                continue
            visible_images = int((group_df[visibility_col].fillna(-1.0) >= visibility_threshold).sum())
            row = {
                "landmark": landmark_name,
                "visible_images": visible_images,
                "total_images": total_images,
                "visibility_pct": visible_images / total_images * 100.0 if total_images else 0.0,
            }
            if group_col:
                row[group_col] = group_value
            rows.append(row)

    summary_df = pd.DataFrame(rows)
    sort_columns = ["landmark"] if not group_col else [group_col, "landmark"]
    return summary_df.sort_values(sort_columns).reset_index(drop=True)


def build_component_status_summary(
    analysis_df: pd.DataFrame,
    *,
    component_columns: list[str] | None = None,
    group_col: str | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    if analysis_df.empty:
        base_columns = ["component", "status", "image_count", "share_pct"]
        if group_col:
            base_columns.insert(0, group_col)
        return pd.DataFrame(columns=base_columns)

    component_columns = component_columns or [
        "shoulder_status",
        "head_status",
        "trunk_status",
        "left_elbow_status",
        "right_elbow_status",
        "overall_status",
    ]
    component_columns = [column for column in component_columns if column in analysis_df.columns]
    rows: list[dict] = []

    if group_col and group_col in analysis_df.columns:
        grouped_items = analysis_df.groupby(group_col, dropna=False)
    else:
        grouped_items = [("ALL", analysis_df)]
        group_col = None

    for group_value, group_df in grouped_items:
        for component_name in component_columns:
            counts = group_df[component_name].value_counts(dropna=False)
            total = int(counts.sum())
            for status_name, image_count in counts.items():
                row = {
                    "component": component_name,
                    "status": status_name,
                    "image_count": int(image_count),
                    "share_pct": float(image_count) / total * 100.0 if normalize and total else None,
                }
                if group_col:
                    row[group_col] = group_value
                rows.append(row)

    summary_df = pd.DataFrame(rows)
    sort_columns = ["component", "status"] if not group_col else [group_col, "component", "status"]
    return summary_df.sort_values(sort_columns).reset_index(drop=True)


def build_label_status_matrix(
    analysis_df: pd.DataFrame,
    *,
    normalize: bool = True,
) -> pd.DataFrame:
    if analysis_df.empty:
        return pd.DataFrame()

    matrix_df = (
        analysis_df.groupby(["group", "overall_status"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    if normalize:
        matrix_df = matrix_df.div(matrix_df.sum(axis=1).replace(0, 1), axis=0) * 100.0
    return matrix_df.reset_index()


def build_metric_quantile_summary(
    analysis_df: pd.DataFrame,
    *,
    metrics: list[str],
    group_col: str = "group",
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> pd.DataFrame:
    if analysis_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    grouped_items = analysis_df.groupby(group_col, dropna=False) if group_col in analysis_df.columns else [("ALL", analysis_df)]
    for group_value, group_df in grouped_items:
        for metric_name in metrics:
            if metric_name not in group_df.columns:
                continue
            series = group_df[metric_name].dropna()
            if series.empty:
                continue
            row = {
                group_col: group_value,
                "metric": metric_name,
                "non_null_count": int(series.shape[0]),
                "mean": float(series.mean()),
            }
            for quantile in quantiles:
                row[f"q{int(quantile * 100):02d}"] = float(series.quantile(quantile))
            rows.append(row)
    return pd.DataFrame(rows).sort_values([group_col, "metric"]).reset_index(drop=True)


def build_reference_threshold_candidates(
    analysis_df: pd.DataFrame,
    *,
    reference_group: str = "looks good",
    metric_thresholds: dict[str, dict[str, float]] | None = None,
    adequate_quantile: float = 0.75,
    improvable_quantile: float = 0.9,
) -> pd.DataFrame:
    if analysis_df.empty:
        return pd.DataFrame()

    metric_thresholds = metric_thresholds or DEFAULT_THRESHOLD_REFERENCE
    reference_df = analysis_df[analysis_df["group"] == reference_group].copy()
    if reference_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for metric_name, current_thresholds in metric_thresholds.items():
        if metric_name not in reference_df.columns:
            continue
        series = reference_df[metric_name].dropna()
        if series.empty:
            continue
        rows.append(
            {
                "metric": metric_name,
                "reference_group": reference_group,
                "non_null_count": int(series.shape[0]),
                "current_adequate_max": current_thresholds.get("adequate_max"),
                "current_improvable_max": current_thresholds.get("improvable_max"),
                "candidate_adequate_max": float(series.quantile(adequate_quantile)),
                "candidate_improvable_max": float(series.quantile(improvable_quantile)),
                "reference_mean": float(series.mean()),
                "reference_median": float(series.median()),
            }
        )
    return pd.DataFrame(rows).sort_values("metric").reset_index(drop=True)


def select_audit_examples(
    merged_df: pd.DataFrame,
    *,
    filters: dict[str, object] | None = None,
    sort_by: str | list[str] | None = None,
    ascending: bool | list[bool] = False,
    limit: int = 9,
) -> pd.DataFrame:
    if merged_df.empty:
        return pd.DataFrame()

    selected_df = merged_df.copy()
    for column_name, expected_value in (filters or {}).items():
        if column_name not in selected_df.columns:
            continue
        if isinstance(expected_value, (list, tuple, set)):
            selected_df = selected_df[selected_df[column_name].isin(expected_value)]
        else:
            selected_df = selected_df[selected_df[column_name] == expected_value]

    if sort_by is not None:
        selected_df = selected_df.sort_values(sort_by, ascending=ascending, na_position="last")

    return selected_df.head(limit).reset_index(drop=True)
