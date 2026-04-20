from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .paths import ERGONOMICS_RESULTS_DIR
from .pose_inference import MediaPipePoseConfig, MediaPipePoseEstimator
from .posture_rules import analyze_pose_row
from .reporting import (
    build_group_status_summary,
    build_metric_summary_by_group,
    build_status_summary,
    save_dataframe,
)


def progress(iterable, **kwargs):
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except Exception:
        return iterable


@dataclass(frozen=True)
class LongRunArtifacts:
    output_dir: Path
    manifest_path: Path
    pose_path: Path
    analysis_path: Path
    status_summary_path: Path
    group_status_summary_path: Path
    metric_summary_path: Path
    processed_images: int


def _append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    write_header = not path.exists()
    frame.to_csv(path, mode="a", header=write_header, index=False)


def _load_processed_image_paths(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        existing = pd.read_csv(path, usecols=["image_path"])
    except Exception:
        return set()
    return set(existing["image_path"].astype(str))


def _load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def run_incremental_long_pipeline(
    records_df: pd.DataFrame,
    *,
    run_label: str,
    pose_config: MediaPipePoseConfig,
    visibility_threshold: float = 0.35,
    checkpoint_every: int = 100,
    resume: bool = True,
    output_root: Path = ERGONOMICS_RESULTS_DIR,
) -> LongRunArtifacts:
    output_dir = output_root / run_label
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "execution_manifest.csv"
    pose_path = output_dir / "pose_landmarks.csv"
    analysis_path = output_dir / "ergonomic_analysis.csv"
    status_summary_path = output_dir / "status_summary.csv"
    group_status_summary_path = output_dir / "group_status_summary.csv"
    metric_summary_path = output_dir / "metric_summary_by_group.csv"

    records_df = records_df.copy()
    records_df["image_path"] = records_df["image_path"].astype(str)
    save_dataframe(records_df, manifest_path)

    if resume:
        processed = _load_processed_image_paths(pose_path)
        pending_df = records_df[~records_df["image_path"].isin(processed)].copy()
    else:
        pending_df = records_df.copy()
        for path in [pose_path, analysis_path, status_summary_path, group_status_summary_path, metric_summary_path]:
            if path.exists():
                path.unlink()

    pose_rows: list[dict] = []
    analysis_rows: list[dict] = []

    with MediaPipePoseEstimator(config=pose_config) as estimator:
        for item in progress(
            pending_df.to_dict(orient="records"),
            total=len(pending_df),
            desc="Long run ergonomico",
        ):
            pose_row = estimator.infer_image(
                item["image_path"],
                metadata={"group": item.get("group"), "split": item.get("split")},
            )
            analysis_row = analyze_pose_row(
                pose_row,
                visibility_threshold=visibility_threshold,
            )
            pose_rows.append(pose_row)
            analysis_rows.append(analysis_row)

            if len(pose_rows) >= checkpoint_every:
                _append_rows(pose_path, pose_rows)
                _append_rows(analysis_path, analysis_rows)
                pose_rows.clear()
                analysis_rows.clear()

    _append_rows(pose_path, pose_rows)
    _append_rows(analysis_path, analysis_rows)

    analysis_df = _load_dataframe(analysis_path)
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

    save_dataframe(status_summary_df, status_summary_path)
    save_dataframe(group_status_summary_df, group_status_summary_path)
    save_dataframe(metric_summary_df, metric_summary_path)

    return LongRunArtifacts(
        output_dir=output_dir,
        manifest_path=manifest_path,
        pose_path=pose_path,
        analysis_path=analysis_path,
        status_summary_path=status_summary_path,
        group_status_summary_path=group_status_summary_path,
        metric_summary_path=metric_summary_path,
        processed_images=len(analysis_df),
    )
