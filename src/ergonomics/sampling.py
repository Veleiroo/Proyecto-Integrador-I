from __future__ import annotations

import random

import pandas as pd

from .datasets import collect_image_records


def select_balanced_sample(
    dataset_key: str,
    *,
    per_group: int = 4,
    splits: tuple[str, ...] = ("train", "valid", "test"),
    seed: int = 7,
) -> pd.DataFrame:
    records = collect_image_records(dataset_key)
    if not records:
        return pd.DataFrame(columns=["image_path", "group", "split"])

    rng = random.Random(seed)
    filtered = [record for record in records if record["split"] in splits]
    grouped: dict[str, list[dict]] = {}
    for record in filtered:
        grouped.setdefault(str(record["group"]), []).append(record)

    selected = []
    for group_name in sorted(grouped):
        group_records = grouped[group_name][:]
        rng.shuffle(group_records)
        selected.extend(group_records[:per_group])

    selected.sort(key=lambda item: (str(item["group"]), str(item["split"]), str(item["image_path"])))
    sample_df = pd.DataFrame(selected)
    if sample_df.empty:
        return pd.DataFrame(columns=["image_path", "group", "split"])
    return sample_df.reset_index(drop=True)


def select_execution_records(
    dataset_key: str,
    *,
    strategy: str = "full_dataset",
    per_group: int = 4,
    max_images: int | None = None,
    splits: tuple[str, ...] = ("train", "valid", "test"),
    seed: int = 7,
) -> pd.DataFrame:
    records = collect_image_records(dataset_key)
    if not records:
        return pd.DataFrame(columns=["image_path", "group", "split"])

    filtered = [record for record in records if record["split"] in splits]
    if strategy == "balanced":
        sample_df = select_balanced_sample(
            dataset_key,
            per_group=per_group,
            splits=splits,
            seed=seed,
        )
    elif strategy == "full_dataset":
        sample_df = pd.DataFrame(filtered)
    else:
        raise ValueError(f"Estrategia desconocida: {strategy}")

    if sample_df.empty:
        return pd.DataFrame(columns=["image_path", "group", "split"])

    if max_images is not None and max_images < len(sample_df):
        rng = random.Random(seed)
        records = sample_df.to_dict(orient="records")
        if strategy == "balanced":
            grouped: dict[tuple[str, str], list[dict]] = {}
            for record in records:
                grouped.setdefault((str(record["split"]), str(record["group"])), []).append(record)
            for items in grouped.values():
                rng.shuffle(items)
            selected: list[dict] = []
            ordered_keys = sorted(grouped)
            while len(selected) < max_images:
                progress_made = False
                for key in ordered_keys:
                    bucket = grouped[key]
                    if not bucket:
                        continue
                    selected.append(bucket.pop())
                    progress_made = True
                    if len(selected) >= max_images:
                        break
                if not progress_made:
                    break
            sample_df = pd.DataFrame(selected)
        else:
            rng.shuffle(records)
            sample_df = pd.DataFrame(records[:max_images])

    return sample_df.sort_values(["group", "split", "image_path"]).reset_index(drop=True)
