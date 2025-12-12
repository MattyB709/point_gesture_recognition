#!/usr/bin/env python3
"""
statistics for the geometric baseline (MediaPipe wrist->index
vectors) and  compare those angular with  model.

1. Run `geo_validition.py <root_dir>` to generate `geo_txt/geo_<name>.txt` files.
2. Run `python geo_stat_tests.py --geo-dir <root_dir>/geo_txt`.
3. Once a model produces its own per-sample .txt files with angular errors,
   pass that directory via `--model-dir` to get paired statistical tests.
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy import stats


@dataclass
class DatasetStats:
    name: str
    angles_deg: np.ndarray


def strip_prefix(value: str, prefix: str) -> str:
    if prefix and value.startswith(prefix):
        return value[len(prefix):]
    return value


def load_angle_map(directory: str, prefix: str) -> Dict[str, float]:
    """Read per-sample angular errors from a directory of .txt files."""
    paths = sorted(glob.glob(os.path.join(directory, "*.txt")))
    angle_map: Dict[str, float] = {}

    for path in paths:
        with open(path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        if not lines:
            continue
        if lines[0] != "1":
            # skip negatives / invalid entries
            continue

        numeric_lines: List[float] = []
        for token in lines[1:]:
            try:
                numeric_lines.append(float(token))
            except ValueError:
                pass

        if len(numeric_lines) < 7:
            continue

        angle_deg = numeric_lines[-1]
        base = os.path.splitext(os.path.basename(path))[0]
        sample_id = strip_prefix(base, prefix)
        angle_map[sample_id] = angle_deg

    return angle_map


def summarize_dataset(name: str, values: np.ndarray) -> DatasetStats:
    """Compute descriptive statistics and print them."""
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    print("=" * 72)
    print(f"{name}: {n} samples")

    if n == 0:
        print("No samples available.")
        print("=" * 72)
        return DatasetStats(name, arr)

    mean = arr.mean()
    median = float(np.median(arr))
    std = arr.std(ddof=1) if n > 1 else 0.0
    min_val = arr.min()
    max_val = arr.max()
    q25, q75 = np.percentile(arr, [25, 75])

    print(f"Mean = {mean:.2f}°, Median = {median:.2f}°, Std = {std:.2f}°")
    print(f"Min/Max = {min_val:.2f}° / {max_val:.2f}°")
    print(f"IQR (25%%-75%%) = [{q25:.2f}°, {q75:.2f}°]")

    if n > 1:
        std_err = std / np.sqrt(n)
        ci_radius = stats.t.ppf(0.975, n - 1) * std_err
        print(f"95% CI for mean: [{mean - ci_radius:.2f}°, {mean + ci_radius:.2f}°]")

        t_stat, p_val = stats.ttest_1samp(arr, popmean=0.0)
        print(f"One-sample t-test vs 0°: t = {t_stat:.3f}, p = {p_val:.3g}")

        try:
            w_stat, w_p = stats.wilcoxon(arr, alternative="greater", zero_method="wilcox")
            print(f"Wilcoxon signed-rank vs 0° (>0): W = {w_stat:.3f}, p = {w_p:.3g}")
        except ValueError as exc:
            print(f"Wilcoxon test skipped: {exc}")

    print("=" * 72)
    return DatasetStats(name, arr)


def summarize_pairwise(model: DatasetStats, geo: DatasetStats) -> None:
    """Run paired tests between model and geometric baselines."""
    model_arr = np.asarray(model.angles_deg, dtype=np.float64)
    geo_arr = np.asarray(geo.angles_deg, dtype=np.float64)

    if model_arr.size == 0 or geo_arr.size == 0:
        print("No overlapping samples between model and geometric datasets.")
        return

    diff = model_arr - geo_arr
    n = diff.size

    print("\nPairwise comparison (Model - Geometric)")
    print(f"Matched samples: {n}")

    mean_diff = diff.mean()
    median_diff = float(np.median(diff))
    std_diff = diff.std(ddof=1) if n > 1 else 0.0

    print(f"Mean diff = {mean_diff:.2f}°, Median diff = {median_diff:.2f}°")
    if n > 1 and std_diff > 0:
        std_err = std_diff / np.sqrt(n)
        ci_radius = stats.t.ppf(0.975, n - 1) * std_err
        print(f"95% CI for diff mean: [{mean_diff - ci_radius:.2f}°, {mean_diff + ci_radius:.2f}°]")
        cohen_d = mean_diff / std_diff
        print(f"Cohen's d (paired) = {cohen_d:.3f}")

        t_stat, p_val = stats.ttest_rel(model_arr, geo_arr)
        print(f"Paired t-test: t = {t_stat:.3f}, p = {p_val:.3g}")

        try:
            w_stat, w_p = stats.wilcoxon(model_arr, geo_arr, zero_method="wilcox")
            print(f"Wilcoxon signed-rank: W = {w_stat:.3f}, p = {w_p:.3g}")
        except ValueError as exc:
            print(f"Wilcoxon test skipped: {exc}")

    better = np.sum(model_arr <= geo_arr)
    print(f"Model better-or-equal samples: {better}/{n} ({better / n * 100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize geometric angular errors and optionally compare them with model outputs."
    )
    parser.add_argument(
        "--geo-dir",
        default="geo_txt",
        help="Directory containing geo_*.txt files from geo_validition.py.",
    )
    parser.add_argument(
        "--geo-prefix",
        default="geo_",
        help="Prefix to strip from filenames inside --geo-dir (default: geo_).",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Optional directory containing per-sample .txt files for the model.",
    )
    parser.add_argument(
        "--model-prefix",
        default="",
        help="Prefix to strip from filenames inside --model-dir.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.geo_dir):
        raise SystemExit(f"Geo directory not found: {args.geo_dir}")

    geo_map = load_angle_map(args.geo_dir, args.geo_prefix)
    print(f"Loaded {len(geo_map)} geometric samples from {args.geo_dir}")
    geo_stats = summarize_dataset("Geometric baseline", list(geo_map.values()))

    if args.model_dir:
        if not os.path.isdir(args.model_dir):
            raise SystemExit(f"Model directory not found: {args.model_dir}")

        model_map = load_angle_map(args.model_dir, args.model_prefix)
        print(f"\nLoaded {len(model_map)} model samples from {args.model_dir}")
        model_stats = summarize_dataset("Model", list(model_map.values()))

        overlap = sorted(set(geo_map) & set(model_map))
        if not overlap:
            print("Warning: no overlapping sample ids found.")
            return

        geo_values = np.array([geo_map[k] for k in overlap], dtype=np.float64)
        model_values = np.array([model_map[k] for k in overlap], dtype=np.float64)

        geo_stats = DatasetStats("Geometric baseline (matched)", geo_values)
        model_stats = DatasetStats("Model (matched)", model_values)
        summarize_pairwise(model_stats, geo_stats)


if __name__ == "__main__":
    main()
