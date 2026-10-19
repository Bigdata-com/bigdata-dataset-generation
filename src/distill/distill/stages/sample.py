"""Stage 9 -- the stratified sample: what the teacher actually labels."""

from __future__ import annotations

from typing import Any

from dataclasses import replace as dc_replace

import numpy as np
import pandas as pd

from .. import frames
from ..config import DistillConfig
from ..manifest import Manifest
from ..paths import RunPaths
from ..theme import ThemeSpec


def bin_edges(values: np.ndarray, *, bins: int, mode: str) -> np.ndarray:
    if mode == "quantile":
        edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    else:
        edges = np.linspace(float(values.min()), float(values.max()), bins + 1)
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([float(values.min()), float(values.max()) + 1e-9])
    return edges


def _apply_ceiling(keep: list[np.ndarray], max_rows: int | None, *,
                   rng: np.random.Generator, label: str) -> list[np.ndarray]:
    """A CEILING on the whole sample, taken proportionally from every bin.

    The teacher is the stage that is paid per row, so a run has to be able to say "at most
    50,000 labels" without also saying how the corpus is shaped. `fraction_per_bin` cannot do
    it: it is a fraction OF THE SUBSET, so the count follows `top_fraction` and a wider subset
    silently buys a bigger teacher bill.

    Proportional, and that is the whole point. Cutting the tail of the ranking instead would
    move the similarity distribution the stratified draw exists to preserve -- the sample would
    stop mirroring the corpus, and every per-bin number downstream would be measuring the cut
    rather than the data. Largest-remainder, so the kept total is exactly `max_rows` and not
    `max_rows` minus twenty roundings.
    """
    total = int(sum(len(part) for part in keep))
    if not max_rows or total <= max_rows:
        return keep
    exact = [len(part) * max_rows / total for part in keep]
    targets = [int(value) for value in exact]
    remainder = max_rows - sum(targets)
    for index in sorted(range(len(keep)), key=lambda i: exact[i] - targets[i], reverse=True):
        if remainder <= 0:
            break
        if targets[index] < len(keep[index]):
            targets[index] += 1
            remainder -= 1
    trimmed = [part if len(part) <= target
               else np.sort(rng.choice(part, size=target, replace=False))
               for part, target in zip(keep, targets)]
    return trimmed


def choose(subset: pd.DataFrame, config: DistillConfig, *, label: str) -> pd.DataFrame:
    rng = np.random.default_rng(config.sample.seed)
    values = pd.to_numeric(subset[config.subset.metric], errors="coerce").to_numpy(dtype=float)
    if np.isnan(values).all():
        raise SystemExit(f"[{label}] {config.subset.metric!r} is NaN on every row: nothing to bin")
    values = np.nan_to_num(values, nan=float(np.nanmin(values)))
    edges = bin_edges(values, bins=config.sample.bins, mode=config.sample.bin_mode)
    n_bins = len(edges) - 1
    if n_bins < config.sample.bins:
        print(f"     note: {config.sample.bins - n_bins} bin edge(s) collapsed on ties "
              f"-- {n_bins} bins")
    which = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, n_bins - 1)

    if config.sample.all_rows:
        description = "EVERY row"
    elif config.sample.count_per_bin is not None:
        description = f"{config.sample.count_per_bin:,} rows from each"
    else:
        description = f"{config.sample.fraction_per_bin:.0%} of each"

    keep: list[np.ndarray] = []
    bounds: dict[int, tuple[float, float]] = {}
    for index in range(n_bins):
        in_bin = np.nonzero(which == index)[0]
        if config.sample.all_rows:
            target = len(in_bin)
        elif config.sample.count_per_bin is not None:
            target = min(config.sample.count_per_bin, len(in_bin))
        else:
            target = int(round(len(in_bin) * (config.sample.fraction_per_bin or 0.0)))
        picked = (rng.choice(in_bin, size=target, replace=False) if target
                  else np.empty(0, dtype=int))
        keep.append(picked)
        bounds[index] = (float(edges[index]), float(edges[index + 1]))

    keep = _apply_ceiling(keep, config.sample.max_rows, rng=rng, label=label)
    indices = np.sort(np.concatenate(keep)) if keep else np.empty(0, dtype=int)
    out = subset.iloc[indices].copy()
    out["bin"] = which[indices]
    out["bin_lo"] = out["bin"].map(lambda b: bounds[b][0])
    out["bin_hi"] = out["bin"].map(lambda b: bounds[b][1])
    return out


def run(config: DistillConfig, theme: ThemeSpec, paths: RunPaths, *,
        force: bool = False, limit: int | None = None,
        max_rows: int | None = None) -> dict[str, Any]:
    if max_rows is not None:
        config.sample = dc_replace(config.sample, max_rows=max_rows)
    out_path = paths.sample_parquet
    if out_path.exists() and not force:
        existing = pd.read_parquet(out_path, columns=["row_key"])
        print(f"sample already written: {out_path} ({len(existing):,} rows). "
              f"Pass --force to redraw -- and note that redrawing invalidates every label and "
              f"every metric measured on the old sample.")
        return {"rows": len(existing), "path": str(out_path), "reused": True}

    subset_path = frames.require_existing(paths.subset_parquet, "the subset", "distill subset")
    manifest = Manifest(stage="sample").add_input("subset", subset_path)
    subset = pd.read_parquet(subset_path)
    content_column = config.input.content_type_column

    parts: list[pd.DataFrame] = []
    for content_type in config.subset.content_types:
        group = subset.loc[subset[content_column] == content_type]
        if group.empty:
            print(f"[{content_type}] no rows -- skipping")
            continue
        sample = choose(group, config, label=content_type)
        if limit is not None:
            sample = sample.sample(n=min(limit, len(sample)), random_state=config.sample.seed)
            print(f"[{content_type}] --limit {limit} -> {len(sample):,} rows")
        parts.append(sample)

    if not parts:
        raise SystemExit("nothing sampled")
    sampled = pd.concat(parts, ignore_index=True)
    frames.write_parquet(sampled, out_path, announce=False)
    print(f"\nto label: {len(sampled):,} rows")
    print(sampled[content_column].value_counts().to_string())

    manifest.config = {
        "sample": {"bins": config.sample.bins, "bin_mode": config.sample.bin_mode,
                   "fraction_per_bin": config.sample.fraction_per_bin,
                   "count_per_bin": config.sample.count_per_bin,
                   "max_rows": config.sample.max_rows,
                   "seed": config.sample.seed, "all_rows": config.sample.all_rows,
                   "limit": limit},
        "subset": {"metric": config.subset.metric},
    }
    manifest.fingerprints = {"theme": theme.fingerprint(), "theme_id": theme.id}
    manifest.counts = {"rows": len(sampled),
                       "by_content_type": frames.class_counts(sampled, content_column)}
    manifest.dry_run = paths.dry_run
    manifest.add_output("sample", out_path)
    if limit is not None:
        manifest.note(f"--limit {limit} was used: this is a TEST sample, not the run's sample.")
    manifest.write(paths.manifest_for(out_path))
    return {"rows": len(sampled), "path": str(out_path)}
