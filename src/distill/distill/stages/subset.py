"""Stage 8 -- the subset: the top N% of the theme's dataset by a relevance metric."""

from __future__ import annotations

from typing import Any

from dataclasses import replace as dc_replace

import pandas as pd
import pyarrow.parquet as pq

from .. import frames
from ..config import REFERENCE_COLUMN, DistillConfig
from ..manifest import Manifest
from ..paths import RunPaths
from ..theme import ThemeSpec


def metric_is_derivable(config: DistillConfig, available: set[str]) -> bool:
    """Whether the subset metric can be computed from the per-sentence similarities.

    A step-7 artifact carries `extracted_sentence_sims`, a list per row, and no scalar
    similarity at all -- the single number the methodology cuts on is the row's maximum. The
    pricing-power run had one only because its input was an already-assembled corpus.
    """
    return (config.subset.metric not in available
            and config.sentence.sims_column in available)


def derive_metric(frame, config: DistillConfig):
    """Add the subset metric as the row maximum of the per-sentence similarities."""
    import numpy as np

    metric, sims = config.subset.metric, config.sentence.sims_column
    frame[metric] = [
        float(np.max(value)) if value is not None and len(value) else float("nan")
        for value in frame[sims]
    ]
    return frame


def apply_max_rows(parts, groups, config: DistillConfig,
                   per_type: dict[str, Any]) -> tuple[list, dict[str, Any] | None]:
    """Trim the subset to `subset.max_rows` by TIGHTENING the cut, not by sampling it.

    `top_fraction` is a share of whatever the search happened to retrieve, so the same 20% is
    290k rows on one theme and 1.4M on another. The ceiling makes the cost predictable, and it
    has to keep the property the cut was chosen for: the rows nearest the theme. So each
    content type keeps its own TOP rows by the metric, and keeps them in proportion to the
    share it already had -- a global cut would quietly change the news/transcripts mix, which
    is the one thing `per_content_type` exists to hold steady.
    """
    ceiling = config.subset.max_rows
    total = sum(len(part) for part in parts)
    if not ceiling or total <= ceiling:
        return parts, None

    metric = config.subset.metric
    labels = [label for label, _ in groups if not _.empty]
    print(f"\nsubset.max_rows: {total:,} rows is over the {ceiling:,} ceiling -- tightening "
          f"the cut, keeping each content type's TOP rows by {metric!r}")
    trimmed, report = [], {"ceiling": ceiling, "before": total, "per_type": {}}
    order = sorted(range(len(parts)), key=lambda i: len(parts[i]))
    budget, remaining = ceiling, total
    keep_by_index: dict[int, int] = {}
    for position, index in enumerate(order):
        part = parts[index]
        if position == len(order) - 1:
            take = min(len(part), budget)
        else:
            take = min(len(part), int(round(ceiling * len(part) / total)))
        keep_by_index[index] = max(take, 0)
        budget -= keep_by_index[index]
        remaining -= len(part)
    for index, part in enumerate(parts):
        take = keep_by_index.get(index, len(part))
        label = labels[index] if index < len(labels) else str(index)
        if take >= len(part):
            trimmed.append(part)
            report["per_type"][label] = {"kept": len(part), "trimmed_from": len(part)}
            continue
        values = pd.to_numeric(part[metric], errors="coerce")
        order_columns = [c for c in config.input.key_fields if c in part.columns]
        ranked = part.assign(_metric=values).sort_values(
            ["_metric", *order_columns], ascending=[False, *[True] * len(order_columns)],
            kind="mergesort")
        kept = ranked.head(take).drop(columns="_metric")
        threshold = float(pd.to_numeric(kept[metric], errors="coerce").min())
        print(f"[{label}] {len(part):,} -> {len(kept):,} "
              f"({100 * len(kept) / max(len(part), 1):.1f}% of its cut), "
              f"{metric} >= {threshold:.4f}")
        trimmed.append(kept)
        report["per_type"][label] = {"kept": len(kept), "trimmed_from": len(part),
                                     "threshold": threshold}
        if label in per_type:
            per_type[label]["kept"] = len(kept)
            per_type[label]["threshold"] = threshold
            per_type[label]["trimmed_by_max_rows"] = True
    report["after"] = sum(len(part) for part in trimmed)
    return trimmed, report


def required_columns(config: DistillConfig) -> list[str]:
    """Columns without which this run cannot proceed."""
    wanted = [
        *config.input.key_fields,
        config.input.content_type_column,
        config.input.entity_name_column,
        config.input.text_column,
        config.input.headline_column,
        config.subset.metric,
        config.input.date_column,
        config.input.timestamp_column,
        config.input.relevance_column,
        config.input.sentiment_column,
    ]
    if config.input.reference_label_column:
        wanted.append(config.input.reference_label_column)
    if "sentence" in config.granularities():
        wanted += [config.sentence.sentences_column, config.sentence.sims_column]
    return list(dict.fromkeys(w for w in wanted if w))


def needed_columns(config: DistillConfig) -> list[str]:
    """Every column any later stage reads, so the subset is self-sufficient."""
    return list(dict.fromkeys([
        *required_columns(config),
        config.sentence.sentences_column,
        config.sentence.sims_column,
        config.sentence.seeds_column,
    ]))


def run(config: DistillConfig, theme: ThemeSpec, paths: RunPaths, *,
        force: bool = False, top_fraction: float | None = None) -> dict[str, Any]:
    if top_fraction is not None:
        config.subset = dc_replace(config.subset, top_fraction=top_fraction)
    out_path = paths.subset_parquet
    if out_path.exists() and not force:
        existing = pd.read_parquet(out_path, columns=["row_key"])
        print(f"subset already written: {out_path} ({len(existing):,} rows). "
              f"Pass --force to rebuild.")
        return {"rows": len(existing), "path": str(out_path), "reused": True}

    manifest = Manifest(stage="subset")
    source = config.source_frame_path()
    manifest.add_input("source", source,
                       irreproducible=config.input.irreproducible_source)

    schema_names = set(pq.ParquetFile(source).schema_arrow.names)
    derivable = metric_is_derivable(config, schema_names)
    missing = [c for c in required_columns(config)
               if c not in schema_names and not (derivable and c == config.subset.metric)]
    if missing:
        raise SystemExit(
            f"the input is missing columns this run needs: {missing}\n"
            f"It has: {sorted(schema_names)}\n"
            f"Either point the matching `input.*` / `backtest.*_column` setting at the right "
            f"name, or use a different input."
        )
    wanted = [c for c in needed_columns(config) if c in schema_names]
    if derivable and config.sentence.sims_column not in wanted:
        wanted.append(config.sentence.sims_column)
    optional_absent = [c for c in needed_columns(config)
                       if c not in schema_names
                       and not (derivable and c == config.subset.metric)]
    if optional_absent:
        print(f"  note: {optional_absent} are absent, so this subset cannot serve a "
              f"sentence-level experiment later")

    frame = frames.read_source(config, columns=wanted)
    if derivable:
        frame = derive_metric(frame, config)
        manifest.note(f"{config.subset.metric} derived as the row maximum of "
                      f"{config.sentence.sims_column}")
    reference = config.input.reference_label_column
    if reference and reference != REFERENCE_COLUMN:
        frame = frame.rename(columns={reference: REFERENCE_COLUMN})
        print(f"  reference label {reference!r} -> {REFERENCE_COLUMN!r} "
              f"(so it cannot collide with this run's own label column)")
    content_column = config.input.content_type_column
    known = set(config.subset.content_types)
    present = set(frame[content_column].dropna().unique())
    unexpected = sorted(present - known)
    if unexpected:
        print(f"  note: dropping {sorted(unexpected)} -- subset.content_types is "
              f"{list(config.subset.content_types)}")
    frame = frame[frame[content_column].isin(known)].copy()

    metric = config.subset.metric
    fraction = config.subset.top_fraction
    parts: list[pd.DataFrame] = []
    per_type: dict[str, Any] = {}

    groups: list[tuple[str, pd.DataFrame]]
    if config.subset.per_content_type:
        groups = [(str(ct), frame[frame[content_column] == ct]) for ct in config.subset.content_types]
    else:
        groups = [("all", frame)]

    for label, group in groups:
        if group.empty:
            print(f"[{label}] no rows")
            per_type[label] = {"available": 0, "kept": 0}
            continue
        if fraction >= 1.0:
            kept = group
            threshold = None
        else:
            values = pd.to_numeric(group[metric], errors="coerce")
            if values.isna().all():
                raise SystemExit(
                    f"[{label}] {metric!r} is not numeric on any row, so a top-{fraction:.0%} cut "
                    f"is meaningless. Set subset.metric to a numeric column."
                )
            threshold = float(values.quantile(1.0 - fraction))
            kept = group.loc[values >= threshold]
        per_type[label] = {"available": len(group), "kept": len(kept),
                           "threshold": threshold, "metric": metric}
        if metric in group.columns:
            # what the quantile above was taken over; the ic stage re-cuts its bins from it
            per_type[label]["scored"] = int(pd.to_numeric(group[metric], errors="coerce")
                                            .notna().sum())
        parts.append(kept)

    if not parts:
        raise SystemExit("the subset is empty")
    parts, capped = apply_max_rows(parts, groups, config, per_type)
    subset = pd.concat(parts, ignore_index=True)
    frames.write_parquet(subset, out_path, announce=False)


    if capped:
        manifest.note(f"subset trimmed from {capped['before']:,} to {capped['after']:,} rows by "
                      f"subset.max_rows={capped['ceiling']:,}")
    manifest.config = {
        "subset": {"metric": metric, "top_fraction": fraction,
                   "max_rows": config.subset.max_rows, "capped": capped,
                   "per_content_type": config.subset.per_content_type,
                   "content_types": list(config.subset.content_types)},
        "input": {"key_fields": list(config.input.key_fields),
                  "irreproducible_source": config.input.irreproducible_source},
    }
    manifest.fingerprints = {"theme": theme.fingerprint(), "theme_id": theme.id}
    manifest.counts = {"rows": len(subset), "by_content_type": per_type}
    manifest.dry_run = paths.dry_run
    manifest.add_output("subset", out_path)
    if config.input.irreproducible_source:
        manifest.note(
            "input.irreproducible_source is set: the source file is not regenerable by any "
            "stage and must be treated as version-controlled source, not as an artifact."
        )
    manifest.write(paths.manifest_for(out_path))
    return {"rows": len(subset), "path": str(out_path), "by_content_type": per_type}
