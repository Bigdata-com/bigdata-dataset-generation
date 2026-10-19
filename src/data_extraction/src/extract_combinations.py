#!/usr/bin/env python3
"""Phase 1 of notebook 2: collapse the search output to one row per (document."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

import workflow_paths as wp
from sentence_extraction import extract_target_sentences

GROUP_KEY = ["doc_id", "chunk_index", "entity_id"]

CARRIED_COLUMNS = ["chunk_text", "detections", "source_name", "content_type", "headline",
                   "date", "timestamp", "url", "chunk_sentiment"]

ROW_CHUNK = 200_000
COLLAPSE_PARTS = 8


def read_parquet_with_bar(path, desc: str, batch_size: int = 250_000) -> pd.DataFrame:
    """``pd.read_parquet`` with a progress bar."""
    parquet = pq.ParquetFile(path)
    batches = []
    with tqdm(total=parquet.metadata.num_rows, unit="row", unit_scale=True, desc=desc,
              leave=False) as bar:
        for batch in parquet.iter_batches(batch_size=batch_size):
            batches.append(batch)
            bar.update(batch.num_rows)
    table = pa.Table.from_batches(batches)
    del batches
    frame = table.to_pandas()
    del table
    gc.collect()
    return frame


def chunked_list(function, total: int, desc: str, chunk: int = ROW_CHUNK) -> list:
    """Build a length-``total`` list by calling ``function(lo, hi)`` per chunk, with a bar."""
    out: list = []
    with tqdm(total=total, unit="row", unit_scale=True, desc=desc, leave=False) as bar:
        for low in range(0, total, chunk):
            high = min(low + chunk, total)
            out.extend(function(low, high))
            bar.update(high - low)
    return out


def collapse_with_bar(frame: pd.DataFrame, group_key: list[str], agg: dict, desc: str,
                      parts: int = COLLAPSE_PARTS) -> pd.DataFrame:
    """``groupby(group_key).agg(agg)`` in hash partitions OF THE KEY, with a progress bar."""
    partition = pd.util.hash_pandas_object(frame[group_key], index=False).to_numpy() % parts
    collected = []
    with tqdm(total=parts, unit="part", desc=desc, leave=False) as bar:
        for index in range(parts):
            subset = frame.loc[partition == index]
            if not subset.empty:
                collected.append(subset.groupby(group_key, sort=False).agg(**agg).reset_index())
            del subset
            gc.collect()
            bar.update(1)
            bar.set_postfix_str(f"{sum(len(c) for c in collected):,} combinations")
    del partition
    return pd.concat(collected, ignore_index=True) if collected else frame.head(0)


def load_entity_names(universe_csv: Path) -> dict[str, str]:
    """id -> name, from the universe CSV named in the config (never a hardcoded filename)."""
    universe_csv = Path(universe_csv)
    if not universe_csv.exists():
        raise FileNotFoundError(f"universe CSV not found: {universe_csv}")
    frame = pd.read_csv(universe_csv, dtype={"id": "string"})
    missing = {"id", "name"} - set(frame.columns)
    if missing:
        raise ValueError(f"{universe_csv.name} must have columns id, name; missing {sorted(missing)}")
    return dict(zip(frame["id"].astype(str).str.strip(), frame["name"].astype(str).str.strip()))


def entity_offsets_for(entity_id, detections) -> list[list[int]]:
    """[[start, end], ...] for this entity's mentions in this chunk's detections."""
    if detections is None:
        return []
    target = str(entity_id).strip()
    out: list[list[int]] = []
    for detection in detections:
        try:
            if detection.get("type", "entity") == "entity" \
                    and str(detection.get("id")).strip() == target:
                out.append([int(detection["start"]), int(detection["end"])])
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    return out


def phase_extract(run_dir: Path, cfg: dict, universe_csv: Path, verbose: bool = True) -> Path:
    """Collapse + extract, appending one content type at a time."""
    out_path = wp.sentences_parquet(run_dir)
    if wp.is_done(out_path):
        rows = wp.read_done(out_path)["rows"]
        print(f"{out_path.name} already complete ({rows:,} rows) — phase skipped. "
              f"Delete it and its .done.json to force a re-extract.")
        return out_path
    wp.clear_output(out_path)

    present = {}
    for content_type in wp.CONTENT_TYPES:
        path = wp.exploded_parquet(run_dir, content_type)
        if path.exists():
            wp.require_done(path, f"{content_type} exploded chunks")
            present[content_type] = path
        else:
            print(f"  [{content_type}] missing, skipped")
    if not present:
        raise FileNotFoundError(
            f"no exploded parquets in {run_dir}. Notebook 1 must write 'exploded_chunks/' "
            f"first; if that folder was deleted to reclaim disk, re-run notebook 1's build step "
            f"(local compute, no API calls)."
        )

    names = {ct: set(pq.ParquetFile(p).schema_arrow.names) | {"content_type"}
             for ct, p in present.items()}
    carried = [c for c in CARRIED_COLUMNS if all(c in n for n in names.values())]
    dropped = [c for c in CARRIED_COLUMNS if c not in carried]
    if dropped:
        print(f"  not carried (absent in at least one content type): {dropped}")

    relevance_fraction = float(cfg.get("relevance_top_fraction", 1.0))
    drop_zero_sentiment = bool(cfg.get("drop_zero_sentiment_rows", False))
    name_by_id = load_entity_names(universe_csv)

    temp_path = Path(str(out_path) + ".part")
    temp_path.unlink(missing_ok=True)
    writer = None
    schema = None
    rows_written = 0
    stats = {}

    try:
        for content_type, path in present.items():
            frame = read_parquet_with_bar(path, desc=f"{content_type}: read")
            frame["content_type"] = content_type
            if "polarity" not in frame.columns:
                frame["polarity"] = ""
            rows_in = len(frame)

            if drop_zero_sentiment and "chunk_sentiment" in frame.columns:
                zero = int((frame["chunk_sentiment"] == 0).sum())
                if zero:
                    frame = frame.loc[frame["chunk_sentiment"] != 0].reset_index(drop=True)
                print(f"dropped {zero:,} rows with chunk_sentiment == 0")

            if "chunk_relevance" in frame.columns and relevance_fraction < 1.0:
                threshold = frame["chunk_relevance"].quantile(1 - relevance_fraction)
                before = len(frame)
                frame = frame.loc[frame["chunk_relevance"] >= threshold].reset_index(drop=True)
                print(f"relevance: {before:,} -> {len(frame):,} rows "
                      f"({len(frame) / max(before, 1) * 100:.1f}%, >= {threshold:.4f})")

            agg = {c: (c, "first") for c in carried}
            if "chunk_relevance" in frame.columns:
                agg["chunk_relevance"] = ("chunk_relevance", "max")
            agg["search_sentences"] = ("sentence", lambda s: sorted(set(s.dropna())))
            agg["search_polarities"] = ("polarity", lambda s: sorted(set(s.dropna())))
            agg["n_retrievals"] = ("sentence", "size")

            retrievals = len(frame)
            frame = collapse_with_bar(frame, GROUP_KEY, agg, desc=f"{content_type}: collapse")
            gc.collect()

            frame["n_search_sentences"] = frame["search_sentences"].map(len)
            frame["has_positive"] = frame["search_polarities"].map(lambda p: "positive" in p)
            frame["has_negative"] = frame["search_polarities"].map(lambda p: "negative" in p)
            frame["is_mixed"] = frame["has_positive"] & frame["has_negative"]

            if "detections" in frame.columns:
                ids = frame["entity_id"].to_numpy()
                detections = frame["detections"].to_numpy()
                frame["entity_offsets"] = chunked_list(
                    lambda low, high: [entity_offsets_for(e, d)
                                       for e, d in zip(ids[low:high], detections[low:high])],
                    len(frame), desc=f"{content_type}: offsets",
                )
                del ids, detections
                frame.drop(columns=["detections"], inplace=True)
                gc.collect()
            else:
                frame["entity_offsets"] = [[] for _ in range(len(frame))]
                print("WARNING: no 'detections' column — every entity_offsets is empty, so no "
                      "sentence can be extracted and the run will produce nothing.")

            frame["entity_name"] = frame["entity_id"].astype(str).str.strip().map(
                lambda x: name_by_id.get(x, x))
            unresolved = int((frame["entity_name"] == frame["entity_id"].astype(str)).sum())
            if unresolved:
                print(f"WARNING: {unresolved:,} rows have no name in the universe CSV and will "
                      f"be sent to the model identified by their id.")
            frame["n_mentions"] = frame["entity_offsets"].map(len)

            texts = frame["chunk_text"].to_numpy()
            offsets = frame["entity_offsets"].to_numpy()
            frame["extracted_sentences"] = chunked_list(
                lambda low, high: [extract_target_sentences(t, o)
                                   for t, o in zip(texts[low:high], offsets[low:high])],
                len(frame), desc=f"{content_type}: extract",
            )
            del texts, offsets
            frame["n_sentences"] = frame["extracted_sentences"].map(len)

            stats[content_type] = {
                "per_retrieval_rows": rows_in,
                "combinations": len(frame),
                "with_sentences": int((frame["n_sentences"] > 0).sum()),
                "sentences_total": int(frame["n_sentences"].sum()),
            }
            print(f"[{content_type}] {rows_in:,} retrievals -> {len(frame):,} "
                  f"combinations, {stats[content_type]['sentences_total']:,} sentences")

            table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(temp_path, schema)
            writer.write_table(table)
            rows_written += len(frame)
            del frame, table, agg
            gc.collect()
    finally:
        if writer is not None:
            writer.close()

    os.replace(temp_path, out_path)
    wp.mark_done(out_path, rows_written, per_content_type={k: v["combinations"]
                                                          for k, v in stats.items()})
    print(f"\nphase complete -> {out_path}  ({rows_written:,} rows, "
          f"{out_path.stat().st_size / 2**30:.2f} GB)")
    return out_path
