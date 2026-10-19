#!/usr/bin/env python3
"""The three steps behind notebook 1: plan the search, run it, and build the chunk dataset."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

import checkpoints as ck
import workflow_paths as wp

EXCLUDED_SOURCE_NAMES = ("PubT",)


def _rss_gb() -> float:
    """Resident memory of this process, in GiB."""
    try:
        with open("/proc/self/statm") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
    except (OSError, ValueError, AttributeError):
        return float("nan")


def is_transcript_source(source_name, transcript_sources) -> bool:
    """True when a chunk comes from an earnings call / presentation, else it is news."""
    if source_name is None or (isinstance(source_name, float) and pd.isna(source_name)):
        return False
    text = str(source_name).strip()
    if text in transcript_sources:
        return True
    lowered = text.lower()
    return ("transcript" in lowered) or ("presentation material" in lowered)


def _to_str_list(value) -> list[str]:
    """Coerce an entity-ids cell (ndarray / list / None) into a clean list of strings."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, np.ndarray):
        items = value.tolist()
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return []
    out = []
    for item in items:
        if item is None or (isinstance(item, float) and pd.isna(item)):
            continue
        out.append(str(item).strip())
    return out


def plan_all(run_dir: Path, statements: list[str], search_cfg: dict,
             universe_ids: list[str]) -> list[dict]:
    """Plan one search per statement."""
    from bigdata_smart_batching import plan_search

    plans_dir = wp.plans_dir(run_dir)
    items = []
    for statement in tqdm(statements, desc="plan", unit="stmt"):
        item = ck.load_plan(plans_dir, statement)
        if item is None:
            plan = plan_search(
                universe=universe_ids,
                start_date=search_cfg["start_date"],
                end_date=search_cfg["end_date"],
                text=statement,
                category=search_cfg["combined_category"],
                min_entities_per_basket=search_cfg.get("min_entities_per_basket", 10),
                min_period_days=search_cfg.get("min_period_days", 90),
                requests_per_minute=search_cfg.get("requests_per_minute", 200),
                max_workers=search_cfg.get("search_max_workers", 40),
            )
            ck.save_plan(plans_dir, statement, plan)
            item = {"statement": statement, "plan": plan}
        items.append(item)
    baskets = [len(item["plan"].get("baskets", [])) for item in items]
    print(f"plans: {len(items):,}/{len(statements):,} | baskets {sum(baskets):,} "
          f"(per statement min {min(baskets)} / max {max(baskets)} / "
          f"avg {sum(baskets) / len(items):.1f})")
    return items


def search_all(run_dir: Path, plan_items: list[dict], search_cfg: dict) -> None:
    """Execute one search per statement."""
    from bigdata_smart_batching import execute_search

    raw_dir = wp.search_dir(run_dir)
    for item in tqdm(plan_items, desc="search", unit="stmt"):
        path = ck.raw_docs_path(raw_dir, item["statement"])
        if path.exists():
            continue
        documents = execute_search(
            item["plan"],
            chunk_percentage=search_cfg.get("chunk_percentage", 0.5),
            requests_per_minute=search_cfg.get("requests_per_minute", 200),
            max_workers=search_cfg.get("search_max_workers", 40),
            second_pass=search_cfg.get("second_pass", False),
        )
        ck.save_pickle_atomic(path, documents)

    documents_total = chunks_total = 0
    for item in plan_items:
        documents = ck.load_pickle(ck.raw_docs_path(raw_dir, item["statement"]))
        if documents is None:
            continue
        documents_total += len(documents)
        chunks_total += sum(len(doc.get("chunks", [])) for doc in documents)
    print(f"search done: {documents_total:,} documents, {chunks_total:,} chunks "
          f"(raw, pre-dedup, across {len(plan_items)} statements)")


def build_exploded(run_dir: Path, plan_items: list[dict], allowed_entity_ids: set,
                   polarity_by_statement: dict, transcript_sources) -> dict[str, Path]:
    """Dedup -> rows -> filter -> explode -> append, one statement at a time."""
    from bigdata_smart_batching import convert_to_dataframe, deduplicate_documents

    transcript_sources = set(transcript_sources or ())
    raw_dir = wp.search_dir(run_dir)
    dedup_dir = wp.dedup_dir(run_dir)
    outputs = {ct: wp.exploded_parquet(run_dir, ct) for ct in wp.CONTENT_TYPES}

    if all(wp.is_done(path) for path in outputs.values()):
        for ct, path in outputs.items():
            print(f"[{ct}] already complete ({wp.read_done(path)['rows']:,} rows) — skipped. "
                  f"Delete {path.name} and its .done.json to force a rebuild.")
        return outputs
    for path in outputs.values():
        wp.clear_output(path)

    sizes = []
    for item in plan_items:
        dedup_path = ck.dedup_docs_path(dedup_dir, item["statement"])
        raw_path = ck.raw_docs_path(raw_dir, item["statement"])
        chosen = dedup_path if dedup_path.exists() else raw_path
        sizes.append(chosen.stat().st_size if chosen.exists() else 0)

    writers: dict[str, pq.ParquetWriter] = {}
    schemas: dict[str, pa.Schema] = {}
    temp_paths = {ct: Path(str(path) + ".part") for ct, path in outputs.items()}
    for path in temp_paths.values():
        path.unlink(missing_ok=True)

    doc_hashes: list[np.ndarray] = []
    chunk_hashes: list[np.ndarray] = []
    entities: set[str] = set()
    rows_pre = rows_post = rows_exploded = 0
    per_type: dict[str, int] = {}

    bar = tqdm(total=sum(sizes) or None, unit="B", unit_scale=True, unit_divisor=1024,
               desc="build", smoothing=0.1)

    def stage(index: int, name: str) -> None:
        bar.set_postfix_str(
            f"stmt {index + 1}/{len(plan_items)} · {name} · out {rows_exploded:,} rows "
            f"· rss {_rss_gb():.1f} GB", refresh=True,
        )

    try:
        for index, item in enumerate(plan_items):
            statement = item["statement"]
            stage(index, "load")
            raw_path = ck.raw_docs_path(raw_dir, statement)
            dedup_path = ck.dedup_docs_path(dedup_dir, statement)
            raw_documents = ck.load_pickle(raw_path)
            if raw_documents is None:
                raise FileNotFoundError(f"missing search checkpoint: {raw_path}")
            documents = ck.load_pickle(dedup_path) if dedup_path.exists() else None
            if documents is None:
                documents = deduplicate_documents(raw_documents)
                ck.save_pickle_atomic(dedup_path, documents)
            del raw_documents

            stage(index, "convert")
            frame = convert_to_dataframe(documents)
            time_by_doc = {d["id"]: d.get("timestamp") for d in documents if d.get("id")}
            frame.insert(
                frame.columns.get_loc("date") + 1,
                "timestamp",
                pd.to_datetime(frame["doc_id"].map(time_by_doc), errors="coerce")
                  .dt.strftime("%H:%M:%S"),
            )
            frame.insert(0, "sentence_key", ck.statement_key(statement))
            frame.insert(0, "polarity", polarity_by_statement.get(statement, ""))
            frame.insert(0, "sentence", statement)
            del documents, time_by_doc
            gc.collect()

            rows_pre += len(frame)
            doc_hashes.append(pd.util.hash_pandas_object(frame["doc_id"], index=False).to_numpy())

            stage(index, "filter")
            if "source_name" in frame.columns:
                frame = frame.loc[
                    ~frame["source_name"].astype("string").isin(EXCLUDED_SOURCE_NAMES)
                ].copy()
            frame["entity_ids"] = frame["entity_ids"].map(
                lambda cell: [e for e in dict.fromkeys(_to_str_list(cell))
                              if e in allowed_entity_ids]
            )
            frame = frame.loc[frame["entity_ids"].map(len) > 0]
            rows_post += len(frame)
            if frame.empty:
                del frame
                gc.collect()
                bar.update(sizes[index])
                continue

            stage(index, "explode")
            frame = (frame.explode("entity_ids", ignore_index=True)
                          .rename(columns={"entity_ids": "entity_id"}))
            frame = frame.drop_duplicates(
                subset=["sentence_key", "entity_id", "doc_id", "chunk_index"], keep="first"
            ).reset_index(drop=True)
            rows_exploded += len(frame)
            entities.update(frame["entity_id"].unique().tolist())
            chunk_hashes.append(
                pd.util.hash_pandas_object(frame[["doc_id", "chunk_index"]], index=False).to_numpy()
            )

            stage(index, "write")
            is_transcript = frame["source_name"].map(
                lambda s: is_transcript_source(s, transcript_sources)
            ).to_numpy()
            for content_type, part in (("news", frame.loc[~is_transcript]),
                                       ("transcripts", frame.loc[is_transcript])):
                if part.empty:
                    continue
                table = pa.Table.from_pandas(part.reset_index(drop=True),
                                             schema=schemas.get(content_type),
                                             preserve_index=False)
                if content_type not in writers:
                    schemas[content_type] = table.schema
                    writers[content_type] = pq.ParquetWriter(temp_paths[content_type],
                                                             table.schema)
                writers[content_type].write_table(table)
                per_type[content_type] = per_type.get(content_type, 0) + len(part)
                del table, part

            del frame, is_transcript
            gc.collect()
            bar.update(sizes[index])
            stage(index, "done")
    finally:
        bar.close()
        for writer in writers.values():
            writer.close()

    documents_unique = int(np.unique(np.concatenate(doc_hashes)).size) if doc_hashes else 0
    chunks_unique = int(np.unique(np.concatenate(chunk_hashes)).size) if chunk_hashes else 0
    del doc_hashes, chunk_hashes
    gc.collect()

    print(f"build: {documents_unique:,} documents, {rows_pre:,} chunk-rows -> {rows_post:,} "
          f"after source/universe filter -> {rows_exploded:,} entity-chunk rows "
          f"({chunks_unique:,} unique doc-chunks, {len(entities):,} entities)")

    written = {}
    for content_type, path in outputs.items():
        if content_type not in writers:
            print(f"[{content_type}] no rows — nothing written")
            temp_paths[content_type].unlink(missing_ok=True)
            continue
        os.replace(temp_paths[content_type], path)
        wp.mark_done(path, per_type.get(content_type, 0))
        written[content_type] = path
        print(f"[{content_type}] {per_type.get(content_type, 0):,} entity-chunk rows -> "
              f"{path.name} ({path.stat().st_size / 2**30:.2f} GB)")
    if not written:
        raise RuntimeError("the build produced no rows at all — check the universe and the seeds")
    return written
