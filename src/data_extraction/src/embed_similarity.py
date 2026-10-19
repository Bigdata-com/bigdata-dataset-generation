#!/usr/bin/env python3
"""The two heavy phases of notebook 2: embed every distinct sentence once."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import workflow_paths as wp  # noqa: E402

SENTENCE_COLUMNS = ("search_sentences", "extracted_sentences")


def shard_paths(shard_dir: Path) -> list[Path]:
    return sorted(Path(shard_dir).glob("shard_*.npz"))


def shard_dim(shards: list[Path]) -> int:
    """Vector width, read from the first shard — never hardcoded."""
    if not shards:
        raise FileNotFoundError("no shards to read the embedding dimension from")
    with np.load(shards[0], allow_pickle=True) as z:
        return int(z["vectors"].shape[1])


def build_missing_set(sentences_parquet: Path, shard_dir: Path, scan_batch: int) -> list[str]:
    """Sentences needed (streamed from the parquet) minus those already embedded (in shards)."""
    needed: set[str] = set()
    parquet = pq.ParquetFile(sentences_parquet)
    with tqdm(total=parquet.metadata.num_rows, unit="row", unit_scale=True,
              desc="scan sentences") as bar:
        for batch in parquet.iter_batches(batch_size=scan_batch, columns=list(SENTENCE_COLUMNS)):
            for column in SENTENCE_COLUMNS:
                for values in batch.column(column).to_pylist():
                    if values:
                        for sentence in values:
                            if isinstance(sentence, str) and sentence.strip():
                                needed.add(sentence)
            bar.update(batch.num_rows)
            bar.set_postfix_str(f"{len(needed):,} distinct")
    print(f"distinct sentences needed : {len(needed):,}")

    shards = shard_paths(shard_dir)
    for shard in tqdm(shards, unit="shard", desc="subtract embedded"):
        with np.load(shard, allow_pickle=True) as z:
            for sentence in z["sentences"]:
                needed.discard(str(sentence))
    print(f"still missing (to embed)  : {len(needed):,}")
    return sorted(needed)


def model_marker_path(run_dir: Path) -> Path:
    return wp.step_dir(run_dir, "embeddings") / "emb_model.json"


def model_of(run_dir: Path) -> str | None:
    path = model_marker_path(run_dir)
    if not path.exists():
        return None
    try:
        return (json.loads(path.read_text(encoding="utf-8")) or {}).get("model")
    except (OSError, json.JSONDecodeError):
        return None


def claim_model(run_dir: Path, model: str) -> None:
    """One embedding folder holds one model's vectors, and this is what holds it to that.

    The shards are keyed on the sentence string and nothing else, so a second model run over
    the same folder would find every sentence already embedded, skip the work, and score the
    corpus with the first model's vectors. Nothing would fail and the numbers would be wrong.
    """
    recorded = model_of(run_dir)
    if recorded and recorded != model:
        raise SystemExit(
            f"this run's embeddings were written with {recorded!r} and the config now asks "
            f"for {model!r}. Vectors from two models cannot share one store: point "
            f"`embed_model` back at {recorded!r}, or delete "
            f"{wp.step_dir(run_dir, 'embeddings')} and encode the corpus again."
        )
    model_marker_path(run_dir).write_text(json.dumps({"model": model}, indent=2),
                                          encoding="utf-8")


def phase_embed(run_dir: Path, cfg: dict, block: int = 20_000, dry_run: bool = False) -> int:
    """Embed everything still missing, one shard per block."""
    from text_embeddings import make_embedder, resolve_model

    sentences_parquet = wp.require_done(wp.sentences_parquet(run_dir), "sentences parquet")
    shard_dir = wp.embeddings_shards(run_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    _, model = resolve_model(cfg)
    claim_model(run_dir, model)

    missing = build_missing_set(sentences_parquet, shard_dir, scan_batch=200_000)
    if dry_run:
        print(f"[dry-run] {len(missing):,} sentences would be embedded with {model!r}. "
              f"Nothing loaded, nothing written.")
        return len(missing)
    if not missing:
        print("nothing to embed — every sentence is already in a shard.")
        return 0

    for leftover in shard_dir.glob("_writing_shard_*.npz"):
        print(f"removing a half-written shard from an interrupted run: {leftover.name}")
        leftover.unlink()

    existing = shard_paths(shard_dir)
    next_index = (int(existing[-1].stem.split("_")[1]) + 1) if existing else 0
    embedder = make_embedder(cfg, models_dir=wp.find_package_root() / "models")
    print(f"encoder: {embedder.describe()}")
    for warning in embedder.warnings():
        print(f"WARNING: {warning}")
    try:
        with tqdm(total=len(missing), unit="sent", unit_scale=True, desc="embed") as bar:
            for start in range(0, len(missing), block):
                chunk = missing[start:start + block]
                vectors = embedder.encode(chunk)
                tmp = shard_dir / f"_writing_shard_{next_index:05d}.npz"
                np.savez_compressed(tmp, sentences=np.array(chunk, dtype=object),
                                    vectors=vectors.astype(np.float16))
                os.replace(tmp, shard_dir / f"shard_{next_index:05d}.npz")
                next_index += 1
                del vectors
                bar.update(len(chunk))
                bar.set_postfix_str(f"shard_{next_index - 1:05d}.npz")
    finally:
        # The cards belong to the next stage the moment this one is done -- including when
        # it is done because it raised.
        embedder.close()
    print(f"embedded {len(missing):,} sentences into {next_index - len(existing)} new shard(s).")
    return len(missing)


def store_meta_path(run_dir: Path) -> Path:
    return wp.step_dir(run_dir, "embeddings") / "emb_store.meta.json"


def build_store(run_dir: Path):
    """Return (sentence -> row index, float16 memmap)."""
    sent_path, vec_path = wp.embedding_store_paths(run_dir)
    shards = shard_paths(wp.embeddings_shards(run_dir))
    meta = None
    if store_meta_path(run_dir).exists():
        try:
            meta = json.loads(store_meta_path(run_dir).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = None

    if sent_path.exists() and vec_path.exists():
        stale = meta is not None and shards and int(meta.get("shards", -1)) != len(shards)
        if not stale:
            sentences = np.load(sent_path, allow_pickle=True)
            index = {str(s): i for i, s in enumerate(sentences)}
            vectors = np.load(vec_path, mmap_mode="r")
            note = "" if shards else "  (shards deleted; store is self-sufficient)"
            print(f"using existing store: {len(index):,} vectors x {vectors.shape[1]}{note}")
            return index, vectors
        print(f"store was built from {meta.get('shards')} shard(s) but {len(shards)} are present "
              f"— rebuilding it.")

    if not shards:
        raise FileNotFoundError(
            f"no embedding store ({vec_path.name}) and no shards in "
            f"{wp.embeddings_shards(run_dir)} — run the embed phase first."
        )

    dim = shard_dim(shards)
    index: dict[str, int] = {}
    for shard in tqdm(shards, unit="shard", desc="store 1/2 (index)"):
        with np.load(shard, allow_pickle=True) as z:
            for sentence in z["sentences"]:
                key = str(sentence)
                if key not in index:
                    index[key] = len(index)
    print(f"building store: {len(index):,} distinct vectors x {dim} (float16 memmap on disk)")
    vectors = np.lib.format.open_memmap(vec_path, mode="w+", dtype=np.float16,
                                        shape=(len(index), dim))
    for shard in tqdm(shards, unit="shard", desc="store 2/2 (fill)"):
        with np.load(shard, allow_pickle=True) as z:
            sentences, block = z["sentences"], z["vectors"]
            rows = np.fromiter((index[str(s)] for s in sentences), dtype=np.int64,
                               count=len(sentences))
            vectors[rows] = block.astype(np.float16)
    vectors.flush()
    np.save(sent_path, np.array(list(index), dtype=object))
    store_meta_path(run_dir).write_text(
        json.dumps({"shards": len(shards), "vectors": len(index), "dim": dim,
                    "model": model_of(run_dir)}, indent=2),
        encoding="utf-8",
    )
    print(f"store written -> {vec_path.name} + {sent_path.name}")
    return index, np.load(vec_path, mmap_mode="r")


def phase_similarity(run_dir: Path, row_batch: int = 25_000) -> Path:
    """Add ``extracted_sentence_sims`` (and the best-matching statement) to every row."""
    sentences_parquet = wp.require_done(wp.sentences_parquet(run_dir), "sentences parquet")
    out_path = wp.similarity_parquet(run_dir)
    tmp_path = out_path.with_name(out_path.name + ".part")
    index, vectors = build_store(run_dir)

    seed_cache: dict[str, np.ndarray] = {}

    def seed_vector(statement: str):
        cached = seed_cache.get(statement)
        if cached is None and statement in index:
            cached = vectors[index[statement]].astype(np.float32)
            seed_cache[statement] = cached
        return cached

    parquet = pq.ParquetFile(sentences_parquet)
    writer = None
    rows_done = 0
    missing_embeddings = 0
    tmp_path.unlink(missing_ok=True)
    bar = tqdm(total=parquet.metadata.num_rows, unit="row", unit_scale=True, desc="similarity")
    try:
        for batch in parquet.iter_batches(batch_size=row_batch):
            seed_lists = batch.column("search_sentences").to_pylist()
            extracted_lists = batch.column("extracted_sentences").to_pylist()

            batch_rows: dict[str, int] = {}
            for values in extracted_lists:
                for sentence in values or ():
                    if sentence in index and sentence not in batch_rows:
                        batch_rows[sentence] = index[sentence]
            if batch_rows:
                ids = np.fromiter(batch_rows.values(), dtype=np.int64, count=len(batch_rows))
                block = vectors[ids].astype(np.float32)
                vector_of = {s: block[i] for i, s in enumerate(batch_rows)}
            else:
                vector_of = {}

            sims_column, best_column = [], []
            for seeds, extracted in zip(seed_lists, extracted_lists):
                seed_vectors = [(s, seed_vector(s)) for s in (seeds or ())
                                if seed_vector(s) is not None]
                sims, best = [], []
                for sentence in extracted or ():
                    vector = vector_of.get(sentence)
                    if vector is None:
                        missing_embeddings += 1
                    if not seed_vectors or vector is None:
                        sims.append(float("nan"))
                        best.append(None)
                    else:
                        score, statement = max(
                            ((float(np.dot(v, vector)), s) for s, v in seed_vectors),
                            key=lambda pair: pair[0],
                        )
                        sims.append(round(score, 6))
                        best.append(statement)
                sims_column.append(sims)
                best_column.append(best)

            table = pa.Table.from_batches([batch])
            table = table.append_column("extracted_sentence_sims",
                                        pa.array(sims_column, pa.list_(pa.float64())))
            table = table.append_column("extracted_best_search",
                                        pa.array(best_column, pa.list_(pa.string())))
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema)
            writer.write_table(table)
            rows_done += len(seed_lists)
            bar.update(len(seed_lists))
            if missing_embeddings:
                bar.set_postfix_str(f"{missing_embeddings:,} sentences without embedding")
    finally:
        bar.close()
        if writer is not None:
            writer.close()

    if missing_embeddings:
        print(f"WARNING: {missing_embeddings:,} extracted sentences had no embedding and scored "
              f"NaN. Re-run the embed phase, then this phase again.")
    os.replace(tmp_path, out_path)
    wp.mark_done(out_path, rows_done, missing_embeddings=missing_embeddings)
    print(f"done: {rows_done:,} rows -> {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", required=True,
                        help="the run folder; its config/data_extraction.yaml is what this "
                             "reads")
    parser.add_argument("--phase", choices=["embed", "similarity", "all"], default="all")
    parser.add_argument("--row-batch", type=int, default=25_000)
    parser.add_argument("--embed-block", type=int, default=20_000)
    parser.add_argument("--dry-run", action="store_true",
                        help="embed phase: report what is missing and stop")
    args = parser.parse_args()

    from dotenv import load_dotenv

    project_root = wp.find_package_root(Path(__file__).resolve().parent)
    load_dotenv(project_root / ".env", override=False)
    run_dir = wp.require_run_dir(args.run, project_root)
    cfg = wp.section(wp.load_run_config(run_dir), "extraction_embeddings")
    print(f"package root: {project_root}\nrun dir     : {run_dir.name}")

    if args.phase in ("embed", "all"):
        print("\n=== PHASE: embed ===")
        phase_embed(run_dir, cfg, args.embed_block, args.dry_run)
    if args.phase in ("similarity", "all") and not args.dry_run:
        print("\n=== PHASE: similarity ===")
        phase_similarity(run_dir, args.row_batch)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
