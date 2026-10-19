#!/usr/bin/env python3
"""2 — Sentence extraction and embeddings."""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=(__doc__ or "").strip())
parser.add_argument("--run", default=None,
                    help="where the pipeline writes. Default: output/ beside config/. Its "
                         "config/data_extraction.yaml is what this step reads.")
parser.add_argument("--yes", action="store_true",
                    help="skip the confirmation before a metered embedding pass (only the "
                         "OpenAI backend has one)")
parser.add_argument("--no-plots", action="store_true",
                    help="skip the diagnostic charts (they are saved as PNG, never displayed)")
parser.add_argument("--drop-embedding-scratch", action="store_true",
                    help="DELETE the embedding shards + float16 store once the similarity "
                         "parquet is complete; a later re-run of phase 3 would have to encode "
                         "the corpus again. Refuses unless phase 3 finished.")
args = parser.parse_args()


def find_package_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(f"pyproject.toml not found above {start}")

PROJECT_ROOT = find_package_root(Path(__file__).resolve().parent)
for _src in (Path(__file__).resolve().parent / "src",):
    if _src.exists() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env", override=False)
print("repository:", PROJECT_ROOT)

import numpy as np
import pyarrow.parquet as pq

import workflow_paths as wp
from extract_combinations import phase_extract
from embed_similarity import phase_embed, phase_similarity
from text_embeddings import resolve_model

SAVE_PLOTS = not args.no_plots
if SAVE_PLOTS:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

RUN_DIR = wp.require_run_dir(args.run, PROJECT_ROOT)
CFG = wp.load_run_config(RUN_DIR)
EXTRACTION = wp.section(CFG, "extraction_embeddings")

PLOT_DIR = RUN_DIR / "diagnostics"
if SAVE_PLOTS:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

EMBED_BLOCK = int(EXTRACTION.get("embed_block", 20_000))
SIM_ROW_BATCH = int(EXTRACTION.get("similarity_row_batch", 25_000))
EMBED_PROVIDER, EMBED_MODEL = resolve_model(EXTRACTION)

print(f"run dir : {RUN_DIR}")
print(f"config  : {wp.config_path(RUN_DIR).relative_to(wp.repo_root(PROJECT_ROOT))}")
print(f"theme   : {(CFG.get('theme') or {}).get('id')}")
print(f"embed   : {EMBED_MODEL}")
print(f"batching: embed_block={EMBED_BLOCK:,}, similarity_row_batch={SIM_ROW_BATCH:,}")
if EMBED_PROVIDER == "openai":
    print(f"rate    : embed_max_workers={EXTRACTION.get('embed_max_workers', 32)}")

SENTENCES_PATH = phase_extract(RUN_DIR, EXTRACTION,
                               wp.universe_csv(CFG, RUN_DIR, PROJECT_ROOT))
gc.collect()

n_missing = phase_embed(RUN_DIR, EXTRACTION, block=EMBED_BLOCK, dry_run=True)

# The cards are free and the pass costs nothing but time, so there is nothing to confirm.
# Sending the corpus to a provider that bills per token is a different matter.
if EMBED_PROVIDER == "openai":
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (environment or .env)")
    if not args.yes:
        input(">>> the dry run above says how many sentences still need embedding. Press Enter "
              "to run the METERED embedding pass, or Ctrl-C to stop. ")

phase_embed(RUN_DIR, EXTRACTION, block=EMBED_BLOCK, dry_run=False)
gc.collect()

SIM_PATH = phase_similarity(RUN_DIR, row_batch=SIM_ROW_BATCH)
gc.collect()

print(f"\n{pq.ParquetFile(SIM_PATH).metadata.num_rows:,} rows, "
      f"{SIM_PATH.stat().st_size / 2**30:.2f} GB -> {SIM_PATH}")

_by_type: dict[str, list] = {}
_parquet = pq.ParquetFile(SIM_PATH)
for _batch in _parquet.iter_batches(batch_size=200_000,
                                    columns=["extracted_sentence_sims", "content_type"]):
    for _ct, _values in zip(_batch.column("content_type").to_pylist(),
                            _batch.column("extracted_sentence_sims").to_pylist()):
        if _values:
            _by_type.setdefault(_ct, []).extend(_values)
_by_type = {k: np.asarray(v, dtype="float32") for k, v in _by_type.items()}
_by_type = {k: v[~np.isnan(v)] for k, v in _by_type.items()}

for _ct, _values in _by_type.items():
    print(f"[{_ct}] n={len(_values):,}  mean {_values.mean():.4f}  "
          f"median {np.median(_values):.4f}  p80 {np.quantile(_values, .8):.4f}  "
          f"p95 {np.quantile(_values, .95):.4f}")

if SAVE_PLOTS and _by_type:
    fig, axes = plt.subplots(1, len(_by_type), figsize=(6 * len(_by_type), 3.2), squeeze=False)
    for _ax, (_ct, _values) in zip(axes[0], _by_type.items()):
        _ax.hist(_values, bins=80, color="#4c78a8", edgecolor="white", linewidth=0.3)
        for _q, _color in ((0.5, "#e45756"), (0.8, "#f58518")):
            _ax.axvline(np.quantile(_values, _q), color=_color, ls="--", lw=1.3,
                        label=f"p{int(_q * 100)}")
        _ax.set_title(f"{_ct} — cosine to best seed statement (n={len(_values):,})")
        _ax.set_xlabel("cosine similarity")
        _ax.legend(fontsize=8)
    axes[0][0].set_ylabel("extracted sentences")
    fig.tight_layout()
    _out = PLOT_DIR / "similarity_distribution.png"
    fig.savefig(_out, dpi=120)
    plt.close(fig)
    print(f"chart -> {_out}")

del _by_type, _parquet
gc.collect()

_sentences_store, _vectors_store = wp.embedding_store_paths(RUN_DIR)
_shard_dir = wp.embeddings_shards(RUN_DIR)
_shards = sorted(_shard_dir.glob("shard_*.npz")) if _shard_dir.exists() else []
_scratch = [*_shards, _sentences_store, _vectors_store]
_size = sum(p.stat().st_size for p in _scratch if p.exists()) / 2**30
print(f"\nembedding scratch on disk: {_size:.2f} GB ({len(_shards)} shard(s) + store)")

if args.drop_embedding_scratch and not wp.is_done(wp.similarity_parquet(RUN_DIR)):
    print("REFUSING to delete: the similarity parquet is missing or incomplete. Re-run phase 3.")
elif args.drop_embedding_scratch:
    for _path in _scratch:
        _path.unlink(missing_ok=True)
    print(f"deleted {_size:.2f} GB of embedding scratch (the similarity parquet is untouched)")
