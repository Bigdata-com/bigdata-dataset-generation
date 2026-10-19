#!/usr/bin/env python3
"""1 — Plan and Search."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

parser = argparse.ArgumentParser(description=(__doc__ or "").strip())
parser.add_argument("--run", default=None,
                    help="where the pipeline writes. Default: output/ beside config/. Its "
                         "config/data_extraction.yaml is what this step reads.")
parser.add_argument("--yes", action="store_true",
                    help="skip the confirmation before the metered search (for unattended runs)")
parser.add_argument("--diagnostics", action="store_true",
                    help="print per-dataset breakdowns at the end; each re-reads a multi-GB "
                         "parquet — useful once, not on every run")
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

import pandas as pd

import search_build as sb
import workflow_paths as wp

RUN_DIR = wp.require_run_dir(args.run, PROJECT_ROOT)
if not os.environ.get("BIGDATA_API_KEY"):
    raise SystemExit("BIGDATA_API_KEY is not set (environment or .env) — the search cannot run.")
CFG = wp.load_run_config(RUN_DIR)
print(f"run dir   : {RUN_DIR}")
print(f"config    : {wp.config_path(RUN_DIR).relative_to(wp.repo_root(PROJECT_ROOT))}   (BIGDATA_API_KEY: set)")

THEME = wp.load_theme(RUN_DIR, PROJECT_ROOT)
print(f"theme     : {THEME.id} — {THEME.title}")
print(f"classes   : {' / '.join(c.name for c in THEME.classes)}")

UNIVERSE_CSV = wp.universe_csv(CFG, RUN_DIR, PROJECT_ROOT)


def load_universe(path: Path) -> pd.DataFrame:
    """One CSV with columns id, name (search universe AND id -> company name)."""
    if not path.exists():
        raise FileNotFoundError(f"universe CSV not found: {path}")
    frame = pd.read_csv(path)
    missing = {"id", "name"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} must have columns id, name; missing {sorted(missing)}. "
                         f"Found: {list(frame.columns)}")
    frame = frame[["id", "name"]].copy()
    frame["id"] = frame["id"].astype("string").str.strip()
    frame["name"] = frame["name"].astype("string").str.strip()
    if frame.empty:
        raise ValueError(f"{path.name} is empty")
    bad_ids = int((frame["id"].isna() | (frame["id"] == "")).sum())
    if bad_ids:
        raise ValueError(f"{path.name}: {bad_ids} rows with an empty id")
    duplicates = int(frame["id"].duplicated().sum())
    if duplicates:
        raise ValueError(f"{path.name}: {duplicates} duplicate ids")
    empty_names = int((frame["name"].isna() | (frame["name"] == "")).sum())
    if empty_names:
        print(f"WARNING: {empty_names} rows have no name — their id will be used instead, and "
              f"the model will see a hex id where it expects a company.")
    return frame

universe = load_universe(UNIVERSE_CSV)
ENTITY_IDS = universe["id"].tolist()
ALLOWED_ENTITY_IDS = set(ENTITY_IDS)
print(f"universe  : {len(ENTITY_IDS):,} entities <- {UNIVERSE_CSV.name}")

SEARCH = dict(wp.section(CFG, "search"))
SEARCH.setdefault("start_date", "2016-01-01")
if not SEARCH.get("end_date"):
    SEARCH["end_date"] = datetime.now(UTC).date().isoformat()
    print(f"end_date not set in the config — using today: {SEARCH['end_date']}")
SEARCH.setdefault("combined_category", {"mode": "INCLUDE",
                                        "values": ["news_premium", "transcripts"]})
TRANSCRIPT_SOURCES = SEARCH.get("transcript_source_names") or []

print(f"window    : {SEARCH['start_date']} -> {SEARCH['end_date']}")
print(f"categories: {SEARCH['combined_category']['values']}")
print(f"baskets   : min_period_days={SEARCH.get('min_period_days', 90)}, "
      f"min_entities_per_basket={SEARCH.get('min_entities_per_basket', 10)}")
print(f"retrieval : chunk_percentage={SEARCH.get('chunk_percentage', 0.5)}")
print(f"rate      : requests_per_minute={SEARCH.get('requests_per_minute', 200)}, "
      f"max_workers={SEARCH.get('search_max_workers', 40)}")

SEEDS = wp.section(CFG, "seeds")
_statements = wp.section(SEEDS, "statements")
POSITIVE_STATEMENTS = list(_statements.get("positive") or [])
NEGATIVE_STATEMENTS = list(_statements.get("negative") or [])
if not POSITIVE_STATEMENTS and not NEGATIVE_STATEMENTS:
    raise SystemExit(f"seeds.statements has no positive and no negative statements in "
                     f"{wp.config_path(RUN_DIR)} — at least one polarity must be populated. "
                     f"They are what the search goes looking for, and they have to describe "
                     f"the same theme as `theme:` in that file.")

STATEMENTS = POSITIVE_STATEMENTS + NEGATIVE_STATEMENTS
if len(set(STATEMENTS)) != len(STATEMENTS):
    duplicated = sorted({s for s in STATEMENTS if STATEMENTS.count(s) > 1})
    raise ValueError(f"duplicate seed statements (they would share a checkpoint): {duplicated}")

POLARITY_BY_STATEMENT = {
    **{s: "positive" for s in POSITIVE_STATEMENTS},
    **{s: "negative" for s in NEGATIVE_STATEMENTS},
}
print(f"statements: {len(STATEMENTS)} ({len(POSITIVE_STATEMENTS)} positive, "
      f"{len(NEGATIVE_STATEMENTS)} negative)")

plan_items = sb.plan_all(RUN_DIR, STATEMENTS, SEARCH, ENTITY_IDS)

if not args.yes:
    input(f">>> about to SEARCH {len(STATEMENTS)} statements over "
          f"{SEARCH['start_date']}..{SEARCH['end_date']} — this is the metered step and takes "
          f"hours. Press Enter to go ahead, or Ctrl-C to stop. ")

sb.search_all(RUN_DIR, plan_items, SEARCH)

outputs = sb.build_exploded(RUN_DIR, plan_items, ALLOWED_ENTITY_IDS, POLARITY_BY_STATEMENT,
                            TRANSCRIPT_SOURCES)

for content_type, path in outputs.items():
    marker = wp.read_done(path)
    print(f"[{content_type}] {marker['rows']:,} rows, {path.stat().st_size / 2**30:.2f} GB "
          f"-> {path.name}")

if args.diagnostics:
    triple = ["doc_id", "chunk_index", "entity_id"]
    for content_type, path in outputs.items():
        frame = pd.read_parquet(path, columns=triple)
        unique = frame.drop_duplicates(subset=triple).shape[0]
        print(f"\n[{content_type}] {len(frame):,} rows -> {unique:,} unique "
              f"(doc_id, chunk_index, entity_id); the difference is chunks retrieved by more "
              f"than one statement, which step 2 collapses into one row each")
        per_chunk = frame.groupby(["doc_id", "chunk_index"])["entity_id"].nunique()
        distribution = per_chunk.value_counts().sort_index()
        table = pd.DataFrame({"n_chunks": distribution,
                              "pct": (distribution / len(per_chunk) * 100).round(1)})
        table.index.name = "companies_per_chunk"
        print(f"companies per chunk: mean {per_chunk.mean():.2f}, "
              f"median {per_chunk.median():.0f}, max {int(per_chunk.max())}")
        print(table.head(10).to_string())
        del frame, per_chunk
