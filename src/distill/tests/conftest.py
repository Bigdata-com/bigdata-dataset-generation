"""Fixtures: a synthetic corpus, and two themes that are shaped differently on purpose."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DISTILL_ROOT = Path(__file__).resolve().parent.parent
if str(DISTILL_ROOT) not in sys.path:
    sys.path.insert(0, str(DISTILL_ROOT))

from distill.theme import ThemeClass, ThemeSpec  # noqa: E402

THEMES_DIR = DISTILL_ROOT / "themes"


@pytest.fixture
def pricing_power() -> ThemeSpec:
    return ThemeSpec.load(THEMES_DIR / "pricing_power.yaml")


@pytest.fixture
def exposure_theme() -> ThemeSpec:
    """A presence pair whose negative side is worth 0."""
    return ThemeSpec(
        id="market_exposure",
        title="market exposure",
        classes=(
            ThemeClass(name="exposed", sign=1.0),
            ThemeClass(name="not_exposed", sign=0.0),
            ThemeClass(name="silent", abstain=True),
        ),
        body="""WHAT IS MARKET EXPOSURE?

Exposure is whether the company's results depend on the named market. Ask it plainly: would
that market moving change the company's numbers?

EXPOSED: the excerpt says the company is exposed. Revenue, supply or cost depends on the
named market.

NOT_EXPOSED: the excerpt asserts the company is NOT exposed — explicitly insulated, hedged,
or out of that market.

SILENT: the excerpt says nothing about exposure. This is not the same as NOT_EXPOSED, which
is a positive assertion of insulation; if the line between the two is left vague the rare
class disappears. When in doubt, classify as SILENT.""",
    )


@pytest.fixture
def gainer_theme() -> ThemeSpec:
    """A polar pair that does NOT use the name `winner`."""
    return ThemeSpec(
        id="share_shift",
        title="share shift",
        classes=(
            ThemeClass(name="gainer", sign=1.0),
            ThemeClass(name="loser", sign=-1.0),
            ThemeClass(name="irrelevant", abstain=True),
        ),
        body="""WHAT IS SHARE SHIFT?

Whether the company is taking share or losing it. Who is gaining, and at whose expense?

GAINER: the excerpt says the company is taking share from competitors.

LOSER: the excerpt says the company is losing share to competitors.

IRRELEVANT: nothing about share. Not index changes, buybacks, or generic growth talk. When
in doubt, classify as IRRELEVANT.""",
    )


def theme_section(theme: ThemeSpec) -> dict:
    """The theme as a config section."""
    return {
        "id": theme.id,
        "title": theme.title,
        "classes": [{"name": c.name, "sign": c.sign, "abstain": c.abstain}
                    for c in theme.classes],
        "body": theme.body,
    }


def make_corpus(n: int = 240, *, seed: int = 3, blank_rows: int = 0) -> pd.DataFrame:
    """A corpus with the shape Part A's step-7 artifact has."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        content_type = "news" if i % 3 == 0 else "transcripts"
        sentences = [f"sentence {i}-{j} about prices and margins." for j in range(1 + i % 3)]
        sims = sorted((0.2 + 0.75 * rng.random() for _ in sentences), reverse=(i % 4 == 0))
        rows.append({
            "doc_id": f"DOC{i:05d}",
            "chunk_index": i % 7,
            "entity_id": f"E{i % 23:03d}",
            "entity_name": f"Company {i % 23}",
            "content_type": content_type,
            "chunk_text": f"Chunk {i}: the company raised prices and customers accepted it.",
            "headline": f"Company {i % 23} reports: headline {i}",
            "sim_top": float(max(sims)),
            "chunk_relevance": float(rng.random()),
            "chunk_sentiment": float(rng.random() * 2 - 1),
            "date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=i % 400),
            "timestamp": "13:45:00" if i % 2 else "20:10:00",
            "classification": ["winner", "loser", "irrelevant"][i % 3],
            "extracted_sentences": sentences,
            "extracted_sentence_sims": [float(s) for s in sims],
            "extracted_best_search": [f"seed {j}" for j in range(len(sentences))],
        })
    frame = pd.DataFrame(rows)
    if blank_rows:
        frame.loc[: blank_rows - 1, "chunk_text"] = None
    return frame


@pytest.fixture
def corpus_parquet(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.parquet"
    make_corpus().to_parquet(path, index=False)
    return path


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "Theme_2026-08-24"
    directory.mkdir()
    return directory


def install_config(run_dir: Path, *, theme: str | None = "market_share") -> Path:
    """The three checked-in config files, BESIDE the run — the production layout.

    `config/` is part of the repository and is never copied into the output; a test lays the
    two out the same way so it reads its own copy and can never reach the real one.
    """
    import shutil

    from distill.paths import CONFIG_FILENAMES, config_dir as repo_config_dir

    dest = run_dir.parent / "config"
    dest.mkdir(parents=True, exist_ok=True)
    for name in CONFIG_FILENAMES:
        shutil.copy(repo_config_dir() / name, dest / name)
    if theme:
        from distill.template import write_extraction_template
        from distill.theme import ThemeSpec
        themes = Path(__file__).resolve().parents[1] / "themes"
        spec = ThemeSpec.load(themes / f"{theme}.yaml")
        write_extraction_template(dest / CONFIG_FILENAMES[0], theme=spec, run_dir=run_dir)
    return dest


def write_price_cache(directory: Path, entity_ids, start: str, end: str) -> Path:
    """A complete price cache on disk, the two files a user brings: one return per company
    per business day, and the metadata that says which dates they cover."""
    import json

    dates = pd.bdate_range(start, end)
    ids = sorted(set(entity_ids))
    rng = np.random.default_rng(0)
    prices = pd.DataFrame({
        "date": np.repeat(dates.date, len(ids)),
        "entity_id": ids * len(dates),
        "return": rng.normal(0, 1.5, len(dates) * len(ids)),
        "volume": 1_000,
    })
    directory.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(directory / "prices.parquet", index=False)
    (directory / "prices.meta.json").write_text(json.dumps(
        {"intervals": [{"start": start, "end": end}]}))
    return directory
