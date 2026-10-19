"""The gate."""

from __future__ import annotations

from pathlib import Path

import yaml

import pytest

from conftest import make_corpus, theme_section
from distill.config import BacktestConfig, DistillConfig
from distill.paths import RunPaths
from distill.preflight import check, check_every_stage
from distill.theme import ThemeSpec

_THEMES = Path(__file__).resolve().parent.parent / "themes"
THEME_SECTION = theme_section(ThemeSpec.load(_THEMES / "pricing_power.yaml"))


def a_run(tmp_path: Path, *, corpus: bool = True, theme=THEME_SECTION,
          extraction_config: bool = True, **overrides):
    run = tmp_path / "output"
    run.mkdir(parents=True, exist_ok=True)
    (run.parent / "config").mkdir(parents=True, exist_ok=True)
    if corpus:
        make_corpus(60).to_parquet(run / "corpus.parquet", index=False)
    if extraction_config:
        (run.parent / "config" / "data_extraction.yaml").write_text(
            yaml.safe_dump({"theme": theme} if theme else {}), encoding="utf-8")
    data = {"input": {"source_parquet": "corpus.parquet"}, "teacher": {"health_attempts": 0}}
    data.update(overrides)
    config = DistillConfig.from_dict(data, base=run, theme=theme)
    config.backtest = BacktestConfig()
    return config, RunPaths.for_run(run)


def test_a_valid_config_passes(tmp_path):
    config, paths = a_run(tmp_path)
    flight, theme = check(config, paths, stage="subset")
    assert flight.passed, flight.report()
    assert theme is not None and theme.id == "pricing_power"


def test_a_scaffold_theme_blocks_everything(tmp_path):
    """The template ships a body saying FILL THIS IN."""
    scaffold = dict(THEME_SECTION)
    scaffold["body"] = "FILL THIS IN. Four things, in this order.\n" * 8
    config, paths = a_run(tmp_path, theme=scaffold)
    flight, _ = check(config, paths, stage="subset")
    assert not flight.passed
    assert any("scaffold" in e for e in flight.errors)


def test_a_missing_theme_section_blocks_everything(tmp_path):
    config, paths = a_run(tmp_path, theme={})
    flight, theme = check(config, paths, stage="subset")
    assert not flight.passed and theme is None
    assert any("no `theme:` section" in e for e in flight.errors)


def test_a_class_without_a_sign_blocks_everything(tmp_path):
    broken = dict(THEME_SECTION)
    broken["classes"] = [{"name": "winner"}, {"name": "loser", "sign": -1},
                         {"name": "irrelevant", "abstain": True}]
    config, paths = a_run(tmp_path, theme=broken)
    flight, _ = check(config, paths, stage="subset")
    assert any("no `sign`" in e for e in flight.errors)


def test_a_missing_input_file_blocks_the_first_stage(tmp_path):
    config, paths = a_run(tmp_path, corpus=False)
    flight, _ = check(config, paths, stage="subset")
    assert not flight.passed
    assert any("not found" in e or "input" in e for e in flight.errors)


def test_an_input_missing_a_column_blocks_before_anything_is_read(tmp_path):
    """Checked from the parquet SCHEMA."""
    import pandas as pd

    config, paths = a_run(tmp_path)
    frame = make_corpus(20).drop(columns=["chunk_sentiment"])
    frame.to_parquet(paths.run_dir / "corpus.parquet", index=False)
    flight, _ = check(config, paths, stage="subset")
    assert not flight.passed
    assert any("chunk_sentiment" in e for e in flight.errors)


def test_a_stage_is_only_blocked_by_what_it_needs(tmp_path):
    """A fresh run must not be blocked by artifacts it has not produced yet."""
    config, paths = a_run(tmp_path)
    assert check(config, paths, stage="subset")[0].passed
    assert not check(config, paths, stage="sample")[0].passed
    assert not check(config, paths, stage="corpus")[0].passed


def test_an_empty_endpoint_blocks_a_stage_that_calls_a_model(tmp_path):
    config, paths = a_run(tmp_path, teacher={"model": "", "health_attempts": 0})
    (paths.run_dir / "sample").mkdir(parents=True)
    (paths.run_dir / "sample" / "sample.parquet").write_bytes(b"x")
    flight, _ = check(config, paths, stage="teacher")
    assert any("teacher.model is empty" in e for e in flight.errors)
    flight, _ = check(config, paths, stage="teacher", dry_run=True)
    assert not any("teacher.model is empty" in e for e in flight.errors)


def test_a_missing_price_cache_is_a_warning_not_a_block(tmp_path):
    """The backtest skips its books cleanly and still writes every signal table."""
    import pandas as pd

    config, paths = a_run(tmp_path)
    (paths.run_dir / "subset").mkdir(parents=True)
    make_corpus(10).assign(row_key="k").to_parquet(
        paths.run_dir / "subset" / "subset.parquet", index=False)
    corpus_dir = paths.run_dir / "corpus_labels" / "chunk.combined"
    corpus_dir.mkdir(parents=True)
    pd.DataFrame({"row_key": ["k"], "classification": ["winner"], "ok": [True]}).to_parquet(
        corpus_dir / "corpus.parquet", index=False)
    flight, _ = check(config, paths, stage="backtest")
    assert flight.passed, flight.report()
    assert any("price cache" in w for w in flight.warnings)


def test_the_gate_raises_and_says_nothing_has_run(tmp_path):
    config, paths = a_run(tmp_path, corpus=False)
    flight, _ = check(config, paths, stage="subset")
    with pytest.raises(SystemExit, match="Nothing has run"):
        flight.require()


def test_every_stage_is_checked_at_once(tmp_path):
    config, paths = a_run(tmp_path)
    flights = check_every_stage(config, paths)
    assert flights["config"].passed
    assert flights["subset"].passed
    assert not flights["backtest"].passed
    assert set(flights) >= {"config", "subset", "sample", "teacher", "split", "export",
                            "evaluate", "corpus", "backtest", "train"}


def test_the_gate_checks_both_granularities(tmp_path):
    config, paths = a_run(tmp_path, experiment={"granularity": "both"})
    flight, _ = check(config, paths, stage="split")
    blocked = " ".join(flight.errors)
    assert "chunk teacher labels" in blocked and "sentence teacher labels" in blocked


def test_an_unfilled_search_is_a_warning_while_it_still_matters(tmp_path):
    """`validate` right after `init-run` should list what the SEARCH still needs too."""
    config, paths = a_run(tmp_path)
    flight, _ = check(config, paths, stage="subset")
    assert flight.passed, flight.report()
    assert any("universe_csv" in w for w in flight.warnings)
    assert any("seeds.statements" in w for w in flight.warnings)


def test_a_finished_search_is_not_nagged_about_its_seeds(tmp_path):
    """Once step 7 exists the retrieval settings are a record, not an input."""
    config, paths = a_run(tmp_path)
    paths.part_a_similarity.parent.mkdir(parents=True, exist_ok=True)
    paths.part_a_similarity.touch()
    flight, _ = check(config, paths, stage="subset")
    assert not any("seeds.statements" in w for w in flight.warnings)


def test_the_seeds_are_counted_when_they_are_there(tmp_path):
    config, paths = a_run(tmp_path)
    path = paths.extraction_config
    data = yaml.safe_load(path.read_text())
    data["universe_csv"] = "universe.csv"
    data["seeds"] = {"statements": {"positive": ["a", "b"], "negative": ["c"]}}
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    flight, _ = check(config, paths, stage="subset")
    assert any("3 seed statements" in item for item in flight.checked)
    assert [w for w in flight.warnings if "completion marker" not in w] == []


def test_a_truncated_input_is_caught_against_its_completion_marker(tmp_path):
    """A partial copy of a parquet still READS -- as a smaller dataset. The marker Part A
    wrote is the only thing that knows how big it was meant to be."""
    import json

    from distill.preflight import check

    config, paths = a_run(tmp_path)
    source = config.source_frame_path()
    real = source.stat().st_size
    (source.parent / (source.name + ".done.json")).write_text(
        json.dumps({"file": source.name, "rows": 999_999, "bytes": real * 12}),
        encoding="utf-8")
    flight, _ = check(config, paths, stage="subset")
    assert not flight.passed
    assert any("copy did not complete" in e for e in flight.errors)


def test_an_input_matching_its_marker_passes(tmp_path):
    import json

    from distill.preflight import check

    config, paths = a_run(tmp_path)
    source = config.source_frame_path()
    (source.parent / (source.name + ".done.json")).write_text(
        json.dumps({"file": source.name, "rows": 60, "bytes": source.stat().st_size}),
        encoding="utf-8")
    flight, _ = check(config, paths, stage="subset")
    assert flight.passed, flight.report()
    assert any("matches its completion marker" in item for item in flight.checked)


def test_no_marker_is_a_warning_not_a_block(tmp_path):
    from distill.preflight import check

    config, paths = a_run(tmp_path)
    flight, _ = check(config, paths, stage="subset")
    assert flight.passed
    assert any("no completion marker" in w for w in flight.warnings)


def test_a_multi_line_blocker_keeps_the_part_that_says_what_to_do():
    """A blocker may span lines: the first says what is wrong, the rest say what to do about
    it. `report()` must carry all of them -- reporting only the first threw that away."""
    from distill.preflight import Preflight

    flight = Preflight(stage="subset")
    flight.error("the input is not where the config says\n"
                 "  - looked in: <run>/similarity/\n"
                 "  - fix: point input.similarity_parquet at the real file")
    report = flight.report()
    assert "the input is not where the config says" in report
    assert "point input.similarity_parquet at the real file" in report
    assert report.count("\n") >= 2


def test_a_theme_that_is_not_the_one_the_seeds_retrieved_warns_at_every_gate(tmp_path):
    """Found on a real run: `run_config.yaml` said `theme: market_share` while all sixty seed
    statements were about operating margins. `init-run` printed a NOTE about it once, into a
    terminal that later carried eight hours of other output. It has to repeat."""
    from distill.preflight import _theme_named_by_retrieval, _check_theme_matches_the_seeds
    from distill.preflight import Preflight
    from distill.paths import RunPaths
    from distill.theme import ThemeSpec

    run = tmp_path / "Margin_2026-08-10"
    run.mkdir()
    (run / "run_config.yaml").write_text("run_name: Margin\ntheme: market_share\n")
    paths = RunPaths.for_run(run)
    named, source = _theme_named_by_retrieval(paths)
    assert (named, source) == ("market_share", "run_config.yaml")

    theme = ThemeSpec.from_dict({"id": "margin", "title": "margin trajectory",
                                 "classes": {"expander": 1, "compressor": -1,
                                             "irrelevant": None},
                                 "body": "x" * 200})
    flight = Preflight("config")
    _check_theme_matches_the_seeds(paths, theme, flight)
    assert any("market_share" in w and "margin" in w for w in flight.warnings), flight.warnings
    assert not flight.errors, "it is a warning: the new theme may be the correction"


def test_no_warning_when_the_retrieval_named_the_same_theme(tmp_path):
    """`market_share` and `MarketShare` are the same theme; a false alarm here trains you to
    ignore the real one."""
    from distill.preflight import _check_theme_matches_the_seeds, Preflight
    from distill.paths import RunPaths
    from distill.theme import ThemeSpec

    run = tmp_path / "MarketShare_2026-07-30"
    run.mkdir()
    (run / "run_config.yaml").write_text("theme: MarketShare\n")
    theme = ThemeSpec.from_dict({"id": "market_share", "title": "market share",
                                 "classes": {"gainer": 1, "loser": -1, "irrelevant": None},
                                 "body": "x" * 200})
    flight = Preflight("config")
    _check_theme_matches_the_seeds(RunPaths.for_run(run), theme, flight)
    assert flight.warnings == []
