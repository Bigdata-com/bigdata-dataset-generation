"""The Information Coefficient stage: bins taken in memory, never materialised."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import theme_section
from distill.config import DistillConfig
from distill.paths import RunPaths
from distill import signals as signals_mod
from distill.stages import ic
from distill.theme import ThemeSpec

THEMES = Path(__file__).resolve().parent.parent / "themes"


def test_the_bins_a_run_can_reach_stop_at_what_it_labelled():
    """A bin wider than the corpus pass is not a smaller number, it is a missing row set."""
    assert ic.bins_for({"news": 0.20, "transcripts": 0.20}) == [1, 2, 5, 10, 20]
    assert ic.bins_for({"news": 1.0, "transcripts": 1.0}) == list(ic.CANONICAL_BINS)
    assert ic.bins_for({"news": 0.20, "transcripts": 0.11}) == [1, 2, 5, 10], \
        "the thinnest content type sets the ceiling -- the cut is per content type"
    assert ic.bins_for({"news": 0.005, "transcripts": 0.005}) == []


def test_coverage_comes_from_the_manifest_not_from_top_fraction(tmp_path):
    """`subset.max_rows` tightens the cut further, and the config would not show it."""
    run = tmp_path / "output"
    (run / "subset").mkdir(parents=True)
    pd.DataFrame({"row_key": ["a"]}).to_parquet(run / "subset" / "subset.parquet", index=False)
    (run / "subset" / "subset.parquet.manifest.json").write_text(json.dumps({
        "counts": {"by_content_type": {
            "news": {"available": 10_000, "kept": 1_000},          # 10%, not the 20% asked for
            "transcripts": {"available": 5_000, "kept": 600},      # 12%
        }}}))
    config = DistillConfig.from_dict({"subset": {"top_fraction": 0.20}}, base=run)
    coverage = ic.coverage_of(RunPaths.for_run(run), config)
    assert coverage == {"news": 0.10, "transcripts": 0.12}
    assert ic.bins_for(coverage) == [1, 2, 5, 10], "10% is the ceiling, not the configured 20%"


def test_a_bin_is_the_row_set_a_real_run_at_that_fraction_would_have_labelled():
    """Each bin is `subset.py`'s own cut over the whole retrieval, ties included, computed from
    the subset alone -- and the rows the student failed on do not move it."""
    config = DistillConfig.from_dict({"subset": {"metric": "sim_top"}}, base=Path("."))
    rng = np.random.default_rng(11)
    sizes = {"news": 1_913, "transcripts": 2_317}
    retrieval = pd.concat([pd.DataFrame({
        "row_key": [f"{ct}{i}" for i in range(n)], "content_type": ct,
        "sim_top": np.round(rng.uniform(0.0, 0.6, n), 2)})          # rounded: plenty of ties
        for ct, n in sizes.items()], ignore_index=True)

    def top(fraction):                    # stages/subset.py, per content type, on everything
        keep = []
        for _, group in retrieval.groupby("content_type"):
            keep.append(group.loc[group["sim_top"] >= group["sim_top"].quantile(1.0 - fraction)])
        return pd.concat(keep)

    subset = top(0.20)
    failed = set(subset["row_key"].sample(frac=0.05, random_state=1))
    labelled = subset.loc[~subset["row_key"].isin(failed)]     # what signal_frame hands over
    bins = [1, 2, 5, 10, 20]
    thresholds = ic.bin_thresholds(subset, sizes, config, bins)

    previous = set()
    for bin_pct in bins:
        got = set(ic.cut_bin(labelled, thresholds, config, bin_pct)["row_key"])
        expected = set(top(bin_pct / 100)["row_key"]) - failed
        assert got == expected, f"{bin_pct}%: not the row set a top_fraction run would label"
        assert previous <= got, "the bins are nested"
        previous = got
    assert previous == set(labelled["row_key"]), "the top bin is the whole subset"


def test_the_smoothing_sweep_runs_to_the_search_end_not_the_ic_end(tmp_path, monkeypatch):
    """The IC stops a month early for its forward returns; the sweep is a backtest and does
    not, so it runs to `search.end_date`, as the backtest stage does."""
    import yaml
    from conftest import write_price_cache

    run, config, theme = _a_labelled_run(tmp_path)
    paths = RunPaths.for_run(run)
    extraction = yaml.safe_load(paths.extraction_config.read_text())
    extraction.setdefault("search", {})["end_date"] = "2024-03-20"
    paths.extraction_config.write_text(yaml.safe_dump(extraction))
    cache = write_price_cache(tmp_path / "cache", ["A", "B", "C"], "2023-12-01", "2024-04-05")
    config.backtest.price_cache = str(cache)
    config.backtest.end = "2024-03-01"
    ends = []
    monkeypatch.setattr(ic, "_sweep", lambda *args, end, **kwargs: ends.append(end) or [])

    ic.run(config, theme, paths, granularities=("chunk",))

    assert ends and set(ends) == {"2024-03-20"}, ends


def test_the_ic_is_signed_the_way_the_signal_is():
    """A signal that leads the return has to come back positive, and the reverse negative.

    `fwd_1d` at date t is the return of t+1, so a signal on t predicting t+1 needs no shift.
    """
    dates = pd.bdate_range("2024-01-01", periods=400)
    entities = [f"E{i}" for i in range(12)]
    rng = np.random.default_rng(7)
    grid = pd.MultiIndex.from_product([dates, entities], names=["date", "entity_id"])
    signal = pd.DataFrame({"signal": rng.choice([-1.0, 1.0], size=len(grid))}, index=grid)

    # tomorrow's return carries today's signal, plus noise: about 1% a day, as a fraction, so
    # no day loses more than 100% and `log1p` stays defined
    tomorrow = signal.copy()
    tomorrow.index = tomorrow.index.set_levels(
        tomorrow.index.levels[0] + pd.Timedelta(days=1), level=0)
    prices = pd.DataFrame({"return": 0.01 * tomorrow["signal"]
                                     + rng.normal(0, 0.004, size=len(tomorrow))}).reset_index()

    forward = ic.forward_returns(prices, {"1d": 1})
    flat = signal.reset_index()
    result = ic.pooled_ic(flat, forward, "1d")
    assert result["ic"] > 0.1, result
    assert result["p_value"] < 0.01, result
    assert ic.pooled_ic(flat.assign(signal=-flat["signal"]), forward, "1d")["ic"] < -0.1


def test_smoothing_decays_a_one_day_label_over_the_half_life():
    """The decay runs forward over the entity's own days, which is what the book holds."""
    days = pd.to_datetime(["2024-01-01", "2024-01-10"])
    signal = pd.DataFrame({"date": days, "entity_id": ["E", "E"], "signal": [1.0, 0.0]})
    assert signals_mod.smooth_signal(signal, 0) is signal, "half-life 0 is the raw signal, untouched"
    decayed = signals_mod.smooth_signal(signal, 2).set_index("date")["signal"]
    assert decayed.loc["2024-01-01"] == pytest.approx(1.0)
    assert decayed.loc["2024-01-03"] == pytest.approx(0.5), "one half-life later, half of it"
    assert decayed.loc["2024-01-05"] == pytest.approx(0.25)


def _a_labelled_run(tmp_path) -> tuple[Path, DistillConfig, ThemeSpec]:
    """A subset, its manifest, and a corpus pass -- everything but the prices."""
    from conftest import install_config

    run = tmp_path / "output"
    run.mkdir(parents=True)
    install_config(run, theme="pricing_power")
    rng = np.random.default_rng(3)
    n = 200
    rows = pd.DataFrame({
        "row_key": [f"k{i:04d}" for i in range(n)],
        "doc_id": [f"d{i:04d}" for i in range(n)], "chunk_index": rng.integers(0, 4, n),
        "content_type": np.where(np.arange(n) % 2 == 0, "news", "transcripts"),
        "entity_id": rng.choice(["A", "B", "C"], n),
        "date": pd.to_datetime(rng.choice(pd.bdate_range("2024-01-02", periods=60), n)).date,
        "timestamp": "12:00:00", "chunk_relevance": rng.uniform(0.3, 1, n),
        "chunk_sentiment": rng.uniform(-1, 1, n), "sim_top": rng.uniform(0.1, 0.5, n)})
    (run / "subset").mkdir(parents=True, exist_ok=True)
    rows.to_parquet(run / "subset" / "subset.parquet", index=False)
    (run / "subset" / "subset.parquet.manifest.json").write_text(json.dumps({
        "counts": {"by_content_type": {"news": {"available": 500, "kept": 100},
                                       "transcripts": {"available": 500, "kept": 100}}}}))
    labels = run / "corpus_labels" / "chunk.combined"
    labels.mkdir(parents=True)
    pd.DataFrame({"row_key": rows["row_key"],
                  "classification": rng.choice(["winner", "loser", "irrelevant"], n),
                  "ok": True}).to_parquet(labels / "corpus.parquet", index=False)

    paths = RunPaths.for_run(run)
    config = DistillConfig.load_run(paths)
    return run, config, ThemeSpec.load(THEMES / "pricing_power.yaml")


def test_without_prices_the_stage_stops_and_says_which_dates_are_missing(tmp_path):
    """A cache that does not cover the signals is the user's to fix, so the stage stops.

    The error names the cache and the window it needed, and nothing is written.
    """
    run, config, theme = _a_labelled_run(tmp_path)
    config.backtest.price_cache = str(tmp_path / "no_such_cache")

    with pytest.raises(SystemExit) as stopped:
        ic.run(config, theme, RunPaths.for_run(run), granularities=("chunk",))
    message = str(stopped.value)

    assert "no_such_cache" in message and "2024-01-02" in message, message
    assert "holds nothing" in message, message
    assert not (run / "ic").exists(), "nothing is written when nothing could be computed"


def test_the_backtest_stops_on_a_cache_that_does_not_cover_the_window(tmp_path):
    from conftest import write_price_cache
    from distill.stages.backtest import BacktestRunner

    _, config, _ = _a_labelled_run(tmp_path)
    cache = write_price_cache(tmp_path / "short_cache", ["E000"], "2024-01-01", "2024-03-31")
    config.backtest.price_cache = str(cache)
    signals = pd.DataFrame({"date": pd.to_datetime(["2024-02-01", "2024-06-03"]),
                            "entity_id": ["E000", "E000"], "signal": [1.0, -1.0]})

    with pytest.raises(SystemExit, match="does not cover") as stopped:
        BacktestRunner(config).run(signals)
    assert "2024-01-01..2024-03-31" in str(stopped.value), "it says what the cache holds"


def test_backtest_end_runs_a_window_the_cache_covers_and_stops_there(tmp_path):
    """Signals run to June, prices to March: `backtest.end` in March makes it a backtest."""
    from conftest import write_price_cache
    from distill.stages.backtest import BacktestRunner

    _, config, _ = _a_labelled_run(tmp_path)
    ids = ["E000", "E001", "E002", "E003"]
    cache = write_price_cache(tmp_path / "short_cache", ids, "2023-12-01", "2024-03-31")
    config.backtest.price_cache = str(cache)
    config.backtest.end = "2024-03-15"
    dates = pd.bdate_range("2024-01-02", "2024-06-03")
    signals = pd.DataFrame({"date": np.repeat(dates, len(ids)), "entity_id": ids * len(dates),
                            "signal": [1.0, -1.0, 1.0, -1.0] * len(dates)})

    performance = BacktestRunner(config).run(signals)

    assert performance is not None and not performance.empty
    assert pd.to_datetime(performance["date"]).max() <= pd.Timestamp("2024-03-15")


def test_ic_end_drops_later_signals_and_keeps_the_month_after_from_the_cache(tmp_path, capsys):
    """Cut at 1 March: the cache holds a month past it but not forty days, and that month is
    what the one-month horizon needs, so it is asked for instead of giving the tail up."""
    from conftest import write_price_cache

    run, config, theme = _a_labelled_run(tmp_path)
    cache = write_price_cache(tmp_path / "cache", ["A", "B", "C"], "2023-12-01", "2024-04-05")
    config.backtest.price_cache = str(cache)
    config.backtest.end = "2024-03-01"

    ic.run(config, theme, RunPaths.for_run(run), granularities=("chunk",))
    out = capsys.readouterr().out

    assert "window ends 2024-03-01 (backtest.end)" in out, out
    assert "asking to 2024-04-01, which still covers the one-month horizon" in out, out
    assert (run / "ic" / "ic.parquet").exists()


def test_backtest_end_needs_prices_only_up_to_the_end(tmp_path):
    """A cache that stops on the end date itself is enough: nothing past it is used."""
    from conftest import write_price_cache
    from distill.stages.backtest import BacktestRunner

    _, config, _ = _a_labelled_run(tmp_path)
    ids = ["E000", "E001", "E002", "E003"]
    cache = write_price_cache(tmp_path / "cache", ids, "2023-12-01", "2024-03-15")
    config.backtest.price_cache = str(cache)
    config.backtest.end = "2024-03-15"
    dates = pd.bdate_range("2024-01-02", "2024-06-03")
    signals = pd.DataFrame({"date": np.repeat(dates, len(ids)), "entity_id": ids * len(dates),
                            "signal": [1.0, -1.0, 1.0, -1.0] * len(dates)})

    performance = BacktestRunner(config).run(signals)

    assert pd.to_datetime(performance["date"]).max() == pd.Timestamp("2024-03-15")
