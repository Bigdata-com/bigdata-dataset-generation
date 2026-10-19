"""Metrics, and the thing the backtest silently got wrong."""

from __future__ import annotations

from datetime import date, time

import pandas as pd
import pytest

from distill import metrics as metrics_mod
from distill import signals as signals_mod

CLASSES = ("winner", "loser", "irrelevant")


def test_a_failed_parse_is_wrong_not_dropped():
    """A model that cannot be parsed cannot be used, and dropping the rows flatters it."""
    pairs = [("winner", "winner"), ("winner", None), ("loser", "loser"), ("irrelevant", None)]
    report = metrics_mod.score(pairs, CLASSES, abstain="irrelevant")
    assert report.rows == 4
    assert report.unparseable == 2
    assert report.accuracy == pytest.approx(0.5), "the two failures count against the model"
    assert report.per_class["winner"].recall == pytest.approx(0.5)
    assert report.confusion["winner"][metrics_mod.FAIL_TOKEN] == 1
    assert sum(report.confusion["irrelevant"].values()) == 1


def test_a_failure_is_never_promoted_to_the_abstain_class():
    report = metrics_mod.score([("irrelevant", None)], CLASSES, abstain="irrelevant")
    assert report.per_class["irrelevant"].predicted == 0
    assert report.accuracy == 0.0


def test_the_abstain_baseline_is_reported_next_to_the_accuracy():
    """Predicting the abstain class everywhere already scores ~61.5% on this corpus."""
    pairs = [("irrelevant", "irrelevant")] * 6 + [("winner", "irrelevant")] * 4
    report = metrics_mod.score(pairs, CLASSES, abstain="irrelevant")
    assert report.abstain_baseline_accuracy == pytest.approx(0.6)
    assert report.macro_f1 < 0.5, "accuracy 0.6 but macro-F1 is what shows it is useless"
    assert "baseline would score" in report.render(abstain="irrelevant")


def test_macro_f1_averages_only_over_classes_the_ruler_used():
    """A class with no gold rows has no measurable F1."""
    pairs = [("winner", "winner"), ("loser", "loser")]
    report = metrics_mod.score(pairs, CLASSES)
    assert report.macro_f1 == pytest.approx(1.0)


def test_the_calibration_block_shows_the_drift_that_chose_epoch_3():
    """0.003 macro-F1 apart, inside the noise."""
    pairs = ([("irrelevant", "irrelevant")] * 90 + [("winner", "irrelevant")] * 10)
    report = metrics_mod.score(pairs, CLASSES, abstain="irrelevant")
    assert report.per_class["winner"].support == 10
    assert report.per_class["winner"].predicted == 0
    text = report.render(abstain="irrelevant")
    assert "calibration" in text
    assert "-10" in text, "the drift against the teacher must be visible as a number"


def test_the_evaluation_carries_a_block_per_content_type():
    evaluation = metrics_mod.Evaluation(tag="t", classes=CLASSES, abstain="irrelevant")
    evaluation.reports["overall"] = metrics_mod.score([("winner", "winner")], CLASSES)
    evaluation.reports["news"] = metrics_mod.score([("winner", "winner")], CLASSES, title="NEWS")
    evaluation.reports["transcripts"] = metrics_mod.score([("loser", "irrelevant")], CLASSES,
                                                          title="TRANSCRIPTS")
    rendered = evaluation.render()
    assert "NEWS" in rendered.upper() and "TRANSCRIPTS" in rendered.upper()
    assert "overall" in evaluation.to_dict()["reports"]


def test_the_comparison_table_is_readable_across_checkpoints():
    evaluations = []
    for tag, correct in (("e3", 9), ("e5", 7)):
        pairs = [("winner", "winner")] * correct + [("winner", "irrelevant")] * (10 - correct)
        evaluation = metrics_mod.Evaluation(tag=tag, classes=CLASSES, abstain="irrelevant",
                                           rows=10)
        evaluation.reports["overall"] = metrics_mod.score(pairs, CLASSES, abstain="irrelevant")
        evaluations.append(evaluation)
    table = metrics_mod.comparison_table(evaluations)
    assert "e3" in table and "e5" in table
    assert "does not choose" in table, "the file must say the pipeline does not pick"
    assert "drift" in table


def test_the_cutoff_and_the_business_day_roll():
    """15:30 New York, then roll forward."""
    cutoff = time(15, 30)
    assert signals_mod.utc_timestamp_to_signal_trade_date(
        pd.Timestamp("2024-01-08 18:00:00", tz="UTC"), cutoff) == date(2024, 1, 8)
    assert signals_mod.utc_timestamp_to_signal_trade_date(
        pd.Timestamp("2024-01-08 21:00:00", tz="UTC"), cutoff) == date(2024, 1, 9)
    assert signals_mod.utc_timestamp_to_signal_trade_date(
        pd.Timestamp("2024-01-12 21:00:00", tz="UTC"), cutoff) == date(2024, 1, 15)
    assert signals_mod.utc_timestamp_to_signal_trade_date(None, cutoff) is None


def _frame(labels: list[str], theme_field: str = "classification") -> pd.DataFrame:
    return pd.DataFrame({
        theme_field: labels,
        "entity_id": [f"E{i % 2}" for i in range(len(labels))],
        "content_type": ["news"] * len(labels),
        "date": [pd.Timestamp("2024-01-08")] * len(labels),
        "timestamp": ["13:00:00"] * len(labels),
        "chunk_relevance": [1.0] * len(labels),
        "chunk_sentiment": [0.5] * len(labels),
    })


def test_the_sign_map_comes_from_the_theme(pricing_power):
    signal = signals_mod.build_signals(_frame(["winner", "winner", "loser"]), pricing_power)
    assert set(signal.columns) == {"date", "entity_id", "signal"}
    assert signal["signal"].sum() == pytest.approx(1.0)


def test_a_missing_sign_is_refused_before_a_signal_is_built(exposure_theme):
    import dataclasses
    broken = dataclasses.replace(
        exposure_theme,
        classes=tuple(dataclasses.replace(c, sign=None) if c.name == "exposed" else c
                      for c in exposure_theme.classes),
    )
    with pytest.raises(SystemExit, match="no `sign`"):
        signals_mod.require_signs(broken)


def test_a_presence_pair_with_a_zero_sign_produces_only_longs(exposure_theme):
    """`not_exposed` is worth 0 for this theme, and that is stated rather than inferred."""
    frame = _frame(["exposed", "not_exposed", "not_exposed"])
    signal = signals_mod.build_signals(frame, exposure_theme)
    assert signal["signal"].min() >= 0
    assert signal["signal"].sum() == pytest.approx(1.0)


def test_an_unknown_class_raises_instead_of_producing_an_empty_book(pricing_power):
    """THE bug."""
    frame = _frame(["exposed", "not_exposed"])
    with pytest.raises(SystemExit, match="classes the theme does not declare"):
        signals_mod.build_signals(frame, pricing_power)


def test_the_signal_carries_only_classes_with_a_non_zero_sign(pricing_power):
    """A class with sign 0 contributes nothing, so its rows must not reach the aggregation."""
    frame = _frame(["winner", "irrelevant", "irrelevant"])
    frame["entity_id"] = ["E0", "E1", "E2"]
    signal = signals_mod.build_signals(frame, pricing_power)
    assert len(signal) == 1, "only the winner's entity-day is a cell"
    assert signal["entity_id"].tolist() == ["E0"]
    assert signals_mod.active_signs(pricing_power) == {"winner": 1.0, "loser": -1.0}


def test_a_zero_sign_class_in_a_presence_pair_creates_no_cells(exposure_theme):
    frame = _frame(["not_exposed", "not_exposed"])
    frame["entity_id"] = ["E0", "E1"]
    assert signals_mod.build_signals(frame, exposure_theme).empty


def test_an_unknown_signal_mode_is_refused(pricing_power):
    with pytest.raises(SystemExit, match="unknown signal_mode"):
        signals_mod.build_signals(_frame(["winner"]), pricing_power, signal_mode="weighted")


def test_relevance_weighting_is_available(pricing_power):
    frame = _frame(["winner", "winner"])
    frame["chunk_relevance"] = [0.5, 1.5]
    frame["entity_id"] = ["E0", "E0"]
    signal = signals_mod.build_signals(frame, pricing_power,
                                       signal_mode="relevance_class_sign")
    assert signal["signal"].sum() == pytest.approx(2.0)


def test_the_sentiment_floor_reads_no_label(pricing_power):
    frame = _frame(["winner", "loser"])
    frame["chunk_sentiment"] = [1.0, -1.0]
    floor = signals_mod.build_signals(frame, pricing_power, signal_mode="relevance_sentiment")
    assert floor["signal"].sum() == pytest.approx(0.0)
    frame["classification"] = ["anything", "else"]
    again = signals_mod.build_signals(frame, pricing_power, signal_mode="relevance_sentiment")
    pd.testing.assert_frame_equal(floor, again)


def test_undateable_rows_are_dropped_from_the_signal_not_from_the_labels(pricing_power):
    frame = _frame(["winner", "winner"])
    frame.loc[0, "timestamp"] = "not a time"
    signal = signals_mod.build_signals(frame, pricing_power)
    assert signal["signal"].sum() == pytest.approx(1.0)


def _price_cache():
    """The vendored `simple_backtest.price_cache`, importable from the tests."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backtesting"))
    from simple_backtest import price_cache

    return price_cache


def test_the_cache_serves_the_window_its_meta_covers(tmp_path):
    """Two files, named `prices`, read as they are: the rows inside the window, sorted."""
    from conftest import write_price_cache

    price_cache = _price_cache()
    write_price_cache(tmp_path / "prices", ["E2", "E1"], "2026-01-01", "2026-03-31")
    cache = price_cache.PriceCache(cache_dir=tmp_path / "prices")

    raw = cache.get_raw_prices("US_ML_SRP2", "2026-02-02", "2026-02-06")

    assert sorted(p.name for p in (tmp_path / "prices").iterdir()) == \
        ["prices.meta.json", "prices.parquet"]
    assert raw["date"].min() == date(2026, 2, 2) and raw["date"].max() == date(2026, 2, 6)
    assert list(raw["entity_id"].iloc[:2]) == ["E1", "E2"], "sorted by date, then entity"


def test_the_cache_never_fills_a_gap_it_stops_and_says_what_it_holds(tmp_path):
    """Nothing fetches prices any more: a window past the meta is the user's to fill."""
    from conftest import write_price_cache

    price_cache = _price_cache()
    write_price_cache(tmp_path / "prices", ["E1"], "2026-01-01", "2026-03-31")
    cache = price_cache.PriceCache(cache_dir=tmp_path / "prices")

    with pytest.raises(price_cache.PriceCacheIncomplete, match="2026-01-01..2026-03-31"):
        cache.get_raw_prices("US_ML_SRP2", "2026-01-01", "2026-04-30")
    with pytest.raises(price_cache.PriceCacheIncomplete, match="holds nothing"):
        price_cache.PriceCache(cache_dir=tmp_path / "empty").get_raw_prices(
            "US_ML_SRP2", "2026-01-01", "2026-01-31")
    assert not (tmp_path / "empty").exists(), "a miss writes nothing"
