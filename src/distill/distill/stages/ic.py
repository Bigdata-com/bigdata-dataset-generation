"""Stage 14 -- is the signal predictive, before a portfolio is built on it.

The Information Coefficient is a rank correlation between the signal and the forward
market-neutral return, computed on signal rows. It touches no leg size and no position cap,
which is why it comes BEFORE the backtest: below roughly 10-20% coverage a fixed-leg book is
thin and one-sided by construction, so its information ratio measures portfolio construction
rather than signal quality. The IC is immune to that.

COVERAGE, AND WHY NOTHING IS WRITTEN PER BIN
--------------------------------------------
The curve's x axis is how much of the RETRIEVAL was labelled. The upstream implementation
materialised one run folder per bin -- config copied, subset symlinked, corpus sliced -- and
read them back. Nothing in that survives the read: the only thing that differs between two
bins is which `row_key` values are in. So the bins are taken here in memory, from the one
corpus pass that exists.

The slice is exact rather than approximate. A bin is cut with `subset.py`'s own rule: per
content type, every row at or above the `(1 - f)` quantile of `subset.metric` over the WHOLE
retrieval, ties included -- the row set a real `top_fraction: f` run would have labelled. That
quantile only reads the top of the distribution, which is in the subset, plus the retrieval's
size, which is in the manifest, so it is computed here without the source. The cut reads every
subset row, failed ones included; a row the student failed on drops out after it. Counting the
surviving rows instead moves each bin by however many rows failed below it. The bins are
nested, and the top one is the subset itself. It holds only because this cuts exactly as
`subset.py` cuts, so the two must not drift apart.

T comes from the subset's own manifest, not from `subset.top_fraction`: `subset.max_rows`
tightens the cut further, and a run that hit that ceiling covers less of the retrieval than
its config claims. The cut is per content type, so T is too.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from .. import frames, signals as signals_mod
from ..config import DistillConfig, search_end
from ..manifest import Manifest
from ..paths import RunPaths
from ..theme import ThemeSpec

CANONICAL_BINS: tuple[int, ...] = (1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
"""Retrieval-coverage bins, in percent. Fixed, so two runs are read on the same axis."""

HORIZONS: dict[str, int] = {"1d": 1, "1w": 5, "1m": 21}
"""Forward windows, in trading days."""

HALF_LIVES: tuple[int, ...] = (0, 1, 2, 7, 30, 90, 180)
"""Holding-period half-lives for the smoothing sweep. 0 is the raw signal."""


def coverage_of(paths: RunPaths, config: DistillConfig) -> dict[str, float]:
    """How much of the retrieval the subset actually kept, per content type.

    Read from the manifest because `subset.top_fraction` is what was ASKED for and
    `subset.max_rows` can have tightened it afterwards. `apply_max_rows` records the trim, and
    a run that hit the ceiling would otherwise be plotted at a coverage it never had.
    """
    manifest_path = paths.manifest_for(paths.subset_parquet)
    if not manifest_path.exists():
        raise SystemExit(
            f"no manifest beside {paths.subset_parquet.name}, so there is no record of how much "
            f"of the retrieval this subset kept. Re-run `distill subset`.")
    counts = json.loads(manifest_path.read_text(encoding="utf-8")).get("counts") or {}
    by_type = counts.get("by_content_type") or {}
    coverage: dict[str, float] = {}
    for content_type, entry in by_type.items():
        available = float(entry.get("available") or 0)
        kept = float(entry.get("kept") or 0)
        if available > 0 and kept > 0:
            coverage[str(content_type)] = kept / available
    if not coverage:
        raise SystemExit("the subset manifest records no per-content-type counts to read "
                         "coverage from")
    return coverage


def bins_for(coverage: dict[str, float], bins: tuple[int, ...] = CANONICAL_BINS) -> list[int]:
    """The canonical bins this run can reach: those at or under its thinnest coverage.

    A bin wider than what was labelled is not a smaller number, it is a number that does not
    exist -- those rows were never sent to the student.
    """
    ceiling = min(coverage.values()) * 100
    return [b for b in bins if b <= ceiling + 1e-9]


def retrieval_sizes(paths: RunPaths) -> dict[str, int]:
    """Rows of the retrieval the subset cut was taken over, per content type.

    `scored` (rows with a metric) when the manifest records it, else `available`: the
    quantile `subset.py` cuts at skips rows without one.
    """
    manifest = json.loads(paths.manifest_for(paths.subset_parquet).read_text(encoding="utf-8"))
    by_type = (manifest.get("counts") or {}).get("by_content_type") or {}
    return {str(ct): int(entry.get("scored", entry.get("available")) or 0)
            for ct, entry in by_type.items()}


def bin_thresholds(subset: pd.DataFrame, sizes: dict[str, int], config: DistillConfig,
                   bins: list[int]) -> dict[str, dict[int, float]]:
    """Per content type and bin, the metric a row needs to be in the bin.

    `subset.py` keeps the rows at or above the `(1 - f)` quantile over the whole retrieval.
    Below the subset sit `n - len(values)` rows the subset never kept; they are all lower, so
    standing in for them with the subset's minimum leaves every quantile up to the subset's own
    cut unchanged. `subset` is every subset row, failed ones included.
    """
    metric, content = config.subset.metric, config.input.content_type_column
    out: dict[str, dict[int, float]] = {}
    for content_type, part in subset.groupby(content):
        values = pd.to_numeric(part[metric], errors="coerce").dropna().to_numpy(dtype=float)
        n = sizes.get(str(content_type), len(values))
        if not len(values):
            out[str(content_type)] = {b: math.inf for b in bins}
            continue
        retrieval = pd.Series(np.concatenate([np.full(max(n - len(values), 0), values.min()),
                                              values]))
        out[str(content_type)] = {
            b: -math.inf if b >= 100 else float(retrieval.quantile(1.0 - b / 100))
            for b in bins}
    return out


def cut_bin(frame: pd.DataFrame, thresholds: dict[str, dict[int, float]],
            config: DistillConfig, bin_pct: int) -> pd.DataFrame:
    """The rows of `frame` a `top_fraction: bin_pct/100` run would have labelled."""
    floor = frame[config.input.content_type_column].astype(str).map(
        {ct: by_bin[bin_pct] for ct, by_bin in thresholds.items()})
    # at 100% `subset.py` keeps the whole group, rows without a metric included
    whole = floor == -math.inf
    return frame.loc[whole | (pd.to_numeric(frame[config.subset.metric], errors="coerce") >= floor)]


def forward_returns(prices: pd.DataFrame,
                    horizons: dict[str, int] = HORIZONS) -> pd.DataFrame:
    """Cumulative forward return per horizon, demeaned by date.

    `return` here is what `process_raw_prices` hands back, NOT what the cache file holds: it
    has already been divided by 100 and already had its cross-sectional mean subtracted. So no
    second `/100` -- that was a bug, and it shrank every return by two orders of magnitude.

    The demeaning below is therefore close to a no-op on a full cross-section, and it is kept
    because the grid is not full: entities enter and leave, and the mean of the rows actually
    present on a date is not the mean the provider subtracted.
    """
    prices = prices.sort_values(["entity_id", "date"]).reset_index(drop=True)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["cum"] = prices.groupby("entity_id")["return"].transform(
        lambda r: np.log1p(pd.to_numeric(r, errors="coerce")).cumsum())
    for tag, days in horizons.items():
        forward = prices.groupby("entity_id")["cum"].shift(-days) - prices["cum"]
        prices[f"fwd_{tag}"] = forward - forward.groupby(prices["date"]).transform("mean")
    return prices


def pooled_ic(signal: pd.DataFrame, prices: pd.DataFrame, tag: str) -> dict[str, Any]:
    """Spearman between the signal and the forward return, pooled over every (date, entity).

    Spearman is Pearson on the ranks, and `rank()` averages ties -- which is what the
    definition says and what `scipy.stats.spearmanr` does. Computing it that way rather than
    through `corr(method="spearman")` is not a flourish: that path imports scipy, and this
    package does not depend on it. Ties are the normal case here, not an edge case: a
    `class_sign` signal takes three values over millions of rows.

    The p-value uses the normal rather than the t. They differ in the fourth decimal past a
    few hundred rows and n here is in the tens of thousands.
    """
    column = f"fwd_{tag}"
    frame = signal.merge(prices[["date", "entity_id", column]],
                         on=["date", "entity_id"], how="inner").dropna(subset=["signal", column])
    n = len(frame)
    if n < 10 or frame["signal"].nunique() < 2:
        return {"n": n, "ic": float("nan"), "t_stat": float("nan"), "p_value": float("nan")}
    ic = float(frame["signal"].rank().corr(frame[column].rank()))
    if not math.isfinite(ic) or abs(ic) >= 1:
        return {"n": n, "ic": ic, "t_stat": float("nan"), "p_value": float("nan")}
    t = ic * math.sqrt((n - 2) / (1 - ic ** 2))
    p = math.erfc(abs(t) / math.sqrt(2))
    return {"n": n, "ic": round(ic, 4), "t_stat": round(t, 2), "p_value": round(p, 6)}


def _backtest_library(config: DistillConfig) -> None:
    """Put the vendored `simple_backtest` on the path, the way the backtest stage does."""
    import sys
    from pathlib import Path

    candidates = [config.repo_path(config.backtest.library_path),
                  Path(__file__).resolve().parents[3] / "backtesting"]
    for library in candidates:
        if library and library.exists() and str(library) not in sys.path:
            sys.path.insert(0, str(library))


def prices_for(config: DistillConfig, start: str, end: str) -> pd.DataFrame:
    """Raw daily returns over the window the signals cover."""
    _backtest_library(config)
    from simple_backtest import PriceCache  # type: ignore

    cache_dir = config.repo_path(config.backtest.price_cache)
    cache = PriceCache(cache_dir=str(cache_dir) if cache_dir else None)
    return cache.get_prices(config.backtest.universe_code, start, end)


def signal_frame(config: DistillConfig, theme: ThemeSpec, paths: RunPaths,
                 granularity: str) -> pd.DataFrame:
    """Corpus labels joined to the columns a signal needs, ready to be ranked and sliced."""
    from .backtest import _enrichment

    topology = config.experiment.topology
    corpus = frames.require_existing(paths.corpus_parquet(granularity, topology),
                                     f"the {granularity} corpus labels", "distill corpus")
    enrich = _enrichment(config, paths)
    metric = config.subset.metric
    if metric not in enrich.columns:
        extra = pd.read_parquet(paths.subset_parquet, columns=["row_key", metric])
        enrich = enrich.merge(extra, on="row_key", how="left")
    labels = pd.read_parquet(corpus, columns=["row_key", theme.label_field, "ok"])
    labels = labels.loc[labels["ok"]]
    return labels.merge(enrich, on="row_key", how="inner", validate="one_to_one")


def run(config: DistillConfig, theme: ThemeSpec, paths: RunPaths, *,
        granularities: tuple[str, ...] | None = None, force: bool = False,
        smooth_bins: int = 3) -> dict[str, Any]:
    """The IC across every coverage bin this run can reach, and a smoothing sweep."""
    out_dir = paths.step_dir("ic")
    out_path = out_dir / "ic.parquet"
    if out_path.exists() and not force:
        existing = pd.read_parquet(out_path)
        print(f"ic already written: {out_path} ({len(existing):,} rows). Pass --force to redo.")
        return {"rows": len(existing), "path": str(out_path), "reused": True}

    covered = granularities or config.granularities()
    coverage = coverage_of(paths, config)
    bins = bins_for(coverage)
    thinnest = min(coverage.values()) * 100
    print("coverage of the retrieval, per content type: "
          + ", ".join(f"{k} {100 * v:.1f}%" for k, v in sorted(coverage.items())))
    print(f"bins this run can reach (<= {thinnest:.1f}%): {bins}")
    if not bins:
        raise SystemExit(
            f"the subset covers {thinnest:.2f}% of the retrieval, under the smallest bin "
            f"({CANONICAL_BINS[0]}%). There is no point on the curve to plot.")

    window = pd.read_parquet(paths.subset_parquet, columns=[config.input.date_column])
    window = pd.to_datetime(window[config.input.date_column], errors="coerce").dropna()
    # The longest horizon looks 21 trading days ahead, so a month of padding past the last
    # signal buys forward returns for the tail. A cache that stops at the last signal cannot
    # serve that, so the padding is an attempt, not a requirement: without it the final month
    # of signals simply has no forward return and drops out. A cache that does not cover the
    # signals themselves stops the stage, and says which dates it holds.
    _backtest_library(config)
    from simple_backtest import PriceCacheIncomplete  # type: ignore

    end = pd.Timestamp(str(config.backtest.end)) if config.backtest.end else None
    last = min(window.max(), end) if end is not None else window.max()
    if end is not None:
        print(f"window ends {end.date()} (backtest.end): signals after it are left out")
    start = str(window.min().date())
    # Forty calendar days is a margin over the month the longest horizon needs. Thirty-one is
    # that month without the margin -- 21 trading days rarely span more -- and is tried before
    # giving the tail up, because a window cut short of the search usually has it in the cache.
    padded = str((last + pd.Timedelta(days=40)).date())
    month = str((last + pd.Timedelta(days=31)).date())
    try:
        try:
            prices = forward_returns(prices_for(config, start, padded))
        except Exception:                         # noqa: BLE001 - the cache may stop earlier
            try:
                prices = forward_returns(prices_for(config, start, month))
                print(f"the price cache does not reach {padded}; asking to {month}, which "
                      f"still covers the one-month horizon")
            except Exception:                     # noqa: BLE001 - nor that far
                plain = str(last.date())
                print(f"the price cache does not reach {padded}; asking only to {plain}, so "
                      f"the last few weeks of signals have no forward return and are left out")
                prices = forward_returns(prices_for(config, start, plain))
    except PriceCacheIncomplete as exc:
        raise SystemExit(f"no IC: {exc}") from exc
    except Exception as exc:                      # noqa: BLE001 - any failure to price
        print(f"\nno prices ({type(exc).__name__}: {exc}).\n"
              f"The IC is a correlation against forward returns, so without them there is "
              f"nothing to compute -- skipping, and leaving every label set on disk untouched. "
              f"Point --price-cache at a cache covering "
              f"{window.min().date()}..{window.max().date()}.")
        return {"skipped": "no prices"}
    print(f"prices: {len(prices):,} rows, {prices['entity_id'].nunique():,} entities, "
          f"{prices['date'].min().date()} .. {prices['date'].max().date()}")

    # The sweep is a backtest: it looks no month ahead, so it runs to the search's end, as the
    # backtest stage does, not to the IC's earlier end -- as far as the prices asked for go.
    sweep_end = pd.Timestamp(search_end(paths) or window.max())
    priced_to = prices["date"].max()
    if sweep_end > priced_to:
        print(f"smoothing sweep runs to {priced_to.date()}, where the prices stop; the search "
              f"runs to {sweep_end.date()}")
        sweep_end = priced_to
    else:
        print(f"smoothing sweep runs to {sweep_end.date()} (search end): a backtest needs no "
              f"forward month")

    manifest = Manifest(stage="ic").add_input("subset", paths.subset_parquet)
    ic_rows: list[dict[str, Any]] = []
    smooth_rows: list[dict[str, Any]] = []
    chosen = bins if len(bins) <= smooth_bins else \
        [bins[0], bins[len(bins) // 2], bins[-1]][:smooth_bins]
    thresholds = bin_thresholds(
        pd.read_parquet(paths.subset_parquet,
                        columns=[config.input.content_type_column, config.subset.metric]),
        retrieval_sizes(paths), config, bins)

    for granularity in covered:
        frame = signal_frame(config, theme, paths, granularity)
        manifest.add_input(f"corpus.{granularity}",
                           paths.corpus_parquet(granularity, config.experiment.topology))
        for bin_pct in bins:
            rows = cut_bin(frame, thresholds, config, bin_pct)
            whole = signals_mod.build_signals(
                rows, theme, signal_mode=config.backtest.signal_mode)
            whole["date"] = pd.to_datetime(whole["date"])
            signal = whole
            if end is not None:
                signal = whole.loc[whole["date"] <= end].reset_index(drop=True)
            print(f"[{granularity} {bin_pct}%] {len(rows):,} labelled rows -> "
                  f"{len(signal):,} signal days")
            for tag in HORIZONS:
                ic_rows.append({"granularity": granularity, "coverage_pct": bin_pct,
                                "horizon": tag, **pooled_ic(signal, prices, tag)})
            if bin_pct in chosen:
                smooth_rows += _sweep(config, whole, prices, granularity, bin_pct,
                                      end=str(sweep_end.date()))

    ic_table = pd.DataFrame(ic_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames.write_parquet(ic_table, out_path, announce=False)
    print("\n" + ic_table.pivot_table(index=["granularity", "coverage_pct"],
                                      columns="horizon", values="ic").to_string())
    outputs = {"ic": str(out_path)}
    if smooth_rows:
        smoothing_path = out_dir / "smoothing.parquet"
        frames.write_parquet(pd.DataFrame(smooth_rows), smoothing_path, announce=False)
        outputs["smoothing"] = str(smoothing_path)
        manifest.add_output("smoothing", smoothing_path)

    manifest.config = {"bins": list(bins), "horizons": dict(HORIZONS),
                       "half_lives": list(HALF_LIVES), "smoothing_bins": list(chosen),
                       "coverage_per_content_type": coverage,
                       "signal_mode": config.backtest.signal_mode,
                       "universe_code": config.backtest.universe_code}
    manifest.fingerprints = {"theme": theme.fingerprint()}
    manifest.counts = {"rows": len(ic_table), "granularities": list(covered)}
    manifest.add_output("ic", out_path)
    manifest.note("every bin is a slice of the one corpus pass, ranked as `subset.py` ranks; "
                  "nothing was re-labelled and no per-bin run folder was written")
    manifest.write(paths.manifest_for(out_path))
    return {"rows": len(ic_table), **outputs}


def _sweep(config: DistillConfig, signal: pd.DataFrame, prices: pd.DataFrame,
           granularity: str, bin_pct: int, *, end: str | None) -> list[dict[str, Any]]:
    """One backtest per half-life, on this bin's signal, up to `end`. CPU, no model."""
    _backtest_library(config)
    try:
        from simple_backtest import (  # type: ignore
            StrategyConfig, compute_metrics, run_backtest,
        )
    except ImportError as exc:
        print(f"  smoothing skipped: simple_backtest is not importable ({exc})")
        return []
    strategy = StrategyConfig(leg_size=config.backtest.leg_size,
                              max_position=config.backtest.max_position,
                              long_weight=config.backtest.long_weight,
                              short_weight=config.backtest.short_weight)
    rows = []
    for half_life in HALF_LIVES:
        performance = run_backtest(
            prices, signals_mod.smooth_signal(signal, half_life), strategy, end_date=end)
        if performance.empty:
            continue
        metrics = compute_metrics(performance)
        rows.append({"granularity": granularity, "coverage_pct": bin_pct,
                     "half_life": half_life,
                     "ir": round(float(metrics.information_ratio), 4),
                     "gross_exposure": round(float(metrics.gross_exposure), 4),
                     "net_exposure": round(float(metrics.net_exposure), 4)})
    print(f"  [{granularity} {bin_pct}%] smoothing: {len(rows)} half-lives")
    return rows
