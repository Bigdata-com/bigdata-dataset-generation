"""Stage 14 -- the backtest, and it is a stage of the run rather than a notebook elsewhere."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from .. import frames, signals as signals_mod
from ..config import REFERENCE_COLUMN, DistillConfig
from ..manifest import Manifest
from ..paths import RunPaths
from ..theme import ThemeSpec


def discover_label_sets(config: DistillConfig, paths: RunPaths) -> dict[str, Path]:
    """Every corpus label set under the run, keyed by `<granularity>.<topology>`."""
    root = paths.step_dir("corpus")
    if not root.exists():
        return {}
    found: dict[str, Path] = {}
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        candidate = directory / "corpus.parquet"
        if candidate.exists():
            found[directory.name] = candidate
    return found


def _enrichment(config: DistillConfig, paths: RunPaths) -> pd.DataFrame:
    """The columns a signal needs and a label set does not carry."""
    subset_path = frames.require_existing(paths.subset_parquet, "the subset", "distill subset")
    columns = ["row_key", config.input.content_type_column, "entity_id",
               config.input.date_column, config.input.timestamp_column,
               config.input.relevance_column, config.input.sentiment_column]
    if config.input.reference_label_column:
        columns.append(REFERENCE_COLUMN)
    available = set(pq.ParquetFile(subset_path).schema_arrow.names)
    columns = [c for c in dict.fromkeys(columns) if c in available]
    return pd.read_parquet(subset_path, columns=columns)


def _load_book_frame(path: Path, enrich: pd.DataFrame, theme: ThemeSpec,
                     config: DistillConfig) -> pd.DataFrame:
    label_field = theme.label_field
    wanted = ["row_key", label_field, "ok"]
    available = set(pq.ParquetFile(path).schema_arrow.names)
    labels = pd.read_parquet(path, columns=[c for c in wanted if c in available])
    labels = labels.loc[labels.get("ok", True) if "ok" in labels.columns else slice(None)]
    merged = labels.merge(enrich, on="row_key", how="inner", validate="one_to_one")
    return merged


def _metrics_row(metrics: Any) -> dict[str, Any]:
    return {
        "annualized_return": round(float(metrics.annualized_returns), 6),
        "volatility": round(float(metrics.volatility), 6),
        "information_ratio": round(float(metrics.information_ratio), 4),
        # What share of the target capital is actually invested. A book at 44% is not cautious,
        # it is half in cash by accident -- and too thin to fill both legs, so it skews long.
        # It is the number a holding-period decision is made on, so it belongs in the table.
        "gross_exposure": round(float(metrics.gross_exposure), 4),
        "net_exposure": round(float(metrics.net_exposure), 4),
        "total_turnover": round(float(metrics.total_turnover), 6),
        "portfolio_size": round(float(metrics.portfolio_size), 2),
        "effective_holding_period": round(float(metrics.effective_holding_period), 3),
        "date_min": str(metrics.date_min),
        "date_max": str(metrics.date_max),
    }


class BacktestRunner:
    """Wraps `simple_backtest`, and says clearly when it cannot run."""

    def __init__(self, config: DistillConfig) -> None:
        self.config = config
        self.available = False
        self.reason = ""
        self._run = None
        self._metrics = None
        self._cache_dir = config.repo_path(config.backtest.price_cache)
        self._strategy = None
        self._import()

    def _import(self) -> None:
        candidates = [self.config.repo_path(self.config.backtest.library_path)]
        candidates.append(Path(__file__).resolve().parents[3] / "backtesting")
        for library in candidates:
            if library and library.exists() and str(library) not in sys.path:
                sys.path.insert(0, str(library))
        try:
            from simple_backtest import (  # type: ignore
                PriceCacheIncomplete, StrategyConfig, compute_metrics, run_backtest_from_signals,
            )
        except ImportError as exc:
            self.reason = (f"simple_backtest is not importable ({exc}). Set "
                           f"backtest.library_path at the vendored package, or pip install it. "
                           f"Signals and label mixes are still written.")
            return
        self._run = run_backtest_from_signals
        self._metrics = compute_metrics
        self._incomplete = PriceCacheIncomplete
        self._strategy = StrategyConfig(
            leg_size=self.config.backtest.leg_size,
            max_position=self.config.backtest.max_position,
            long_weight=self.config.backtest.long_weight,
            short_weight=self.config.backtest.short_weight,
        )
        self.available = True

    def run(self, signals: pd.DataFrame) -> pd.DataFrame | None:
        """Run one book, or skip cleanly and say why."""
        if not self.available or self._run is None:
            return None
        try:
            end = self.config.backtest.end
            last = pd.to_datetime(signals["date"]).max()
            if end and pd.Timestamp(str(end)) > last:
                end = last.date()   # the data stops before the window does
            return self._run(signals, universe_code=self.config.backtest.universe_code,
                             config=self._strategy, cache_dir=self._cache_dir,
                             end_date=str(end) if end else None)
        except Exception as exc:                      # noqa: BLE001 - any failure to price
            if isinstance(exc, self._incomplete):
                raise SystemExit(f"no backtest: {exc}") from exc
            self.available = False
            self.reason = (
                f"prices could not be obtained ({type(exc).__name__}: {exc}). Skipping the "
                f"books; every signal table and label mix is still written."
            )
            print(f"\nBACKTEST SKIPPED: {self.reason}\n")
            return None

    def metrics(self, performance: pd.DataFrame) -> dict[str, Any] | None:
        if self._metrics is None or performance is None or performance.empty:
            return None
        return _metrics_row(self._metrics(performance))


def run(config: DistillConfig, theme: ThemeSpec, paths: RunPaths, *,
        only: tuple[str, ...] = (), force: bool = False) -> dict[str, Any]:
    if not config.backtest.enabled:
        print("backtest.enabled is false -- nothing to do")
        return {"skipped": "disabled"}

    signs = signals_mod.require_signs(theme)
    cutoff = signals_mod.parse_cutoff(config.backtest.cutoff_ny)
    print(f"class signs from the theme: {signs}")
    print(f"trading-date rule: {config.backtest.cutoff_ny} America/New_York cut-off, then roll "
          f"forward to the next business day")
    if config.backtest.end:
        print(f"window ends {config.backtest.end} (backtest.end): signals after it are left out")

    label_sets = discover_label_sets(config, paths)
    if only:
        label_sets = {k: v for k, v in label_sets.items() if k in only}
    if not label_sets:
        raise SystemExit(
            f"no corpus label sets under {paths.step_dir('corpus')}. Run `distill corpus` first."
        )
    print(f"label sets found: {list(label_sets)}")

    manifest = Manifest(stage="backtest")
    enrich = _enrichment(config, paths)
    manifest.add_input("subset", paths.subset_parquet)

    runner = BacktestRunner(config)
    if not runner.available:
        print(f"\nBACKTEST SKIPPED: {runner.reason}\n")

    books: dict[str, dict[str, Any]] = {}
    label_mix: dict[str, dict[str, int]] = {}
    label_mix_totals: dict[str, int] = {}
    signal_classes = set(theme.signal_classes)
    floors: dict[str, pd.DataFrame] = {}

    for name, path in label_sets.items():
        print("\n" + "=" * 72)
        print(f"BOOK  {name}   from {path}")
        print("=" * 72)
        manifest.add_input(f"labels.{name}", path)
        frame = _load_book_frame(path, enrich, theme, config)
        label_mix[name] = frames.class_counts(frame, theme.label_field)
        label_mix_totals[name] = int(sum(v for k, v in label_mix[name].items()
                                         if k in signal_classes))
        print(f"{len(frame):,} labelled rows joined to their dates; label mix {label_mix[name]}")
        print(f"{label_mix_totals[name]:,} of them carry a signal class")
        signal = signals_mod.build_signals(
            frame, theme, signal_mode=config.backtest.signal_mode, cutoff=cutoff,
            date_column=config.input.date_column,
            timestamp_column=config.input.timestamp_column,
            relevance_column=config.input.relevance_column,
            sentiment_column=config.input.sentiment_column,
        )
        books[name] = _write_book(config, paths, name, signal, runner, manifest, force=force)
        for smoothed_name, smoothed in _smoothed(config, name, signal):
            books[smoothed_name] = _write_book(config, paths, smoothed_name, smoothed, runner,
                                               manifest, force=force)

        if config.backtest.sentiment_floor:
            floors[name] = signals_mod.build_signals(
                frame, theme, signal_mode="relevance_sentiment", cutoff=cutoff,
                date_column=config.input.date_column,
                timestamp_column=config.input.timestamp_column,
                relevance_column=config.input.relevance_column,
                sentiment_column=config.input.sentiment_column,
            )

    reference = REFERENCE_COLUMN if config.input.reference_label_column else None
    if reference and reference in enrich.columns:
        print("\n" + "=" * 72)
        print(f"BENCHMARK  {config.backtest.reference_name}  (the labels the corpus arrived with)")
        print("=" * 72)
        reference_frame = enrich.rename(columns={reference: theme.label_field})
        known = set(theme.class_names)
        present = set(reference_frame[theme.label_field].dropna().astype(str).str.lower())
        if not present & known:
            print(f"the reference column {reference!r} holds {sorted(present)[:6]}, none of which "
                  f"are this theme's classes {sorted(known)} -- not a comparable benchmark, "
                  f"skipped")
        else:
            reference_frame = reference_frame.loc[
                reference_frame[theme.label_field].astype(str).str.lower().isin(known)
            ]
            label_mix[config.backtest.reference_name] = frames.class_counts(
                reference_frame, theme.label_field)
            label_mix_totals[config.backtest.reference_name] = int(sum(
                v for k, v in label_mix[config.backtest.reference_name].items()
                if k in signal_classes))
            signal = signals_mod.build_signals(
                reference_frame, theme, signal_mode=config.backtest.signal_mode, cutoff=cutoff,
                date_column=config.input.date_column,
                timestamp_column=config.input.timestamp_column,
                relevance_column=config.input.relevance_column,
                sentiment_column=config.input.sentiment_column,
            )
            books[config.backtest.reference_name] = _write_book(
                config, paths, f"benchmarks/{config.backtest.reference_name}", signal, runner,
                manifest, force=force,
            )
            for smoothed_name, smoothed in _smoothed(config, config.backtest.reference_name,
                                                     signal):
                books[smoothed_name] = _write_book(
                    config, paths, f"benchmarks/{smoothed_name}", smoothed, runner, manifest,
                    force=force,
                )
    else:
        print(f"\nno reference label column ({reference!r}) in the subset -- the reference "
              f"benchmark is unavailable for this theme")

    if floors:
        print("\n" + "=" * 72)
        print("BENCHMARK  sentiment_only  (chunk_relevance x chunk_sentiment -- reads no label)")
        print("=" * 72)
        base = floors[next(iter(floors))]
        books["sentiment_only"] = _write_book(
            config, paths, "benchmarks/sentiment_only", base, runner, manifest, force=force,
        )

    rows = []
    for name, book in books.items():
        row: dict[str, Any] = {"book": name,
                               "labelled_rows": label_mix_totals.get(name),
                               "signal_rows": book["signal_rows"],
                               "nonzero_rows": book["nonzero_rows"],
                               "entities": book["entities"], **book.get("breadth", {})}
        row.update(book.get("metrics") or {})
        rows.append(row)
    comparison = pd.DataFrame(rows)
    paths.backtest_dir(create=True)
    comparison.to_csv(paths.backtest_comparison, index=False)
    print("\n" + "=" * 72)
    print("COMPARISON")
    print("=" * 72)
    print(comparison.to_string(index=False))
    print(f"\n-> {paths.backtest_comparison}")
    if not runner.available:
        print("(no IR columns: the books were skipped for want of prices, which is not a failure)")

    if config.backtest.chart and runner.available:
        chart = _plot(books, paths, config, theme)
        if chart:
            print(f"-> {chart}")

    manifest.config = {
        "half_life": config.backtest.half_life,
        "backtest": {k: v for k, v in config.backtest.__dict__.items()},
        "experiment": {"granularity": config.experiment.granularity,
                       "topology": config.experiment.topology},
    }
    manifest.fingerprints = {"theme": theme.fingerprint()}
    manifest.counts = {"label_mix": label_mix,
                       "signal_class_rows": label_mix_totals,
                       "books": {k: {"metrics": v.get("metrics"),
                                     "signal_rows": v.get("signal_rows"),
                                     "nonzero_rows": v.get("nonzero_rows"),
                                     "breadth": v.get("breadth")}
                                 for k, v in books.items()}}
    manifest.model = {"class_signs": signs, "signal_mode": config.backtest.signal_mode}
    manifest.add_output("comparison", paths.backtest_comparison)
    if not runner.available:
        manifest.note(f"books skipped: {runner.reason}")
    manifest.write(paths.manifest_for(paths.backtest_dir()))
    return {"books": {k: v.get("metrics") for k, v in books.items()},
            "label_mix": label_mix,
            "comparison": str(paths.backtest_comparison),
            "priced": runner.available}


def _smoothed(config: DistillConfig, name: str, signal: pd.DataFrame):
    """The same book with a half-life, BESIDE the raw one rather than instead of it.

    Replacing the raw book would hide the trade it makes. Smoothing buys gross exposure -- a
    raw 20%-coverage book ran at 44% of target on one measured run -- and it does not buy it
    for free: on that run the chunk book's information ratio fell while the sentence book's
    rose. Both rows have to be on the table for that to be visible.

    The same half-life goes on every labelled book. Choosing one per book to flatter each one
    is tuning after the fact, not a rule. The sentiment floor is the exception, and stays raw:
    it reads a sentiment score rather than a sparse label, so it already runs nearly fully
    deployed and smoothing only decays it. A smoothed floor would lower the bar.
    """
    half_life = config.backtest.half_life
    if not half_life:
        return
    yield f"{name}.h{half_life}", signals_mod.smooth_signal(signal, half_life)


def _write_book(config: DistillConfig, paths: RunPaths, name: str, signal: pd.DataFrame,
                runner: BacktestRunner, manifest: Manifest, *, force: bool) -> dict[str, Any]:
    directory = paths.backtest_book_dir(name, create=True)
    signals_path = directory / "signals.parquet"
    frames.write_parquet(signal, signals_path)
    breadth = signals_mod.signal_breadth(signal)
    entities = int(signal["entity_id"].nunique()) if not signal.empty else 0
    nonzero = int((signal["signal"] != 0).sum()) if not signal.empty else 0
    print(f"signal rows {len(signal):,} ({nonzero:,} non-zero)   entities {entities}   "
          f"breadth {breadth}")

    out: dict[str, Any] = {"signal_rows": len(signal), "nonzero_rows": nonzero,
                           "entities": entities, "breadth": breadth,
                           "signals": str(signals_path)}
    performance = runner.run(signal)
    if performance is not None and not performance.empty:
        performance_path = directory / "performance.parquet"
        frames.write_parquet(performance, performance_path)
        metrics = runner.metrics(performance)
        out["metrics"] = metrics
        out["performance"] = str(performance_path)
        (directory / "metrics.json").write_text(
            json.dumps({"book": name, "breadth": breadth, "signal_rows": len(signal),
                        "entities": entities, "metrics": metrics}, indent=2),
            encoding="utf-8",
        )
        print(f"IR {metrics['information_ratio']:.2f}   annualized "
              f"{100 * metrics['annualized_return']:.2f}%   vol "
              f"{100 * metrics['volatility']:.2f}%")
        manifest.add_output(f"performance.{name}", performance_path)
    manifest.add_output(f"signals.{name}", signals_path)
    return out


def _plot(books: dict[str, dict[str, Any]], paths: RunPaths, config: DistillConfig,
          theme: ThemeSpec) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib is not installed -- no chart")
        return None
    series = [(name, Path(book["performance"])) for name, book in books.items()
              if book.get("performance")]
    if not series:
        return None
    figure, plot = plt.subplots(figsize=(11, 5))
    for name, path in series:
        performance = pd.read_parquet(path).sort_values("date")
        plot.plot(pd.to_datetime(performance["date"]),
                  np.exp(performance["portfolio_return"].cumsum()), label=name)
    plot.set_title(f"{theme.title} long/short -- cumulative "
                   f"({config.backtest.signal_mode}, leg={config.backtest.leg_size})")
    plot.set_ylabel("growth of $1")
    plot.set_xlabel("date")
    plot.grid(alpha=0.3)
    plot.legend()
    figure.tight_layout()
    paths.backtest_dir(create=True)
    figure.savefig(paths.backtest_chart, dpi=140)
    plt.close(figure)
    return paths.backtest_chart
