"""Main backtesting engine."""

from __future__ import annotations

from datetime import date, timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from simple_backtest.price_cache import PriceCache
from simple_backtest.strategy import StrategyConfig, compute_weights, prepare_merged_for_weights


def run_backtest_from_signals(
    signals: pd.DataFrame,
    universe_code: str,
    config: StrategyConfig,
    *,
    price_cache: PriceCache | None = None,
    cache_dir: Path | str | None = None,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
) -> pd.DataFrame:
    """Fetch prices for signal dates and execute a strategy backtest."""
    required_columns: set[str] = {"date", "entity_id", "signal"}
    missing_columns: set[str] = required_columns - set(signals.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in signals data: {sorted(missing_columns)}. "
            f"Available columns: {list(signals.columns)}"
        )
    if signals.empty:
        raise ValueError("signals must contain at least one row")
    if price_cache is not None and cache_dir is not None:
        raise ValueError("Pass either price_cache or cache_dir, not both")

    signals_clean: pd.DataFrame = signals.copy()
    signals_clean["date"] = pd.to_datetime(signals_clean["date"]).dt.date

    signal_start_date: date = signals_clean["date"].min()
    signal_end_date: date = signals_clean["date"].max()
    performance_start_date: date = _coerce_backtest_date(start_date) or signal_start_date
    performance_end_date: date = _coerce_backtest_date(end_date) or signal_end_date
    if performance_start_date > performance_end_date:
        raise ValueError("start_date must be less than or equal to end_date")

    # Nothing past the end is used: the last day kept earns its own return, on the positions of
    # the day before, and what a position on the last day would earn falls on a day that is
    # cut. So prices are asked for up to the end and no further -- measured identical to
    # asking for ten more days.
    prices_start_date = performance_start_date - timedelta(days=10)
    prices_end_date = performance_end_date

    cache: PriceCache = price_cache or PriceCache(cache_dir=cache_dir)
    prices: pd.DataFrame = cache.get_prices(
        universe_code,
        prices_start_date,
        prices_end_date,
    )
    return run_backtest(
        prices=prices,
        signals=signals_clean,
        config=config,
        start_date=performance_start_date,
        end_date=performance_end_date,
    )


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    config: StrategyConfig,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
) -> pd.DataFrame:
    """Execute backtest and return daily performance."""
    merged = _merge_data(prices, signals)
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "portfolio_return",
                "turnover",
                "portfolio_size",
                "long_exposure",
                "short_exposure",
                "gross_exposure",
                "net_exposure",
            ]
        )

    merged_for_weights, weight_config = prepare_merged_for_weights(merged, config)
    weighted = compute_weights(merged_for_weights, weight_config)

    portfolio_returns = _calculate_portfolio_returns(weighted)
    turnover = _calculate_turnover(weighted)
    portfolio_size = _calculate_portfolio_size(weighted)
    exposures = _calculate_exposures(weighted)

    performance = portfolio_returns.merge(turnover, on="date", how="left")
    performance = performance.merge(portfolio_size, on="date", how="left")
    performance = performance.merge(exposures, on="date", how="left")

    performance = _filter_date_range(performance, start_date, end_date)

    return performance.sort_values("date").reset_index(drop=True)


def daily_active_entities(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    config: StrategyConfig,
) -> pd.DataFrame:
    """Return one row per (date, entity_id) with a non-zero portfolio weight."""
    merged = _merge_data(prices, signals)
    if merged.empty:
        return pd.DataFrame(columns=["date", "entity_id"])

    merged_for_weights, weight_config = prepare_merged_for_weights(merged, config)
    weighted = compute_weights(merged_for_weights, weight_config)
    active_mask = weighted["weight"].notna() & (weighted["weight"] != 0)
    return (
        weighted.loc[active_mask, ["date", "entity_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def _coerce_backtest_date(value: date | str | None) -> date | None:
    """Convert optional user-provided date values to ``datetime.date``."""
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


def _merge_data(prices: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Merge signal and price data; keep only tradeable (date, entity_id) pairs."""
    prices_clean = prices.copy()
    signals_clean = signals.copy()

    prices_clean["date"] = pd.to_datetime(prices_clean["date"]).dt.date
    signals_clean["date"] = pd.to_datetime(signals_clean["date"]).dt.date

    return signals_clean.merge(
        prices_clean,
        on=["date", "entity_id"],
        how="inner",
    )


def _calculate_portfolio_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily portfolio returns from weighted positions."""
    result = df.copy()

    result["weighted_return"] = result["weight"].fillna(0) * result["t1_return"].fillna(0)

    daily_returns = (
        result.groupby("date")["weighted_return"]
        .sum()
        .reset_index()
        .rename(columns={"weighted_return": "T1_RETURN"})
    )

    daily_returns['T0_RETURN'] = daily_returns['T1_RETURN'].shift(1).fillna(0)
    daily_returns["portfolio_return"] = np.log(1 + daily_returns["T0_RETURN"])

    return daily_returns[["date", "portfolio_return"]]


def _calculate_turnover(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily portfolio turnover."""
    data = df[["date", "entity_id", "weight", "t1_return"]].copy()

    unique_dates = sorted(data["date"].unique())
    unique_entities = data["entity_id"].unique()

    if len(data) != len(unique_dates) * len(unique_entities):
        full_grid = pd.DataFrame(
            product(unique_dates, unique_entities),
            columns=["date", "entity_id"],
        )
        data = full_grid.merge(data, on=["date", "entity_id"], how="left").fillna(0)

    data["next_day_weight"] = data["weight"] * (1 + data["t1_return"].fillna(0))

    for sign, comparison in [("positive", "gt"), ("negative", "lt")]:
        mask = getattr(data["weight"], comparison)(0)
        if mask.any():
            daily_sum = data[mask].groupby("date")["next_day_weight"].transform("sum")
            original_sum = data[mask].groupby("date")["weight"].transform("sum")

            if sign == "negative":
                daily_sum = daily_sum.abs()
                original_sum = original_sum.abs()

            scaling = original_sum / daily_sum.replace(0, np.nan)
            data.loc[mask, "next_day_weight"] = data.loc[mask, "next_day_weight"] * scaling

    data["prev_next_day_weight"] = data.groupby("entity_id")["next_day_weight"].shift(1)
    data["weight_change"] = np.abs(data["weight"] - data["prev_next_day_weight"])

    turnover = (
        data[data["weight_change"].notna()]
        .groupby("date")["weight_change"]
        .sum()
        .div(2)
        .reset_index()
        .rename(columns={"weight_change": "turnover"})
    )

    return turnover


def _calculate_portfolio_size(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily number of non-zero positions."""
    if df.empty:
        return pd.DataFrame(columns=["date", "portfolio_size"])

    portfolio_size = (
        df.groupby("date")
        .apply(lambda g: ((g["weight"].notna()) & (g["weight"] != 0)).sum())
        .reset_index()
        .rename(columns={0: "portfolio_size"})
    )
    return portfolio_size


def _calculate_exposures(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily long, short, gross, and net exposures from portfolio weights."""
    if df.empty:
        return pd.DataFrame(
            columns=["date", "long_exposure", "short_exposure", "gross_exposure", "net_exposure"]
        )

    data = df[["date", "weight"]].copy()
    data["weight"] = data["weight"].fillna(0.0)
    grouped = data.groupby("date")["weight"]
    out = pd.DataFrame(
        {
            "long_exposure": grouped.apply(lambda s: s[s > 0].sum()),
            "short_exposure": grouped.apply(lambda s: s[s < 0].sum()),
            "net_exposure": grouped.sum(),
            "gross_exposure": grouped.apply(lambda s: s.abs().sum()),
        }
    ).reset_index()
    return out[["date", "long_exposure", "short_exposure", "gross_exposure", "net_exposure"]]


def _filter_date_range(
    df: pd.DataFrame,
    start_date: date | str | None,
    end_date: date | str | None,
) -> pd.DataFrame:
    """Filter DataFrame to specified date range."""
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.date

    if start_date is not None:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        result = result[result["date"] >= start_date]

    if end_date is not None:
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()
        result = result[result["date"] <= end_date]

    return result
