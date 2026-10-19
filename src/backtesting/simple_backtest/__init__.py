"""Simple Backtesting Framework."""

from __future__ import annotations

from simple_backtest.backtest import daily_active_entities, run_backtest, run_backtest_from_signals
from simple_backtest.data import load_prices, load_signals, process_raw_prices
from simple_backtest.metrics import compute_metrics
from simple_backtest.price_cache import PriceCache, PriceCacheIncomplete
from simple_backtest.strategy import StrategyConfig

__version__ = "0.1.0"

__all__ = [
    "load_prices",
    "load_signals",
    "process_raw_prices",
    "PriceCache",
    "PriceCacheIncomplete",
    "daily_active_entities",
    "run_backtest",
    "run_backtest_from_signals",
    "compute_metrics",
    "StrategyConfig",
]
