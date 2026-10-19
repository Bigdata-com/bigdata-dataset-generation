"""Performance metrics calculation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from datetime import date

TRADING_DAYS_PER_YEAR = 252


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    annualized_returns: float
    volatility: float
    information_ratio: float
    portfolio_size: float
    long_exposure: float
    short_exposure: float
    gross_exposure: float
    net_exposure: float
    total_turnover: float
    effective_holding_period: float
    date_min: date
    date_max: date

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "annualized_returns": self.annualized_returns,
            "volatility": self.volatility,
            "information_ratio": self.information_ratio,
            "portfolio_size": self.portfolio_size,
            "long_exposure": self.long_exposure,
            "short_exposure": self.short_exposure,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "total_turnover": self.total_turnover,
            "effective_holding_period": self.effective_holding_period,
            "date_min": self.date_min,
            "date_max": self.date_max,
        }


def compute_metrics(performance: pd.DataFrame) -> PerformanceMetrics:
    """Compute performance metrics from backtest results."""
    df = performance.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date")

    mean_return = df["portfolio_return"].mean()
    std_return = df["portfolio_return"].std()

    annualized_returns = mean_return * TRADING_DAYS_PER_YEAR
    volatility = std_return * np.sqrt(TRADING_DAYS_PER_YEAR)

    if std_return > 0:
        information_ratio = np.sqrt(TRADING_DAYS_PER_YEAR) * mean_return / std_return
    else:
        information_ratio = 0.0

    total_turnover = df["turnover"].mean() if "turnover" in df.columns else 0.0
    portfolio_size = df["portfolio_size"].mean() if "portfolio_size" in df.columns else 0.0
    long_exposure = df["long_exposure"].mean() if "long_exposure" in df.columns else 0.0
    short_exposure = df["short_exposure"].mean() if "short_exposure" in df.columns else 0.0
    gross_exposure = df["gross_exposure"].mean() if "gross_exposure" in df.columns else 0.0
    net_exposure = df["net_exposure"].mean() if "net_exposure" in df.columns else 0.0
    effective_holding_period = 1 / total_turnover if total_turnover > 0 else float("inf")

    return PerformanceMetrics(
        annualized_returns=annualized_returns,
        volatility=volatility,
        information_ratio=information_ratio,
        portfolio_size=portfolio_size,
        long_exposure=long_exposure,
        short_exposure=short_exposure,
        gross_exposure=gross_exposure,
        net_exposure=net_exposure,
        total_turnover=total_turnover,
        effective_holding_period=effective_holding_period,
        date_min=df["date"].min(),
        date_max=df["date"].max(),
    )


def compute_rolling_metrics(
    performance: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """Compute rolling performance metrics."""
    df = performance.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date")

    df["rolling_return"] = (
        df["portfolio_return"].rolling(window=window).mean() * TRADING_DAYS_PER_YEAR
    )
    df["rolling_volatility"] = (
        df["portfolio_return"].rolling(window=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    )
    df["rolling_ir"] = df["rolling_return"] / df["rolling_volatility"]

    return df[["date", "rolling_return", "rolling_volatility", "rolling_ir"]].dropna()


def compute_cumulative_returns(performance: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative return series."""
    df = performance.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date")

    df["cumulative_return"] = df["portfolio_return"].cumsum()

    return df[["date", "cumulative_return"]]
