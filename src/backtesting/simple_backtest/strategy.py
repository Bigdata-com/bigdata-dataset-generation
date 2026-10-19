"""Strategy configuration and weight computation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd

SelectionSide = Literal["long", "short", "long_short"]


@dataclass
class StrategyConfig:
    """Configuration for the backtesting strategy."""

    leg_size: int | None = None
    max_position: float = 0.05
    long_weight: float = 0.5
    short_weight: float = 0.5
    equal_weight: bool = False
    selection_side: SelectionSide = "long_short"

    def __post_init__(self) -> None:
        if self.long_weight + self.short_weight != 1.0:
            raise ValueError(
                f"long_weight + short_weight must equal 1.0, "
                f"got {self.long_weight} + {self.short_weight} = "
                f"{self.long_weight + self.short_weight}"
            )
        if self.equal_weight and self.leg_size is None:
            raise ValueError("equal_weight=True requires leg_size to be set")
        if self.max_position <= 0 or self.max_position > 1:
            raise ValueError(f"max_position must be in (0, 1], got {self.max_position}")


def prepare_merged_for_weights(
    merged: pd.DataFrame,
    config: StrategyConfig,
) -> tuple[pd.DataFrame, StrategyConfig]:
    """Apply one-sided post-merge leg selection before ``compute_weights``."""
    if config.leg_size is None or config.selection_side == "long_short":
        return merged, config

    filtered = _select_one_side_signals(
        merged,
        leg_size=config.leg_size,
        side=config.selection_side,
    )
    weight_config = replace(config, leg_size=None)
    return filtered, weight_config


def _select_one_side_signals(
    df: pd.DataFrame,
    *,
    leg_size: int,
    side: Literal["long", "short"],
) -> pd.DataFrame:
    """Keep top ``leg_size`` (long) or bottom ``leg_size`` (short) signals per date."""
    out = df.copy()
    ranks = out.groupby("date")["signal"].rank(method="average")
    max_rank = out.groupby("date")["signal"].transform("count")
    if side == "long":
        keep = ranks > (max_rank - leg_size)
    else:
        keep = ranks <= leg_size
    out.loc[~keep, "signal"] = np.nan
    return out


def compute_weights(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Convert signals to portfolio weights."""
    result = df.copy()
    result["weight"] = result["signal"]

    if config.leg_size is not None:
        result = _select_top_bottom(result, config.leg_size)

    result["weight"] = result["weight"].fillna(0)
    result = _rescale_weights(result, config)
    result = apply_position_cap(result, config.max_position)

    return result


def _select_top_bottom(df: pd.DataFrame, leg_size: int) -> pd.DataFrame:
    """Select top and bottom positions based on signal ranking."""
    result = df[df["weight"].notna()].copy()
    result["daily_rank"] = result.groupby("date")["weight"].rank(method="average")
    result["daily_max"] = result.groupby("date")["daily_rank"].transform("max")

    middle_mask = (
        (result["daily_max"] > 2 * leg_size)
        & (result["daily_rank"] > leg_size)
        & (result["daily_rank"] < result["daily_max"] - leg_size + 1)
    )
    result.loc[middle_mask, "weight"] = np.nan

    return result


def _rescale_weights(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Rescale weights to match target long/short exposures."""
    result = df.copy()

    if config.equal_weight:
        if "daily_rank" not in result.columns or "daily_max" not in result.columns:
            raise ValueError("equal_weight requires daily_rank from select_top_bottom")

        leg_size = config.leg_size
        short_mask = result["daily_rank"] <= leg_size
        long_mask = result["daily_rank"] > (result["daily_max"] - leg_size)

        result.loc[short_mask, "weight"] = -1
        result.loc[long_mask, "weight"] = 1
        result.loc[~short_mask & ~long_mask, "weight"] = 0

        short_sum = result[result["weight"] < 0].groupby("date")["weight"].transform("sum")
        result.loc[result["weight"] < 0, "weight"] = (
            result.loc[result["weight"] < 0, "weight"] / short_sum.abs() * config.short_weight
        )

        long_sum = result[result["weight"] > 0].groupby("date")["weight"].transform("sum")
        result.loc[result["weight"] > 0, "weight"] = (
            result.loc[result["weight"] > 0, "weight"] / long_sum * config.long_weight
        )
    else:
        short_mask = result["weight"] < 0
        long_mask = result["weight"] > 0

        if short_mask.any():
            short_sum = result[short_mask].groupby("date")["weight"].transform("sum")
            result.loc[short_mask, "weight"] = (
                result.loc[short_mask, "weight"] / short_sum.abs() * config.short_weight
            )

        if long_mask.any():
            long_sum = result[long_mask].groupby("date")["weight"].transform("sum")
            result.loc[long_mask, "weight"] = (
                result.loc[long_mask, "weight"] / long_sum * config.long_weight
            )

    return result


def apply_position_cap(df: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Apply position size constraints and redistribute excess."""
    result = df.copy()

    if result["weight"].isna().all() or result["weight"].abs().max() <= cap:
        return result

    result = _cap_positive_weights(result, cap)
    result = _cap_negative_weights(result, cap)

    return result


def _cap_positive_weights(df: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Cap positive weights and redistribute excess."""
    if not df["weight"].gt(cap).any():
        return df

    dates_to_adjust = df.loc[df["weight"] > cap, "date"].unique()
    mask = df["date"].isin(dates_to_adjust) & (df["weight"] > 0)
    pos_adj = df[mask].copy()

    pos_adj["original_exposure"] = pos_adj.groupby("date")["weight"].transform("sum")

    max_iterations = len(pos_adj)
    iteration = 0

    while pos_adj["weight"].gt(cap).any() and iteration < max_iterations:
        pos_adj.loc[pos_adj["weight"] > cap, "weight"] = cap

        current_exposure = pos_adj.groupby("date")["weight"].transform("sum")
        uncapped_exposure = (
            pos_adj[pos_adj["weight"] < cap].groupby("date")["weight"].transform("sum")
        )

        excess = pos_adj["original_exposure"] - current_exposure
        adjustment_factor = excess / uncapped_exposure + 1

        uncapped_mask = pos_adj["weight"] < cap
        pos_adj.loc[uncapped_mask, "weight"] = (
            pos_adj.loc[uncapped_mask, "weight"] * adjustment_factor.loc[uncapped_mask]
        )

        iteration += 1

    df.loc[pos_adj.index, "weight"] = pos_adj["weight"]
    return df


def _cap_negative_weights(df: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Cap negative weights and redistribute excess."""
    if not df["weight"].lt(-cap).any():
        return df

    dates_to_adjust = df.loc[df["weight"] < -cap, "date"].unique()
    mask = df["date"].isin(dates_to_adjust) & (df["weight"] < 0)
    neg_adj = df[mask].copy()

    neg_adj["original_exposure"] = neg_adj.groupby("date")["weight"].transform("sum")

    max_iterations = len(neg_adj)
    iteration = 0

    while neg_adj["weight"].lt(-cap).any() and iteration < max_iterations:
        neg_adj.loc[neg_adj["weight"] < -cap, "weight"] = -cap

        current_exposure = neg_adj.groupby("date")["weight"].transform("sum")
        uncapped_exposure = (
            neg_adj[neg_adj["weight"] > -cap].groupby("date")["weight"].transform("sum")
        )

        excess = neg_adj["original_exposure"] - current_exposure
        adjustment_factor = excess / uncapped_exposure + 1

        uncapped_mask = neg_adj["weight"] > -cap
        neg_adj.loc[uncapped_mask, "weight"] = (
            neg_adj.loc[uncapped_mask, "weight"] * adjustment_factor.loc[uncapped_mask]
        )

        iteration += 1

    df.loc[neg_adj.index, "weight"] = neg_adj["weight"]
    return df
