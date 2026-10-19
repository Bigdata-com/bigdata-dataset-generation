"""Data loading utilities for the backtesting framework."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from typing import Literal


def process_raw_prices(
    df: pd.DataFrame,
    *,
    return_column: str = "return",
    entity_column: str = "entity_id",
    date_column: str = "date",
    volume_column: str = "volume",
) -> pd.DataFrame:
    """Apply the standard price processing pipeline to raw price rows."""
    required_columns = {date_column, entity_column, return_column, volume_column}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in prices data: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column]).dt.date
    df.loc[df[volume_column].eq(0), return_column] = np.nan
    exclude_dates = (
        df[df[return_column].isna()].groupby(date_column)[entity_column].count()
        / df.groupby(date_column)[entity_column].count()
    ).reset_index()
    exclude_dates = exclude_dates[exclude_dates[entity_column].ge(0.75)].copy()
    df = (
        df[~df[date_column].isin(exclude_dates[date_column].unique())]
        .sort_values(date_column)
        .reset_index(drop=True)
    )

    df[return_column] = df[return_column] / 100
    df.loc[np.abs(df[return_column]) > 1, return_column] = np.nan
    df["original"] = True
    full_grid = pd.DataFrame(
        product(*[df[date_column].unique(), df[entity_column].unique()]),
        columns=[date_column, entity_column],
    )

    df = full_grid.merge(df, on=[date_column, entity_column], how="left")
    df["original"] = df["original"].fillna(False)
    df[date_column] = pd.to_datetime(df[date_column])
    df["MRET"] = df.groupby(date_column)[return_column].transform("mean")
    df[return_column] = df[return_column] - df["MRET"]
    df[f"t1_{return_column}"] = df.groupby(entity_column)[return_column].shift(-1)
    df = df[df["original"]]
    df = df.drop(columns=["MRET", volume_column, "original"])

    return df.sort_values([date_column, entity_column]).reset_index(drop=True)


def load_prices(
    path: str | Path,
    date_column: str = "date",
    entity_column: str = "entity_id",
    return_column: str = "return",
    volume_column: str = "volume",
) -> pd.DataFrame:
    """Load price/return data from CSV or Parquet file."""
    df = _load_file(path)
    df = _validate_and_rename_columns(
        df,
        column_mapping={
            date_column: "date",
            entity_column: "entity_id",
            return_column: "return",
            volume_column: "volume",
        },
        file_type="prices",
    )
    return process_raw_prices(df)


def load_signals(
    path: str | Path,
    date_column: str = "date",
    entity_column: str = "entity_id",
    signal_column: str = "signal",
) -> pd.DataFrame:
    """Load signal/factor data from CSV or Parquet file."""
    df = _load_file(path)
    df = _validate_and_rename_columns(
        df,
        column_mapping={
            date_column: "date",
            entity_column: "entity_id",
            signal_column: "signal",
        },
        file_type="signals",
    )
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values(["date", "entity_id"]).reset_index(drop=True)


def _load_file(path: str | Path) -> pd.DataFrame:
    """Load data from CSV or Parquet file based on extension."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .csv or .parquet")


def _validate_and_rename_columns(
    df: pd.DataFrame,
    column_mapping: dict[str, str],
    file_type: Literal["prices", "signals"],
) -> pd.DataFrame:
    """Validate required columns exist and rename to standard names."""
    missing = [col for col in column_mapping if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {file_type} file: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return df.rename(columns=column_mapping)[list(column_mapping.values())]
