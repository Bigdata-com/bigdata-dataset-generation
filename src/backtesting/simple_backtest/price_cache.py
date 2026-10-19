"""Local parquet cache of raw universe prices, provided by the user and read as it is."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from simple_backtest.data import process_raw_prices

CACHE_STEM = "prices"

DateInput = str | date | datetime


class PriceCacheIncomplete(RuntimeError):
    """The cache does not cover the window asked for."""

    def __init__(self, cache_dir: Path, universe_code: str, requested: Interval,
                 coverage: list[Interval]) -> None:
        held = (", ".join(f"{i.start}..{i.end}" for i in coverage) if coverage
                else "nothing")
        super().__init__(
            f"the price cache in {cache_dir} does not cover {universe_code} over "
            f"{requested.start}..{requested.end}: its {CACHE_STEM}.meta.json holds {held}. "
            f"Add the missing dates to {CACHE_STEM}.parquet and {CACHE_STEM}.meta.json, or "
            f"point the price cache at a directory that has them."
        )


@dataclass(frozen=True)
class Interval:
    """Inclusive date interval."""

    start: date
    end: date


@dataclass
class CacheMeta:
    """The dates the cache's parquet covers, as its `.meta.json` records them."""

    intervals: list[Interval]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheMeta:
        return cls(intervals=[
            Interval(start=_parse_date(item["start"]), end=_parse_date(item["end"]))
            for item in data.get("intervals", [])
        ])


def _parse_date(value: DateInput) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    """Merge overlapping or adjacent inclusive date intervals."""
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda interval: (interval.start, interval.end))
    merged: list[Interval] = []
    current = sorted_intervals[0]

    for interval in sorted_intervals[1:]:
        if interval.start <= current.end + timedelta(days=1):
            current = Interval(start=current.start, end=max(current.end, interval.end))
            continue
        merged.append(current)
        current = interval

    merged.append(current)
    return merged


def missing_intervals(requested: Interval, coverage: list[Interval]) -> list[Interval]:
    """Return sub-intervals of ``requested`` not covered by ``coverage``."""
    if requested.start > requested.end:
        return []

    missing: list[Interval] = []
    cursor = requested.start

    for interval in merge_intervals(coverage):
        if interval.end < requested.start:
            continue
        if interval.start > requested.end:
            break

        covered_start = max(interval.start, requested.start)
        covered_end = min(interval.end, requested.end)
        if cursor < covered_start:
            missing.append(Interval(start=cursor, end=covered_start - timedelta(days=1)))
        cursor = max(cursor, covered_end + timedelta(days=1))
        if cursor > requested.end:
            return missing

    if cursor <= requested.end:
        missing.append(Interval(start=cursor, end=requested.end))

    return missing


class PriceCache:
    """Serve processed prices from a cache directory: `prices.parquet` and `prices.meta.json`.

    The cache is the user's: nothing here fetches or writes prices. A window the meta does not
    cover raises `PriceCacheIncomplete`, which says what the cache holds.
    """

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "simple_backtest" / "prices"
        self.cache_dir = Path(cache_dir)

    def get_raw_prices(
        self,
        universe_code: str,
        start_date: DateInput,
        end_date: DateInput,
    ) -> pd.DataFrame:
        """Return raw price rows for the inclusive date range."""
        requested = Interval(start=_parse_date(start_date), end=_parse_date(end_date))
        if requested.start > requested.end:
            raise ValueError("start_date must be less than or equal to end_date")

        parquet_path, meta_path = self._paths()
        meta = self._load_meta(meta_path)
        if missing_intervals(requested, meta.intervals):
            raise PriceCacheIncomplete(self.cache_dir, universe_code, requested, meta.intervals)

        cached = self._read_cached(parquet_path)
        if cached.empty:
            return cached

        cached["date"] = pd.to_datetime(cached["date"]).dt.date
        mask = (cached["date"] >= requested.start) & (cached["date"] <= requested.end)
        return cached.loc[mask].sort_values(["date", "entity_id"]).reset_index(drop=True)

    def get_prices(
        self,
        universe_code: str,
        start_date: DateInput,
        end_date: DateInput,
    ) -> pd.DataFrame:
        """Return processed prices for the inclusive date range."""
        raw = self.get_raw_prices(universe_code, start_date, end_date)
        if raw.empty:
            return pd.DataFrame(columns=["date", "entity_id", "return", "t1_return"])
        return process_raw_prices(raw)

    def _paths(self) -> tuple[Path, Path]:
        stem = self.cache_dir / CACHE_STEM
        return stem.with_suffix(".parquet"), stem.with_suffix(".meta.json")

    def _load_meta(self, meta_path: Path) -> CacheMeta:
        if not meta_path.exists():
            return CacheMeta(intervals=[])
        raw = meta_path.read_text(encoding="utf-8").strip()
        if not raw:
            return CacheMeta(intervals=[])
        return CacheMeta.from_dict(json.loads(raw))

    def _read_cached(self, parquet_path: Path) -> pd.DataFrame:
        if not parquet_path.exists():
            return pd.DataFrame(columns=["date", "entity_id", "return", "volume"])
        return pd.read_parquet(parquet_path)
