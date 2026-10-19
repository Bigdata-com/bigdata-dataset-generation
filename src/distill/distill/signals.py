"""Labelled chunks -> a daily (date, entity_id, signal) table."""

from __future__ import annotations

import warnings
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .theme import ThemeSpec

_NY = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")


def parse_cutoff(text: str) -> time:
    hour, _, minute = str(text).partition(":")
    return time(int(hour), int(minute or 0))


def utc_timestamp_to_signal_trade_date(ts: Any, cutoff: time) -> date | None:
    """UTC timestamp -> US trading date (NY cut-off, then roll to a business day)."""
    if ts is None:
        return None
    parsed = pd.Timestamp(ts)
    if pd.isna(parsed):
        return None
    parsed = parsed.tz_localize(_UTC) if parsed.tzinfo is None else parsed.tz_convert(_UTC)
    local = parsed.tz_convert(_NY)
    calendar_day = date(local.year, local.month, local.day)
    clock = time(local.hour, local.minute, local.second, local.microsecond)
    anchor = datetime.combine(calendar_day, time(0, 0), tzinfo=_NY)
    if clock > cutoff:
        anchor += timedelta(days=1)
    return pd.offsets.BDay().rollforward(anchor).date()


def require_signs(theme: ThemeSpec) -> dict[str, float]:
    """The sign map, or a refusal."""
    signs = theme.signs()
    missing = [c for c in theme.signal_classes if c not in signs]
    if missing:
        raise SystemExit(
            f"theme {theme.id!r}: classes {missing} have no `sign`, so no signal can be built "
            f"from them. Add `sign:` to each class in the theme file -- for a polar pair that is "
            f"+1 and -1; for a presence pair, whether 'the text asserts absence' is worth -1 or 0 "
            f"is a design decision and the code must not guess it."
        )
    active = {name: value for name, value in signs.items() if value}
    if not active:
        raise SystemExit(
            f"theme {theme.id!r}: every class has sign 0, so the signal is identically zero. "
            f"That is exactly the all-zero book the old hardcoded sign map produced silently."
        )
    return signs


def active_signs(theme: ThemeSpec) -> dict[str, float]:
    """The classes that actually contribute to a signal: sign != 0."""
    return {name: value for name, value in require_signs(theme).items() if value}


def build_signals(
    frame: pd.DataFrame,
    theme: ThemeSpec,
    *,
    signal_mode: str = "class_sign",
    cutoff: time | None = None,
    date_column: str = "date",
    timestamp_column: str = "timestamp",
    relevance_column: str = "chunk_relevance",
    sentiment_column: str = "chunk_sentiment",
    label_column: str | None = None,
    content_types: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """(date, entity_id, signal), one row per entity per trading day."""
    cutoff = cutoff or time(15, 30)
    label_column = label_column or theme.label_field
    work = frame.copy()
    if content_types is not None:
        work = work[work["content_type"].isin(content_types)].copy()

    work["entity_id"] = work["entity_id"].astype(str)
    for column in (relevance_column, sentiment_column):
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    if relevance_column in work.columns:
        work[relevance_column] = work[relevance_column].fillna(0.0)

    day = pd.to_datetime(work[date_column], errors="coerce").dt.strftime("%Y-%m-%d")
    if timestamp_column in work.columns:
        stamp = day + " " + work[timestamp_column].astype(str).str.strip()
    else:
        stamp = day
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)
        utc = pd.to_datetime(stamp, errors="coerce", utc=True)
    work["signal_date"] = utc.map(lambda ts: utc_timestamp_to_signal_trade_date(ts, cutoff))
    dropped = int(work["signal_date"].isna().sum())
    work = work[work["signal_date"].notna()].copy()
    if dropped:
        print(f"     note: {dropped:,} rows had no usable timestamp and cannot be dated to a "
              f"trading day -- dropped from the signal, not from the label set")

    if signal_mode == "relevance_sentiment":
        if sentiment_column not in work.columns:
            raise SystemExit(f"signal_mode=relevance_sentiment needs a {sentiment_column!r} column")
        work = work[work[sentiment_column].notna()].copy()
        work["_score"] = work[relevance_column] * work[sentiment_column]
    elif signal_mode in ("class_sign", "relevance_class_sign"):
        declared = require_signs(theme)
        signs = active_signs(theme)
        canonical = {name.lower(): name for name in declared}
        work[label_column] = (work[label_column].astype(str).str.strip().str.lower()
                              .map(canonical).fillna(work[label_column].astype(str).str.strip()))
        unknown = sorted(set(work[label_column].unique()) - set(declared)
                         - {"nan", "None", "none", ""})
        if unknown:
            raise SystemExit(
                f"the label set contains classes the theme does not declare: {unknown}. "
                f"The theme has {sorted(declared)}. Filtering them out silently is what produced "
                f"an empty book with no error; refusing is the fix."
            )
        inert = sorted(set(declared) - set(signs))
        before = len(work)
        work = work[work[label_column].isin(set(signs))].copy()
        if inert:
            print(f"     {before - len(work):,} of {before:,} rows are {inert} "
                  f"and carry no position")
        work["class_sign"] = work[label_column].map(signs)
        work["_score"] = work["class_sign"]
        if signal_mode == "relevance_class_sign":
            work["_score"] = work[relevance_column] * work["_score"]
    else:
        raise SystemExit(f"unknown signal_mode {signal_mode!r}")

    daily = (work.groupby(["signal_date", "entity_id"], as_index=False)["_score"]
                 .sum()
                 .rename(columns={"signal_date": "date", "_score": "signal"})
                 .sort_values(["date", "entity_id"]))
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    return daily.reset_index(drop=True)


def signal_breadth(signals: pd.DataFrame) -> dict[str, float]:
    """Average number of long / short names per day."""
    if signals.empty:
        return {}
    side = (signals.assign(side=np.sign(signals["signal"]))
                   .groupby("date")["side"].value_counts().unstack(fill_value=0).mean())
    return {{-1.0: "short", 0.0: "flat", 1.0: "long"}.get(k, str(k)): round(float(v), 1)
            for k, v in side.items()}


def smooth_signal(signal: pd.DataFrame, half_life_days: int) -> pd.DataFrame:
    """Let a label decay instead of firing for one day, with a half-life in calendar days.

    A label is an event: the passage was published on one day, and the raw signal holds nothing
    the day after. Smoothing asks whether the information outlives the event -- and it is the
    difference between a book that turns over every day and one that holds.

    The decay runs forward over each entity's own span. Weekends get a row and no price, so
    they cost nothing; what they do is carry the weight across to Monday instead of dropping it.

    The `date` column comes back in the dtype it went in as. `build_signals` hands out
    `datetime.date` and the backtest merges on it; converting it to a Timestamp here would
    silently match nothing.
    """
    if not half_life_days:
        return signal
    import numpy as np

    as_dates = not pd.api.types.is_datetime64_any_dtype(signal["date"])
    work = signal.assign(date=pd.to_datetime(signal["date"]))
    length = min(365, int(np.ceil(10 * half_life_days)))
    weights = 2.0 ** (-np.arange(length) / half_life_days)
    weights[weights < 1e-10] = 0.0

    out = []
    for entity_id, group in work.groupby("entity_id"):
        index = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        values = group.set_index("date")["signal"].reindex(index, fill_value=0.0).to_numpy()
        decayed = np.convolve(values, weights, mode="full")[:len(values)]
        out.append(pd.DataFrame({"date": index, "entity_id": entity_id, "signal": decayed}))
    result = pd.concat(out, ignore_index=True)
    # The reindex above gives every entity a row on every calendar day between its first and
    # last event, and past ten half-lives the decay is numerically dead. Those rows take no
    # position -- a zero weight and a missing row build the same book -- but there are two
    # orders of magnitude more of them than of real ones, and they make the breadth read as
    # hundreds of flat names a day. Dropping them is not an approximation.
    result = result[result["signal"].abs() > 1e-9]
    result = result.sort_values(["date", "entity_id"])
    if as_dates:
        result["date"] = result["date"].dt.date
    return result.reset_index(drop=True)
