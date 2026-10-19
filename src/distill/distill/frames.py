"""Frame helpers shared by every stage: the row key, the merges, the summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .config import DistillConfig


def add_row_key(frame: pd.DataFrame, key_fields: Sequence[str]) -> pd.DataFrame:
    """`row_key` = the key fields joined, and it must be unique."""
    missing = [f for f in key_fields if f not in frame.columns]
    if missing:
        raise SystemExit(
            f"the input has no {missing} column(s), so no stable row key can be built. "
            f"Set input.key_fields in the config to columns the file actually has "
            f"(it has: {sorted(frame.columns)[:20]}...)."
        )
    parts: list[pd.Series] = []
    for field in key_fields:
        column = frame[field]
        if pd.api.types.is_integer_dtype(column) or pd.api.types.is_float_dtype(column):
            parts.append(column.astype("int64").astype(str))
        else:
            parts.append(column.astype(str))
    out = frame.copy()
    out["row_key"] = parts[0]
    for part in parts[1:]:
        out["row_key"] = out["row_key"] + ":" + part
    duplicated = int(out["row_key"].duplicated().sum())
    if duplicated:
        example = out.loc[out["row_key"].duplicated(keep=False), "row_key"].iloc[0]
        raise SystemExit(
            f"row_key is not unique in the input: {duplicated:,} duplicates (e.g. {example!r}). "
            f"Checkpoints would collide and every merge downstream would multiply rows. Fix the "
            f"input or widen input.key_fields."
        )
    return out


def read_source(config: DistillConfig, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """The one input Part B may assume, with `row_key` attached."""
    path = config.source_frame_path()
    wanted = list(dict.fromkeys([*(columns or []), *config.input.key_fields])) if columns else None
    frame = pd.read_parquet(path, columns=wanted)
    if config.input.irreproducible_source:
        print("  input.irreproducible_source: TRUE -- this file is SOURCE, not an artifact. "
              "No stage may assume it can be rebuilt.")
    return add_row_key(frame, config.input.key_fields)


def merge_labels(rows: pd.DataFrame, labels: pd.DataFrame, *, how: str = "inner") -> pd.DataFrame:
    """Join label records back onto their rows, 1:1 or not at all."""
    overlap = (set(rows.columns) & set(labels.columns)) - {"row_key"}
    if overlap:
        rows = rows.drop(columns=sorted(overlap))
    return rows.merge(labels, on="row_key", how=how, validate="one_to_one")


def class_counts(frame: pd.DataFrame, label_column: str) -> dict[str, int]:
    if label_column not in frame.columns:
        return {}
    return {str(k): int(v) for k, v in frame[label_column].value_counts(dropna=False).items()}


def write_parquet(frame: pd.DataFrame, path: str | Path, *, announce: bool = True) -> Path:
    """`announce=False` where the caller reports the result better than a path can."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    if announce:
        print(f"wrote {path}  ({len(frame):,} rows, {len(frame.columns)} columns)")
    return path


def require_existing(path: str | Path, what: str, produced_by: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{what} not found: {path}\nRun `{produced_by}` first.")
    return path


def rows_as_dicts(frame: pd.DataFrame, columns: Iterable[str]) -> list[dict[str, Any]]:
    present = [c for c in columns if c in frame.columns]
    return frame[present].to_dict("records")


def confirm(question: str, *, flag: str = "--yes") -> None:
    """Ask before a metered pass, and say what to do when there is nobody to ask.

    Under `nohup` or any detached launch stdin is closed, and a bare `input()` then dies with
    `OSError: Bad file descriptor` -- a traceback that says nothing about the flag that would
    have avoided it. A long pass is exactly the one you launch detached.
    """
    try:
        input(question)
    except (OSError, EOFError):
        raise SystemExit(
            f"there is no terminal to confirm on -- stdin is closed, which is what `nohup` and "
            f"any detached launch do. Pass {flag} to skip the question:\n"
            f"  nohup distill ... {flag} > log 2>&1 &"
        ) from None
