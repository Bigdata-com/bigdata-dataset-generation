"""Append-only per-row checkpoints, and the one concurrent labelling loop."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .parsing import error_result


class Checkpoint:
    """One append-only JSONL of per-row records."""

    def __init__(self, path: str | Path, *, retry_errors: bool = False) -> None:
        self.path = Path(path)
        self.retry_errors = retry_errors

    def load(self) -> dict[str, dict[str, Any]]:
        """row_key -> the record already obtained."""
        if not self.path.exists():
            return {}
        done: dict[str, dict[str, Any]] = {}
        truncated = 0
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    truncated += 1
                    continue
                key = record.get("row_key")
                if key is None:
                    truncated += 1
                    continue
                if record.get("ok") or not self.retry_errors:
                    done[key] = record
                else:
                    done.pop(key, None)
        if truncated:
            print(f"     note: ignored {truncated} unusable line(s) in {self.path.name} "
                  f"(a truncated tail is what a killed process leaves behind)")
        return done

    def open_appender(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return self.path.open("a", encoding="utf-8")


def label_rows(
    rows: Sequence[Any],
    *,
    label_one: Callable[[Any], dict[str, Any]],
    key_of: Callable[[Any], str],
    checkpoint: Checkpoint,
    workers: int,
    desc: str,
    label_field: str = "classification",
    progress: bool = True,
    warmup_rows: int = 150,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Label `rows` concurrently, checkpointing every answer as it arrives."""
    done = checkpoint.load()
    todo = [row for row in rows if key_of(row) not in done]
    print(f"[{desc}] {len(rows):,} rows — {len(done):,} already in {checkpoint.path.name}, "
          f"{len(todo):,} to label")

    counts: dict[str, int] = {}
    elapsed = 0.0
    steady_start: float | None = None
    steady_done = 0
    steady_rate: float | None = None
    if todo:
        bar = _progress_bar(len(todo), desc) if progress else None
        started = time.monotonic()
        with checkpoint.open_appender() as fh:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = {pool.submit(label_one, row): key_of(row) for row in todo}
                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        result = error_result(f"worker exception: {exc}", kind="http")
                        record = {"row_key": key, label_field: None, "ok": False,
                                  "error": result.error, "error_kind": result.error_kind}
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fh.flush()
                    done[key] = record
                    bucket = record.get(label_field) or "ERROR"
                    counts[bucket] = counts.get(bucket, 0) + 1
                    finished = sum(counts.values())
                    if finished == warmup_rows:
                        steady_start = time.monotonic()
                        steady_done = 0
                    elif steady_start is not None:
                        steady_done += 1
                    if bar is not None:
                        bar.update(1)
        if bar is not None:
            bar.close()
        elapsed = time.monotonic() - started
        rate = len(todo) / max(elapsed, 1e-9)
        print(f"[{desc}] {len(todo):,} calls in {elapsed / 60:.1f} min ({rate:.2f} calls/s at "
              f"{workers} workers)")
        if steady_start is not None and steady_done > 0:
            steady_rate = steady_done / max(time.monotonic() - steady_start, 1e-9)
            print(f"[{desc}] {steady_rate:.1f} rows/s steady, over {steady_done:,} rows")
        elif len(todo) < warmup_rows:
            print(f"[{desc}] fewer than {warmup_rows} rows, so the rate above is a COLD-START "
                  f"reading and not a throughput measurement")

    records = [done[key_of(row)] for row in rows if key_of(row) in done]
    stats = {
        "rows": len(rows),
        "labelled": len(records),
        "newly_labelled": len(todo),
        "errors": int(sum(1 for r in records if not r.get("ok", True))),
        "seconds": round(elapsed, 1),
        "calls_per_second": round(len(todo) / elapsed, 2) if elapsed > 0 else None,
        "steady_calls_per_second": round(steady_rate, 2) if steady_rate else None,
        "steady_rows": steady_done if steady_rate else None,
        "workers": workers,
        "checkpoint": str(checkpoint.path),
    }
    return records, stats


def _progress_bar(total: int, desc: str):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(total=total, unit="row", desc=desc)


def iter_records(path: str | Path) -> Iterable[dict[str, Any]]:
    """Stream a checkpoint file."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
