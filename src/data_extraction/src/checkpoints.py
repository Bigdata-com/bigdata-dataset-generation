#!/usr/bin/env python3
"""Per-statement checkpoints for the plan / search / dedup steps of notebook 1."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

_JSON_SUFFIX = ".json"
_TMP_SUFFIX = ".tmp"


def statement_key(statement: str) -> str:
    """Stable 16-char hex id for a seed statement."""
    return hashlib.sha1(statement.encode("utf-8")).hexdigest()[:16]


def plan_path(plan_dir: Path, statement: str) -> Path:
    return Path(plan_dir) / f"plan_{statement_key(statement)}{_JSON_SUFFIX}"


def raw_docs_path(raw_dir: Path, statement: str) -> Path:
    return Path(raw_dir) / f"docs_{statement_key(statement)}.pkl"


def dedup_docs_path(dedup_dir: Path, statement: str) -> Path:
    return Path(dedup_dir) / f"dedup_{statement_key(statement)}.pkl"


def save_json_atomic(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + _TMP_SUFFIX)
    tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


def save_pickle_atomic(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + _TMP_SUFFIX)
    with tmp.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_json(path: Path) -> dict | None:
    """Load a JSON object; ``None`` if missing, unreadable, or not an object."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def load_pickle(path: Path) -> Any | None:
    """Load a pickle; ``None`` if missing or unreadable."""
    try:
        with Path(path).open("rb") as handle:
            return pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError):
        return None


def save_plan(plan_dir: Path, statement: str, plan: dict) -> Path:
    """Persist ``{"statement", "plan"}``."""
    out = plan_path(plan_dir, statement)
    save_json_atomic(out, {"statement": statement, "sentence": statement, "plan": plan})
    return out


def load_plan(plan_dir: Path, statement: str) -> dict | None:
    """Return ``{"statement", "plan"}`` for this statement, or ``None`` if not planned yet."""
    wrapper = load_json(plan_path(plan_dir, statement))
    if wrapper is None:
        return None
    plan = wrapper.get("plan")
    if not isinstance(plan, dict) or "baskets" not in plan:
        return None
    text = wrapper.get("statement") or wrapper.get("sentence") or statement
    return {"statement": str(text), "plan": plan}
