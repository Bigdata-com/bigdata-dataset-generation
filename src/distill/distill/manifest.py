"""Manifest sidecars — an artifact that cannot say where it came from cannot be handed over."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_VERSION = "1"
DEFAULT_MAX_HASH_BYTES = 2 * 1024 ** 3


def file_digest(path: str | Path, *, max_bytes: int = DEFAULT_MAX_HASH_BYTES) -> dict[str, Any]:
    """Identity of one input file: size, mtime, and sha256 when it is worth computing."""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "present": False}
    stat = path.stat()
    out: dict[str, Any] = {
        "path": str(path),
        "present": True,
        "bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
    }
    if path.is_dir():
        out["kind"] = "directory"
        out["sha256"] = None
        out["sha256_skipped"] = "directory"
        return out
    if stat.st_size > max_bytes:
        out["sha256"] = None
        out["sha256_skipped"] = f"larger than {max_bytes:,} bytes"
        return out
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    out["sha256"] = digest.hexdigest()[:32]
    return out


def config_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _git_revision(start: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass
class Manifest:
    """Accumulated while a stage runs, written once at the end."""

    stage: str
    experiment: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    fingerprints: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    travels: bool = True
    dry_run: bool = False
    _started: float = field(default_factory=time.monotonic, repr=False)

    def add_input(self, name: str, path: str | Path, **extra: Any) -> Manifest:
        entry = file_digest(path)
        entry.update(extra)
        self.inputs[name] = entry
        return self

    def add_output(self, name: str, path: str | Path, **extra: Any) -> Manifest:
        entry = file_digest(path)
        entry.update(extra)
        self.outputs[name] = entry
        return self

    def note(self, text: str) -> Manifest:
        self.notes.append(text)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": MANIFEST_VERSION,
            "stage": self.stage,
            "experiment": self.experiment,
            "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "wall_clock_seconds": round(time.monotonic() - self._started, 1),
            "dry_run": self.dry_run,
            "travels": self.travels,
            "config_hash": config_hash(self.config),
            "config": self.config,
            "fingerprints": self.fingerprints,
            "model": self.model,
            "counts": self.counts,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "notes": self.notes,
            "environment": {
                "host": socket.gethostname(),
                "user": os.environ.get("USER") or os.environ.get("USERNAME"),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "distill_git_rev": _git_revision(Path(__file__).resolve().parent),
            },
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
        return path


def read_manifest(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
