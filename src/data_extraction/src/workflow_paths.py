#!/usr/bin/env python3
"""Folder layout + config loading for the retrieval half — steps 1 to 7 of one run folder."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

OUTPUT_DIR_NAME = "output"
CONFIG_DIR_NAME = "config"
EXTRACTION_CONFIG_NAME = "data_extraction.yaml"

ROOT_MARKERS = ("pyproject.toml",)

STEP = {
    "plans":          "plans",
    "search":         "search",
    "dedup":          "dedup",
    "exploded":       "exploded_chunks",
    "sentences":      "extracted_sentences",
    "embeddings":     "embeddings",
    "similarity":     "similarity",
}

CONTENT_TYPES = ("news", "transcripts")

def find_package_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (default: this file's parent) to the package root."""
    start = (start or Path(__file__).resolve().parent).resolve()
    for candidate in (start, *start.parents):
        if any((candidate / marker).exists() for marker in ROOT_MARKERS):
            return candidate
    raise FileNotFoundError(
        f"package root not found above {start}: none of {ROOT_MARKERS} present. "
        "Run the scripts from inside the package folder."
    )


def repo_root(project_root) -> Path:
    """The folder that holds `config/`, `input/`, `models/`, `output/` and `src/`.

    Identity, since `pyproject.toml` sits at the repository root and that is the marker
    `find_package_root` walks up to. It stays a function because every caller names what it
    wants, and because the two were genuinely different while the package lived one level in.
    """
    return Path(project_root)


def default_output_dir(project_root, create: bool = False) -> Path:
    """`output/` beside `config/` — where every stage writes. One workflow, one folder."""
    root = repo_root(project_root) / OUTPUT_DIR_NAME
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def require_run_dir(run_dir, project_root=None) -> Path:
    """The run folder a step works on, checked before anything else happens."""
    if not run_dir:
        if project_root is None:
            raise SystemExit("--run is required when the package root is unknown")
        return default_output_dir(project_root, create=True).resolve()
    path = Path(run_dir)
    if not path.is_absolute() and project_root is not None:
        candidate = Path(project_root) / path
        path = candidate if candidate.exists() or not path.exists() else path.resolve()
    path = path.resolve()
    if not path.is_dir():
        raise SystemExit(f"--run points at a folder that does not exist: {path}")
    config = config_path(path)
    if not config.exists():
        raise SystemExit(
            f"{path.name} has no {CONFIG_DIR_NAME}/{EXTRACTION_CONFIG_NAME} — that file IS the "
            f"config for this run, and it has to be there before the first API call.\n"
            f"  the config it reads is config/, beside it"
        )
    return path

def step_dir(run_dir, key: str, create: bool = True) -> Path:
    if key not in STEP:
        raise KeyError(f"unknown step {key!r}; known: {sorted(STEP)}")
    d = Path(run_dir) / STEP[key]
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def plans_dir(run_dir, create: bool = True) -> Path:
    return step_dir(run_dir, "plans", create)


def search_dir(run_dir, create: bool = True) -> Path:
    return step_dir(run_dir, "search", create)


def dedup_dir(run_dir, create: bool = True) -> Path:
    return step_dir(run_dir, "dedup", create)


def exploded_parquet(run_dir, content_type: str) -> Path:
    return (step_dir(run_dir, "exploded")
            / f"search_chunks_by_sentence_exploded_{content_type}.parquet")


def sentences_parquet(run_dir) -> Path:
    return step_dir(run_dir, "sentences") / "combinations_with_sentences.parquet"


def embeddings_shards(run_dir) -> Path:
    return step_dir(run_dir, "embeddings") / "sentence_embeddings_shards"


def embedding_store_paths(run_dir) -> tuple[Path, Path]:
    """(sentences index .npy, float16 vector memmap .npy)."""
    d = step_dir(run_dir, "embeddings")
    return d / "emb_sentences.npy", d / "emb_vectors.f16.npy"


def similarity_parquet(run_dir) -> Path:
    return step_dir(run_dir, "similarity") / "combinations_with_similarity.parquet"


def done_marker(path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".done.json")


def mark_done(path, rows: int, **extra) -> Path:
    path = Path(path)
    marker = done_marker(path)
    payload = {
        "file": path.name,
        "rows": int(rows),
        "bytes": path.stat().st_size if path.exists() else 0,
        "completed_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **extra,
    }
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return marker


def read_done(path) -> dict | None:
    marker = done_marker(path)
    if not marker.exists():
        return None
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def is_done(path, expected_rows: int | None = None) -> bool:
    """True only if the output exists, its marker exists, and the row counts agree."""
    path = Path(path)
    data = read_done(path)
    if not path.exists() or data is None:
        return False
    if expected_rows is not None and int(data.get("rows", -1)) != int(expected_rows):
        return False
    return True


def require_done(path, what: str = "input") -> Path:
    """Raise unless ``path`` exists AND carries a completion marker."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing {what}: {path}")
    if read_done(path) is None:
        raise RuntimeError(
            f"{path.name} exists but has no completion marker "
            f"({done_marker(path).name}) — the step that writes it did not finish. "
            f"Delete the file and re-run that step; a half-written parquet still reads fine "
            f"and would silently give you a truncated dataset."
        )
    return path


def clear_output(path) -> None:
    """Delete an output and its marker (used to force a step to re-run)."""
    path = Path(path)
    path.unlink(missing_ok=True)
    done_marker(path).unlink(missing_ok=True)


def config_path(run_dir) -> Path:
    """`config/` BESIDE the output folder — checked in, edited by hand, never copied into it."""
    return Path(run_dir).parent / CONFIG_DIR_NAME / EXTRACTION_CONFIG_NAME


def load_run_config(run_dir) -> dict:
    """This run's own configuration."""
    path = config_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{CONFIG_DIR_NAME}/{EXTRACTION_CONFIG_NAME} not found in {run_dir}. "
            f"The three config files are checked in under config/, beside the output."
        )
    with open(path) as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must be a mapping at the top level")
    return cfg


def section(config: dict, name: str) -> dict:
    """One config section as a dict ({} when absent)."""
    value = (config or {}).get(name)
    return value if isinstance(value, dict) else {}


def load_theme(run_dir, project_root=None):
    """This run's theme, as `distill`'s own ThemeSpec — one implementation, not two."""
    import sys

    root = Path(project_root) if project_root else find_package_root()
    for candidate in (root / "distill", root.parent / "distill"):
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    try:
        from distill.theme import ThemeSpec
    except ImportError as exc:                                        # pragma: no cover
        raise SystemExit(
            f"cannot import the theme definition ({exc}). The retrieval half and the "
            f"labelling half read the SAME theme, from the same code, so that what was "
            f"searched for and what gets labelled cannot drift apart. Install the package "
            f"(`uv sync` at {root}) or run from inside it."
        ) from exc
    raw = load_run_config(run_dir).get("theme")
    if isinstance(raw, str):
        raise SystemExit(
            f"`theme:` in {config_path(run_dir)} is a name ({raw!r}), but it has to be the "
            f"theme itself: id, title, classes and body. Seed it with "
            f"write the theme inline in config/data_extraction.yaml."
        )
    if not raw:
        raise SystemExit(
            f"no `theme:` section in {config_path(run_dir)}. The search has to know what it is "
            f"looking for, and it is the same definition the labelling uses."
        )
    return ThemeSpec.from_dict(raw).require_valid()


def universe_csv(config: dict, run_dir, project_root) -> Path:
    """The universe CSV this run searches over — the one place that resolves it.

    A relative path is looked for at the repository root first, because that is where
    `config/`, `input/`, `models/` and `output/` live and so where a user drops it. Both
    scripts call this: resolving it twice is how they came to disagree.
    """
    raw = config.get("universe_csv")
    if not raw:
        raise SystemExit(
            f"`universe_csv:` is not set in {config_path(run_dir)} — the search needs a "
            f"universe (a CSV with columns id, name)."
        )
    path = Path(raw)
    if path.is_absolute():
        return path
    root = repo_root(project_root)
    for candidate in (root / path, Path(run_dir) / path, Path(project_root) / path):
        if candidate.exists():
            return candidate
    raise SystemExit(
        f"universe CSV not found: {raw!r} from {config_path(run_dir)}.\n"
        f"  looked in {root}, {run_dir} and {project_root}"
    )
