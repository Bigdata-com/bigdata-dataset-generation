"""Preflight — the gate every command passes before it does anything."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import DistillConfig
from .paths import EXTRACTION_CONFIG_NAME, RunPaths
from .theme import ThemeSpec

CALLING_STAGES = {"teacher", "evaluate", "corpus"}

SCAFFOLD_MARKERS = ("FILL THIS IN", "my_theme")


def _indented(message: str) -> str:
    """A multi-line message, kept whole and lined up under the BLOCKED marker.

    Only the first line used to survive. For the folder-collision check that meant losing
    every actionable part of it -- which folders clash, and the two ways out.
    """
    head, *rest = message.strip().splitlines()
    return "\n".join([head, *(f"{' ' * 13}{line}" for line in rest)])


@dataclass
class Preflight:
    stage: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def ok(self, message: str) -> None:
        self.checked.append(message)

    @property
    def passed(self) -> bool:
        return not self.errors

    def report(self, *, verbose: bool = False) -> str:
        lines: list[str] = []
        if verbose:
            for item in self.checked:
                lines.append(f"  ok       {item}")
        for item in self.warnings:
            lines.append(f"  WARNING  {item}")
        for item in self.errors:
            lines.append(f"  BLOCKED  {item}")
        return "\n".join(lines)

    def require(self) -> Preflight:
        """Stop here if anything is missing."""
        if self.errors:
            raise SystemExit(
                f"preflight failed for `{self.stage}` — {len(self.errors)} problem(s). "
                f"Nothing has run.\n" + self.report(verbose=False)
                + "\n\nFix the config and try again. `distill check-config` checks "
                  "every stage at once."
            )
        return self


def _check_theme(theme_data: dict[str, Any], flight: Preflight, *,
                 where: Path | None = None, paths: RunPaths | None = None) -> ThemeSpec | None:
    if not theme_data:
        if where is not None and not where.exists():
            flight.error(f"no {EXTRACTION_CONFIG_NAME} in this run's config/ — it is the file "
                         f"that carries the theme. It is checked in under config/. "
                         f"writes one.")
        else:
            flight.error(f"no `theme:` section in {EXTRACTION_CONFIG_NAME} — the run has "
                         f"nothing to label with")
        return None
    try:
        theme = ThemeSpec.from_dict(theme_data)
    except SystemExit as exc:
        flight.error(f"theme: {exc}")
        return None
    for issue in theme.validate():
        flight.error(f"theme: {issue}")
    body = theme.body
    for marker in SCAFFOLD_MARKERS:
        if marker in body or marker == theme.id:
            flight.error(
                f"the theme in {EXTRACTION_CONFIG_NAME} is still the scaffold ({marker!r} is in "
                f"it). Write the body before running anything — see "
                f"distill/distill/prompt_guidelines.md."
            )
            break
    if not flight.errors:
        flight.ok(f"theme {theme.id!r}: {len(theme.class_names)} classes, "
                  f"{len(body.strip()):,}-char body, signs {theme.signs()}")
    if paths is not None:
        _check_theme_matches_the_seeds(paths, theme, flight)
    return theme


def _check_theme_matches_the_seeds(paths: RunPaths, theme, flight: Preflight) -> None:
    """The theme has to be the one the SEEDS retrieved this corpus with. Nothing can prove it.

    Which is why this is a warning that repeats, and not a note printed once. A run was found
    on 2026-09-02 whose `run_config.yaml` said `theme: market_share` while all sixty of its
    seed statements were about operating margins -- so the old pipeline would have labelled a
    margin corpus with market share's prompts. It does not fail: it produces numbers, and they
    answer a question nobody asked. It was printed as a NOTE once, into a terminal
    that later carried eight hours of other output.

    The seeds are the record of what was actually searched and paid for. The theme is what the
    excerpt will be judged against. Only a person can say whether they are the same subject, so
    the check puts them side by side at every gate and asks.
    """
    named, source = _theme_named_by_retrieval(paths)
    if not named:
        return
    squashed = lambda text: "".join(c for c in str(text).lower() if c.isalnum())
    if squashed(named) == squashed(theme.id):
        return
    flight.warn(
        f"{source} retrieved this corpus under theme {named!r}, and it is about to be labelled "
        f"as {theme.id!r}. One of three things is true and only you can say which: the old key "
        f"was wrong and {theme.id!r} is the correction; the theme was renamed; or this corpus "
        f"is about to be labelled with the wrong prompts. Read `seeds.statements` in "
        f"{EXTRACTION_CONFIG_NAME} -- they are what actually selected these rows -- and check "
        f"they are asking {theme.id!r}'s question."
    )


def _theme_named_by_retrieval(paths: RunPaths) -> tuple[str, str]:
    """The theme key the retrieval half recorded, from whichever file carries it."""
    import yaml

    for path in (paths.run_dir / "run_config.yaml", paths.extraction_config):
        if not path.exists():
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:                                # noqa: BLE001 - _check_theme reports it
            continue
        named = data.get("theme") if isinstance(data, dict) else None
        if isinstance(named, str) and named.strip():
            return named.strip(), path.name
    return "", ""


def _check_retrieval(paths: RunPaths, flight: Preflight) -> None:
    """The retrieval half of `data_extraction.yaml`, checked only while it still matters."""
    from .config import read_theme

    path = paths.extraction_config
    if not path.exists() or paths.part_a_similarity.exists():
        return
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:                                    # noqa: BLE001 - reported by _check_theme
        return
    if not isinstance(data, dict):
        return
    if not data.get("universe_csv"):
        flight.warn(f"{path.name}: `universe_csv` is not set, so the search has no universe. "
                    f"Only needed if you are running the retrieval half here.")
    statements = ((data.get("seeds") or {}).get("statements") or {}) \
        if isinstance(data.get("seeds"), dict) else {}
    total = sum(len(statements.get(polarity) or []) for polarity in ("positive", "negative"))
    if not total:
        flight.warn(f"{path.name}: `seeds.statements` is empty, so the search has nothing to "
                    f"look for. They must describe the same theme as `theme:` above them.")
    else:
        flight.ok(f"{path.name}: {total} seed statements")


def _check_config(config: DistillConfig, flight: Preflight) -> None:
    for issue in config.validate():
        if issue.startswith("theme:"):
            continue
        flight.error(issue)
    if not flight.errors:
        flight.ok(f"config: {config.experiment.granularity}.{config.experiment.topology}, "
                  f"subset by {config.subset.metric}, seed {config.sample.seed}")


DEAD_MARKER_KEYS = ("bytes", "rows")


def _check_completion_marker(path: Path, flight: Preflight) -> bool:
    """Compare the artifact against the completion marker its producer wrote beside it.

    Part A records the byte count and the row count when a step finishes. A run copied
    between machines is where that pays for itself: a truncated parquet is not a corrupt
    file with a clear error, it is a file whose first N row groups read perfectly, so a
    partial transfer looks like a smaller dataset. The marker is the only thing that knows
    how big it was supposed to be.
    """
    marker = path.with_name(path.name + ".done.json")
    if not marker.exists():
        flight.warn(f"{path.name} has no completion marker beside it, so there is nothing to "
                    f"check its size against. A truncated copy would read as a smaller "
                    f"dataset.")
        return True
    import json

    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        flight.warn(f"{marker.name} cannot be read")
        return True
    expected = int(data.get("bytes") or 0)
    actual = path.stat().st_size
    if expected and actual != expected:
        percent = 100 * actual / expected
        flight.error(
            f"{path.name} is {actual:,} bytes but {marker.name} says it finished at "
            f"{expected:,} ({percent:.1f}% of it is here). The copy did not complete. "
            f"Re-transfer it -- and note that a truncated parquet often still READS, just "
            f"with fewer rows, so nothing downstream would have told you."
        )
        return False
    if expected:
        flight.ok(f"{path.name}: {actual:,} bytes, matches its completion marker "
                  f"({int(data.get('rows') or 0):,} rows)")
    return True


def _check_input(config: DistillConfig, flight: Preflight, *, needed: bool) -> None:
    named = [name for name, value in (("input.similarity_parquet", config.input.similarity_parquet),
                                      ("input.source_parquet", config.input.source_parquet))
             if value]
    if not named:
        message = ("no input: set `input.similarity_parquet` (the data-extraction output, "
                   "relative to the run) or `input.source_parquet`")
        flight.error(message) if needed else flight.warn(message)
        return
    if len(named) == 2:
        flight.warn(f"both {named[0]} and {named[1]} are set; source_parquet wins")
    try:
        path = config.source_frame_path()
    except SystemExit as exc:
        message = str(exc).splitlines()[0]
        flight.error(message) if needed else flight.warn(message)
        return
    if not needed:
        flight.ok(f"input present: {path}")
        return
    if not _check_completion_marker(path, flight):
        return

    from .stages.subset import required_columns

    try:
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(path).schema_arrow.names)
    except Exception as exc:                      # noqa: BLE001 - any unreadable input
        flight.error(f"input {path} cannot be read as a parquet: {type(exc).__name__}: {exc}")
        return
    from .stages.subset import metric_is_derivable

    derivable = metric_is_derivable(config, available)
    if derivable:
        flight.ok(f"subset.metric {config.subset.metric!r} is not a column, and will be derived "
                  f"as the row maximum of {config.sentence.sims_column!r} -- which is what a "
                  f"step-7 artifact carries")
    missing = [c for c in required_columns(config)
               if c not in available and not (derivable and c == config.subset.metric)]
    if missing:
        flight.error(
            f"input {path.name} is missing columns this run needs: {missing}. It has: "
            f"{sorted(available)[:12]}{'...' if len(available) > 12 else ''}. Point the "
            f"matching `input.*` / `backtest.*_column` setting at the right name."
        )
    else:
        flight.ok(f"input {path.name}: every column this run reads is present")


def _check_endpoint(config: DistillConfig, flight: Preflight, role: str) -> None:
    settings = getattr(config, role)
    if not settings.url.strip():
        flight.error(f"{role}.url is empty, and this stage calls a model")
    if not settings.model.strip():
        flight.error(f"{role}.model is empty — the served model name is what the endpoint "
                     f"matches the request against")
    if settings.api_key_env:
        import os

        if not os.environ.get(settings.api_key_env, "").strip():
            flight.error(f"{role}.api_key_env names ${settings.api_key_env} and it is empty")
    if settings.workers < 1:
        flight.error(f"{role}.workers is {settings.workers}")
    if not flight.errors:
        flight.ok(f"{role} endpoint: {settings.model!r} at {settings.url}, "
                  f"{settings.workers} workers")


def _check_serving_recipe(config: DistillConfig, flight: Preflight, role: str) -> None:
    settings = getattr(config, role)
    if settings.serve is None:
        flight.warn(f"no `{role}.serve` block, so `distill serve-cmd {role}` has no recipe to "
                    f"print. The docker line is configuration, not a runbook.")
        return
    if not settings.serve.model_path:
        flight.warn(f"{role}.serve.model_path is empty — fill it in before serving. For a "
                    f"student that is the MERGED directory, and `du -sh` it first: a real 4B "
                    f"merge is ~7.5 G and an aborted one is 4 K.")
    from .serving import card_warnings, resolve_chat_template, serve_warnings

    others = tuple(other.serve.container
                   for name, other in (("teacher", config.teacher), ("student", config.student))
                   if name != role and other.serve is not None and other.serve.container)
    serve = settings.serve
    if role == "student" and config.training.chat_template:
        # a null chat_template is filled with the package's own copy when the student is served
        serve = resolve_chat_template(serve, config.training)
    for warning in serve_warnings(serve, role=role, other_containers=others,
                                  settings_model=settings.model):
        flight.warn(f"{role}.serve: {warning}")
    for warning in card_warnings(settings.serve, role=role):
        flight.warn(f"{role}.serve: {warning}")


def _need(path: Path, what: str, produced_by: str, flight: Preflight) -> None:
    if path.exists():
        flight.ok(f"{what} present")
    else:
        flight.error(f"{what} is missing ({path}). Run `{produced_by}` first.")


def _check_stage_inputs(config: DistillConfig, theme: ThemeSpec | None, paths: RunPaths,
                        flight: Preflight, *, dry_run: bool,
                        granularities: tuple[str, ...] | None = None) -> None:
    stage = flight.stage
    topology = config.experiment.topology
    granularities = granularities or config.granularities()

    if stage in ("subset",):
        _check_input(config, flight, needed=True)
    elif stage in ("sample",):
        _need(paths.subset_parquet, "the subset", "distill subset", flight)
    elif stage == "teacher":
        _need(paths.sample_parquet, "the sample", "distill sample", flight)
    elif stage == "split":
        for granularity in granularities:
            _need(paths.teacher_labels(granularity), f"the {granularity} teacher labels",
                  "distill teacher", flight)
    elif stage == "export":
        for granularity in granularities:
            _need(paths.train_split(granularity), f"the {granularity} train split",
                  "distill split", flight)
    elif stage in ("train", "evaluate"):
        for granularity in granularities:
            _need(paths.heldout_split(granularity), f"the {granularity} held-out split",
                  "distill split", flight)
            for dataset in config.datasets_for_topology():
                _need(paths.train_jsonl(granularity, topology, dataset),
                      f"the {granularity} training set for {dataset!r}", "distill export",
                      flight)
    elif stage == "corpus":
        _need(paths.subset_parquet, "the subset", "distill subset", flight)
        for granularity in granularities:
            for dataset in config.datasets_for_topology():
                _need(paths.train_jsonl(granularity, topology, dataset),
                      f"the {granularity} training set for {dataset!r} (its prompt fingerprint "
                      f"is the gate)", "distill export", flight)
    elif stage == "ic":
        _need(paths.subset_parquet, "the subset", "distill subset", flight)
        _need(paths.manifest_for(paths.subset_parquet),
              "the subset's manifest, which records how much of the retrieval it kept",
              "distill subset", flight)
        for granularity in granularities:
            _need(paths.corpus_parquet(granularity, topology),
                  f"the {granularity} corpus labels", "distill corpus", flight)
    elif stage == "backtest":
        _need(paths.subset_parquet, "the subset", "distill subset", flight)
        from .stages.backtest import discover_label_sets

        found = discover_label_sets(config, paths)
        if not found:
            flight.error(f"no corpus label sets under {paths.step_dir('corpus')}. Run "
                         f"`distill corpus` first.")
        else:
            flight.ok(f"label sets to back-test: {sorted(found)}")
        if theme is not None:
            from .signals import require_signs

            try:
                require_signs(theme)
            except SystemExit as exc:
                flight.error(str(exc).splitlines()[0])

    if stage in CALLING_STAGES and not dry_run:
        role = "teacher" if stage == "teacher" else "student"
        _check_endpoint(config, flight, role)
    if stage in ("train",):
        _check_serving_recipe(config, flight, "student")
    if stage == "corpus":
        _check_specialists(config, flight)
    if stage in ("ic", "backtest") and config.backtest.enabled:
        cache = config.repo_path(config.backtest.price_cache)
        if not (cache and cache.exists()):
            flight.warn("no price cache: the IC and the backtest need daily returns, and stop "
                        "without them. Set backtest.price_cache at a cache covering the window")


def _check_specialists(config: DistillConfig, flight: Preflight) -> None:
    """The two specialists run AT THE SAME TIME, so what they share is a blocker, not a nit."""
    if config.experiment.topology not in ("specialist", "both"):
        return
    from .serving import _specialist_collisions, card_warnings

    types = tuple(config.subset.content_types)
    for problem in _specialist_collisions(config, types):
        flight.error(f"specialists: {problem}")
    for content_type in types:
        settings = config.student_for(content_type)
        if settings.serve is None:
            continue
        for warning in card_warnings(settings.serve, role="student", cards=1):
            flight.warn(f"student_overrides.{content_type}.serve: {warning}")


def check(config: DistillConfig, paths: RunPaths, *, stage: str,
          dry_run: bool = False,
          granularities: tuple[str, ...] | None = None) -> tuple[Preflight, ThemeSpec | None]:
    """Run every check that applies."""
    flight = Preflight(stage=stage)
    theme = _check_theme(config.theme, flight, where=paths.extraction_config, paths=paths)
    _check_retrieval(paths, flight)
    _check_config(config, flight)
    if not paths.run_dir.exists():
        flight.error(f"the run folder {paths.run_dir} does not exist")

    if stage not in ("subset",):
        _check_input(config, flight, needed=False)
    _check_stage_inputs(config, theme, paths, flight, dry_run=dry_run,
                        granularities=granularities)
    return flight, theme


def check_every_stage(config: DistillConfig, paths: RunPaths) -> dict[str, Preflight]:
    """What `distill check-config` reports: the config once, then each stage's own needs."""
    from .stages import ORDER

    return {stage: check(config, paths, stage=stage)[0]
            for stage in ("config", *ORDER, "split", "train")}
