"""What a notebook calls, so that the notebook itself holds no logic.

§14.11 puts the logic in `.py` and the viewing in the notebook, and the reason was concrete:
the old backtest notebook kept the signal construction inside its cells, so its results existed
only as scrollback. A notebook that drives three days of GPU leans the same way, hard. The
compromise that holds is one call per cell -- which is only possible if every call it needs
already exists here.

Four things live in this module and nothing else does:

  `parameters`   the run's config, edited from the notebook's parameter cell, with a refusal
                 when the change would invalidate work that is already on disk
  `stage`        one CLI stage, streamed, so the notebook runs exactly what DISTILLATION.md runs
  `model`        a bring-up whose teardown is in a `finally`
  `state`        what this run has already produced, for the cell that asks "where am I"
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

from .config import DistillConfig
from .paths import RunPaths

PARAMETERS: dict[str, tuple[str, str]] = {
    "granularity": ("experiment.granularity", "sample"),
    "top_fraction": ("subset.top_fraction", "subset"),
    "fraction_per_bin": ("sample.fraction_per_bin", "sample"),
    "sample_max_rows": ("sample.max_rows", "sample"),
    "teacher_model_path": ("teacher.serve.model_path", "teacher"),
    "stratify_on": ("split.stratify_on", "split"),
    "evaluate_epochs": ("training.evaluate_epochs", "train"),
    "epochs": ("training.epochs", "train"),
}
"""The parameters the notebook exposes, mapped to their config key and the stage that reads them.

A parameter changed after its stage has produced output leaves a run whose artifacts and whose
config no longer describe the same experiment, which is why `parameters` refuses it.

Four of them are less obvious than their names suggest:

`teacher_model_path` is the 120B snapshot on THIS machine. It is a path rather than a
judgement, and a notebook that cannot set it sends the reader to a shell mid-run.

`stratify_on` is how the held-out set is drawn. Small runs need a coarser table: at 1,700 rows
`[content_type, label, bin]` is 120 strata, every one too thin to hold anything out, and the
held-out comes back empty with nothing for `evaluate` to score.

`evaluate_epochs` is not `epochs`. Every epoch is trained and checkpointed, which is free; this
is which of them get merged, served and scored, and each one costs a 7.4 GB merge plus a
bring-up.

`epochs` is how many are trained. Checkpoints being free, the default of five costs only the
wall clock of the last two, and lowering it is how a trial run is shortened.
"""

STAGE_ORDER: tuple[str, ...] = (
    "subset", "sample", "teacher", "split", "export", "train", "evaluate", "corpus", "ic",
    "backtest")
"""What each stage writes, in order. Read by `reset` and by `state`."""


def _get(config: DistillConfig, dotted: str) -> Any:
    target = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
        if target is None:
            return None
    return getattr(target, parts[-1])


def _set(config: DistillConfig, dotted: str, value: Any) -> None:
    target = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
        if target is None:
            raise SystemExit(f"cannot set {dotted}: `{part}` is not in the config")
    setattr(target, parts[-1], value)


EXTRACTION_STEPS: tuple[tuple[str, str], ...] = (
    ("plans", "plan_and_search.py"),
    ("search", "plan_and_search.py"),
    ("dedup", "plan_and_search.py"),
    ("exploded_chunks", "plan_and_search.py"),
    ("extracted_sentences", "extract_and_embed.py"),
    ("embeddings", "extract_and_embed.py"),
    ("similarity", "extract_and_embed.py"),
)
"""The folders the retrieval half writes, and which of its two scripts writes each.

That half is `data_extraction/` and is where a theme starts; `distill` picks the run up from
the single artifact it leaves behind, `SIMILARITY_PARQUET`.
"""

SIMILARITY_PARQUET = "similarity/combinations_with_similarity.parquet"
PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def extraction_state(run: str | Path) -> list[dict[str, Any]]:
    """What the retrieval half left behind, folder by folder.

    The handover between the two halves is one file, and this is the cell that shows whether it
    is there. A run folder with `plans` and no `similarity` is a step 2 that did not
    finish, which `distill subset` would otherwise report as a missing input.
    """
    run = Path(run).resolve()
    rows = []
    for folder, script in EXTRACTION_STEPS:
        path = run / folder
        files = sorted(path.rglob("*")) if path.exists() else []
        rows.append({"step": folder, "written by": script, "there": path.exists(),
                     "files": sum(1 for f in files if f.is_file())})
    handover = run / SIMILARITY_PARQUET
    rows.append({"step": SIMILARITY_PARQUET, "written by": "extract_and_embed.py",
                 "there": handover.exists(),
                 "files": _human(handover) if handover.exists() else 0})
    return rows


def extraction(run: str | Path, script: str, *args: str, check: bool = True) -> int:
    """One of the two retrieval scripts, on this run folder.

    They are their own programs with their own dependencies, run from inside
    `data_extraction/`, and they are idempotent: each step skips what it has already written.
    So these cells are safe to leave in place after the retrieval is done -- they cost the time
    it takes to notice the outputs exist.
    """
    path = PACKAGE_ROOT / "data_extraction" / script
    if not path.exists():
        raise SystemExit(f"no {path} -- the retrieval half is not in this copy of the package")
    command = [sys.executable, str(path), "--run", str(Path(run).resolve()), *args]
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"$ {printable}\n", flush=True)
    result = subprocess.run(command, cwd=path.parent)
    if check and result.returncode != 0:
        raise SystemExit(f"extraction step failed ({result.returncode}): {printable}")
    return result.returncode


SEARCH_OUTPUTS = tuple(f"exploded_chunks/search_chunks_by_sentence_exploded_{content_type}.parquet"
                       for content_type in ("news", "transcripts"))
"""What a finished search leaves behind, each with its `.done.json` marker."""


def search_done(run: str | Path) -> bool:
    """True once the search has written its outputs, so running it again costs nothing.

    Any one output with its marker is enough: the build marks them all at its very end, and a
    content type with no rows gets no file at all -- a news-only search never writes the
    transcripts one, and waiting for it would block 1.2 forever.
    """
    run = Path(run).resolve()
    return any((run / name).exists() and (run / f"{name}.done.json").exists()
               for name in SEARCH_OUTPUTS)


def search(run: str | Path, *args: str, confirm: bool = False, check: bool = True) -> int | None:
    """Step 1.1 -- plan one search per seed statement, then run them."""
    if not (confirm or search_done(run)):
        print("The search did not start.\n"
              "Set RUN_METERED_SEARCH = True in this cell and run it again.")
        return None
    # The first step that writes, so it makes the run folder: `output/` is gitignored, a fresh
    # clone has none, and the script refuses a --run that does not exist.
    Path(run).mkdir(parents=True, exist_ok=True)
    return extraction(run, "plan_and_search.py", "--yes", *args, check=check)


def embed(run: str | Path, *args: str, check: bool = True) -> int | None:
    """Step 1.2 -- sentences out of the retrieved passages, embedded and scored."""
    if not search_done(run):
        print("The search has not run yet, so there is nothing to embed.\n"
              "Run 1.1 with RUN_METERED_SEARCH = True first.")
        return None
    return extraction(run, "extract_and_embed.py", "--yes", *args, check=check)


def granularities(choice: str) -> tuple[str, ...]:
    """`"chunk"`, `"sentence"` or `"both"` -> the granularities to run, in order.

    The notebook names one string and loops over this, so the choice reads as a choice rather
    than as a tuple comprehension in the middle of a cell.
    """
    from .theme import GRANULARITIES as KNOWN

    if choice == "both":
        return tuple(KNOWN)
    if choice not in KNOWN:
        raise SystemExit(f"granularity {choice!r} is not one of {[*KNOWN, 'both']}")
    return (choice,)


def state(run: str | Path, config: DistillConfig | None = None) -> list[dict[str, Any]]:
    """One row per stage: what it should have written, and whether it is there.

    Read from the artifacts rather than from a log, because the artifacts are what the next
    stage will actually find. A stage that says `done` here will be skipped by the pipeline.

    The run is resolved first because the paths below are reported relative to it, and a
    notebook runs from inside the folder it names as ".".
    """
    run = Path(run).resolve()
    paths = RunPaths.for_run(run)
    config = config or DistillConfig.load_run(paths)
    granularities = config.granularities()
    topology = config.experiment.topology
    rows: list[dict[str, Any]] = []

    def add(stage: str, artifact: Path, note: str = "") -> None:
        rows.append({"stage": stage, "artifact": _short(artifact, run),
                     "done": artifact.exists(),
                     "size": _human(artifact) if artifact.exists() else "",
                     "note": note})

    add("subset", paths.subset_parquet)
    add("sample", paths.sample_parquet)
    for granularity in granularities:
        add("teacher", paths.teacher_labels(granularity), granularity)
        add("split", paths.heldout_split(granularity), granularity)
        for dataset in config.datasets_for_topology():
            add("export", paths.train_jsonl(granularity, topology, dataset),
                f"{granularity}.{dataset}")
        add("evaluate", paths.eval_comparison(granularity, topology), granularity)
        add("corpus", paths.corpus_parquet(granularity, topology), granularity)
    return rows


def _short(path: Path, run: Path) -> str:
    try:
        return str(path.relative_to(run))
    except ValueError:
        return str(path)


def _human(path: Path) -> str:
    size = path.stat().st_size if path.is_file() else sum(
        p.stat().st_size for p in path.rglob("*") if p.is_file())
    for unit in ("B", "K", "M", "G"):
        if size < 1024 or unit == "G":
            return f"{size:.0f}{unit}"
        size /= 1024
    return ""


def _stage_outputs(config: DistillConfig, paths: RunPaths) -> dict[str, list[Path]]:
    """The directory each stage owns. Deleting one un-does exactly that stage.

    Three stages do not own a directory of their own. `split` writes into the teacher's folder,
    and `export` and `train` share the training folder with each other, so none of them can be
    reset alone: resetting any of them takes the whole folder. That is named here rather than
    pretended otherwise, because a reset that silently left half a stage behind is the failure
    this exists to prevent.
    """
    topology = config.experiment.topology
    step = lambda key: paths.step_dir(key)
    outputs: dict[str, list[Path]] = {
        "subset": [step("subset")],
        "sample": [step("sample")],
        "teacher": [step("teacher")],
        "split": [],
        "export": [step("training")],
        "train": [step("training")],
        "evaluate": [step("evaluations")],
        "corpus": [step("corpus")],
        "ic": [step("ic")],
        "backtest": [paths.backtest_dir()],
    }
    for granularity in config.granularities():
        outputs["split"] += [paths.train_split(granularity), paths.heldout_split(granularity)]
    outputs["split"].append(paths.heldout_keys_parquet)
    return outputs


def reset(run: str | Path, stage: str, *, apply: bool = False) -> list[Path]:
    """Delete what `stage` wrote and everything after it, so the run can be redone from there.

    This exists because the alternative is a shell. Changing `top_fraction` after the subset is
    refused by `parameters` -- correctly, the subset on disk would no longer match the config --
    and the only way forward was `rm -rf` on four folders, in a terminal, remembering which
    ones. Forgetting `training/` once meant `export` reused a 100-row JSONL under a
    1,520-row run and a GPU was spent on it.

    Read-only by default: it lists what it would remove and how big. `apply=True` removes it.

    Adapters and merged models are written INSIDE the trainer container and are root-owned, so
    a host `rm` fails with Permission denied halfway through and leaves the folder in a state
    that is neither there nor gone. Those go through `docker exec` -- the same hands that made
    them.
    """
    from . import serving

    run = Path(run).resolve()
    paths = RunPaths.for_run(run)
    config = DistillConfig.load_run(paths)
    if stage not in STAGE_ORDER:
        raise SystemExit(f"unknown stage {stage!r}. One of: {', '.join(STAGE_ORDER)}")

    outputs = _stage_outputs(config, paths)
    doomed: list[Path] = []
    for later in STAGE_ORDER[STAGE_ORDER.index(stage):]:
        for path in outputs.get(later, []):
            if path.exists() and path not in doomed:
                doomed.append(path)

    if not doomed:
        print(f"nothing to reset: no artifact from `{stage}` onward exists")
        return []
    print(f"reset from `{stage}` -- {len(doomed)} path(s):")
    for path in doomed:
        print(f"  {_short(path, run)}   {_human(path)}")
    if not apply:
        print("\nread-only: pass apply=True to delete these")
        return doomed

    container = config.training.container if config.training else ""
    for path in doomed:
        shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)
        if path.exists() and container and serving.docker_available():
            subprocess.run(["docker", "exec", container, "rm", "-rf", str(path)], check=False)
        print(f"  {'still there' if path.exists() else 'removed'}  {_short(path, run)}")
    left = [p for p in doomed if p.exists()]
    if left:
        raise SystemExit(
            f"{len(left)} path(s) survived the reset, which is worse than not trying: a stage "
            f"will find half of its old output and reuse it. They are root-owned and this "
            f"machine has no docker to reach them with:\n  "
            + "\n  ".join(str(p) for p in left))
    return doomed


def parameters(run: str | Path, *, apply: bool = False, **values: Any) -> DistillConfig:
    """Read the notebook's parameters against the run's config, and optionally write them.

    Read-only by default. `apply=True` is a deliberate act, because this rewrites the file that
    every stage reads and that the manifests fingerprint -- a config edited silently by a
    notebook is the fastest way to a run whose artifacts nobody can account for.

    It refuses when a parameter's own stage has already produced output. That is not caution
    for its own sake: a `top_fraction` changed after the subset leaves a subset on disk that no
    longer matches the config that describes it, and everything downstream inherits the lie.
    Delete the artifact, or start a new run folder -- which is cheap and is what run folders
    are for.
    """
    run = Path(run)
    paths = RunPaths.for_run(run)
    config = DistillConfig.load_run(paths)
    done = {row["stage"] for row in state(run, config) if row["done"]}

    changes, blocked = [], []
    for name, value in values.items():
        if value is None:
            continue
        if name not in PARAMETERS:
            raise SystemExit(f"unknown parameter {name!r}. Known: {sorted(PARAMETERS)}")
        dotted, stage = PARAMETERS[name]
        current = _get(config, dotted)
        if _same(current, value):
            continue
        changes.append((name, dotted, current, value))
        if stage in done:
            blocked.append((dotted, current, value, stage))

    if not changes:
        print("parameters: the config already says all of this -- nothing to change")
        _print_parameters(config)
        return config

    for name, dotted, current, value in changes:
        print(f"  {dotted}: {current!r} -> {value!r}")
    if blocked:
        lines = "\n".join(
            f"  {dotted} ({current!r} -> {value!r}) is an input to `{stage}`, which has already "
            f"written its output" for dotted, current, value, stage in blocked)
        raise SystemExit(
            f"refusing to change a parameter whose stage is already done:\n{lines}\n"
            f"The artifact on disk was built with the old value and nothing downstream would "
            f"know. Either delete that artifact, or run this in a new run folder -- "
            f"point it at a new output folder instead; the config is shared.")
    if not apply:
        print("\nread-only: pass apply=True to write these into the run's config")
        return config

    for _, dotted, _, value in changes:
        _set(config, dotted, value)
    _narrow_epoch_lists(config)
    config.save(paths.distill_config)
    print(f"\nwritten to {paths.distill_config}")
    _print_parameters(config)
    return config


def _same(current: Any, value: Any) -> bool:
    """Equality that does not care whether a sequence is a tuple or a list."""
    if isinstance(current, (tuple, list)) and isinstance(value, (tuple, list)):
        return list(current) == list(value)
    return bool(current == value)


def _narrow_epoch_lists(config: DistillConfig) -> None:
    """Three epoch settings that nest, and the config gate refuses them if they do not.

    `epochs` are trained; `evaluate_epochs` of those are merged, served and scored;
    `keep_merged` of THOSE are kept on disk. An epoch that is not trained cannot be scored, and
    one that is not scored is never merged, so each list can only be narrowed to the one above
    it -- there is nothing for a person to decide. Doing it here rather than letting `validate`
    refuse is what keeps `EPOCHS = 3` a one-line change in a notebook instead of an error and a
    trip to the config.

    It prints what it did. A config a notebook edits in silence is how you get a run nobody can
    account for.
    """
    trained = int(config.training.epochs)
    evaluated = tuple(config.training.evaluate_epochs)
    within = tuple(e for e in evaluated if 1 <= e <= trained)
    if within != evaluated:
        config.training.evaluate_epochs = within
        print(f"  training.evaluate_epochs: {list(evaluated)} -> {list(within)}   (an epoch "
              f"past {trained} is never trained, so it cannot be scored)")
    kept = tuple(config.training.keep_merged)
    narrowed = tuple(e for e in kept if e in tuple(config.training.evaluate_epochs))
    if narrowed != kept:
        config.training.keep_merged = narrowed
        print(f"  training.keep_merged: {list(kept)} -> {list(narrowed)}   (an epoch that is "
              f"not evaluated is never merged, so it cannot be kept)")


def _print_parameters(config: DistillConfig) -> None:
    print()
    for name, (dotted, _) in PARAMETERS.items():
        print(f"  {dotted:<28} {_get(config, dotted)!r}")


def stage(run: str | Path, *args: Any, check: bool = True) -> int:
    """One `distill` stage, streamed into the cell.

    The notebook shells out to the same CLI the guides use, on purpose. A second entry point
    into the stages -- python calls from the notebook, CLI from the shell -- would be two code
    paths to keep honest, and the preflight, the gates and the printed provenance all hang off
    this one.

    The child process is given `PYTHONPATH` so it can find `distill`. An installed package is
    the normal case, but a kernel that has it on `sys.path` without a `pip install -e` is not
    unusual, and there the notebook would import this module happily and then fail on every
    stage cell, which reads as a broken package rather than as a path.

    An option whose value is `None` is left out, flag and all, so a parameters cell can say
    `SAMPLE_ROWS = None` and the stage falls back to its config instead of receiving "None".
    """
    kept: list[Any] = []
    for arg in args:
        if arg is None:
            if kept and str(kept[-1]).startswith("--"):
                kept.pop()
            continue
        kept.append(arg)
    command = [sys.executable, "-m", "distill.cli", "--run", str(run),
               *(str(a) for a in kept)]
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"$ {printable}\n", flush=True)
    env = dict(os.environ)
    root = str(PACKAGE_ROOT / "distill")
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run(command, env=env)
    if check and result.returncode != 0:
        raise SystemExit(f"stage failed ({result.returncode}): {printable}")
    return result.returncode


@contextmanager
def model(run: str | Path, role: str, *, content_type: str = "", reuse: bool = True,
          leave_up: bool = False):
    """Bring a model up for the block, and take it down whatever happens inside it.

    Wraps `serving.up`, which is where the docker line and the health gate live. The teardown
    is in a `finally` there, so an exception -- including the KeyboardInterrupt of a stopped
    cell -- still gives the cards back. Never wrap `train`: it serves and tears down its own
    student for every checkpoint it scores.
    """
    from . import serving

    paths = RunPaths.for_run(Path(run))
    config = DistillConfig.load_run(paths)
    with serving.up(config, role, content_type=content_type, reuse=reuse, leave_up=leave_up):
        yield config


def sample_bins(run: str | Path) -> Any:
    """What the stratified draw actually took, per bin -- read back from the sample itself.

    `sample` prints this while it runs, and a printed table is scrollback. Reading it from the
    parquet is what makes the ceiling auditable: if `sample.max_rows` had trimmed the tail
    instead of trimming proportionally, the `taken` column would slope and this is where it
    would show.
    """
    import pandas as pd

    paths = RunPaths.for_run(Path(run))
    path = paths.sample_parquet
    if not path.exists():
        print(f"not yet: {path}")
        return pd.DataFrame()
    columns = ["bin", "bin_lo", "bin_hi"]
    frame = pd.read_parquet(path, columns=columns + ["content_type"]
                            if _has_column(path, "content_type") else columns)
    grouped = frame.groupby("bin", as_index=False).agg(
        lo=("bin_lo", "min"), hi=("bin_hi", "max"), taken=("bin", "size"))
    grouped["share"] = (grouped["taken"] / grouped["taken"].sum()).round(4)
    return grouped


def _has_column(path: Path, name: str) -> bool:
    import pyarrow.parquet as pq

    return name in pq.ParquetFile(path).schema.names


def _tag_of(path: Path, directory: Path) -> str:
    """The checkpoint tag behind an `<tag>.metrics.json`.

    `evaluate` names the file after the tag and reads its own output back with
    `*.metrics.json`. This reads the same files, so it has to spell the suffix the same way:
    globbing for a bare `metrics.json` matched nothing and the table came back empty, which
    reads as a training that scored nothing rather than as a pattern that never matched.
    """
    if path.parent != directory:
        return path.parent.name
    return path.name.removesuffix(".metrics.json")


def checkpoint_table(run: str | Path, granularity: str) -> Any:
    """Every scored checkpoint for one granularity, as a frame.

    Reads `evaluations/*/metrics.json`, which is what `evaluate` wrote. No metric is computed
    here: the notebook must not be able to disagree with the run's own numbers.
    """
    import json

    import pandas as pd

    paths = RunPaths.for_run(Path(run))
    config = DistillConfig.load_run(paths)
    directory = paths.evaluations_dir(granularity, config.experiment.topology)
    rows = []
    if directory.exists():
        for path in sorted(directory.rglob("*.metrics.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            payload["tag"] = _tag_of(path, directory)
            rows.append(payload)
    return pd.DataFrame(rows)


def scores(run: str | Path, granularities: tuple[str, ...]) -> Any:
    """One row per checkpoint, across granularities: the table you choose from.

    The tags `evaluate` writes are `both.combined.both.e3` -- the granularity is in the folder
    name, not in the tag, so reading two of these side by side you cannot tell which model is
    which. This puts it in a column.

    Per class it reports PRECISION and RECALL for every class the theme has, the abstain class
    included. Recall on a rare class is how many of them the student finds at all; precision is
    how many of the ones it claims are really there. The counts those are computed from are in
    `metrics.json` beside the predictions, for when the question is how many positions the book
    would take rather than how well the class was found.
    """
    import json
    import re

    import pandas as pd

    paths = RunPaths.for_run(Path(run))
    config = DistillConfig.load_run(paths)
    rows = []
    for granularity in granularities:
        directory = paths.evaluations_dir(granularity, config.experiment.topology)
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.metrics.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            tag = _tag_of(path, directory)
            epoch = re.search(r"e(\d+)", tag)
            overall = payload.get("reports", {}).get("overall", {}) or {}
            row = {
                "granularity": granularity,
                "epoch": int(epoch.group(1)) if epoch else None,
                "macro_f1": payload.get("macro_f1"),
                # What you get for free by answering the abstain class to everything, and the
                # reason accuracy is not in this table. On a corpus this skewed the do-nothing
                # model scores higher on accuracy than a model that finds things: one measured
                # run had the collapsed epoch at 95.2% against 91.7% for the best checkpoint.
                # A macro-F1 has to clear this floor to mean anything, and that reads.
                "floor": overall.get("abstain_baseline_macro_f1"),
            }
            per_class = overall.get("per_class", {})
            for name, metrics in per_class.items():
                row[f"{name} f1"] = metrics.get("f1")
                row[f"{name} precision"] = metrics.get("precision")
                row[f"{name} recall"] = metrics.get("recall")
            rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.sort_values(["granularity", "epoch"]).reset_index(drop=True) if len(frame) \
        else frame


def label_mix(run: str | Path, granularities: tuple[str, ...]) -> Any:
    """What the teacher labelled and what the student labelled, side by side.

    One row per granularity, stage and content type, with the class shares as columns. The
    teacher's row is the sample it saw; the student's is the whole subset. Reading them next
    to each other is how you see whether the student kept the teacher's proportions -- a
    student that finds half the `loser` the teacher did takes half the shorts.
    """
    import pandas as pd

    paths = RunPaths.for_run(Path(run))
    config = DistillConfig.load_run(paths)
    topology = config.experiment.topology
    label = config.theme_spec().label_field
    content_column = config.input.content_type_column
    rows = []

    for granularity in granularities:
        sources = {
            "teacher": paths.teacher_labels(granularity),
            "student": paths.corpus_parquet(granularity, topology),
        }
        for stage, path in sources.items():
            if not Path(path).exists():
                continue
            columns = [label, content_column] + (["ok"] if stage == "student" else [])
            frame = pd.read_parquet(path, columns=columns)
            if "ok" in frame:
                frame = frame.loc[frame["ok"]]
            for content_type in [*sorted(frame[content_column].unique()), "all"]:
                part = frame if content_type == "all" else \
                    frame.loc[frame[content_column] == content_type]
                share = part[label].value_counts(normalize=True).mul(100).round(2)
                rows.append({"granularity": granularity, "labelled by": stage,
                             "content": content_type, "rows": len(part),
                             **{name: share.get(name, 0.0) for name in config.theme_spec().class_names}})
    return pd.DataFrame(rows)


def point_student_at(run: str | Path, granularity: str, epoch: int, *, dataset: str = "both",
                     apply: bool = False) -> DistillConfig:
    """Write the chosen checkpoint into `student.model` / `serve.model_path` / the served name.

    §15.7 decision 6, and it is three fields rather than one because they are three different
    things: what the CLIENT asks for, where the WEIGHTS are, and what the SERVER calls them.
    They have to agree, and §16.4 is what happens when they silently do not -- so the name
    carries the granularity and the epoch, which is the only signal on the wire.

    What goes into the config is the symlink `train` made, not the merged directory, built from
    the same helper under the same name so the two cannot drift apart. The link does not
    necessarily remove the space in `training`: with `link_root: null` it is placed inside it,
    and quoting is what carries the path. It is used because it is the path `train` already
    served and scored that epoch through.
    """
    from . import serving

    run = Path(run)
    paths = RunPaths.for_run(run)
    config = DistillConfig.load_run(paths)
    topology = config.experiment.topology
    merged = paths.merged_dir(granularity, topology, dataset, epoch)
    served = f"{config.theme_id}_{granularity}_{dataset}_e{epoch}".replace("-", "_")
    model_path, link_command = serving.link_free_of_spaces(
        merged, link_root=config.training.link_root,
        name=f"{config.theme_id}_{granularity}_{topology}_{dataset}_e{epoch}")

    print(f"  student.model                    {config.student.model!r} -> {served!r}")
    print(f"  student.serve.served_model_name  "
          f"{(config.student.serve.served_model_name if config.student.serve else None)!r} "
          f"-> {served!r}")
    print(f"  student.serve.model_path         -> {str(model_path)!r}")
    if link_command:
        print(f"  (a symlink, because the merged path has a space in it: {merged})")
    if not merged.exists():
        also = (" It also makes the symlink above." if link_command else "")
        print(f"\n  NOTE  {merged} does not exist yet. `train` writes it when it merges that "
              f"epoch, and only for the epochs in `training.evaluate_epochs`.{also} So run "
              f"this AFTER train, not before.")
    if not apply:
        print("\nread-only: pass apply=True to write this into the run's config")
        return config

    config.student.model = served
    if config.student.serve is None:
        raise SystemExit("student.serve is not in the config: there is no recipe to point")
    config.student.serve = dc_replace(config.student.serve, served_model_name=served,
                                      model_path=str(model_path))
    config.save(paths.distill_config)
    print(f"\nwritten to {paths.distill_config}")
    return config


def labelling_speed(run: str | Path, granularity: str | None = None) -> Any:
    """How much faster the distilled student labels than the teacher it learned from.

    Both rates come from the same estimator on the same two cards: `steady_calls_per_second`,
    which `label_rows` starts measuring only after the first 150 rows. That matters because a
    restarted vLLM serves its first few hundred rows at a fraction of its steady rate, and
    reading that as throughput is how a model gets credited with being slow. Using the same
    estimator on both sides is also what lets the ratio stand on its own: whatever warm-up
    still leaks into the number leaks into both, and largely cancels.

    Chunk by default, because there one row is one call and the two models are doing the same
    work per row. Sentence falls back to it only when chunk was never run: there the walk stops
    at the first non-abstain answer, so a row costs 1..N calls and the two models do not make
    the same number of them.

    The student's hours are what it actually took, projected from its own measured rate; the
    teacher's are what the same corpus would have cost at the rate it was measured at while it
    labelled the sample. One side is a measurement, the other an extrapolation from one.
    """
    import pandas as pd

    from .manifest import read_manifest

    paths = RunPaths.for_run(Path(run))
    config = DistillConfig.load_run(paths)
    topology = config.experiment.topology

    def rate_of(entry: dict[str, Any]) -> tuple[float | None, bool]:
        """Rows per second, and whether the warm-up had to be left in."""
        steady = entry.get("steady_calls_per_second")
        if steady:
            return float(steady), False
        cold = entry.get("calls_per_second")
        return (float(cold), True) if cold else (None, True)

    for one in ([granularity] if granularity else list(config.granularities())):
        teacher = read_manifest(paths.manifest_for(paths.teacher_labels(one)))
        student = read_manifest(paths.manifest_for(paths.corpus_parquet(one, topology)))
        if not teacher or not student:
            continue
        teacher_runs = (teacher.get("counts") or {}).get("per_content_type_run") or {}
        student_runs = (student.get("model") or {}).get("per_content_type") or {}
        rows = []
        for content_type in sorted(set(teacher_runs) & set(student_runs)):
            teacher_rate, teacher_cold = rate_of(teacher_runs[content_type])
            student_rate, student_cold = rate_of(student_runs[content_type])
            corpus_rows = int(student_runs[content_type].get("rows") or 0)
            if not (teacher_rate and student_rate and corpus_rows):
                continue
            rows.append({
                "content_type": content_type,
                "rows": corpus_rows,
                "teacher rows/s": round(teacher_rate, 2),
                "student rows/s": round(student_rate, 2),
                "teacher hours": corpus_rows / teacher_rate / 3600,
                "student hours": corpus_rows / student_rate / 3600,
                "warm-up in the rate": teacher_cold or student_cold,
            })
        if not rows:
            continue
        frame = pd.DataFrame(rows).set_index("content_type")
        frame.loc["total"] = {
            "rows": frame["rows"].sum(),
            "teacher rows/s": round(frame["rows"].sum()
                                    / frame["teacher hours"].sum() / 3600, 2),
            "student rows/s": round(frame["rows"].sum()
                                    / frame["student hours"].sum() / 3600, 2),
            "teacher hours": frame["teacher hours"].sum(),
            "student hours": frame["student hours"].sum(),
            "warm-up in the rate": bool(frame["warm-up in the rate"].any()),
        }
        frame["rows"] = frame["rows"].astype(int)
        frame.attrs["granularity"] = one
        frame.attrs["speedup"] = (frame.loc["total", "teacher hours"]
                                  / frame.loc["total", "student hours"])
        return frame.round(2)

    raise SystemExit(
        "no granularity has BOTH a teacher manifest and a corpus manifest, so there are no two "
        "rates to compare. Run `teacher` and `corpus` on the same granularity first.")


def headline(run: str | Path) -> Any:
    """The one table a reader is shown: every book, raw and smoothed, against the floor.

    `comparison.csv` carries a row per book and encodes the holding period in the name --
    `chunk.combined` beside `chunk.combined.h1`. That is the right shape on disk, where a book
    is a folder, and the wrong one to read: the comparison being made is raw against smoothed,
    and it reads as a column, not as a suffix.

    The floor is reported RAW even when the books are smoothed, and that is deliberate rather
    than an oversight. The sentiment floor already fires on nearly every day it could -- it
    reads a sentiment score, not a sparse label -- so it starts near fully deployed and
    smoothing only decays it. Comparing a smoothed book against a smoothed floor would lower
    the bar; the raw floor is the harder one to clear.
    """
    import pandas as pd

    paths = RunPaths.for_run(Path(run))
    frame = pd.read_csv(paths.backtest_comparison)
    holding = frame["book"].str.extract(r"\.h(\d+)$")[0]
    frame["signal"] = holding.map(lambda h: f"smoothed_h{h}" if pd.notna(h) else "raw")
    frame["book"] = frame["book"].str.replace(r"\.h\d+$", "", regex=True)

    columns = {"information_ratio": "ir", "annualized_return": "ann_return",
               "volatility": "volatility", "gross_exposure": "gross_exposure",
               "net_exposure": "net_exposure"}
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise SystemExit(
            f"{paths.backtest_comparison.name} has no {missing} — the books were skipped for "
            f"want of prices, so there is no result to table.")
    out = frame[["book", "signal", *columns]].rename(columns=columns)

    floor = out["book"] == "sentiment_only"
    out = out[~floor | (floor & (out["signal"] == "raw"))]
    # smoothed first, then the raw books, then the floor: the order the reader needs them in
    rank = out["signal"].ne("raw").map({True: 0, False: 1}) + floor[out.index].map(
        {True: 1, False: 0})
    return (out.assign(_rank=rank).sort_values(["_rank", "book"])
               .drop(columns="_rank").reset_index(drop=True).round(4))


def cumulative(run: str | Path) -> Any:
    """One column per book, cumulative return from 0, for the chart the headline table sits on.

    The per-book `performance.parquet` holds a daily log return; this compounds it. Starting at
    0 rather than at 1 is the convention the published series uses, and it reads better: the
    number at any date is what the book has made since inception.

    Columns are named the way a legend needs them -- `Chunk-level (1-day smoothed)` beside
    `Chunk-level (raw, for reference)` -- because the comparison is between a book and its own
    raw self, and a reader should not have to decode a `.h1` suffix to see that.

    The smoothed floor is left out, as in `headline`: it reads a sentiment score rather than a
    sparse label, so it already runs nearly fully deployed and smoothing only decays it.
    """
    import numpy as np
    import pandas as pd

    paths = RunPaths.for_run(Path(run))
    root = paths.backtest_dir()
    series: dict[str, Any] = {}
    for path in sorted(root.rglob("performance.parquet")):
        name = str(path.parent.relative_to(root))
        base, _, holding = name.partition(".h")
        floor = base.endswith("sentiment_only")
        if floor and holding:
            continue
        if floor:
            label = "Sentiment only (floor)"
        else:
            granularity = base.split(".")[0].capitalize()
            label = (f"{granularity}-level ({holding}-day smoothed)" if holding
                     else f"{granularity}-level (raw, for reference)")
        frame = pd.read_parquet(path, columns=["date", "portfolio_return"]).sort_values("date")
        series[label] = pd.Series(
            np.exp(frame["portfolio_return"].cumsum()).to_numpy() - 1.0,
            index=pd.to_datetime(frame["date"]).to_numpy())
    if not series:
        raise SystemExit(
            f"no performance.parquet under {root} — the books were skipped for want of prices, "
            f"so there is no series to chart.")
    order = sorted(series, key=lambda name: ("smoothed" not in name, "floor" in name, name))
    return pd.DataFrame(series)[order].sort_index()
