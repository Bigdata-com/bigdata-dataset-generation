"""`distill <stage>` -- the whole pipeline, one subcommand per stage."""

from __future__ import annotations

import argparse
import shlex
import sys
from datetime import date
from pathlib import Path

from .config import DistillConfig, search_end
from .paths import (
    CONFIG_DIR_NAME, DISTILL_CONFIG_NAME, EXTRACTION_CONFIG_NAME,
    PART_A_SIMILARITY, RunPaths, default_output_dir,
)
from .theme import GRANULARITIES, ThemeSpec

DRY_RUN_HELP = ("make no calls and write nothing outside _dry_run/ -- a dry run cannot name a "
                "real artifact")


def _run_dir(args: argparse.Namespace) -> Path:
    """The run folder every command works against."""
    explicit = getattr(args, "config", None)
    if explicit:
        path = Path(explicit).resolve()
        if not path.exists():
            raise SystemExit(f"--config {path} does not exist")
        return path.parent.parent if path.parent.name == CONFIG_DIR_NAME else path.parent
    run = getattr(args, "run", None) or getattr(args, "run_dir", None)
    if run:
        path = Path(run).resolve()
        if not path.is_dir():
            raise SystemExit(f"--run {path} is not a directory")
        return path
    return default_output_dir()


def _config_path(run_dir: Path, args: argparse.Namespace) -> Path:
    explicit = getattr(args, "config", None)
    if explicit:
        return Path(explicit).resolve()
    candidate = RunPaths.for_run(run_dir).distill_config
    if candidate.exists():
        return candidate
    raise SystemExit(
        f"{candidate} not found. The pipeline reads two checked-in files from "
        f"{CONFIG_DIR_NAME}/:\n"
        f"  {EXTRACTION_CONFIG_NAME}   THE THEME, the universe, the search\n"
        f"  {DISTILL_CONFIG_NAME}      labelling, distillation, fine-tuning"
    )


def _selected_granularities(config, chosen: str | None, label: str) -> tuple[str, ...]:
    """The granularities a command should work on, narrowed by `--granularity` if given.

    With `granularity: both`, half the run can already be finished -- and re-running the
    finished half is not idempotent: the trainer now refuses a used output directory rather
    than continuing an old run under a new name.
    """
    covered = config.granularities()
    if not chosen:
        return covered
    if chosen not in covered:
        raise SystemExit(
            f"--granularity {chosen!r} is not one this run covers ({list(covered)}). "
            f"experiment.granularity is {config.experiment.granularity!r}."
        )
    print(f"--granularity {chosen}: {label} on that one only, of {list(covered)}")
    return (chosen,)


def _announce(covered: tuple[str, ...], label: str):
    """Yield each granularity, announcing it when there is more than one."""
    for index, one in enumerate(covered):
        if len(covered) > 1:
            print("\n" + "#" * 100)
            print(f"# {label.upper()}, GRANULARITY {one.upper()}  "
                  f"({index + 1} of {len(covered)})")
            print("#" * 100)
        yield one


def load(args: argparse.Namespace) -> tuple[DistillConfig, ThemeSpec, RunPaths]:
    run_dir = _run_dir(args)
    paths = RunPaths.for_run(run_dir, dry_run=bool(getattr(args, "dry_run", False)))
    config_path = _config_path(run_dir, args)
    config = DistillConfig.load_run(paths)
    if getattr(args, "price_cache", None):
        config.backtest.price_cache = str(args.price_cache)

    from .preflight import check

    stage = getattr(args, "stage_name", "config")
    chosen = getattr(args, "granularity", None)
    flight, theme = check(config, paths, stage=stage,
                          dry_run=bool(getattr(args, "dry_run", False)),
                          granularities=(chosen,) if chosen else None)
    for warning in flight.warnings:
        print(f"  WARNING  {warning}")
    flight.require()
    assert theme is not None
    print(f"config : {config_path}")
    print(f"theme  : {theme.id} ({theme.title})   definition {theme.fingerprint()}")
    print(paths.describe())
    if hasattr(args, "granularity"):
        print(f"experiment: {chosen or config.granularities()[0]}."
              f"{config.experiment.topology}")
    print()
    return config, theme, paths


def _themes_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "themes"


def cmd_check_config(args: argparse.Namespace) -> int:
    """Read the three config files, and say what each stage is still waiting for.

    Quiet on success on purpose: a stage that is ready has nothing to tell you, and a wall of
    `ready` lines is what made the blocked one easy to miss.
    """
    from .preflight import check_every_stage
    from .signals import require_signs
    from .stages.teacher import build_prompts

    config, theme, paths = load(args)

    missing = [p.name for p in (paths.extraction_config, paths.distill_config)
               if not p.exists()]
    classes = ", ".join(require_signs(theme))
    print("config valid: both files present, classes " + classes if not missing
          else "MISSING: " + ", ".join(missing))
    print("prompts " + ", ".join(
        f"{g} {build_prompts(config, theme, g).fingerprint()}"
        for g in config.granularities()))

    flights = check_every_stage(config, paths)
    blocked = {stage: flight for stage, flight in flights.items() if not flight.passed}
    root = str(paths.config_dir.parent) + "/"

    config_flight = flights.get("config")
    if config_flight is not None and not config_flight.passed:
        for issue in config_flight.errors:
            print(f"invalid   {issue.replace(root, '')}")
        raise SystemExit("\nthe configuration itself is invalid — nothing will run until it "
                         "is fixed")

    first, flight = next(iter(blocked.items()), (None, None))
    expected = {issue for issue in (flight.errors if flight else ())}

    for warning in (config_flight.warnings if config_flight else ()):
        if warning not in expected:
            print(f"warning   {warning.replace(root, '')}")

    if first is None:
        print("\nevery stage ready")
        return 0
    print(f"\n{len(blocked)} of {len(flights)} stages have not run yet, starting with "
          f"`{first}`:")
    for issue in flight.errors:
        print(f"  - {issue.replace(root, '')}")
    return 0


def cmd_show_prompts(args: argparse.Namespace) -> int:
    """Every system message the four combinations will send, before a call goes out."""
    if args.theme:
        theme = ThemeSpec.resolve(args.theme, themes_dir=_themes_dir()).require_valid()
        granularities = (args.granularity,) if args.granularity else ("chunk", "sentence")
    else:
        config, theme, _ = load(args)
        granularities = (args.granularity,) if args.granularity else config.granularities()
    from .prompts import PromptSet
    for granularity in granularities:
        prompts = PromptSet(theme=theme, granularity=granularity)
        print(prompts.describe())
        print()
    return 0


def cmd_subset(args: argparse.Namespace) -> int:
    from .stages import subset
    config, theme, paths = load(args)
    subset.run(config, theme, paths, force=args.force, top_fraction=args.top_fraction)
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    from .stages import sample
    config, theme, paths = load(args)
    sample.run(config, theme, paths, force=args.force, limit=args.limit,
               max_rows=args.max_rows)
    return 0


def cmd_teacher(args: argparse.Namespace) -> int:
    from .stages import teacher
    config, theme, paths = load(args)
    teacher.run(config, theme, paths, dry_run=args.dry_run, assume_yes=args.yes,
                retry_errors=args.retry_errors, force=args.force, limit=args.limit,
                granularities=_selected_granularities(config, args.granularity, "label"))
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    from .stages import teacher
    config, theme, paths = load(args)
    teacher.run_split(config, theme, paths, new_split=args.new_split,
                      granularities=_selected_granularities(config, args.granularity, "split"))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .stages import export
    config, theme, paths = load(args)
    for granularity in _announce(_selected_granularities(config, args.granularity, "export"),
                                 "export"):
        export.run(config, theme, paths, granularity=granularity, force=args.force)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .stages import evaluate
    config, theme, paths = load(args)
    for granularity in _selected_granularities(config, args.granularity, "evaluate"):
        evaluate.run(config, theme, paths, granularity=granularity, dataset=args.dataset,
                     epoch=args.epoch, role=args.role, model_label=args.model_label,
                     prompt_style=args.prompt_style, train_jsonl=args.train_jsonl,
                     no_stub_check=args.no_stub_check,
                     expect_fingerprint=args.expect_fingerprint,
                     allow_fingerprint_mismatch=args.allow_fingerprint_mismatch,
                     dry_run=args.dry_run, limit=args.limit, content_type=args.content_type)
    return 0


def cmd_corpus(args: argparse.Namespace) -> int:
    from .stages import corpus
    config, theme, paths = load(args)
    for granularity in _selected_granularities(config, args.granularity, "corpus"):
        corpus.run(config, theme, paths, granularity=granularity,
                   content_type=args.content_type, dataset=args.dataset,
                   dry_run=args.dry_run, assume_yes=args.yes, retry_errors=args.retry_errors,
                   no_stub_check=args.no_stub_check,
                   allow_fingerprint_mismatch=args.allow_fingerprint_mismatch,
                   limit=args.limit, assemble_only=args.assemble_only)
    return 0


def cmd_ic(args: argparse.Namespace) -> int:
    """Information Coefficient across every coverage bin this run can reach."""
    from .stages import ic

    config, theme, paths = load(args)
    if getattr(args, "price_cache", None):
        config.backtest.price_cache = str(args.price_cache)
    if args.end is not None:
        config.backtest.end = str(args.end)
    ic.run(config, theme, paths, force=args.force, smooth_bins=args.smooth_bins,
           granularities=_selected_granularities(config, args.granularity, "ic"))
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    from .stages import backtest
    config, theme, paths = load(args)
    if args.half_life is not None:
        config.backtest.half_life = args.half_life
    if args.end is not None:
        config.backtest.end = str(args.end)
    elif not config.backtest.end:
        # No end given: the backtest stops where the search stops.
        config.backtest.end = search_end(paths)
    backtest.run(config, theme, paths, only=tuple(args.only or ()), force=args.force)
    return 0


def _train_all(config, theme, paths, *, print_only: bool | None, skip_training: bool,
               wait_minutes: int, datasets: tuple, epochs: tuple, run_dir,
               covered: tuple[str, ...]) -> int:
    """Place the trainings on the cards the way `training.gpus` says, and say what it did.

    One code path for `train` and `train-cmd`, so the plan you read is the plan that runs.
    """
    from .stages import train

    mode = config.training.gpus
    common = dict(datasets=datasets, epochs=epochs, print_only=print_only,
                  skip_training=skip_training)

    if mode == "per_granularity" and len(covered) > 1 and not skip_training:
        cards = list(config.training.devices)
        print(f"training.gpus = per_granularity: {len(covered)} trainings, one per card "
              f"{cards[:len(covered)]}, launched together")
        for granularity, card in zip(covered, cards):
            print("\n" + "#" * 100)
            print(f"# {granularity.upper()} ON CARD {card}")
            print("#" * 100)
            train.run(config, theme, paths, granularity=granularity, device=card,
                      wait_minutes=0, **common)
        print("\n" + "=" * 72)
        print("Both trainers run at once, one per card. When they finish, the same command "
              "merges and scores them:")
        print(f"  distill --run {run_dir} train --skip-training")
        print("=" * 72)
        return 0

    for index, granularity in enumerate(covered):
        if len(covered) > 1:
            print("\n" + "#" * 100)
            print(f"# TRAIN, GRANULARITY {granularity.upper()}  "
                  f"({index + 1} of {len(covered)})")
            print("#" * 100)
        result = train.run(config, theme, paths, granularity=granularity,
                           wait_minutes=wait_minutes, **common)
        if result.get("_trainer_running") and len(covered) > 1:
            print("\n" + "=" * 72)
            print(f"STOPPING HERE. The {granularity} trainer is running detached, and "
                  f"training.gpus is {mode!r}.")
            print("When it finishes, the same command picks up where this left off:")
            print(f"  distill --run {run_dir} train")
            print("Or block until it is done:  ... train --wait 720")
            print("Or set training.gpus: per_granularity to run the two on separate cards at "
                  "once.")
            print("=" * 72)
            break
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train, then merge / du -sh / serve / score / tear down for every checkpoint named."""
    config, theme, paths = load(args)
    return _train_all(config, theme, paths,
                      print_only=(True if args.print_only
                                  else (False if args.execute else None)),
                      skip_training=args.skip_training, wait_minutes=args.wait,
                      datasets=tuple(args.dataset or ()), epochs=tuple(args.epoch or ()),
                      run_dir=paths.run_dir,
                      covered=_selected_granularities(config, args.granularity, "train"))


def cmd_serve_cmd(args: argparse.Namespace) -> int:
    """Every model this role needs, with the whole sequence to bring it up."""
    from . import serving
    config, _, paths = load(args)
    serving.print_recipe(config, args.role, run_dir=paths.run_dir)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Bring one role's model up and wait until it answers. Idempotent: reuses a healthy one."""
    from . import serving
    config, _, _ = load(args)
    serving.serve_model(config, args.role, content_type=args.content_type or "",
                        reuse=not args.restart)
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    """Take one role's container down, and report what the cards are holding afterwards."""
    from . import serving
    config, _, _ = load(args)
    serving.stop_model(config, args.role, content_type=args.content_type or "")
    return 0


def cmd_use_checkpoint(args: argparse.Namespace) -> int:
    """Point the student at a chosen checkpoint: the three fields that must agree."""
    from . import driver
    driver.point_student_at(_run_dir(args), args.granularity, args.epoch,
                            dataset=args.dataset, apply=not args.dry_run)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="distill", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run", default=None,
                        help="where the pipeline writes. Default: output/ beside config/")
    parser.add_argument("--config", default=None,
                        help="an explicit distill.yaml, for one kept outside the run")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler, help_text: str,
            stage: str = "config") -> argparse.ArgumentParser:
        child = sub.add_parser(name, help=help_text, description=handler.__doc__)
        child.set_defaults(handler=handler, stage_name=stage)
        return child

    add("check-config", cmd_check_config,
        "read the three config files and report what each stage is waiting for")

    prompts = add("show-prompts", cmd_show_prompts, "print every prompt before spending anything")
    prompts.add_argument("--theme", default=None, help="a theme id or path, instead of a config")
    prompts.add_argument("--granularity", choices=("chunk", "sentence"), default=None)

    stage = add("subset", cmd_subset, "keep the most relevant rows of the corpus", "subset")
    stage.add_argument("--force", action="store_true")
    stage.add_argument("--top-fraction", type=float, default=None,
                       help="how much of the corpus to keep, per content type")
    stage.add_argument("--dry-run", action="store_true", help=DRY_RUN_HELP)

    stage = add("sample", cmd_sample, "draw the rows the teacher will label", "sample")
    stage.add_argument("--force", action="store_true")
    stage.add_argument("--limit", type=int, default=None)
    stage.add_argument("--max-rows", type=int, default=None,
                       help="ceiling on the sample, per content type")
    stage.add_argument("--dry-run", action="store_true", help=DRY_RUN_HELP)

    stage = add("teacher", cmd_teacher, "the teacher labels the sample",
                "teacher")
    stage.add_argument("--dry-run", action="store_true", help=DRY_RUN_HELP)
    stage.add_argument("--yes", action="store_true", help="skip the pause after the smoke test")
    stage.add_argument("--retry-errors", action="store_true",
                       help="re-attempt rows whose checkpoint record is an error")
    stage.add_argument("--force", action="store_true")
    stage.add_argument("--limit", type=int, default=None)
    stage.add_argument("--granularity", choices=GRANULARITIES, default=None,
                       help="run this one only, of the ones the config covers")


    stage = add("split", cmd_split, "the train/held-out split -- written once, never regenerated",
                "split")
    stage.add_argument("--new-split", action="store_true",
                       help="refused while a split is on disk; every measured number hangs off it")
    stage.add_argument("--dry-run", action="store_true", help=DRY_RUN_HELP)
    stage.add_argument("--granularity", choices=GRANULARITIES, default=None,
                       help="run this one only, of the ones the config covers")


    stage = add("export", cmd_export, "the training JSONL, and the training commands",
                "export")
    stage.add_argument("--force", action="store_true")
    stage.add_argument("--dry-run", action="store_true", help=DRY_RUN_HELP)
    stage.add_argument("--granularity", choices=GRANULARITIES, default=None,
                       help="run this one only, of the ones the config covers")


    stage = add("evaluate", cmd_evaluate, "score one checkpoint on the held-out",
                "evaluate")
    stage.add_argument("--dataset", default="both",
                       help="which half the model was trained on: news, transcripts or both")
    stage.add_argument("--epoch", type=int, default=None, help="names the checkpoint in the tag")
    stage.add_argument("--granularity", choices=GRANULARITIES, default=None,
                       help="which ruler to score against; default: every one the run covers")
    stage.add_argument("--role", choices=("student", "teacher"), default="student")
    stage.add_argument("--model-label", default=None,
                       help="extra identity for the tag, e.g. a hosted model's name")
    stage.add_argument("--prompt-style", choices=("stub", "full"), default=None,
                       help="default: stub for a local student, full for a hosted model")
    stage.add_argument("--train-jsonl", default=None,
                       help="the file the student was really trained on, for the byte check")
    stage.add_argument("--no-stub-check", action="store_true")
    stage.add_argument("--expect-fingerprint", default=None)
    stage.add_argument("--allow-fingerprint-mismatch", action="store_true")
    stage.add_argument("--dry-run", action="store_true")
    stage.add_argument("--limit", type=int, default=None)
    stage.add_argument("--content-type", default=None)

    stage = add("corpus", cmd_corpus, "the student labels the whole subset",
                "corpus")
    stage.add_argument("--granularity", choices=GRANULARITIES, default=None,
                       help="label with ONE granularity's model; default: every one the run "
                            "covers -- but they need different models served, so one at a time")
    stage.add_argument("--content-type", default=None,
                       help="label one half only (the specialist topology, one server at a time)")
    stage.add_argument("--dataset", default=None,
                       help="which training file to check the fingerprint against")
    stage.add_argument("--assemble-only", action="store_true",
                       help="make no calls: rebuild the parquet from the checkpoints")
    stage.add_argument("--dry-run", action="store_true")
    stage.add_argument("--yes", action="store_true")
    stage.add_argument("--retry-errors", action="store_true")
    stage.add_argument("--no-stub-check", action="store_true")
    stage.add_argument("--allow-fingerprint-mismatch", action="store_true")
    stage.add_argument("--limit", type=int, default=None)

    stage = add("ic", cmd_ic, "is the signal predictive -- IC across the coverage bins", "ic")
    stage.add_argument("--granularity", choices=GRANULARITIES, default=None,
                       help="one granularity only; default: every one the run covers")
    stage.add_argument("--price-cache", default=None,
                       help="directory holding the price cache")
    stage.add_argument("--end", type=date.fromisoformat, default=None,
                       help="last signal date to use, YYYY-MM-DD; overrides backtest.end")
    stage.add_argument("--smooth-bins", type=int, default=3,
                       help="how many coverage bins get the holding-period sweep; each one "
                            "costs seven backtests per granularity, on CPU")
    stage.add_argument("--force", action="store_true")

    stage = add("backtest", cmd_backtest, "signals, both benchmarks, the comparison",
                "backtest")
    stage.add_argument("--price-cache", default=None,
                       help="directory holding the price cache")
    stage.add_argument("--end", type=date.fromisoformat, default=None,
                       help="last signal date to use, YYYY-MM-DD; overrides backtest.end")
    stage.add_argument("--half-life", type=int, default=None,
                       help="calendar days a label keeps half its weight. 0 (the default) is "
                            "the raw signal, firing for one day. Applied to every labelled "
                            "book, beside its raw self; the sentiment floor stays raw.")
    stage.add_argument("--only", nargs="*", default=None, help="label sets to include")
    stage.add_argument("--force", action="store_true")
    stage.add_argument("--dry-run", action="store_true", help=DRY_RUN_HELP)

    stage = add("train", cmd_train,
                "train, then merge / du -sh / serve / score / tear down per "
                "checkpoint", "train")
    stage.add_argument("--dataset", action="append", default=None,
                       help="which halves to train (repeatable); default: what the topology needs")
    stage.add_argument("--epoch", action="extend", nargs="+", type=int, default=None,
                       help="which checkpoints to score, e.g. --epoch 3 5; default: "
                            "training.evaluate_epochs")
    stage.add_argument("--skip-training", action="store_true",
                       help="the adapter already exists: go straight to merge/serve/score")
    stage.add_argument("--wait", type=int, default=0,
                       help="block up to N minutes for the detached trainer to finish")
    stage.add_argument("--granularity", choices=GRANULARITIES, default=None,
                       help="work on one granularity only; default: every one the run covers")
    stage.add_argument("--print-only", action="store_true", help="print the sequence, run nothing")
    stage.add_argument("--execute", action="store_true",
                       help="run it even though docker was not detected")
    stage.add_argument("--dry-run", action="store_true", help=DRY_RUN_HELP)

    serve = add("serve-cmd", cmd_serve_cmd, "print one model's docker recipe from the config")
    serve.add_argument("role", choices=("teacher", "student"))

    up = add("serve", cmd_serve, "bring one role's model up and wait until it answers", "config")
    up.add_argument("role", choices=("teacher", "student"))
    up.add_argument("--content-type", default=None,
                    help="for a specialist student: which of the two models")
    up.add_argument("--restart", action="store_true",
                    help="replace a healthy container instead of reusing it (a 120B reload)")

    down = add("stop", cmd_stop, "take one role's container down", "config")
    down.add_argument("role", choices=("teacher", "student"))
    down.add_argument("--content-type", default=None)

    use = add("use-checkpoint", cmd_use_checkpoint,
              "point the student at a chosen checkpoint, all three fields at once", "config")
    use.add_argument("--granularity", choices=GRANULARITIES, required=True)
    use.add_argument("--epoch", type=int, required=True)
    use.add_argument("--dataset", default="both",
                     help="which half's student, under a specialist topology")
    use.add_argument("--dry-run", action="store_true",
                     help="print what it would set and write nothing")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted -- the checkpoint holds everything answered so far; the same "
              "command resumes and re-labels nothing")
        return 130

if __name__ == "__main__":
    sys.exit(main())
