"""Serving and training recipes, rendered from the config instead of copied by hand."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

from .config import DistillConfig, ServeConfig, TrainingConfig

MIN_MERGED_BYTES = 1 * 1024 ** 3

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def resolve_chat_template(serve: ServeConfig, training: TrainingConfig) -> ServeConfig:
    """Fill in a null `chat_template` with the package's own copy."""
    if serve.chat_template:
        return serve
    from dataclasses import replace as dc_replace

    return dc_replace(serve, chat_template=script_path(
        training, training.chat_template, package_root=PACKAGE_ROOT))


def _volume(spec: str) -> str:
    """`models:/models` — a relative host path is resolved against the repository root.

    Docker needs an absolute host path. Keeping the config relative is what lets `models/`,
    `config/` and `output/` be three siblings you can move together.

    A spec with NO colon is mounted at its own absolute path, host and container alike: `.`
    puts the repository at the same path inside. That is not a convenience -- `model_path`
    and `--chat-template` are absolute HOST paths, so a student whose merged checkpoint lives
    under `output/` cannot be served unless that exact path resolves inside the container.
    It fails as a health gate that never goes green, with the reason inside the container.
    """
    from .paths import config_dir
    host, sep, rest = spec.partition(":")
    if not host.startswith(("/", "~", "$")):
        host = str((config_dir().parent / host).resolve())
    if not sep:
        return f"{host}:{host}"
    return f"{host}{sep}{rest}"


def _env_value(value: str) -> str:
    """Expand `${VAR}` against the environment, so a secret stays out of the config file."""
    return os.path.expandvars(str(value))


def render_serve_command(serve: ServeConfig) -> str:
    """The `docker run` line for one model, as a copy-pasteable string."""
    if not serve.model_path:
        raise SystemExit("serve.model_path is empty: there is nothing to serve")
    parts = ["docker", "run"]
    if serve.detach:
        parts.append("-d")
    if serve.container:
        parts += ["--name", serve.container]
    parts += ["--gpus", serve.gpus if serve.gpus == "all" else f'"{serve.gpus}"']
    for volume in serve.volumes or ():
        parts += ["-v", shlex.quote(_volume(volume))]
    parts += ["--ipc=host", "-p", f"{serve.port}:{serve.port}"]
    for key, value in (serve.env or {}).items():
        parts += ["-e", f"{key}={_env_value(value)}"]
    parts.append(serve.image)
    parts += ["--model", shlex.quote(serve.model_path)]
    if serve.revision:
        parts += ["--revision", shlex.quote(serve.revision),
                  "--tokenizer-revision", shlex.quote(serve.revision)]
    if serve.served_model_name:
        parts += ["--served-model-name", shlex.quote(serve.served_model_name)]
    if serve.chat_template:
        parts += ["--chat-template", shlex.quote(serve.chat_template)]
    parts += [str(a) for a in (serve.args or ())]
    parts += ["--port", str(serve.port)]
    return " ".join(parts)


def data_parallel_size(serve: ServeConfig) -> int | None:
    """The `--data-parallel-size` in the rendered args, or None if it is not set."""
    args = [str(a) for a in (serve.args or ())]
    for index, arg in enumerate(args):
        if arg == "--data-parallel-size" and index + 1 < len(args):
            try:
                return int(args[index + 1])
            except ValueError:
                return None
        if arg.startswith("--data-parallel-size="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                return None
    return None


def single_card(serve: ServeConfig) -> str | None:
    """The one card this recipe pins itself to (`device=N`), or None for all of them."""
    gpus = (serve.gpus or "").strip()
    if gpus.startswith("device="):
        devices = gpus.split("=", 1)[1].split(",")
        return devices[0] if len(devices) == 1 else None
    return None


def card_warnings(serve: ServeConfig, *, role: str, cards: int = 2) -> list[str]:
    """Whether this recipe will actually use the machine it is being run on."""
    warnings: list[str] = []
    pinned = single_card(serve)
    parallel = data_parallel_size(serve)
    if pinned is not None:
        if parallel and parallel > 1:
            warnings.append(
                f"gpus is pinned to device={pinned} but --data-parallel-size is {parallel}: "
                f"{parallel} replicas cannot fit on one card. Either widen gpus to all, or "
                f"drop the flag (which is right for a specialist -- one model per card)."
            )
        return warnings
    if parallel is None:
        warnings.append(
            f"no --data-parallel-size, so this serves ONE replica and card 1 stays idle at 0% "
            f"-- the pass will take about twice as long as the measured rate, and nothing will "
            f"say so. Add `--data-parallel-size {cards}` unless you are deliberately keeping a "
            f"card free (a trainer, or the other specialist)."
        )
    elif parallel == 1:
        warnings.append(
            f"--data-parallel-size is 1: one replica, one card, the other idle. Set it to "
            f"{cards} for a labelling pass, or pin `gpus: \"device=N\"` if leaving a card "
            f"free is deliberate."
        )
    elif parallel > cards:
        warnings.append(f"--data-parallel-size is {parallel} on a {cards}-card machine.")
    return warnings


def render_sequence(serve: ServeConfig, *, role: str, label: str = "",
                    then: str = "") -> list[str]:
    """The whole ordered sequence for one model, as lines to run."""
    name = serve.container or f"tx-{role}"
    port = serve.port
    what = f" ({label})" if label else ""
    lines = [
        f"# 1. tear down anything already holding this name, this port and this memory",
        f"docker rm -f {shlex.quote(name)} 2>/dev/null || true",
        f"",
        f"# 2. start it{what} -- detached, so a dropped SSH session does not take it with it",
        render_serve_command(serve),
        f"",
        f"# 3. wait for it to be ready ({'a 120B takes minutes' if role == 'teacher' else 'a 4B is quick, but not instant'}); do not race it",
        f"until curl -sf http://localhost:{port}/v1/models >/dev/null; do sleep 5; done; "
        f"echo ready",
        f"",
    ]
    pinned = single_card(serve)
    parallel = data_parallel_size(serve)
    if pinned is not None:
        lines += [
            f"# 4. confirm it took card {pinned}, and ONLY that card",
            f"nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv",
            f"",
        ]
    else:
        expected = parallel or 1
        lines += [
            f"# 4. confirm BOTH cards are loaded. {expected} replica(s) expected -- if only one "
            f"card shows memory in use, the pass will run at half the measured rate and "
            f"nothing else will tell you",
            f"nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv",
            f"",
        ]
    if then:
        lines += ["# 5. label. This is the long part", then, ""]
    lines += [
        f"# {'6' if then else '5'}. give the cards back",
        f"docker rm -f {shlex.quote(name)}",
    ]
    return lines


def serve_warnings(serve: ServeConfig, *, role: str,
                   other_containers: tuple[str, ...] = (),
                   settings_model: str = "") -> list[str]:
    warnings: list[str] = []
    if serve.container and serve.container in other_containers:
        warnings.append(
            f"container name {serve.container!r} is used by more than one role. Starting the "
            f"second one KILLS the first: the recipes begin with `docker rm -f <name>`, which "
            f"is how a second manual teacher launch once killed the run in progress."
        )
    if not serve.detach:
        warnings.append("serve.detach is false: without -d a dropped session takes the container "
                        "with it. One labelling run and two containers were lost this way.")
    if "--rm" in (serve.args or ()):
        warnings.append("--rm is in serve.args. Remove it: a container that dies at startup has to "
                        "leave logs behind.")
    if settings_model:
        served = serve.served_model_name or serve.model_path
        if served and served != settings_model:
            how = ("serve.served_model_name" if serve.served_model_name
                   else "serve.model_path, because served_model_name is empty and vLLM then "
                        "answers under the path")
            warnings.append(
                f"{role}.model is {settings_model!r} but the server will call it {served!r} "
                f"({how}). The first is what the CLIENT asks for and the second is what the "
                f"SERVER calls it: they have to be the same string."
            )
    if role == "student" and not serve.chat_template:
        warnings.append("no chat_template for the student. A merged model carries the base model's "
                        "template, whose default is reasoning-on -- you will get a reasoning trace "
                        "instead of the JSON the student was trained to emit.")
    if not any(str(a).startswith("--max-num-seqs") for a in (serve.args or ())):
        warnings.append("no --max-num-seqs. On a hybrid Mamba teacher this is a hard cap (~165 MiB "
                        "of SSM state per sequence, and it does not shrink with context); vLLM "
                        "would default to 1024 and reserve ~165 GiB of state alone.")
    return warnings


def directory_size(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def human(size: int) -> str:
    value = float(size)
    for unit in ("B", "K", "M", "G", "T"):
        if value < 1024 or unit == "T":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}T"


def check_merged_model(path: str | Path, *, minimum: int = MIN_MERGED_BYTES) -> int:
    """`du -sh` before serving."""
    path = Path(path)
    size = directory_size(path)
    if not path.exists():
        raise SystemExit(
            f"merged model not found: {path}\nMerge the adapter first; `distill train-cmd` prints "
            f"the command."
        )
    if size < minimum:
        raise SystemExit(
            f"merged model at {path} is only {human(size)} ({size:,} bytes).\nThe merge script "
            f"creates the output directory BEFORE it reads the adapter, so an aborted merge looks "
            f"like a success. Re-merge and check `du -sh` before serving -- an empty merge was "
            f"served once and scored as a model."
        )
    return size


def script_path(training: TrainingConfig, script: str, *, package_root: Path) -> str:
    """Resolve one of the package's own scripts to a path visible INSIDE the container."""
    candidate = Path(script)
    resolved = candidate if candidate.is_absolute() else (package_root / candidate)
    if training.container_mount:
        return str(Path(training.container_mount) / candidate)
    return str(resolved)


def render_train_command(training: TrainingConfig, *, dataset_path: Path, output_dir: Path,
                         package_root: Path | None = None, log_path: Path | None = None,
                         detached: bool = True, device: int | None = None) -> str:
    """The trainer, launched DETACHED inside the container.

    `training.gpus` decides how the cards are used. `ddp` puts torchrun in front so one
    training spans both; `per_granularity` pins this one to `device` so another can have the
    other card. Both are the user's choice, because which is faster depends on how many
    datasets there are to place.
    """
    root = package_root or Path(__file__).resolve().parent.parent
    script = shlex.quote(script_path(training, training.trainer_script, package_root=root))
    if training.gpus == "ddp":
        launcher = ["python3", "-m", "torch.distributed.run",
                    f"--nproc_per_node={len(training.devices)}", script]
    else:
        launcher = ["python3", script]
    inner = " ".join([
        *launcher,
        "--dataset", shlex.quote(str(dataset_path)),
        "--output-dir", shlex.quote(str(output_dir)),
        "--model", shlex.quote(training.base_model),
        *(["--revision", shlex.quote(training.base_revision)] if training.base_revision else []),
        "--epochs", str(training.epochs),
        "--lora-r", str(training.lora_r),
        "--batch-size", str(training.batch_size),
        "--lr", str(training.learning_rate),
        "--max-seq-length", str(training.max_seq_length),
        "--val-ratio", str(training.val_ratio),
        "--no-resume",
        *(["--grad-checkpointing",
           "on" if training.gradient_checkpointing else "off"]
          if training.gradient_checkpointing is not None else []),
    ])
    inner = f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {inner}"
    if device is not None and training.gpus != "ddp":
        inner = f"CUDA_VISIBLE_DEVICES={device} {inner}"
    if not detached:
        return f"docker exec {training.container} {inner}"
    if log_path is not None:
        inner += f" >> {shlex.quote(str(log_path))} 2>&1"
    return f'docker exec -d {training.container} bash -c {shlex.quote(inner)}'


def render_train_checks(training: TrainingConfig) -> list[str]:
    """What to run after launching a trainer, and how to stop it."""
    return [
        f"docker exec {training.container} ps aux | grep -c '[t]rain_lora_sft'"
        f"   # 6: wrapper + main + 4 dataloader workers",
        "nvidia-smi --query-gpu=index,memory.used --format=csv"
        "   # ~15 GB for ONE trainer; a multiple of that means several",
        f"docker exec {training.container} pkill -f train_lora_sft.py"
        f"   # the only way to actually stop it",
    ]


def render_merge_command(training: TrainingConfig, *, adapter: Path, output: Path,
                         package_root: Path | None = None, force: bool = False) -> str:
    root = package_root or Path(__file__).resolve().parent.parent
    return " ".join([
        "docker", "exec", training.container, "python3",
        shlex.quote(script_path(training, training.merge_script, package_root=root)),
        "--adapter", shlex.quote(str(adapter)),
        "--out", shlex.quote(str(output)),
        "--base", shlex.quote(training.base_model),
        *(["--revision", shlex.quote(training.base_revision)] if training.base_revision else []),
        *(["--force"] if force else []),
    ])


def merged_state(path: Path, *, minimum: int = MIN_MERGED_BYTES) -> tuple[str, int]:
    """Whether a merged model is already there, and whether it is real.

    "absent", "good" or "aborted". The last one is the reason this is a size check and not an
    existence check: `merge_adapter.py` creates the output directory before it reads the
    adapter, so an aborted merge leaves a 4 KB directory that looks exactly like a success.
    """
    if not path.exists():
        return "absent", 0
    size = directory_size(path)
    return ("good" if size >= minimum else "aborted"), size


def checkpoint_for_epoch(adapter_dir: Path, epoch: int, *, rows: int,
                         effective_batch: int) -> tuple[Path, str]:
    """Resolve "epoch N" to the `checkpoint-<STEP>` directory the trainer actually wrote."""
    if adapter_dir.exists():
        found = sorted((p for p in adapter_dir.glob("checkpoint-*") if p.is_dir()),
                       key=lambda p: int(p.name.rsplit("-", 1)[-1]))
        if len(found) >= epoch:
            return found[epoch - 1], "read from disk"
        if found:
            return (adapter_dir / "checkpoint-?",
                    f"only {len(found)} checkpoint(s) present ({[p.name for p in found]}) -- "
                    f"epoch {epoch} has not been written yet")
    steps_per_epoch = max(1, -(-rows // max(effective_batch, 1))) if rows else 0
    if not steps_per_epoch:
        return adapter_dir / "checkpoint-?", "unknown: the training JSONL is not here to count"
    return (adapter_dir / f"checkpoint-{steps_per_epoch * epoch}",
            f"estimated from {rows:,} rows / batch {effective_batch} = {steps_per_epoch} "
            f"steps per epoch -- confirm with ls")


def link_free_of_spaces(path: Path, *, link_root: str | None, name: str) -> tuple[Path, str | None]:
    """A path `docker run --model` can take, plus the symlink command that makes it exist."""
    if " " not in str(path):
        return path, None
    root = Path(link_root) if link_root else path.parent.parent / "_links"
    link = root / name
    return link, f"mkdir -p {shlex.quote(str(root))} && ln -sfn {shlex.quote(str(path))} {shlex.quote(str(link))}"


def training_estimate(rows: int, training: TrainingConfig) -> dict[str, Any]:
    """Steps and a WIDE hour range, on purpose."""
    steps = max(1, round(rows / max(training.effective_batch, 1))) * training.epochs
    low, high = training.seconds_per_step
    return {"rows": rows, "steps": steps,
            "hours_low": round(steps * low / 3600, 1), "hours_high": round(steps * high / 3600, 1)}


def print_recipe(config: DistillConfig, role: str, *, run_dir: Path | None = None) -> None:
    """Print the serving instructions for one role: every model it needs, in order."""
    if role == "student" and config.experiment.topology in ("specialist", "both"):
        types = tuple(config.subset.content_types)
        print("#" * 100)
        print(f"# TOPOLOGY {config.experiment.topology.upper()}: the two halves of the corpus "
              f"are labelled by TWO DIFFERENT models,")
        print(f"# one per card, both up at the same time. {len(types)} recipes follow.")
        print("#" * 100)
        collisions = _specialist_collisions(config, types)
        for content_type in types:
            print()
            _print_one_recipe(config, config.student_for(content_type), role="student",
                              label=f"{content_type} specialist", run_dir=run_dir,
                              cards=1, stage="corpus",
                              key=f"student_overrides.{content_type}")
        if collisions:
            print("\n" + "!" * 100)
            for problem in collisions:
                print(f"! {problem}")
            print("!" * 100)
        if config.experiment.topology == "both":
            print("\n" + "#" * 100)
            print("# ...and the combined student, which uses BOTH cards on its own -- so it "
                  "runs")
            print("# in its own pass, after the two specialists are down.")
            print("#" * 100 + "\n")
            _print_one_recipe(config, config.student, role="student", label="combined",
                              run_dir=run_dir, cards=2, stage="corpus")
        return
    settings = getattr(config, role, None)
    if settings is None:
        raise SystemExit(f"no `{role}` section in the config (expected teacher or student)")
    granularities = config.granularities()
    if role == "student" and len(granularities) > 1:
        print("#" * 100)
        print(f"# This run covers {list(granularities)}, and each granularity has its OWN "
              f"trained model.")
        print("# One at a time: point `student.model` and `student.serve` at one, label with "
              "it, tear it down.")
        print("#" * 100)
    _print_one_recipe(config, settings, role=role, run_dir=run_dir, cards=2,
                      stage="teacher" if role == "teacher" else "corpus",
                      granularity=granularities[0] if len(granularities) == 1 else "")


def _specialist_collisions(config: DistillConfig, types: tuple[str, ...]) -> list[str]:
    """What would make two simultaneous specialists fight each other."""
    problems: list[str] = []
    seen: dict[str, dict[str, str]] = {"card": {}, "port": {}, "container": {}, "url": {}}
    for content_type in types:
        settings = config.student_for(content_type)
        serve = settings.serve
        if serve is None:
            problems.append(f"{content_type}: no `serve` block, so there is no recipe for it. "
                            f"Give each specialist its own under "
                            f"`student_overrides.{content_type}.serve`.")
            continue
        for what, value in (("card", single_card(serve) or "all"),
                            ("port", str(serve.port)),
                            ("container", serve.container),
                            ("url", settings.url)):
            if not value:
                continue
            other = seen[what].get(value)
            if other:
                extra = (" -- and the second `docker run` starts with `rm -f`, so it KILLS "
                         "the first") if what == "container" else ""
                problems.append(f"{content_type} and {other} both use {what} {value!r}, and "
                                f"they run at the same time{extra}.")
            seen[what][value] = content_type
    return problems


def granularity_evidence(settings, granularity: str) -> tuple[str, str] | None:
    """Whether the model this config names looks like the OTHER granularity's.

    Nothing in an HTTP request says which weights are behind an endpoint, and the fingerprint
    gate cannot see it either: it compares the prompt the config renders against the prompt
    frozen in the training JSONL, and with `granularity: both` there is a matching pair for
    each. So a chunk pass can be run, cleanly and silently, against sentence weights -- which
    is what happened on 2026-08-28, and the only symptom was an 8-point drop in agreement with
    labels the same student had already produced.

    The names are the only signal, which is why they carry the granularity. Only the SHORT
    names are read -- the served name and the merge's own directory name -- never the whole
    `model_path`, because the run folder above it may itself contain a granularity word.

    Returns ("error", message) on positive evidence of a mismatch, ("warn", message) when the
    names carry no granularity at all and nothing can be concluded, None when they agree.
    """
    from .theme import GRANULARITIES

    names = [settings.model or ""]
    if settings.serve is not None:
        names.append(settings.serve.served_model_name or "")
        if settings.serve.model_path:
            names.append(Path(settings.serve.model_path).name)
    haystack = " ".join(names).lower()
    mine = granularity.lower() in haystack
    theirs = [g for g in GRANULARITIES if g != granularity and g.lower() in haystack]

    if theirs and not mine:
        return ("error", (
            f"this is a {granularity!r} pass, but the model the config names looks like "
            f"{theirs[0]!r}: {' / '.join(n for n in names if n)}.\n"
            f"With `granularity: both` each granularity has its own trained model, and the "
            f"config points at ONE at a time -- so this is usually the model left over from "
            f"the last pass. Point `student.model`, `student.serve.served_model_name` and "
            f"`student.serve.model_path` at the {granularity} merge, or pass "
            f"--granularity {theirs[0]} if that model is really the one you meant."
        ))
    if not mine and not theirs:
        return ("warn", (
            f"the model named here carries no granularity, so nothing checks that it is the "
            f"{granularity!r} one: {' / '.join(n for n in names if n) or '(unnamed)'}. "
            f"Nothing in an HTTP request says which weights are behind the endpoint."
        ))
    return None


def require_matching_granularity(settings, granularity: str, *, role: str = "student") -> None:
    """`granularity_evidence`, as a gate. Raises on positive evidence of a mismatch."""
    verdict = granularity_evidence(settings, granularity)
    if verdict is None:
        return
    severity, message = verdict
    if severity == "error":
        raise SystemExit(f"{role}: {message}")
    print(f"WARNING  {role}: {message}")


def _print_one_recipe(config: DistillConfig, settings, *, role: str, label: str = "",
                      run_dir: Path | None = None, cards: int = 2,
                      stage: str = "", key: str = "", granularity: str = "") -> None:
    print("=" * 72)
    title = f"{role.upper()}{' / ' + label if label else ''}"
    print(f"{title} -- endpoint {settings.url}   served model name {settings.model!r}")
    print(f"  workers {settings.workers}   max output tokens {settings.max_output_tokens}   "
          f"timeout {settings.timeout}s   retries {settings.retries}")
    if settings.extra_body:
        print(f"  extra body: {settings.extra_body}")
    print("=" * 72)
    key = key or role
    if settings.serve is None:
        print(f"no `{key}.serve` block in the config, so there is no recipe to print. Add one -- "
              f"the docker line is part of the configuration, not part of a runbook.")
        return
    if not settings.serve.model_path:
        print(f"`{key}.serve.model_path` is empty, so there is no command to render yet. Fill it "
              f"in with the model to serve -- for a student that is the MERGED directory, and "
              f"`du -sh` it first: a real 4B merge is ~7.5 G and an aborted one is 4 K.")
        return
    if role == "student":
        settings = dc_replace(settings, serve=resolve_chat_template(settings.serve,
                                                                   config.training))
    then = ""
    if stage:
        target = shlex.quote(str(run_dir)) if run_dir else "$R"
        then = f"distill --run {target} {stage}"
        if granularity:
            then += f" --granularity {granularity}"
        if label and label.endswith("specialist"):
            then += "   # ...after the other specialist is up too"
    for line in render_sequence(settings.serve, role=role, label=label, then=then):
        print(line)
    print()
    for warning in card_warnings(settings.serve, role=role, cards=cards):
        print(f"CARDS: {warning}")
    others = tuple(
        other.serve.container
        for name, other in (("teacher", config.teacher), ("student", config.student))
        if name != role and other.serve is not None and other.serve.container
    )
    warnings = serve_warnings(settings.serve, role=role, other_containers=others,
                              settings_model=settings.model)
    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    if settings.serve.model_path and Path(settings.serve.model_path).exists():
        size = directory_size(settings.serve.model_path)
        print(f"\nmodel path on this machine: {settings.serve.model_path} -- {human(size)}")
        if size < MIN_MERGED_BYTES:
            print("  that is far too small for a merged 4B. Check the merge before serving it.")
    else:
        print(f"\nmodel path {settings.serve.model_path!r} is not on this machine -- expected if "
              f"this is the laptop and the model lives on the box.")


def settings_for(config: DistillConfig, role: str, *, content_type: str = ""):
    """The endpoint settings for a role, including a specialist's per-content-type override."""
    if role == "student" and content_type:
        return config.student_for(content_type)
    settings = getattr(config, role, None)
    if settings is None:
        raise SystemExit(f"no `{role}` section in the config (expected teacher or student)")
    return settings


def gpu_memory() -> list[tuple[int, int]] | None:
    """[(card index, MiB used)] from nvidia-smi, or None where it cannot be asked."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    cards = []
    for line in out.stdout.splitlines():
        index, _, used = line.partition(",")
        try:
            cards.append((int(index.strip()), int(used.strip())))
        except ValueError:
            continue
    return cards or None


def endpoint_report(settings, serve: ServeConfig, *, expect_cards: int | None = None) -> None:
    """After the health gate: what the endpoint answers to, and which cards are holding memory.

    Printed rather than asserted, with ONE exception -- a served name that is not the one the
    client will ask for is fatal, because every request would come back "model not found" and
    the 7.4 GiB has already been loaded by then.

    The card line is the cheap end of the question §16.4 is about. It cannot say which weights
    are behind the endpoint, but it does say whether the work is spread the way the recipe
    intends: `--data-parallel-size 2` with memory on one card only is one replica doing
    everything, and that is a 2x you would otherwise discover from the throughput eight hours
    later, if at all.
    """
    from .client import endpoint_names

    names = endpoint_names(settings.url)
    if names is None:
        print("served models: the endpoint did not answer /v1/models (it passed the health "
              "gate, so this is unusual -- carrying on)")
    else:
        print(f"served models: {names}")
        if settings.model and settings.model not in names:
            raise SystemExit(
                f"the server answers to {names} but {settings.model!r} is what the client will "
                f"ask for. Every call would come back 'model not found', after the load. Fix "
                f"`served_model_name` or the role's `model` -- they are one string in two files."
            )
    cards = gpu_memory()
    if cards is None:
        return
    busy = [index for index, used in cards if used > 1024]
    print("cards: " + ", ".join(f"GPU{index} {used} MiB" for index, used in cards)
          + f"   (holding memory: {busy or 'none'})")
    wanted = expect_cards if expect_cards is not None else (data_parallel_size(serve) or 1)
    if wanted > 1 and len(busy) < wanted:
        print(f"WARNING  the recipe asks for {wanted} replicas but only {len(busy)} card(s) "
              f"hold memory. One replica doing the work of two is a silent 2x: check the "
              f"throughput the stage reports against the number you know.")


def serve_model(config: DistillConfig, role: str, *, content_type: str = "",
                reuse: bool = True, verbose: bool = True) -> ServeConfig:
    """Bring one role's model up and return once it answers. Idempotent by default.

    `reuse` is what makes this usable from a notebook. A container outlives the kernel, so a
    restarted kernel would otherwise tear a healthy 120B down and pay minutes to load it again.
    Reuse is granted only when the container is running AND the endpoint answers AND it answers
    to the name this config will ask for -- the third condition is the one that matters, since
    a container of the right NAME serving the previous granularity is exactly the failure
    §16.4 records.
    """
    settings = settings_for(config, role, content_type=content_type)
    serve = settings.serve
    if serve is None:
        raise SystemExit(
            f"`{role}.serve` is not in the config, so there is no recipe to start. It is not "
            f"optional: the docker line is configuration, not something a caller invents.")
    if role == "student":
        serve = resolve_chat_template(serve, config.training)
    if not serve.model_path:
        raise SystemExit(
            f"`{role}.serve.model_path` is empty, so there is nothing to bring up. For the "
            f"teacher that is the 120B snapshot on this machine; for the student it is the "
            f"merged checkpoint -- `driver.point_student_at(run, granularity, epoch, "
            f"apply=True)` writes it, along with the two names that have to agree with it.")
    if not docker_available():
        print(f"docker is not on this machine -- this is a PRINT of the bring-up, not a run:\n"
              f"  {render_serve_command(serve)}")
        return serve

    for warning in serve_warnings(serve, role=role, settings_model=settings.model):
        print(f"  WARNING: {warning}")
    for warning in card_warnings(serve, role=role):
        print(f"  WARNING: {warning}")

    from .client import endpoint_names

    if reuse and container_running(serve.container):
        names = endpoint_names(settings.url, timeout=5.0)
        if names and (not settings.model or settings.model in names):
            print(f"{role}: {serve.container} is already up and answers to {names} -- reusing "
                  f"it. Pass reuse=False (or `distill stop {role}`) to force a restart.")
            endpoint_report(settings, serve)
            return serve
        print(f"{role}: {serve.container} is running but "
              + ("does not answer /v1/models yet" if names is None else
                 f"answers to {names}, not {settings.model!r}")
              + " -- replacing it.")

    _docker(f"docker rm -f {shlex.quote(serve.container)} 2>/dev/null || true", check=False)
    command = render_serve_command(serve)
    if verbose:
        print(command)
    _docker(command)
    from .client import wait_for_url

    try:
        wait_for_url(settings.url, attempts=settings.health_attempts,
                     delay=settings.health_delay, verbose=verbose)
    except SystemExit:
        print(report_container_failure(serve.container))
        raise
    endpoint_report(settings, serve)
    return serve


def stop_model(config: DistillConfig, role: str, *, content_type: str = "",
               verbose: bool = True) -> bool:
    """Take one role's container down. Never raises: a teardown that fails must not mask the
    error that sent us here, and a container that is already gone is the outcome we wanted."""
    settings = settings_for(config, role, content_type=content_type)
    serve = settings.serve
    if serve is None or not serve.container:
        return False
    if not docker_available():
        print(f"docker rm -f {serve.container}   # (not run: no docker here)")
        return False
    _docker(f"docker rm -f {shlex.quote(serve.container)}", check=False)
    if verbose:
        print(f"{role}: {serve.container} removed")
        cards = gpu_memory()
        if cards:
            print("cards: " + ", ".join(f"GPU{i} {used} MiB" for i, used in cards))
    return True


@contextmanager
def up(config: DistillConfig, role: str, *, content_type: str = "", reuse: bool = True,
       leave_up: bool = False):
    """`serve_model` ... `stop_model`, with the teardown in a `finally`.

    The teardown belongs here and not in the caller's next line, because the failure mode is
    the interesting one: a stage that raises must not leave a model holding both cards, or the
    stage after it fails the health gate -- or worse, is answered by the wrong model.
    `KeyboardInterrupt` runs `finally` too, which is what makes this safe from a notebook.

    Never wrap `train` in this. It serves and tears down its own student for every checkpoint
    it scores, so a second container would be on the same cards. Only `teacher` and `corpus`
    expect a model that someone else started.
    """
    serve_model(config, role, content_type=content_type, reuse=reuse)
    try:
        yield
    finally:
        if leave_up:
            print(f"{role}: left up on purpose (leave_up=True). `distill --run <run> stop "
                  f"{role}` when you are done with it.")
        else:
            stop_model(config, role, content_type=content_type)


def report_container_failure(name: str, *, tail: int = 60) -> str:
    """Why a container is not answering: its state, and the end of its log.

    A health gate that times out says only that nothing answered. The reason is always in
    the log, and the log dies with the container -- so this is read while it still exists.
    """
    if not docker_available():
        return f"docker is not on this machine, so {name} cannot be inspected."
    lines = [f"{'=' * 100}", f"{name} did not come up. Its state and the last {tail} log lines:",
             f"{'=' * 100}"]
    try:
        state = subprocess.run(
            ["docker", "inspect", "-f",
             "status={{.State.Status}} exitcode={{.State.ExitCode}} oom={{.State.OOMKilled}}",
             name],
            capture_output=True, text=True, timeout=15, check=False)
        lines.append((state.stdout or state.stderr or "").strip() or "no state reported")
    except (OSError, subprocess.SubprocessError) as exc:
        lines.append(f"could not inspect: {exc}")
    try:
        logs = subprocess.run(["docker", "logs", "--tail", str(tail), name],
                              capture_output=True, text=True, timeout=30, check=False)
        body = ((logs.stdout or "") + (logs.stderr or "")).strip()
        lines.append(body or "the container produced no log output at all")
    except (OSError, subprocess.SubprocessError) as exc:
        lines.append(f"could not read logs: {exc}")
    lines.append("=" * 72)
    return "\n".join(lines)


def _docker(command: str, *, check: bool = True) -> int:
    """Shell out, streaming to wherever we are -- a terminal or a notebook cell."""
    result = subprocess.run(command, shell=True)
    if check and result.returncode != 0:
        raise SystemExit(f"failed ({result.returncode}): {command}")
    return result.returncode


def docker_available() -> bool:
    return shutil.which("docker") is not None


def container_running(name: str) -> bool | None:
    """True/False, or None when docker cannot be asked (the laptop)."""
    if not docker_available():
        return None
    try:
        out = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                             capture_output=True, text=True, timeout=10, check=False)
        return name in out.stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
