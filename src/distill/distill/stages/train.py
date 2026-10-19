"""Stage 11b -- train, then merge / size-check / serve / score / tear down, per checkpoint."""

from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

from .. import frames, serving
from ..config import DistillConfig
from ..manifest import Manifest
from ..paths import RunPaths
from ..theme import ThemeSpec

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def adapter_for(config: DistillConfig, theme: ThemeSpec, paths: RunPaths, dataset: str,
                granularity: str) -> Path:
    topology = config.experiment.topology
    if config.training.adapters_root:
        return Path(config.training.adapters_root) / f"{theme.id}_{granularity}_{dataset}"
    return paths.adapter_dir(granularity, topology, dataset)


def merged_for(config: DistillConfig, theme: ThemeSpec, paths: RunPaths, dataset: str,
               epoch: int, granularity: str) -> Path:
    topology = config.experiment.topology
    if config.training.merged_root:
        return (Path(config.training.merged_root)
                / f"{theme.id}_{granularity}_{dataset}_e{epoch}")
    return paths.merged_dir(granularity, topology, dataset, epoch)


def _run(command: str, *, print_only: bool, check: bool = True) -> int:
    print(f"$ {command}")
    if print_only:
        return 0
    result = subprocess.run(command, shell=True)
    if check and result.returncode != 0:
        raise SystemExit(f"command failed with {result.returncode}: {command}")
    return result.returncode


def _trainers_on_card(container: str, device: int | None) -> int:
    """Trainer processes pinned to one card, or all of them when `device` is None.

    "One card, one trainer" is the rule, not "one container, one trainer" -- two trainers on
    two different cards is the point of `per_granularity`, and two on the SAME card is the
    failure that stacked four of them once.
    """
    if device is None:
        return _trainer_processes(container)
    try:
        out = subprocess.run(
            ["docker", "exec", container, "bash", "-c",
             f"ps eww -o command | grep '[t]rain_lora_sft' | "
             f"grep -c 'CUDA_VISIBLE_DEVICES={device}\\b'"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return int((out.stdout or "0").strip() or 0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return -1


def _trainer_processes(container: str) -> int:
    try:
        out = subprocess.run(
            ["docker", "exec", container, "bash", "-c",
             "ps aux | grep -c '[t]rain_lora_sft'"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return int((out.stdout or "0").strip() or 0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return -1


def run(config: DistillConfig, theme: ThemeSpec, paths: RunPaths, *,
        granularity: str | None = None,
        datasets: tuple[str, ...] = (), epochs: tuple[int, ...] = (),
        print_only: bool | None = None, skip_training: bool = False,
        wait_minutes: int = 0, device: int | None = None) -> dict[str, Any]:
    training = config.training
    granularity, topology = (granularity or config.granularities()[0],
                             config.experiment.topology)
    if print_only is None:
        print_only = not serving.docker_available()
        if print_only:
            print("docker is not on this machine, so this is a PRINT of the sequence, not a run. "
                  "That is the expected result on the laptop.")
    wanted_datasets = datasets or config.datasets_for_topology()
    wanted_epochs = epochs or tuple(training.evaluate_epochs)

    manifest = Manifest(stage="train", experiment=f"{granularity}.{topology}",
                        dry_run=paths.dry_run)
    results: dict[str, Any] = {}

    for dataset in wanted_datasets:
        jsonl = paths.train_jsonl(granularity, topology, dataset)
        if not jsonl.exists():
            raise SystemExit(f"{jsonl} does not exist -- run `distill export` first.")
        rows = sum(1 for line in jsonl.open(encoding="utf-8") if line.strip())
        adapter = adapter_for(config, theme, paths, dataset, granularity)
        log = paths.train_log(granularity, topology, dataset)
        estimate = serving.training_estimate(rows, training)
        steps_per_epoch = max(1, -(-rows // max(training.effective_batch, 1)))

        print(f"\ntraining {granularity}.{topology} on {rows:,} rows, "
              f"{estimate['steps']:,} steps"
              + (f", {len(training.devices)} cards" if training.gpus == "ddp" else ""))
        manifest.add_input(f"train.{dataset}", jsonl)

        if not skip_training:
            log.parent.mkdir(parents=True, exist_ok=True)
            if not print_only:
                running = _trainers_on_card(training.container, device)
                if running > 0:
                    where = f"on card {device}" if device is not None else \
                        f"in {training.container}"
                    raise SystemExit(
                        f"a trainer is ALREADY running {where} ({running} processes). One card "
                        f"takes one trainer: two of them share the GPU, write into their own "
                        f"output-dirs at a third of the speed, and every timing afterwards "
                        f"measures a shared card. Four got stacked once -- 138 GB of 140 in "
                        f"use, 21 s/step.\n"
                        f"  wait for it:  tail -f {log}\n"
                        f"  or stop it:   docker exec {training.container} "
                        f"pkill -f train_lora_sft.py"
                    )
            _run(serving.render_train_command(training, dataset_path=jsonl, output_dir=adapter,
                                              package_root=PACKAGE_ROOT, log_path=log,
                                              device=device),
                 print_only=print_only)
            if print_only:
                for check in serving.render_train_checks(training):
                    print(f"# {check}")
            if not print_only:
                time.sleep(5)
                count = _trainer_processes(training.container)
                print(f"  running detached ({count} processes). Watch: tail -f {log}   "
                      f"Stop: docker exec {training.container} pkill -f train_lora_sft.py")
                if count > 8:
                    raise SystemExit(
                        f"{count} trainer processes are running. More than one trainer on one "
                        f"card is how four got stacked once -- 138 GB of 140 in use, 21 s/step, "
                        f"two of them writing into the same output-dir, and every timing "
                        f"afterwards was measuring a shared GPU. Stop them with "
                        f"`docker exec {training.container} pkill -f train_lora_sft.py`."
                    )
                if wait_minutes:
                    deadline = time.monotonic() + wait_minutes * 60
                    while time.monotonic() < deadline:
                        if _trainer_processes(training.container) <= 1:
                            break
                        time.sleep(30)
                else:
                    print("Not waiting. Re-run with --skip-training --wait 0 once it is done, or "
                          "pass --wait <minutes> to block here.")
                    results[dataset] = {"launched": True, "log": str(log)}
                    results["_trainer_running"] = True
                    continue

        for epoch in wanted_epochs:
            base_run = epoch == 0
            if base_run:
                # Epoch 0 is the model before any of this: the base 4B, straight out of
                # `models/`. There is nothing to merge and nothing to link, so it skips to
                # serving. What makes it a comparison rather than a separate number is that it
                # answers the SAME held-out rows as every checkpoint below -- the split is
                # written once and never regenerated.
                checkpoint, how, merged = Path(training.base_model), "the base model", None
                model_path, link_command = training.base_model, ""
                served_name = f"{theme.id}-{granularity}-base"
                print(f"\n  epoch 0: {training.base_model}, before fine-tuning")
            else:
                checkpoint, how = serving.checkpoint_for_epoch(
                    adapter, epoch, rows=rows, effective_batch=training.effective_batch)
                merged = merged_for(config, theme, paths, dataset, epoch, granularity)
            if not base_run and "?" in checkpoint.name:
                print("cannot name the checkpoint, so nothing can be merged for this epoch. "
                      "`ls` the adapter directory on the box and pass --epoch explicitly.")
                continue
            if not base_run and not print_only and not checkpoint.exists():
                if skip_training and not adapter.exists():
                    print(f"--skip-training was passed, but {adapter} does not exist: this "
                          f"granularity has never been trained. Drop the flag to train it.")
                else:
                    print(f"{checkpoint} is not on disk, so there is nothing to merge for "
                          f"epoch {epoch}. Trained checkpoints here: "
                          f"{sorted(p.name for p in adapter.glob('checkpoint-*')) or 'none'}.")
                break

            state, size = ("absent", 0) if (print_only or base_run) \
                else serving.merged_state(merged)
            if base_run:
                pass
            elif state == "good":
                print(f"  epoch {epoch}: merge already there ({serving.human(size)})")
            else:
                if state == "aborted":
                    print(f"{merged} is only {serving.human(size)} -- an aborted merge, which "
                          f"looks like a success because the script makes the directory before "
                          f"it reads the adapter. Overwriting it.")
                _run(serving.render_merge_command(training, adapter=checkpoint, output=merged,
                                                  package_root=PACKAGE_ROOT,
                                                  force=state == "aborted"),
                     print_only=print_only)
            if not base_run:
                model_path, link_command = serving.link_free_of_spaces(
                    merged, link_root=training.link_root,
                    name=f"{theme.id}_{granularity}_{topology}_{dataset}_e{epoch}",
                )
            if link_command:
                print("# `docker run --model` gets a path with no spaces:")
                _run(f"docker exec {shlex.quote(training.container)} bash -c "
                     f"{shlex.quote(link_command)}", print_only=print_only)
            if base_run:
                pass
            elif print_only:
                print(f"du -sh {shlex.quote(str(merged))}   # ~7.5 G; 4 K is an aborted merge")
            else:
                serving.check_merged_model(merged)

            if not base_run:
                served_name = f"{theme.id}-{granularity}-{dataset}-e{epoch}"
            serve = config.student.serve
            if serve is None:
                print("no `student.serve` block in the config, so this checkpoint cannot be "
                      "served automatically. Add one -- the docker line is configuration.")
                continue
            serve = serving.resolve_chat_template(
                dc_replace(serve, model_path=str(model_path), served_model_name=served_name,
                           revision=training.base_revision if base_run else None),
                training)
            for warning in serving.serve_warnings(serve, role="student"):
                print(f"  WARNING: {warning}")
            _run(f"docker rm -f {serve.container} 2>/dev/null || true", print_only=print_only,
                 check=False)
            _run(serving.render_serve_command(serve), print_only=print_only)

            scored: dict[str, Any] | None = None
            if print_only:
                print(f"distill --run {shlex.quote(str(paths.run_dir))} evaluate "
                      f"--granularity {granularity} --dataset {dataset} --epoch {epoch}"
                      + (" --prompt-style full --model-label base" if base_run else ""))
            else:
                from . import evaluate
                student = dc_replace(config.student, model=served_name)
                scoring_config = dc_replace(config, student=student)
                scoring_config._base = config.base
                try:
                    scored = evaluate.run(
                        scoring_config, theme, paths, granularity=granularity,
                        dataset=dataset, epoch=epoch,
                        # The base model was never trained on the stub, so asking it with one
                        # would only measure that. It gets the theme in full -- the most
                        # generous prompt there is -- and the gap to the fine-tuned model is
                        # then a result rather than a tautology. (Epoch 0 skips the fingerprint
                        # gate in `evaluate` itself: there is no training prompt to match.)
                        **({"prompt_style": "full", "model_label": "base"} if base_run else {}))
                finally:
                    _run(f"docker rm -f {serve.container}", print_only=False, check=False)
                    _run("nvidia-smi --query-gpu=index,memory.used --format=csv",
                         print_only=False, check=False)
            if print_only:
                _run(f"docker rm -f {serve.container}", print_only=True, check=False)

            if not base_run and epoch not in tuple(training.keep_merged):
                print(f"# training.keep_merged does not list epoch {epoch}: reap the 7.5 G. The "
                      f"ADAPTER is what is worth keeping -- it is small and it is the only thing "
                      f"that cannot be rebuilt without training again. merged/ is root-owned "
                      f"inside the container, so:")
                _run(f"docker exec {training.container} rm -rf {shlex.quote(str(merged))}",
                     print_only=True, check=False)

            results.setdefault(dataset, {})[f"epoch{epoch}"] = {
                "checkpoint": str(checkpoint), "checkpoint_resolved_by": how,
                "merged": str(merged) if merged else None, "served_as": served_name,
                "metrics": scored,
            }
            manifest.add_output(f"adapter.{dataset}", adapter, travels=False)


    manifest.config = {
        "experiment": {"granularity": granularity, "topology": topology},
        "training": training.__dict__,
        "datasets": list(wanted_datasets),
        "epochs": list(wanted_epochs),
        "print_only": print_only,
    }
    manifest.fingerprints = {"theme": theme.fingerprint()}
    manifest.counts = results
    manifest.travels = False
    manifest.note("adapters and merged models are box-only. The adapter is the artifact worth "
                  "keeping; a merged model is derived and can be rebuilt from it.")
    if not print_only:
        manifest.write(paths.manifest_for(
            paths.training_dir(granularity, topology, create=True)))
    return results
