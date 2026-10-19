"""The box-side half: the vendored scripts, and the per-checkpoint loop."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from distill.config import DistillConfig, ServeConfig, TrainingConfig
from distill.serving import (
    checkpoint_for_epoch, link_free_of_spaces, render_merge_command, render_serve_command,
    render_train_checks, render_train_command, script_path, serve_warnings,
)

from conftest import theme_section  # noqa: E402
from distill.theme import ThemeSpec  # noqa: E402

_THEMES = Path(__file__).resolve().parent.parent / "themes"
THEME_SECTION = theme_section(ThemeSpec.load(_THEMES / "pricing_power.yaml"))

DISTILL_ROOT = Path(__file__).resolve().parent.parent
OPS = DISTILL_ROOT / "ops"


def test_the_package_carries_the_scripts_it_needs():
    """`/home/nvidia/code/train_lora_sft.py` is a HOST file the container executes."""
    for name in ("train_lora_sft.py", "merge_adapter.py", "n34b_nothink_template.jinja"):
        assert (OPS / name).exists(), name
    defaults = TrainingConfig()
    assert defaults.trainer_script == "ops/train_lora_sft.py"
    assert defaults.merge_script == "ops/merge_adapter.py"
    assert defaults.chat_template == "ops/n34b_nothink_template.jinja"


def test_the_vendored_scripts_are_importable_and_have_a_cli():
    for name in ("train_lora_sft.py", "merge_adapter.py"):
        out = subprocess.run([sys.executable, "-m", "py_compile", str(OPS / name)],
                             capture_output=True, text=True, check=False)
        assert out.returncode == 0, f"{name}: {out.stderr}"


def test_the_trainer_no_longer_hardcodes_winner_loser_irrelevant():
    """The literal `labels = ("winner"."""
    import ast

    source = (OPS / "train_lora_sft.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    literals = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Tuple)
        and [getattr(e, "value", None) for e in node.elts] == ["winner", "loser", "irrelevant"]
    ]
    assert literals == [], "the taxonomy is still hardcoded in the trainer"
    assert "def dataset_labels(" in source, "the class list must come from the data"
    assert "are not classes in this dataset" in source, "an unknown class must be an error"


def test_the_trainer_refuses_a_val_ratio_that_would_be_discarded():
    """It defaulted to 0.1 and the split it made never reached the trainer."""
    source = (OPS / "train_lora_sft.py").read_text(encoding="utf-8")
    assert '"--val-ratio", type=float, default=0.0' in source
    assert "--allow-discarded-val" in source
    assert "NEVER give them to the trainer" in source


def test_the_trainer_states_the_epoch_to_checkpoint_mapping():
    source = (OPS / "train_lora_sft.py").read_text(encoding="utf-8")
    assert "checkpoints this run will write" in source
    assert "steps_per_epoch" in source


def test_the_merge_script_asserts_before_it_writes():
    """A wrong key-prefix remap must fail before anything is written."""
    source = (OPS / "merge_adapter.py").read_text(encoding="utf-8")
    assert "delta keys are not in the checkpoint" in source
    assert 'backbone.' in source, "the backbone./model. remap"
    assert "MIN_PLAUSIBLE_BYTES" in source, "the du -sh check after writing"


def test_the_trainer_is_launched_detached_inside_the_container():
    """A foreground `docker exec` does not forward signals into the container."""
    command = render_train_command(
        TrainingConfig(), dataset_path=Path("/run/11. T/train.both.jsonl"),
        output_dir=Path("/adapters/both"), log_path=Path("/run/11. T/train.both.log"),
    )
    assert command.startswith("docker exec -d tx-train bash -c")
    assert ">>" in command and "2>&1" in command, "the log has to be captured"
    checks = " ".join(render_train_checks(TrainingConfig()))
    assert "grep -c" in checks and "pkill -f train_lora_sft.py" in checks
    assert "6:" in checks, "the expected process count, not 1"


def test_the_commands_point_at_the_packages_own_scripts():
    training = TrainingConfig()
    assert script_path(training, training.trainer_script, package_root=DISTILL_ROOT) \
        == str(OPS / "train_lora_sft.py")
    mounted = TrainingConfig(container_mount="/work/distill")
    assert script_path(mounted, mounted.merge_script, package_root=DISTILL_ROOT) \
        == "/work/distill/ops/merge_adapter.py"
    command = render_merge_command(mounted, adapter=Path("/a/checkpoint-2049"),
                                  output=Path("/m/e3"), package_root=DISTILL_ROOT)
    assert "/work/distill/ops/merge_adapter.py" in command


def test_epoch_three_is_checkpoint_2049_not_checkpoint_3(tmp_path):
    """Checkpoints are named by STEP, not by epoch."""
    path, how = checkpoint_for_epoch(tmp_path / "absent", 3, rows=43671, effective_batch=64)
    assert path.name == "checkpoint-2049"
    assert "estimated" in how and "683" in how

    adapter = tmp_path / "adapters"
    for step in (683, 1366, 2049, 2732, 3415):
        (adapter / f"checkpoint-{step}").mkdir(parents=True)
    path, how = checkpoint_for_epoch(adapter, 3, rows=43671, effective_batch=64)
    assert path.name == "checkpoint-2049" and how == "read from disk"
    path, _ = checkpoint_for_epoch(adapter, 5, rows=43671, effective_batch=64)
    assert path.name == "checkpoint-3415"


def test_an_epoch_that_has_not_been_written_is_named_as_such(tmp_path):
    adapter = tmp_path / "adapters"
    (adapter / "checkpoint-683").mkdir(parents=True)
    path, how = checkpoint_for_epoch(adapter, 4, rows=43671, effective_batch=64)
    assert "?" in path.name
    assert "has not been written yet" in how


def test_a_model_path_with_a_space_gets_a_symlink():
    """`docker run --model` cannot be handed a path with a space. The step folders no longer
    have one, but a run folder the user named does, so the guard still has to work."""
    target = Path("/runs/Market Share 2026/training/chunk.combined/merged/both/epoch3")
    link, command = link_free_of_spaces(target, link_root="/box/links", name="pp_e3")
    assert " " not in str(link)
    assert command is not None and "ln -sfn" in command and "Market Share" in command
    unchanged, no_command = link_free_of_spaces(Path("/merged/e3"), link_root=None, name="x")
    assert unchanged == Path("/merged/e3") and no_command is None


def test_the_step_folders_no_longer_need_a_symlink_at_all():
    """What the rename bought: a path built only from step folders is already space-free, so
    no link is made and `docker run --model` can take the run folder's own path."""
    target = Path("/runs/MarketShare/training/chunk.combined/merged/both/epoch3")
    link, command = link_free_of_spaces(target, link_root="/box/links", name="pp_e3")
    assert link == target and command is None


def test_two_roles_sharing_a_container_name_is_a_warning():
    """A shared container name means launching the second one kills the first."""
    serve = ServeConfig(model_path="/m", chat_template="/t", container="tx-model",
                        args=("--max-num-seqs", "512"))
    assert serve_warnings(serve, role="student", other_containers=("tx-model",))
    assert serve_warnings(serve, role="student", other_containers=("tx-other",)) == []


def measured_config(tmp_path: Path) -> DistillConfig:
    """The two templates as `init-run` writes them, loaded back as one config."""
    import shutil

    from distill.paths import config_dir as repo_config_dir
    from distill.template import write_extraction_template
    from distill.theme import ThemeSpec
    theme = ThemeSpec.load(DISTILL_ROOT / "themes" / "pricing_power.yaml")
    config_dir = tmp_path / "config"
    write_extraction_template(config_dir / "data_extraction.yaml", theme=theme,
                              run_dir=tmp_path)
    path = shutil.copy(repo_config_dir() / "distill.yaml", config_dir / "distill.yaml")
    return DistillConfig.load(path)


def test_the_template_ships_the_measured_teacher_configuration(tmp_path):
    """7.13 calls/s: both cards, 96 sequences per replica, 192 client threads."""
    config = measured_config(tmp_path)
    assert config.teacher.workers == 192
    args = list(config.teacher.serve.args)
    assert args[args.index("--data-parallel-size") + 1] == "2"
    assert args[args.index("--max-num-seqs") + 1] == "96"
    assert "--max-num-batched-tokens" in args
    assert config.teacher.serve.detach is True


def test_the_template_ships_the_measured_student_configuration(tmp_path):
    """189 rows/s over 484,982 rows in 43 min: DP=2."""
    config = measured_config(tmp_path)
    assert config.student.workers == 384
    args = list(config.student.serve.args)
    assert args[args.index("--max-num-seqs") + 1] == "512"
    assert args[args.index("--max-model-len") + 1] == "4096", (
        "config.json declares max_position_embeddings 262144; leaving it free reserves an "
        "absurd KV cache"
    )


def test_the_template_is_valid_and_uses_the_packages_own_scripts(tmp_path):
    config = measured_config(tmp_path)
    assert config.validate() == []
    assert config.training.trainer_script == "ops/train_lora_sft.py"
    assert config.training.merge_script == "ops/merge_adapter.py"
    assert config.training.val_ratio == 0.0


def test_a_null_chat_template_falls_back_to_the_packages_own(tmp_path):
    """Not optional: the base template opens a <think> block even at enable_thinking=False."""
    config = measured_config(tmp_path)
    assert config.student.serve.chat_template is None
    resolved = script_path(config.training, config.training.chat_template,
                           package_root=DISTILL_ROOT)
    assert Path(resolved).exists()


def test_the_per_checkpoint_loop_prints_every_step_in_order(tmp_path, capsys, pricing_power):
    from conftest import make_corpus
    from distill.paths import RunPaths
    from distill.stages import export, sample, subset, teacher, train

    run_dir = tmp_path / "a box path" / "Theme_2026-08-24"
    run_dir.mkdir(parents=True)
    make_corpus(200).to_parquet(run_dir / "corpus.parquet", index=False)
    config = DistillConfig.from_dict(theme=THEME_SECTION, data={
        "input": {"source_parquet": "corpus.parquet"},
        "sample": {"bins": 3, "fraction_per_bin": 0.6, "seed": 17},
        "split": {"heldout_fraction": 0.2},
        "teacher": {"workers": 4, "health_attempts": 0},
        "student": {"serve": {"container": "tx-student", "port": 8000,
                              "args": ["--max-num-seqs", "512"]}},
        "training": {"epochs": 5, "evaluate_epochs": [3], "keep_merged": []},
    }, base=run_dir).require_valid()
    paths = RunPaths.for_run(run_dir, dry_run=True)
    subset.run(config, pricing_power, paths)
    sample.run(config, pricing_power, paths)
    teacher.run(config, pricing_power, paths, dry_run=True, assume_yes=True)
    teacher.run_split(config, pricing_power, paths)
    export.run(config, pricing_power, paths)
    capsys.readouterr()

    train.run(config, pricing_power, paths, print_only=True)
    out = capsys.readouterr().out
    order = ["docker exec -d tx-train", "merge_adapter.py", "ln -sfn", "du -sh",
             "docker rm -f tx-student", "docker run -d --name tx-student",
             "evaluate --granularity chunk --dataset both --epoch 3"]
    positions = [out.index(fragment) for fragment in order]
    assert positions == sorted(positions), f"out of order: {list(zip(order, positions))}"
    assert "n34b_nothink_template.jinja" in out, "the chat template is not optional"
    assert "keep_merged does not list epoch 3" in out, "an unkept merge must be reaped"
    assert "pkill -f train_lora_sft.py" in out


def test_epoch_zero_serves_the_base_model_and_merges_nothing(tmp_path, capsys, pricing_power):
    """Epoch 0 is the model before fine-tuning, on the same held-out rows."""
    from conftest import make_corpus
    from distill.paths import RunPaths
    from distill.stages import export, sample, subset, teacher, train

    run_dir = tmp_path / "Theme"
    run_dir.mkdir(parents=True)
    make_corpus(200).to_parquet(run_dir / "corpus.parquet", index=False)
    config = DistillConfig.from_dict(theme=THEME_SECTION, data={
        "input": {"source_parquet": "corpus.parquet"},
        "sample": {"bins": 3, "fraction_per_bin": 0.6, "seed": 17},
        "split": {"heldout_fraction": 0.2},
        "teacher": {"workers": 4, "health_attempts": 0},
        "student": {"serve": {"container": "tx-student", "port": 8000,
                              "args": ["--max-num-seqs", "512"]}},
        "training": {"epochs": 5, "evaluate_epochs": [3], "keep_merged": []},
    }, base=run_dir).require_valid()
    paths = RunPaths.for_run(run_dir, dry_run=True)
    subset.run(config, pricing_power, paths)
    sample.run(config, pricing_power, paths)
    teacher.run(config, pricing_power, paths, dry_run=True, assume_yes=True)
    teacher.run_split(config, pricing_power, paths)
    export.run(config, pricing_power, paths)
    capsys.readouterr()

    train.run(config, pricing_power, paths, print_only=True, epochs=(0, 3))
    out = capsys.readouterr().out
    # the epoch-0 block runs from its own banner to its own evaluate line
    start = out.index("epoch 0: nvidia")
    end = out.index("--epoch 0 --prompt-style full")
    base, later = out[start:end], out[end:]

    assert "epoch 0: nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16, before fine-tuning" in base
    assert "merge_adapter.py" not in base, "epoch 0 has nothing to merge"
    assert "ln -sfn" not in base, "epoch 0 has nothing to link"
    assert "--model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16" in base
    assert f"--served-model-name {pricing_power.id}-chunk-base" in base
    assert "--model-label base" in later
    assert "keep_merged" not in base, "there is no merge to reap for epoch 0"
    # and the real checkpoint below it still merges
    assert "merge_adapter.py" in later, "epoch 3 must still merge its adapter"


def test_the_loop_does_not_run_docker_when_there_is_none(monkeypatch, tmp_path, pricing_power):
    """On the laptop this has to print, not fail."""
    from distill import serving
    from distill.paths import RunPaths
    from distill.stages import train

    monkeypatch.setattr(serving, "docker_available", lambda: False)
    run_dir = tmp_path / "Theme_2026-08-24"
    (run_dir / "x").mkdir(parents=True)
    config = DistillConfig.from_dict(theme=THEME_SECTION, data={
                                      "input": {"source_parquet": "corpus.parquet"}},
                                     base=run_dir)
    paths = RunPaths.for_run(run_dir)
    with pytest.raises(SystemExit, match="run `distill export` first"):
        train.run(config, pricing_power, paths)


def a_serving_run(tmp_path: Path, *, news_card="0", trans_card="1", news_port=8000,
                  trans_port=8001, news_container="tx-spec-news",
                  trans_container="tx-spec-trans"):
    """A run with both specialists configured, for the collision checks."""
    from distill.paths import RunPaths

    run = tmp_path / "R_2026-01-01"
    (run.parent / "config").mkdir(parents=True)

    def spec(card, port, container, name):
        return {"url": f"http://localhost:{port}/v1/chat/completions", "model": name,
                "workers": 256, "health_attempts": 0,
                "serve": {"container": container, "gpus": f"device={card}", "port": port,
                          "model_path": f"/m/{name}", "served_model_name": name,
                          "args": ["--max-num-seqs", "512"]}}

    config = DistillConfig.from_dict({
        "input": {"source_parquet": "corpus.parquet"},
        "experiment": {"granularity": "chunk", "topology": "specialist"},
        "teacher": {"model": "t", "health_attempts": 0,
                    "serve": {"container": "tx-teacher", "port": 8107,
                                            "model_path": "/m/120b", "gpus": "all",
                                            "args": ["--data-parallel-size", "2",
                                                     "--max-num-seqs", "96"]}},
        "student": {"model": "s", "health_attempts": 0,
                    "serve": {"container": "tx-student", "port": 8000,
                                            "model_path": "/m/merged", "gpus": "all",
                                            "args": ["--data-parallel-size", "2",
                                                     "--max-num-seqs", "512"]}},
        "student_overrides": {"news": spec(news_card, news_port, news_container, "news-e5"),
                              "transcripts": spec(trans_card, trans_port, trans_container,
                                                  "trans-e5")},
    }, base=run, theme=THEME_SECTION)
    return config, RunPaths.for_run(run)


def test_gpus_all_without_data_parallel_is_flagged_as_one_idle_card():
    """The silent failure: one replica on card 0."""
    from distill.config import ServeConfig
    from distill.serving import card_warnings

    serve = ServeConfig(container="tx", model_path="/m", gpus="all",
                        args=("--max-num-seqs", "96"))
    warnings = card_warnings(serve, role="teacher")
    assert warnings and "ONE replica" in warnings[0]
    assert "--data-parallel-size 2" in warnings[0]


def test_data_parallel_two_on_all_cards_is_clean():
    from distill.config import ServeConfig
    from distill.serving import card_warnings, data_parallel_size

    serve = ServeConfig(container="tx", model_path="/m", gpus="all",
                        args=("--data-parallel-size", "2", "--max-num-seqs", "96"))
    assert data_parallel_size(serve) == 2
    assert card_warnings(serve, role="teacher") == []


def test_a_pinned_card_with_data_parallel_cannot_fit():
    """A specialist is one model on one card."""
    from distill.config import ServeConfig
    from distill.serving import card_warnings, single_card

    serve = ServeConfig(container="tx", model_path="/m", gpus="device=0",
                        args=("--data-parallel-size", "2"))
    assert single_card(serve) == "0"
    warnings = card_warnings(serve, role="student", cards=1)
    assert warnings and "cannot fit on one card" in warnings[0]


def test_a_specialist_pinned_to_one_card_is_clean():
    from distill.config import ServeConfig
    from distill.serving import card_warnings

    serve = ServeConfig(container="tx", model_path="/m", gpus="device=1",
                        args=("--max-num-seqs", "512"))
    assert card_warnings(serve, role="student", cards=1) == []


def test_two_specialists_sharing_anything_is_a_blocker(tmp_path):
    """They run at the same time."""
    from distill.preflight import check

    config, paths = a_serving_run(tmp_path, news_card="0", trans_card="0",
                                  news_port=8000, trans_port=8000,
                                  news_container="tx-spec", trans_container="tx-spec")
    flight, _ = check(config, paths, stage="corpus", dry_run=True)
    problems = " ".join(flight.errors)
    assert "card" in problems and "port" in problems
    assert "KILLS the first" in problems


def test_two_specialists_on_their_own_cards_pass(tmp_path):
    from distill.preflight import check

    config, paths = a_serving_run(tmp_path)
    flight, _ = check(config, paths, stage="corpus", dry_run=True)
    assert not any(e.startswith("specialists:") for e in flight.errors), flight.errors


def test_the_printed_student_recipe_carries_the_chat_template(capsys, tmp_path):
    """`train` resolved a null chat_template to the package's own copy."""
    from distill import serving

    config, _ = a_serving_run(tmp_path)
    config.experiment.topology = "combined"
    serving.print_recipe(config, "student")
    out = capsys.readouterr().out
    assert "--chat-template" in out
    assert "n34b_nothink_template.jinja" in out


def test_the_printed_sequence_tears_down_first_and_looks_at_the_cards(capsys, tmp_path):
    from distill import serving

    config, _ = a_serving_run(tmp_path)
    config.experiment.topology = "combined"
    serving.print_recipe(config, "teacher")
    out = capsys.readouterr().out
    assert "docker rm -f" in out
    assert "/v1/models" in out
    assert "nvidia-smi" in out
    assert out.count("docker rm -f") >= 2


def test_a_second_trainer_is_refused_before_it_is_launched(tmp_path, monkeypatch):
    """Counting 5 s AFTER launch sees the wrapper and the main process but not the four
    dataloader workers -- 2, then 4, under any threshold. So the guard has to run BEFORE."""
    from distill.stages import train

    config, paths = a_serving_run(tmp_path)
    config.experiment.topology = "combined"
    jsonl = paths.train_jsonl("chunk", "combined", "both")
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write_text('{"messages": []}\n' * 64, encoding="utf-8")
    monkeypatch.setattr(train, "_trainer_processes", lambda container: 6)
    monkeypatch.setattr(train, "_run", lambda *a, **k: pytest.fail("launched anyway"))

    from distill.theme import ThemeSpec
    theme = ThemeSpec.from_dict(THEME_SECTION).require_valid()
    with pytest.raises(SystemExit, match="ALREADY running"):
        train.run(config, theme, paths, granularity="chunk", print_only=False)


def test_a_docker_that_cannot_be_asked_does_not_block_the_launch(tmp_path, monkeypatch):
    """`_trainer_processes` returns -1 when docker cannot be reached. That is not "a trainer is
    running" -- treating it as one would refuse to start anything at all."""
    from distill.stages import train

    config, paths = a_serving_run(tmp_path)
    config.experiment.topology = "combined"
    jsonl = paths.train_jsonl("chunk", "combined", "both")
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write_text('{"messages": []}\n' * 64, encoding="utf-8")
    launched = []
    monkeypatch.setattr(train, "_trainer_processes", lambda container: -1)
    monkeypatch.setattr(train, "_run", lambda command, **k: launched.append(command))
    monkeypatch.setattr(train.time, "sleep", lambda seconds: None)

    from distill.theme import ThemeSpec
    theme = ThemeSpec.from_dict(THEME_SECTION).require_valid()
    train.run(config, theme, paths, granularity="chunk", print_only=False)
    assert launched, "nothing was launched"


def test_the_merged_symlink_is_made_inside_the_container(tmp_path, monkeypatch):
    """`merged/` is written by the container and owned by root, so a host `mkdir` inside it
    fails with Permission denied -- after a 7.4 GiB merge has already succeeded. The link has
    to be made by the same hands that made the model."""
    import contextlib

    from distill.stages import train
    from distill.theme import ThemeSpec

    config, paths = a_serving_run(tmp_path / "a run folder")
    config.experiment.topology = "combined"
    config.training.evaluate_epochs = (3,)
    jsonl = paths.train_jsonl("chunk", "combined", "both")
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write_text('{"messages": []}\n' * 64, encoding="utf-8")
    adapter = paths.adapter_dir("chunk", "combined", "both")
    (adapter / "checkpoint-1").mkdir(parents=True)
    (adapter / "checkpoint-2").mkdir()
    (adapter / "checkpoint-3").mkdir()

    ran: list[str] = []
    monkeypatch.setattr(train, "_run", lambda command, **k: ran.append(command))
    monkeypatch.setattr(train.serving, "check_merged_model", lambda path, **k: 8 * 1024 ** 3)
    monkeypatch.setattr(train.serving, "wait_for_endpoint", lambda *a, **k: False,
                        raising=False)
    theme = ThemeSpec.from_dict(THEME_SECTION).require_valid()
    with contextlib.suppress(SystemExit):
        train.run(config, theme, paths, granularity="chunk", print_only=False,
                  skip_training=True)

    links = [c for c in ran if "ln -sfn" in c]
    assert links, "no symlink command was issued"
    for command in links:
        assert command.startswith("docker exec "), f"the link is made on the host: {command}"


def test_skip_training_on_an_untrained_granularity_says_so(tmp_path, monkeypatch, capsys):
    """`checkpoint_for_epoch` falls back to an ESTIMATED name when the adapter is absent, and
    the name comes out perfectly well-formed -- so the merge used to be attempted against a
    path that does not exist. With `granularity: both`, that is what --skip-training does to
    the half that was never trained."""
    from distill.stages import train
    from distill.theme import ThemeSpec

    config, paths = a_serving_run(tmp_path / "a run folder")
    config.experiment.topology = "combined"
    config.training.evaluate_epochs = (3,)
    jsonl = paths.train_jsonl("chunk", "combined", "both")
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write_text('{"messages": []}\n' * 64, encoding="utf-8")

    ran: list[str] = []
    monkeypatch.setattr(train, "_run", lambda command, **k: ran.append(command))
    theme = ThemeSpec.from_dict(THEME_SECTION).require_valid()
    train.run(config, theme, paths, granularity="chunk", print_only=False, skip_training=True)

    out = capsys.readouterr().out
    assert "has never been trained" in out
    assert not [c for c in ran if "merge_adapter" in c], "it tried to merge anyway"


def _adapter_with_checkpoints(paths, granularity="chunk", dataset="both", n=5):
    """A trained adapter and the training set it came from, as `train` expects to find them."""
    jsonl = paths.train_jsonl(granularity, "combined", dataset)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write_text('{"messages": []}\n' * 64, encoding="utf-8")
    adapter = paths.adapter_dir(granularity, "combined", dataset)
    for step in range(1, n + 1):
        (adapter / f"checkpoint-{step}").mkdir(parents=True, exist_ok=True)
    return adapter


def _a_merged_model(path, *, gib: float):
    """A merged model of a plausible SIZE, sparse. `directory_size` reads st_size, so a hole
    is as good as 1.2 GiB of zeros -- and writing the zeros made the suite take minutes and
    left gigabytes in tmp."""
    import os

    path.mkdir(parents=True, exist_ok=True)
    weights = path / "model.safetensors"
    weights.touch()
    os.truncate(weights, int(gib * 1024 ** 3))


def test_a_merge_that_is_already_good_is_not_redone(tmp_path, monkeypatch, capsys):
    """The step after the merge can fail -- it did, on a permission error -- and re-running
    then died on "is not empty. Pass --force" instead of picking up where it stopped."""
    import contextlib

    from distill.stages import train
    from distill.theme import ThemeSpec

    config, paths = a_serving_run(tmp_path)
    config.experiment.topology = "combined"
    config.training.evaluate_epochs = (3,)
    _adapter_with_checkpoints(paths)
    _a_merged_model(paths.merged_dir("chunk", "combined", "both", 3), gib=1.2)

    ran: list[str] = []
    monkeypatch.setattr(train, "_run", lambda command, **k: ran.append(command))
    theme = ThemeSpec.from_dict(THEME_SECTION).require_valid()
    with contextlib.suppress(SystemExit):
        train.run(config, theme, paths, granularity="chunk", print_only=False,
                  skip_training=True)

    assert "merge already there" in capsys.readouterr().out
    assert not [c for c in ran if "merge_adapter" in c], "it merged again"


def test_an_aborted_merge_is_overwritten_with_force(tmp_path, monkeypatch, capsys):
    """4 KB is what an aborted merge leaves, because the script makes the output directory
    before it reads the adapter. That must not be reused as a model."""
    import contextlib

    from distill.stages import train
    from distill.theme import ThemeSpec

    config, paths = a_serving_run(tmp_path)
    config.experiment.topology = "combined"
    config.training.evaluate_epochs = (3,)
    _adapter_with_checkpoints(paths)
    aborted = paths.merged_dir("chunk", "combined", "both", 3)
    aborted.mkdir(parents=True)

    ran: list[str] = []
    monkeypatch.setattr(train, "_run", lambda command, **k: ran.append(command))
    theme = ThemeSpec.from_dict(THEME_SECTION).require_valid()
    with contextlib.suppress(SystemExit):
        train.run(config, theme, paths, granularity="chunk", print_only=False,
                  skip_training=True)

    out = capsys.readouterr().out
    assert "aborted merge" in out
    merges = [c for c in ran if "merge_adapter" in c]
    assert merges and "--force" in merges[0], merges


def _training(**kwargs) -> TrainingConfig:
    settings = dict(batch_size=8, effective_batch=64)
    settings.update(kwargs)
    return TrainingConfig(**settings)


def test_one_card_is_the_baseline():
    command = render_train_command(_training(gpus="one"), dataset_path=Path("/d.jsonl"),
                                   output_dir=Path("/o"), package_root=Path("/p"))
    assert "torch.distributed.run" not in command
    assert "CUDA_VISIBLE_DEVICES" not in command


def test_ddp_puts_torchrun_in_front():
    command = render_train_command(_training(gpus="ddp"), dataset_path=Path("/d.jsonl"),
                                   output_dir=Path("/o"), package_root=Path("/p"))
    assert "python3 -m torch.distributed.run --nproc_per_node=2" in command
    assert "CUDA_VISIBLE_DEVICES" not in command, "DDP wants both cards, not one"


def test_per_granularity_pins_one_card():
    command = render_train_command(_training(gpus="per_granularity"),
                                   dataset_path=Path("/d.jsonl"), output_dir=Path("/o"),
                                   package_root=Path("/p"), device=1)
    assert "CUDA_VISIBLE_DEVICES=1" in command
    assert "torch.distributed.run" not in command


def test_the_trainer_is_always_told_not_to_resume():
    """The script resumes from output-dir/checkpoint-* on its own, so a used directory
    silently continues an old run under a new name."""
    for mode in ("one", "ddp", "per_granularity"):
        command = render_train_command(_training(gpus=mode), dataset_path=Path("/d.jsonl"),
                                       output_dir=Path("/o"), package_root=Path("/p"))
        assert "--no-resume" in command, mode


def test_checkpointing_is_only_passed_when_it_is_decided():
    """null leaves the script's own per-family default alone; false turns it off explicitly."""
    auto = render_train_command(_training(gradient_checkpointing=None),
                                dataset_path=Path("/d.jsonl"), output_dir=Path("/o"),
                                package_root=Path("/p"))
    assert "--grad-checkpointing" not in auto
    off = render_train_command(_training(gradient_checkpointing=False),
                               dataset_path=Path("/d.jsonl"), output_dir=Path("/o"),
                               package_root=Path("/p"))
    assert "--grad-checkpointing off" in off


def test_ddp_halves_the_accumulation_so_the_effective_batch_holds():
    """torchrun runs one process per GPU and each takes a full per-device batch, so the
    effective batch is batch x accumulation x processes. Left out of the arithmetic,
    --nproc_per_node=2 silently DOUBLES it -- a different run under the same settings."""
    assert _training(gpus="one").processes() == 1
    assert _training(gpus="ddp").processes() == 2
    assert _training(gpus="per_granularity").processes() == 1
    for mode, accum in (("one", 8), ("ddp", 4)):
        training = _training(gpus=mode)
        world = training.processes()
        assert 64 // (training.batch_size * world) == accum, mode


def test_a_batch_size_that_does_not_divide_the_effective_batch_is_refused(tmp_path):
    """The trainer computes accumulation as effective // (batch x processes). If they
    disagree, the epoch-to-checkpoint mapping is wrong."""
    from distill.config import DistillConfig

    config = DistillConfig.from_dict(
        {"training": {"batch_size": 5, "effective_batch": 64}},
        base=tmp_path, theme=THEME_SECTION)
    assert any("does not divide" in issue for issue in config.validate())


def test_per_granularity_needs_two_granularities(tmp_path):
    from distill.config import DistillConfig

    one = DistillConfig.from_dict(
        {"experiment": {"granularity": "chunk"}, "training": {"gpus": "per_granularity"}},
        base=tmp_path, theme=THEME_SECTION)
    assert any("one granularity" in issue for issue in one.validate())
    both = DistillConfig.from_dict(
        {"experiment": {"granularity": "both"}, "training": {"gpus": "per_granularity"}},
        base=tmp_path, theme=THEME_SECTION)
    assert not [i for i in both.validate() if "granularity" in i], both.validate()


def test_an_unknown_gpu_mode_is_refused(tmp_path):
    from distill.config import DistillConfig

    config = DistillConfig.from_dict({"training": {"gpus": "both"}}, base=tmp_path,
                                     theme=THEME_SECTION)
    assert any("training.gpus must be one of" in issue for issue in config.validate())


def test_two_trainers_on_two_cards_are_allowed_but_not_on_one(tmp_path, monkeypatch):
    """"One card, one trainer" is the rule, not "one container, one trainer" -- two on two
    cards is the point of per_granularity."""
    from distill.stages import train

    seen: list[int | None] = []
    monkeypatch.setattr(train, "_trainer_processes", lambda container: 6)

    def on_card(container, device):
        seen.append(device)
        return 6 if device == 0 else 0

    monkeypatch.setattr(train, "_trainers_on_card", on_card)
    assert train._trainers_on_card("tx-train", 0) == 6
    assert train._trainers_on_card("tx-train", 1) == 0
    assert seen == [0, 1]


def test_train_can_be_pointed_at_one_granularity(tmp_path):
    """With `granularity: both`, half the run can already be finished -- and re-running the
    finished half is not idempotent now that the trainer refuses a used output directory."""
    from distill.cli import _selected_granularities
    from distill.config import DistillConfig

    config = DistillConfig.from_dict({"experiment": {"granularity": "both"}}, base=tmp_path,
                                     theme=THEME_SECTION)
    assert _selected_granularities(config, None, "train") == ("chunk", "sentence")
    assert _selected_granularities(config, "sentence", "train") == ("sentence",)
    with pytest.raises(SystemExit, match="not one this run covers"):
        chunk_only = DistillConfig.from_dict({"experiment": {"granularity": "chunk"}},
                                             base=tmp_path, theme=THEME_SECTION)
        _selected_granularities(chunk_only, "sentence", "train")


def test_the_step_count_is_the_same_in_every_gpu_mode():
    """5,255 steps and epoch 3 at checkpoint-3153, whatever the mode. Each rank takes a full
    per-device batch and DDP shards the sampler, so an epoch is rows / (batch x accum x world).
    Counted the old way, a 2-process run printed twice the steps it would run and named
    checkpoints that never appear."""
    rows, target, epochs = 67_254, 64, 5
    expected = None
    for world, batch in ((1, 4), (1, 8), (2, 8), (2, 4)):
        accum = max(1, target // (batch * world))
        effective = batch * accum * world
        assert effective == target, (world, batch, effective)
        steps = max(1, -(-rows // effective))
        expected = expected or steps
        assert steps == expected, (world, batch, steps)
    assert expected == 1_051
    assert expected * epochs == 5_255
    assert expected * 3 == 3_153


def test_the_trainer_script_turns_off_the_unused_parameter_search():
    """torch walks the autograd graph an extra time every iteration looking for parameters a
    LoRA adapter never has -- it says so itself in the log."""
    source = (DISTILL_ROOT / "ops" / "train_lora_sft.py").read_text()
    assert 'if world > 1 or family == "gpt-oss":' in source
    assert 'sft_kwargs["ddp_find_unused_parameters"] = False' in source


def test_the_allocator_is_told_to_expand_segments():
    """An OOM that says "7.56 GiB is reserved but unallocated" is fragmentation, and expandable
    segments is what the error message itself recommends."""
    for mode in ("one", "ddp", "per_granularity"):
        command = render_train_command(_training(gpus=mode), dataset_path=Path("/d.jsonl"),
                                       output_dir=Path("/o"), package_root=Path("/p"))
        assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in command, mode


def test_the_pinned_card_comes_before_the_allocator_setting():
    """Both are env prefixes on one shell command; the order has to leave a runnable line."""
    command = render_train_command(_training(gpus="per_granularity"),
                                   dataset_path=Path("/d.jsonl"), output_dir=Path("/o"),
                                   package_root=Path("/p"), device=1)
    body = command.split("bash -c ", 1)[1].strip("'")
    assert body.startswith("CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF="), body[:80]


def test_train_scores_the_granularity_it_trained(tmp_path, monkeypatch):
    """Without this, `evaluate` fell back to the run's FIRST granularity: a sentence student
    was served correctly and then scored with chunk's prompt against chunk's gold, into
    chunk's folder, overwriting chunk's real numbers. The fingerprint gate cannot catch it --
    it compares the prompt with the training JSONL, and both were chunk's."""
    import contextlib

    from distill.stages import train
    from distill.theme import ThemeSpec

    config, paths = a_serving_run(tmp_path)
    config.experiment.granularity = "both"
    config.experiment.topology = "combined"
    config.training.evaluate_epochs = (3,)
    _adapter_with_checkpoints(paths, granularity="sentence")
    _a_merged_model(paths.merged_dir("sentence", "combined", "both", 3), gib=1.2)

    seen: list[str | None] = []
    monkeypatch.setattr(train, "_run", lambda command, **k: None)

    def fake_evaluate(config, theme, paths, **kwargs):
        seen.append(kwargs.get("granularity"))
        return {"macro_f1": 0.5}

    from distill.stages import evaluate as evaluate_module
    monkeypatch.setattr(evaluate_module, "run", fake_evaluate)
    theme = ThemeSpec.from_dict(THEME_SECTION).require_valid()
    with contextlib.suppress(SystemExit):
        train.run(config, theme, paths, granularity="sentence", print_only=False,
                  skip_training=True)

    assert seen == ["sentence"], f"scored {seen} while training sentence"


def test_the_printed_plan_names_the_granularity_too(tmp_path):
    """`train-cmd` prints the evaluate command for a human to run. Without --granularity it
    printed one that would score the wrong ruler."""
    import contextlib
    import io
    from contextlib import redirect_stdout

    from distill.stages import train
    from distill.theme import ThemeSpec

    config, paths = a_serving_run(tmp_path)
    config.experiment.granularity = "both"
    config.experiment.topology = "combined"
    config.training.evaluate_epochs = (3,)
    _adapter_with_checkpoints(paths, granularity="sentence")
    theme = ThemeSpec.from_dict(THEME_SECTION).require_valid()
    out = io.StringIO()
    with redirect_stdout(out), contextlib.suppress(SystemExit):
        train.run(config, theme, paths, granularity="sentence", print_only=True)
    assert "evaluate --granularity sentence" in out.getvalue()


def test_the_served_name_distinguishes_the_granularities(tmp_path, monkeypatch):
    """Without the granularity in it, chunk-e3 and sentence-e3 are both
    `market_share-both-e3`: `student.model` cannot tell them apart and vLLM answers for
    whichever is up. That is how a sentence student got scored on chunk's ruler."""
    import contextlib

    from distill.stages import train
    from distill.theme import ThemeSpec

    names: list[str] = []
    for granularity in ("chunk", "sentence"):
        config, paths = a_serving_run(tmp_path / granularity)
        config.experiment.topology = "combined"
        config.training.evaluate_epochs = (3,)
        _adapter_with_checkpoints(paths, granularity=granularity)
        _a_merged_model(paths.merged_dir(granularity, "combined", "both", 3), gib=1.2)
        ran: list[str] = []
        monkeypatch.setattr(train, "_run", lambda command, **k: ran.append(command))
        theme = ThemeSpec.from_dict(THEME_SECTION).require_valid()
        with contextlib.suppress(SystemExit):
            train.run(config, theme, paths, granularity=granularity, print_only=False,
                      skip_training=True)
        served = [c for c in ran if "--served-model-name" in c]
        assert served, granularity
        name = served[0].split("--served-model-name ")[1].split()[0]
        names.append(name)
        assert granularity in name, name
    assert names[0] != names[1], names


def test_the_two_model_names_have_to_match():
    """`role.model` is what the CLIENT asks for and `role.serve.served_model_name` is what the
    SERVER calls it. Different strings mean every request comes back "model not found" --
    after the model has finished loading."""
    from distill.config import ServeConfig
    from distill.serving import serve_warnings

    serve = ServeConfig(container="tx", model_path="/m", served_model_name="student",
                        chat_template="t.jinja",
                        args=("--max-num-seqs", "512", "--data-parallel-size", "2"))
    mismatched = serve_warnings(serve, role="student",
                                settings_model="market_share-chunk-both-e5")
    assert any("what the CLIENT asks for" in w for w in mismatched), mismatched
    assert serve_warnings(serve, role="student", settings_model="student") == []

    unnamed = ServeConfig(container="tx", model_path="/m", chat_template="t.jinja",
                          args=("--max-num-seqs", "512", "--data-parallel-size", "2"))
    warned = serve_warnings(unnamed, role="student", settings_model="market_share-chunk-both-e5")
    assert any("answers under the path" in w for w in warned), warned
    assert serve_warnings(unnamed, role="student", settings_model="/m") == []


def test_the_gate_reports_a_model_name_mismatch(tmp_path):
    from distill.preflight import check

    config, paths = a_serving_run(tmp_path)
    config.experiment.topology = "combined"
    config.student.model = "market_share-chunk-both-e5"
    flight, _ = check(config, paths, stage="train", dry_run=True)
    assert any("what the CLIENT asks for" in w for w in flight.warnings), flight.warnings


def test_volume_spec_without_colon_mounts_at_its_own_path():
    """The student is served by absolute HOST paths, so those paths must resolve inside.

    `--model <repo>/output/.../merged/epoch3` and `--chat-template <repo>/src/...`
    are both written by the pipeline as host paths. Mounting only `models:/models` leaves the
    container unable to see either, and the sole symptom is a health gate that never goes
    green -- the container's log has the reason and the teardown deletes it.
    """
    from distill.serving import _volume

    host, _, container = _volume(".").partition(":")
    assert host == container, "a colon-less spec must mount at the same path on both sides"
    assert Path(host).is_absolute()
    assert (Path(host) / "config").is_dir(), "`.` must resolve to the repository root"

    assert _volume("models:/models").endswith(":/models")
    assert _volume("/abs/path:/models") == "/abs/path:/models"
