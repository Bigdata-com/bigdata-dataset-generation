"""The notebook that drives a run, and the module it calls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from distill import driver, serving
from distill.cli import main
from conftest import theme_section
from distill.config import DistillConfig, ServeConfig
from conftest import install_config
from distill.paths import RunPaths
from distill.stages.sample import _apply_ceiling


def a_run(tmp_path: Path) -> Path:
    run = tmp_path / "output"
    run.mkdir(parents=True, exist_ok=True)
    install_config(run, theme="pricing_power")
    return run


def test_the_sample_ceiling_is_taken_in_proportion_from_every_bin():
    """Trimming the tail instead would move the distribution the stratified draw preserves."""
    rng = np.random.default_rng(0)
    bins = [np.arange(start, start + size) for start, size in
            ((0, 1000), (1000, 2000), (3000, 3000), (6000, 4000))]
    trimmed = _apply_ceiling(bins, 1000, rng=rng, label="t")

    assert sum(len(part) for part in trimmed) == 1000, "the ceiling is exact, not approximate"
    before = np.array([len(part) for part in bins], dtype=float) / 10000
    after = np.array([len(part) for part in trimmed], dtype=float) / 1000
    assert np.allclose(before, after, atol=0.001), "each bin keeps the share it had"


def test_the_ceiling_does_nothing_when_the_sample_is_already_under_it():
    rng = np.random.default_rng(0)
    bins = [np.arange(0, 10), np.arange(10, 20)]
    assert _apply_ceiling(bins, 500, rng=rng, label="t") is bins


def test_the_ceiling_never_asks_a_bin_for_more_rows_than_it_holds():
    """Largest-remainder hands out the leftovers, and a small bin must not be handed one."""
    rng = np.random.default_rng(0)
    bins = [np.arange(0, 1), np.arange(1, 999)]
    trimmed = _apply_ceiling(bins, 500, rng=rng, label="t")
    assert sum(len(part) for part in trimmed) == 500
    assert all(len(part) <= len(source) for part, source in zip(trimmed, bins))


def test_the_ceiling_is_per_content_type_and_the_manifest_records_it(tmp_path, corpus_parquet):
    """Two things a reader of a finished run needs and could not get before.

    That the ceiling is PER CONTENT TYPE -- `choose` runs once per type, so 4 writes 8 on a
    two-type run, and someone budgeting a teacher stage has to know that. And that the number
    is in the manifest: a sample capped at 500 and one capped at 50,000 were indistinguishable
    afterwards, which is exactly what a manifest exists to prevent.
    """
    import json

    from distill.paths import RunPaths
    from distill.stages import sample as sample_stage, subset as subset_stage
    from distill.theme import ThemeSpec

    theme = ThemeSpec.load(Path(__file__).resolve().parent.parent / "themes" /
                           "pricing_power.yaml")
    run = tmp_path / "Theme_2026-01-01"
    run.mkdir()
    config = DistillConfig.from_dict(
        {"input": {"source_parquet": str(corpus_parquet)},
         "subset": {"metric": "sim_top", "top_fraction": 0.9, "per_content_type": True},
         "sample": {"bins": 4, "fraction_per_bin": 1.0, "max_rows": 4, "seed": 17}},
        base=tmp_path, theme=theme_section(theme))
    paths = RunPaths.for_run(run)
    subset_stage.run(config, theme, paths)
    sample_stage.run(config, theme, paths)

    import pandas as pd

    sampled = pd.read_parquet(paths.sample_parquet)
    per_type = sampled["content_type"].value_counts().to_dict()
    assert set(per_type.values()) == {4}, f"the ceiling is per content type: {per_type}"
    assert len(sampled) == 4 * len(per_type)

    manifest = json.loads(paths.manifest_for(paths.sample_parquet).read_text())
    assert manifest["config"]["sample"]["max_rows"] == 4


def test_max_rows_is_a_config_key_and_a_typo_is_refused(tmp_path):
    config = DistillConfig.from_dict({"sample": {"max_rows": 50000}}, base=tmp_path)
    assert config.sample.max_rows == 50000
    with pytest.raises(SystemExit, match="unknown key"):
        DistillConfig.from_dict({"sample": {"max_row": 50000}}, base=tmp_path)


def test_parameters_is_read_only_until_it_is_told_otherwise(tmp_path, capsys):
    run = a_run(tmp_path)
    driver.parameters(run, top_fraction=0.05)
    assert "read-only" in capsys.readouterr().out
    assert DistillConfig.load_run(RunPaths.for_run(run)).subset.top_fraction != 0.05

    driver.parameters(run, top_fraction=0.05, apply=True)
    assert DistillConfig.load_run(RunPaths.for_run(run)).subset.top_fraction == 0.05


def test_a_parameter_cannot_be_changed_once_its_stage_has_written(tmp_path):
    """The artifact on disk was built with the old value and nothing downstream would know."""
    run = a_run(tmp_path)
    subset = RunPaths.for_run(run).subset_parquet
    subset.parent.mkdir(parents=True, exist_ok=True)
    subset.write_bytes(b"pretend this is the subset")

    with pytest.raises(SystemExit, match="already written its output"):
        driver.parameters(run, top_fraction=0.05, apply=True)

def test_reset_removes_the_stage_and_everything_after_it(tmp_path):
    """The alternative was a shell, four `rm -rf`, and remembering which four.

    Forgetting `training/` once meant `export` reused a 100-row JSONL under a 1,520-row run
    and a GPU was spent training on it.
    """
    from distill.paths import RunPaths

    run = a_run(tmp_path)
    paths = RunPaths.for_run(run)
    planted = {}
    for key in ("subset", "sample", "teacher", "training", "evaluations", "corpus"):
        directory = paths.step_dir(key, create=True)
        (directory / "artifact.parquet").write_bytes(b"x" * 16)
        planted[key] = directory

    listed = driver.reset(run, "sample")
    assert planted["subset"] not in listed, "the stage BEFORE must survive"
    assert planted["sample"] in listed and planted["corpus"] in listed
    assert all(p.exists() for p in planted.values()), "read-only by default"

    driver.reset(run, "sample", apply=True)
    assert planted["subset"].exists(), "still the stage before"
    for key in ("sample", "teacher", "training", "evaluations", "corpus"):
        assert not planted[key].exists(), key


def test_reset_refuses_an_unknown_stage(tmp_path):
    run = a_run(tmp_path)
    with pytest.raises(SystemExit, match="unknown stage"):
        driver.reset(run, "labelling")


def test_reset_on_a_clean_run_says_so_instead_of_failing(tmp_path, capsys):
    run = a_run(tmp_path)
    assert driver.reset(run, "subset") == []
    assert "nothing to reset" in capsys.readouterr().out


def test_the_two_config_parameters_a_run_could_not_set_from_the_notebook(tmp_path):
    """The 120B snapshot and the split's strata. Both had to be `sed` into the config mid-run.

    `stratify_on` matters at small scale: at 1,700 rows the default three keys make 120 strata,
    every one too thin to hold anything out, and the held-out comes back EMPTY.
    """
    run = a_run(tmp_path)
    driver.parameters(run, teacher_model_path="/models/snap",
                      stratify_on=["content_type", "label"], apply=True)
    config = DistillConfig.load_run(RunPaths.for_run(run))
    assert config.teacher.serve.model_path == "/models/snap"
    assert tuple(config.split.stratify_on) == ("content_type", "label")
    assert config.validate() == []


def test_a_list_that_equals_the_configs_tuple_is_not_a_change(tmp_path, capsys):
    """The config holds tuples; a notebook cell writes lists. `==` said they differed.

    Harmless before anything had run, and a refusal afterwards: on the second pass `split` was
    done, and a parameter "changed" to exactly what it already held blocked the cell.
    """
    run = a_run(tmp_path)
    driver.parameters(run, stratify_on=["content_type", "label", "bin"], apply=True)
    config = DistillConfig.load_run(RunPaths.for_run(run))
    assert tuple(config.split.stratify_on) == ("content_type", "label", "bin")

    heldout = RunPaths.for_run(run).heldout_split("chunk")
    heldout.parent.mkdir(parents=True, exist_ok=True)
    heldout.write_bytes(b"the split has run")

    driver.parameters(run, stratify_on=["content_type", "label", "bin"], apply=True)
    assert "already says all of this" in capsys.readouterr().out


def test_an_unknown_parameter_is_named_rather_than_ignored(tmp_path):
    run = a_run(tmp_path)
    with pytest.raises(SystemExit, match="unknown parameter"):
        driver.parameters(run, granularty="chunk")


def test_state_reads_the_artifacts_because_that_is_what_the_next_stage_finds(tmp_path):
    run = a_run(tmp_path)
    rows = {row["stage"]: row for row in driver.state(run)}
    assert rows["subset"]["done"] is False
    subset = RunPaths.for_run(run).subset_parquet
    subset.parent.mkdir(parents=True, exist_ok=True)
    subset.write_bytes(b"x" * 2048)
    rows = {row["stage"]: row for row in driver.state(run)}
    assert rows["subset"]["done"] is True and rows["subset"]["size"] == "2K"


def test_use_checkpoint_is_a_command_and_not_three_edits_by_hand(tmp_path):
    """Following the guide, this was the one step with no command behind it.

    It said "point student.model / served_model_name / serve.model_path at your choice" in a
    comment, in a workflow that is otherwise entirely commands, and the only thing that did it
    was a Python one-liner.
    """
    from distill.cli import main

    run = a_run(tmp_path)
    assert main(["--run", str(run), "use-checkpoint", "--granularity", "chunk",
                 "--epoch", "3", "--dry-run"]) == 0
    config = DistillConfig.load_run(RunPaths.for_run(run))
    assert config.student.model == "student", "--dry-run wrote nothing"

    assert main(["--run", str(run), "use-checkpoint", "--granularity", "chunk",
                 "--epoch", "3"]) == 0
    config = DistillConfig.load_run(RunPaths.for_run(run))
    assert config.student.model == config.student.serve.served_model_name
    assert "chunk" in config.student.model and "e3" in config.student.model


def test_point_student_at_moves_all_three_fields_together(tmp_path):
    """What the client asks for, where the weights are, what the server calls them."""
    run = a_run(tmp_path)
    driver.point_student_at(run, "chunk", 5, apply=True)
    config = DistillConfig.load_run(RunPaths.for_run(run))
    assert config.student.model == config.student.serve.served_model_name
    assert "chunk" in config.student.model and "e5" in config.student.model
    assert config.student.serve.model_path.endswith("/merged/both/epoch5"), \
        "the merged directory train writes for that epoch"
    assert serving.granularity_evidence(config.student, "chunk") is None
    severity, _ = serving.granularity_evidence(config.student, "sentence")
    assert severity == "error", "the name is the only signal, and it must carry the granularity"


def a_serve(**overrides) -> ServeConfig:
    fields = {"image": "vllm:v1", "container": "tx-test", "gpus": "all", "port": 8000,
              "model_path": "/merged/e5", "served_model_name": "e5",
              "args": ("--max-num-seqs", "96")}
    fields.update(overrides)
    return ServeConfig(**fields)


def test_serve_does_not_demand_the_artifacts_of_the_stage_it_serves(tmp_path):
    """Bringing a model up is not a stage and must not inherit one's preflight.

    Registered against `corpus` at first, so `distill serve teacher` on a fresh run refused for
    want of `training/chunk.combined/train.both.jsonl` -- on a run whose whole purpose at
    that moment was to bring the teacher up before anything else existed. Starting a container
    cannot corrupt an artifact.
    """
    import argparse

    from distill.cli import build_parser

    parser = build_parser()
    for name in ("serve", "stop"):
        args = parser.parse_args(["--run", str(tmp_path), name, "teacher"])
        assert args.stage_name == "config", f"{name} runs the {args.stage_name!r} preflight"


def test_the_container_comes_down_even_when_the_stage_raises(tmp_path, monkeypatch):
    """The whole reason the teardown is in a `finally` and not in the caller's next line."""
    run = a_run(tmp_path)
    config = DistillConfig.load_run(RunPaths.for_run(run))
    config.student.serve = a_serve()
    config.student.model = "e5"
    calls: list[str] = []

    monkeypatch.setattr(serving, "docker_available", lambda: True)
    monkeypatch.setattr(serving, "container_running", lambda name: False)
    monkeypatch.setattr(serving, "_docker", lambda command, check=True: calls.append(command))
    monkeypatch.setattr(serving, "endpoint_report", lambda *a, **k: None)
    monkeypatch.setattr("distill.client.wait_for_url", lambda *a, **k: None)

    with pytest.raises(ValueError):
        with serving.up(config, "student"):
            raise ValueError("the stage exploded")

    assert any(c.startswith("docker run") for c in calls), "it was brought up"
    assert calls[-1].startswith("docker rm -f"), "and taken down last, despite the exception"


def test_a_healthy_container_of_the_right_model_is_reused(tmp_path, monkeypatch):
    """A container outlives the kernel, and a 120B reload is minutes."""
    run = a_run(tmp_path)
    config = DistillConfig.load_run(RunPaths.for_run(run))
    config.student.serve = a_serve()
    config.student.model = "e5"
    calls: list[str] = []

    monkeypatch.setattr(serving, "docker_available", lambda: True)
    monkeypatch.setattr(serving, "container_running", lambda name: True)
    monkeypatch.setattr(serving, "_docker", lambda command, check=True: calls.append(command))
    monkeypatch.setattr(serving, "endpoint_report", lambda *a, **k: None)
    monkeypatch.setattr("distill.client.endpoint_names", lambda url, timeout=5.0: ["e5"])

    serving.serve_model(config, "student")
    assert calls == [], "nothing was started and nothing was torn down"


def test_a_container_serving_something_else_is_replaced_not_reused(tmp_path, monkeypatch):
    """The right container NAME in front of the previous granularity is exactly §16.4."""
    run = a_run(tmp_path)
    config = DistillConfig.load_run(RunPaths.for_run(run))
    config.student.serve = a_serve()
    config.student.model = "chunk_e5"
    calls: list[str] = []

    monkeypatch.setattr(serving, "docker_available", lambda: True)
    monkeypatch.setattr(serving, "container_running", lambda name: True)
    monkeypatch.setattr(serving, "_docker", lambda command, check=True: calls.append(command))
    monkeypatch.setattr(serving, "endpoint_report", lambda *a, **k: None)
    monkeypatch.setattr("distill.client.endpoint_names",
                        lambda url, timeout=5.0: ["sentence_e3"])
    monkeypatch.setattr("distill.client.wait_for_url", lambda *a, **k: None)

    serving.serve_model(config, "student")
    assert any(c.startswith("docker rm -f") for c in calls), "the stale one is removed"
    assert any(c.startswith("docker run") for c in calls), "and the right one started"


def test_the_served_name_must_be_the_one_the_client_will_ask_for(monkeypatch):
    """Every call would come back 'model not found', after the 7.4 GiB has loaded."""
    from distill.config import EndpointSettings

    settings = EndpointSettings(url="http://localhost:8000/v1/chat/completions", model="e5")
    monkeypatch.setattr("distill.client.endpoint_names", lambda url, timeout=5.0: ["e3"])
    monkeypatch.setattr(serving, "gpu_memory", lambda: None)
    with pytest.raises(SystemExit, match="model not found"):
        serving.endpoint_report(settings, a_serve())


def test_scores_reads_the_files_evaluate_actually_writes(tmp_path):
    """`evaluate` names them `<tag>.metrics.json`, and this reads the same folder.

    Globbing for a bare `metrics.json` matched none of them, so the table came back empty --
    which reads as a training that scored nothing rather than as a pattern that never matched.
    The path here comes from `eval_metrics_json`, the same helper the writer uses, so the two
    cannot drift apart again.
    """
    run = a_run(tmp_path)
    paths = RunPaths.for_run(run)
    topology = DistillConfig.load_run(paths).experiment.topology
    tag = f"chunk.{topology}.both.e3"
    path = paths.eval_metrics_json("chunk", topology, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "macro_f1": 0.81, "accuracy": 0.93, "unparseable": 0, "abstain": "irrelevant",
        "reports": {"overall": {"per_class": {
            "gainer": {"precision": 0.83, "recall": 0.77, "predicted": 120, "support": 130},
            "irrelevant": {"precision": 0.98, "recall": 0.99, "predicted": 900,
                           "support": 890},
        }}},
    }), encoding="utf-8")

    frame = driver.scores(run, ("chunk",))
    assert len(frame) == 1, "the one scored checkpoint on disk has to show up"
    row = frame.iloc[0]
    assert row["granularity"] == "chunk"
    assert row["epoch"] == 3
    assert row["macro_f1"] == 0.81

    assert row["gainer precision"] == 0.83 and row["gainer recall"] == 0.77
    assert "accuracy" not in frame.columns, \
        "the do-nothing model scores higher on accuracy than a model that finds things"
    assert row["irrelevant precision"] == 0.98 and row["irrelevant recall"] == 0.99, \
        "precision and recall are reported for every class, abstain included"
    assert not [c for c in frame.columns if "drift" in c or c == "unparseable"], \
        "the table carries what was scored; counts and failures are in metrics.json"

    assert driver.checkpoint_table(run, "chunk").iloc[0]["tag"] == tag


def test_the_header_names_the_granularity_the_command_will_use(tmp_path, capsys):
    """It printed the CONFIG's value, which is `both` on a run that covers both.

    So a cell that asked for `chunk` was answered `experiment: both.combined`, and `subset`
    and `sample` -- which write one artifact every granularity shares -- announced one they
    do not have.
    """
    from argparse import Namespace

    from distill.cli import load

    run = a_run(tmp_path)

    load(Namespace(run=str(run), stage_name="config"))
    assert "experiment:" not in capsys.readouterr().out, \
        "a stage with no --granularity must not claim one"

    load(Namespace(run=str(run), stage_name="config", granularity="sentence"))
    assert "experiment: sentence.combined" in capsys.readouterr().out

    load(Namespace(run=str(run), stage_name="config", granularity=None))
    printed = capsys.readouterr().out
    assert "experiment: chunk.combined" in printed, \
        "unasked, it is the first granularity the run covers -- the one the stage falls back to"


def test_the_backtest_takes_the_price_cache_from_the_command(tmp_path):
    """It used to be a third config file, `sed`-ed in from the notebook.

    Everything else in `BacktestConfig` is a default no run ever changed, so it stays in the
    code; the price cache names a machine, so it is named by the command that runs on it.
    """
    from argparse import Namespace

    from distill.cli import load

    run = a_run(tmp_path)
    cache = tmp_path / "prices"
    cache.mkdir()

    config, _, _ = load(Namespace(run=str(run), stage_name="config"))
    assert config.backtest.price_cache is None, "unasked, there is no cache"

    config, _, _ = load(Namespace(run=str(run), stage_name="config", price_cache=str(cache)))
    assert config.repo_path(config.backtest.price_cache) == cache.resolve()


def test_the_metered_search_does_not_start_without_the_flag(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(driver, "extraction", lambda *a, **k: calls.append(a) or 0)
    assert driver.search(tmp_path) is None
    assert "Set RUN_METERED_SEARCH = True" in capsys.readouterr().out
    assert calls == [], "nothing may be called, not even the planning"
    assert driver.embed(tmp_path) is None, "and there is nothing to embed yet"
    assert "Run 1.1 with RUN_METERED_SEARCH = True first" in capsys.readouterr().out
    assert calls == []

    driver.search(tmp_path, confirm=True)
    assert len(calls) == 1


def test_a_finished_search_runs_without_the_flag_because_it_costs_nothing(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(driver, "extraction", lambda *a, **k: calls.append(a) or 0)
    for name in driver.SEARCH_OUTPUTS:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"")
    assert driver.search(tmp_path) is None            # outputs without markers: not finished
    for name in driver.SEARCH_OUTPUTS:
        (tmp_path / f"{name}.done.json").write_text("{}")
    driver.search(tmp_path)
    assert len(calls) == 1


def test_the_search_makes_the_run_folder_a_fresh_clone_does_not_have(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(driver, "extraction", lambda *a, **k: calls.append(a) or 0)
    run = tmp_path / "output"
    assert driver.search(run) is None
    assert not run.exists(), "without the flag nothing is written, not even the folder"
    driver.search(run, confirm=True)
    assert run.is_dir() and len(calls) == 1


def test_a_news_only_search_is_finished_without_a_transcripts_file(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(driver, "extraction", lambda *a, **k: calls.append(a) or 0)
    news = next(name for name in driver.SEARCH_OUTPUTS if name.endswith("_news.parquet"))
    (tmp_path / news).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / news).write_bytes(b"")
    (tmp_path / f"{news}.done.json").write_text("{}")
    assert driver.search_done(tmp_path), "the build writes no file for a type with no rows"
    driver.embed(tmp_path)
    assert len(calls) == 1


def test_an_option_set_to_none_is_left_out_of_the_stage_command(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(driver.subprocess, "run",
                        lambda command, env: commands.append(command) or
                        type("Done", (), {"returncode": 0})())
    driver.stage(tmp_path, "sample", "--max-rows", None)
    driver.stage(tmp_path, "sample", "--max-rows", 15000)
    assert commands[0][-1] == "sample", "None drops the flag with it"
    assert commands[1][-2:] == ["--max-rows", "15000"]
