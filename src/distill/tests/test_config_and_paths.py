"""The config file and the run folder."""

from __future__ import annotations

from pathlib import Path

import pytest

from distill.config import DistillConfig, TrainingConfig
from distill.paths import RunPaths, experiment_key
from distill.serving import ServeConfig, render_serve_command, serve_warnings

from conftest import theme_section  # noqa: E402
from distill.theme import ThemeSpec  # noqa: E402

_THEMES = Path(__file__).resolve().parent.parent / "themes"
THEME_SECTION = theme_section(ThemeSpec.load(_THEMES / "pricing_power.yaml"))
def a_config(tmp_path: Path, **overrides) -> DistillConfig:
    """`backtest` is its own file, so it is set on the object rather than passed in."""
    from distill.config import BacktestConfig

    backtest = overrides.pop("backtest", None)
    data = {"input": {"source_parquet": "corpus.parquet"}}
    data.update(overrides)
    config = DistillConfig.from_dict(data, base=tmp_path, theme=THEME_SECTION)
    if backtest is not None:
        config.backtest = BacktestConfig(**backtest)
    return config


def test_an_unknown_key_is_refused_rather_than_ignored(tmp_path):
    """A typo that silently keeps the default is a run not doing what its config says."""
    with pytest.raises(SystemExit, match="unknown key"):
        DistillConfig.from_dict({"sample": {"binz": 20}}, theme=THEME_SECTION,
                                base=tmp_path)


def test_evaluate_epochs_outside_the_training_range_is_refused(tmp_path):
    config = a_config(tmp_path, training={"epochs": 3, "evaluate_epochs": [3, 5],
                                          "keep_merged": [3]})
    assert any("outside 1..3" in issue for issue in config.validate())


def test_keep_merged_must_be_something_that_gets_merged(tmp_path):
    config = a_config(tmp_path, training={"epochs": 5, "evaluate_epochs": [3],
                                          "keep_merged": [3, 5]})
    assert any("not in evaluate_epochs" in issue for issue in config.validate())

def test_an_unknown_signal_mode_is_refused(tmp_path):
    """An error, not a silent fallback to the default."""
    config = a_config(tmp_path, backtest={"signal_mode": "weighted"})
    assert any("class_sign|relevance_class_sign" in issue for issue in config.validate())
    assert a_config(tmp_path, backtest={"signal_mode": "class_sign"}).validate() == []


def test_the_topology_decides_which_training_sets_are_needed(tmp_path):
    assert a_config(tmp_path, experiment={"topology": "combined"}).datasets_for_topology() \
        == ("both",)
    assert a_config(tmp_path, experiment={"topology": "specialist"}).datasets_for_topology() \
        == ("news", "transcripts")
    assert a_config(tmp_path, experiment={"topology": "both"}).datasets_for_topology() \
        == ("news", "transcripts", "both")


def test_student_overrides_state_only_the_differences(tmp_path):
    """The specialist corpus pass labels its two halves with two models on two endpoints."""
    config = a_config(
        tmp_path,
        student={"url": "http://localhost:8000/v1/chat/completions", "model": "combined",
                 "workers": 384},
        student_overrides={"news": {"url": "http://localhost:8001/v1/chat/completions",
                                    "model": "news-e5"}},
    )
    news = config.student_for("news")
    assert news.model == "news-e5" and news.url.endswith("8001/v1/chat/completions")
    assert news.workers == 384, "unstated settings come from `student`"
    assert config.student_for("transcripts").model == "combined"
    assert config.student_for(None).model == "combined"


def test_student_override_keys_must_be_content_types(tmp_path):
    config = a_config(tmp_path, student_overrides={"e3": {}})
    assert any("student_overrides keys" in issue for issue in config.validate())


def test_relative_paths_resolve_against_the_config_file(tmp_path):
    config = a_config(tmp_path)
    assert config.path("a/b.parquet") == tmp_path / "a/b.parquet"
    assert config.path("/abs/b.parquet") == Path("/abs/b.parquet")
    assert config.path(None) is None


def test_a_config_round_trips_through_yaml(tmp_path):
    config = a_config(tmp_path, sample={"bins": 8, "seed": 99})
    path = config.save(tmp_path / "distill.yaml")
    reloaded = DistillConfig.load(path)
    assert reloaded.sample.bins == 8 and reloaded.sample.seed == 99
    assert reloaded.subset.content_types == ("news", "transcripts")


def test_artifact_paths_come_from_the_experiment_not_from_a_tag(run_dir):
    paths = RunPaths.for_run(run_dir)
    assert experiment_key("chunk", "combined") == "chunk.combined"
    combined = paths.corpus_parquet("chunk", "combined")
    specialist = paths.corpus_parquet("chunk", "specialist")
    assert combined != specialist
    assert "chunk.combined" in str(combined) and "chunk.specialist" in str(specialist)
    assert "chunk.combined" in str(paths.corpus_checkpoint("chunk", "combined", "news"))


def test_a_dry_run_cannot_name_a_real_artifact(run_dir):
    """A `--dry-run --limit 20` once destroyed a real 4,829-row evaluation."""
    real = RunPaths.for_run(run_dir)
    dry = RunPaths.for_run(run_dir, dry_run=True)
    assert real.subset_parquet != dry.subset_parquet
    relative = lambda path: path.relative_to(run_dir).parts
    assert relative(dry.subset_parquet)[0] == "_dry_run"
    assert relative(real.subset_parquet)[0] != "_dry_run"
    for method in ("sample_parquet", "heldout_keys_parquet"):
        assert relative(getattr(dry, method))[0] == "_dry_run"
    assert relative(dry.eval_metrics_json("chunk", "combined", "t"))[0] == "_dry_run"


def test_the_split_lives_with_the_labels_that_define_it(run_dir):
    paths = RunPaths.for_run(run_dir)
    assert paths.train_split("chunk").parent == paths.teacher_dir("chunk")
    assert paths.heldout_split("chunk") == paths.heldout_split("chunk")
    assert paths.heldout_split("chunk") != paths.heldout_split("sentence")


def test_manifests_sit_beside_their_artifacts(run_dir):
    paths = RunPaths.for_run(run_dir)
    assert paths.manifest_for(paths.subset_parquet).name == "subset.parquet.manifest.json"
    directory = paths.backtest_dir()
    assert paths.manifest_for(directory).name == "manifest.json"


def test_the_docker_line_is_rendered_from_the_config():
    serve = ServeConfig(image="vllm/vllm-openai:v0.20.0", container="tx-stud", gpus="all",
                        port=8000, model_path="/merged/e3", served_model_name="e3",
                        chat_template="/code/nothink.jinja",
                        args=("--max-model-len", "4096", "--max-num-seqs", "512"))
    command = render_serve_command(serve)
    assert command.startswith("docker run -d --name tx-stud")
    assert "--rm" not in command, "a crash at startup must leave logs"
    assert "--chat-template /code/nothink.jinja" in command
    assert serve_warnings(serve, role="student") == []


def test_a_student_without_a_chat_template_is_warned_about():
    """The merged directory carries the base model's template, whose default is reasoning-on."""
    serve = ServeConfig(model_path="/merged/e3", args=("--max-num-seqs", "512"))
    warnings = serve_warnings(serve, role="student")
    assert any("chat_template" in w for w in warnings)


def test_rm_and_a_missing_seq_cap_are_warned_about():
    serve = ServeConfig(model_path="/m", chat_template="/t", args=("--rm",), detach=False)
    warnings = " ".join(serve_warnings(serve, role="student"))
    assert "--rm" in warnings and "max-num-seqs" in warnings and "-d" in warnings


def test_printed_commands_quote_paths_that_contain_spaces():
    """The step folders no longer carry a space, but a run folder the user named can, and an
    unquoted path with a space silently becomes two arguments."""
    from distill.config import TrainingConfig
    from distill.serving import render_merge_command, render_train_command
    command = render_train_command(
        TrainingConfig(), dataset_path=Path("/a run/training/x/train.both.jsonl"),
        output_dir=Path("/a run/training/x/adapters/both"),
    )
    assert "'/a run/training/x/train.both.jsonl'" in command
    assert "'/a run/training/x/adapters/both'" in command
    merge = render_merge_command(TrainingConfig(), adapter=Path("/a run/training/a"),
                                 output=Path("/a run/training/m"))
    assert "'/a run/training/a'" in merge and "'/a run/training/m'" in merge


def test_the_cli_help_renders_for_every_subcommand():
    """argparse %-formats help strings."""
    import contextlib
    import io

    from distill.cli import build_parser
def test_the_configuration_lives_beside_the_output(run_dir):
    """`config/` is the sibling of the output folder: three files, one per stage, checked in
    and never copied into the output."""
    paths = RunPaths.for_run(run_dir)
    assert paths.config_dir == run_dir.parent / "config"
    assert paths.extraction_config == run_dir.parent / "config" / "data_extraction.yaml"
    assert paths.distill_config == run_dir.parent / "config" / "distill.yaml"
    assert not hasattr(paths, "theme_file"), "the theme is a section of distill.yaml"
    assert not hasattr(paths, "backtest_config"), \
        "the backtest holds two settings, and they are flags on the command"


def test_relative_paths_in_the_config_resolve_against_the_run(run_dir, pricing_power):
    """`input.similarity_parquet: "7."""
    import shutil

    from distill.paths import config_dir as repo_config_dir

    config_dir = run_dir.parent / "config"
    config_dir.mkdir()
    shutil.copy(repo_config_dir() / "distill.yaml", config_dir / "distill.yaml")
    config = DistillConfig.load(config_dir / "distill.yaml", base=run_dir)
    assert config.base == run_dir
    assert config.path(config.input.similarity_parquet) == \
        run_dir / "similarity" / "combinations_with_similarity.parquet"
def test_the_theme_comes_out_of_the_config_beside_the_run(tmp_path, capsys):
    """No `--theme` on any stage: the theme is a section of the one config/, and editing it
    there is what every stage reads."""
    import dataclasses

    from conftest import install_config
    from distill.template import write_extraction_template

    run = tmp_path / "output"
    run.mkdir()
    install_config(run, theme="pricing_power")
    capsys.readouterr()

    config_path = run.parent / "config" / "distill.yaml"
    seeded = DistillConfig.load(config_path, base=run).theme_spec()
    edited = dataclasses.replace(seeded, body=seeded.body + "\nAN EDIT ONLY THIS RUN HAS.")
    write_extraction_template(run.parent / "config" / "data_extraction.yaml", theme=edited)

    reread = DistillConfig.load(config_path, base=run).theme_spec()
    assert "AN EDIT ONLY THIS RUN HAS." in reread.body
    assert reread.fingerprint() != seeded.fingerprint(), \
        "editing the body has to change the theme's fingerprint"

def test_a_config_naming_a_theme_instead_of_carrying_it_is_refused(tmp_path):
    """The old shape."""
    import yaml

    run = tmp_path / "R_2026-01-01"
    (run.parent / "config").mkdir(parents=True)
    (run.parent / "config" / "data_extraction.yaml").write_text(
        yaml.safe_dump({"theme": "pricing_power"}), encoding="utf-8")
    path = run.parent / "config" / "distill.yaml"
    path.write_text(yaml.safe_dump({"input": {"source_parquet": "x.parquet"}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="has to be the theme itself"):
        DistillConfig.load(path)


def test_the_theme_in_the_wrong_config_file_says_which_file_it_belongs_in(tmp_path):
    """Two files could plausibly hold it, so being wrong has to name the right one."""
    import yaml

    run = tmp_path / "R_2026-01-01"
    (run.parent / "config").mkdir(parents=True)
    path = run.parent / "config" / "distill.yaml"
    path.write_text(yaml.safe_dump({"theme": {"id": "pricing_power"}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="data_extraction.yaml"):
        DistillConfig.load(path)


def test_a_run_with_no_theme_anywhere_is_refused(tmp_path):
    """No data_extraction.yaml at all: distill.yaml alone cannot say what the run means."""
    import yaml

    run = tmp_path / "R_2026-01-01"
    (run.parent / "config").mkdir(parents=True)
    path = run.parent / "config" / "distill.yaml"
    path.write_text(yaml.safe_dump({"input": {"source_parquet": "x.parquet"}}), encoding="utf-8")
    config = DistillConfig.load(path)
    assert any("no `theme:` section" in issue for issue in config.validate())
def test_a_run_folder_without_a_config_says_how_to_make_one(tmp_path):
    import argparse

    from distill.cli import load
def test_the_market_share_theme_in_the_library_is_valid():
    """Every theme shipped in `distill/themes/` has to pass the gate it will be checked by."""
    from distill.theme import ThemeSpec

    themes = Path(__file__).resolve().parent.parent / "themes"
    for path in sorted(themes.glob("*.yaml")):
        theme = ThemeSpec.load(path).require_valid()
        assert theme.id == path.stem, f"{path.name} declares id {theme.id!r}"
        assert len(theme.body.strip()) > 500, f"{path.name} has a thin body"


def test_a_run_folder_can_be_moved(tmp_path):
    """A run has to survive being moved between machines and between folders. Nothing in its
    config may name its own location: `run_dir` did, it was in the config hash, and moving a
    run therefore looked like a different experiment."""
    import dataclasses
    import shutil

    from conftest import install_config, make_corpus
    from distill.manifest import config_hash
    from distill.paths import RunPaths

    first = tmp_path / "old" / "output"
    (first / "similarity").mkdir(parents=True)
    make_corpus(30).to_parquet(first / "similarity"
                               / "combinations_with_similarity.parquet", index=False)
    install_config(first)
    before = DistillConfig.load_run(RunPaths.for_run(first))

    (tmp_path / "new").mkdir()
    shutil.move(str(tmp_path / "old" / "config"), str(tmp_path / "new" / "config"))
    second = tmp_path / "new" / "output"
    shutil.move(str(first), str(second))

    after = DistillConfig.load_run(RunPaths.for_run(second))
    def configured(cfg):
        """Everything the config SAYS, with the private fields that record where it was
        loaded from left out -- those are the location, not the experiment."""
        return {f.name: (dataclasses.asdict(getattr(cfg, f.name))
                         if dataclasses.is_dataclass(getattr(cfg, f.name))
                         else getattr(cfg, f.name))
                for f in dataclasses.fields(cfg) if not f.name.startswith("_")}

    assert config_hash(configured(after)) == config_hash(configured(before)), \
        "moving a run must not change what it is"
    assert after.path(after.input.similarity_parquet) == \
        second / "similarity" / "combinations_with_similarity.parquet"

def test_a_config_that_still_carries_run_dir_is_accepted_and_ignored(tmp_path):
    """Configs written before the field was dropped must keep loading."""
    import yaml

    from distill.config import DistillConfig

    run = tmp_path / "R_2026-01-01"
    (run.parent / "config").mkdir(parents=True)
    (run.parent / "config" / "distill.yaml").write_text(yaml.safe_dump({
        "run_dir": "/somewhere/that/no/longer/exists",
        "input": {"source_parquet": "corpus.parquet"},
    }), encoding="utf-8")
    config = DistillConfig.load(run.parent / "config" / "distill.yaml")
    assert not hasattr(config, "run_dir")
    assert "run_dir" not in config.payload()


def test_no_subcommand_turns_the_run_folder_into_a_flag():
    """`--run <folder>` is THE argument every stage takes. A subcommand that redefines `--run`
    as a boolean silently replaces the folder with True/False, and that stage becomes
    unreachable: `distill --run <folder> train` could not work at all.

    `init-run` redefining it as a PATH is fine -- same kind of argument, and it is the one
    command whose folder need not exist yet."""
    import argparse

    from distill.cli import build_parser

    subparsers = next(a for a in build_parser()._actions
                      if isinstance(a, argparse._SubParsersAction))
    offenders = []
    for name, sub in subparsers.choices.items():
        for action in sub._actions:
            if "--run" in action.option_strings and action.nargs == 0:
                offenders.append(f"{name} defines --run as a flag, not a path")
    assert not offenders, offenders



def test_repo_paths_and_run_paths_resolve_from_different_roots(tmp_path, pricing_power):
    """The price cache sits beside config/; the similarity parquet is written inside the run.

    `load_run` bases every relative path on the RUN folder, which the similarity parquet
    needs. Applied to `backtest.price_cache` it would look for the cache inside `output/`.
    """
    from distill.config import DistillConfig
    from distill.paths import RunPaths

    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    (root / "output").mkdir()
    (root / "prices_cache").mkdir()
    (root / "config" / "distill.yaml").write_text(
        "input:\n  similarity_parquet: similarity/combinations_with_similarity.parquet\n",
        encoding="utf-8")
    import yaml
    (root / "config" / "data_extraction.yaml").write_text(
        yaml.safe_dump({"theme": theme_section(pricing_power)}), encoding="utf-8")

    config = DistillConfig.load_run(RunPaths.for_run(root / "output"))
    config.backtest.price_cache = "prices_cache"

    cache = config.repo_path(config.backtest.price_cache)
    assert cache == (root / "prices_cache").resolve()
    assert cache.exists()

    similarity = config.path(config.input.similarity_parquet)
    assert similarity.parent.parent == (root / "output").resolve()

    assert config.repo_path(None) is None
    assert config.repo_path("/absolute/elsewhere") == Path("/absolute/elsewhere")


def test_a_hand_built_corpus_is_enough_to_start_part_two(tmp_path):
    """The notebook tells a reader which columns to bring, and this is that list.

    Written out, a corpus assembled elsewhere has to reach `subset` without the retrieval
    having run at all -- and without a similarity column, which is derived from the
    per-sentence scores rather than required.
    """
    import numpy as np
    import pandas as pd

    from conftest import install_config
    from distill.cli import main

    run = tmp_path / "output"
    run.mkdir(parents=True)
    install_config(run, theme="market_share")

    rng = np.random.default_rng(0)
    rows = 120
    documented = pd.DataFrame({
        "doc_id": [f"DOC{i // 2:05d}" for i in range(rows)],
        "chunk_index": [i % 7 for i in range(rows)],
        "entity_id": [f"E{i % 20:04d}" for i in range(rows)],
        "content_type": rng.choice(["news", "transcripts"], rows),
        "entity_name": [f"Company {i % 20}" for i in range(rows)],
        "chunk_text": ["It took share from its rivals."] * rows,
        "headline": ["Quarterly update"] * rows,
        "date": ["2024-06-03"] * rows,
        "timestamp": ["12:00:00"] * rows,
        "chunk_relevance": rng.random(rows),
        "chunk_sentiment": rng.uniform(-1, 1, rows),
        "extracted_sentences": [["It took share.", "Margins held."]] * rows,
        "extracted_sentence_sims": [list(rng.random(2)) for _ in range(rows)],
    })
    similarity = run / "similarity"
    similarity.mkdir(parents=True)
    documented.to_parquet(similarity / "combinations_with_similarity.parquet", index=False)

    assert main(["--run", str(run), "subset", "--top-fraction", "1"]) == 0

    subset = pd.read_parquet(run / "subset" / "subset.parquet")
    assert len(subset) == rows
    assert "sim_top" in subset.columns, "the cut metric is derived, not required"
    assert "row_key" in subset.columns
    assert subset["row_key"].is_unique


def test_every_model_is_pinned_and_the_download_cell_fetches_those_same_revisions():
    """Loaded by name alone, a model is whatever was uploaded last under that name."""
    import json
    import re
    import sys

    root = Path(__file__).resolve().parents[3]
    config = DistillConfig.load(root / "config" / "distill.yaml")
    sys.path.insert(0, str(root / "src" / "data_extraction" / "src"))
    from encoders import MODELS

    pinned = {
        MODELS["nemotron-1b"].hf_id: MODELS["nemotron-1b"].revision,
        config.teacher.serve.model_path: config.teacher.serve.revision,
        config.training.base_model: config.training.base_revision,
    }
    assert all(re.fullmatch(r"[0-9a-f]{40}", r or "") for r in pinned.values()), pinned

    notebook = json.loads((root / "notebooks" / "dataset-generation-pipeline.ipynb").read_text())
    cell = next("".join(c["source"]) for c in notebook["cells"] if "hf download" in "".join(c["source"]))
    downloaded = dict(re.findall(r"hf download (\S+) \\\s+--revision (\S+)", cell))
    assert downloaded == pinned


def test_the_revision_reaches_every_command_that_loads_a_base_model(tmp_path):
    from distill import serving

    serve = ServeConfig(image="vllm/vllm-openai:v0.20.0", model_path="nvidia/teacher",
                        revision="a" * 40)
    assert f"--model nvidia/teacher --revision {'a' * 40}" in serving.render_serve_command(serve)

    training = TrainingConfig(base_model="nvidia/student", base_revision="b" * 40)
    train = serving.render_train_command(training, dataset_path=tmp_path / "d.jsonl",
                                         output_dir=tmp_path / "out")
    assert f"--revision {'b' * 40}" in train
    merge = serving.render_merge_command(training, adapter=tmp_path / "a", output=tmp_path / "m")
    assert f"--revision {'b' * 40}" in merge

    unpinned = TrainingConfig(base_model="nvidia/student")
    assert "--revision" not in serving.render_merge_command(unpinned, adapter=tmp_path / "a",
                                                            output=tmp_path / "m")
