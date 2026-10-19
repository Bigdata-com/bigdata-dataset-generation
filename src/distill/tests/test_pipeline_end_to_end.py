"""The whole pipeline, end to end, with no server and no GPU."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from distill.config import BacktestConfig, DistillConfig
from distill.paths import RunPaths
from distill.stages import backtest, corpus, evaluate, export, sample, subset, teacher
from distill.theme import ThemeSpec

from conftest import theme_section  # noqa: E402
from distill.theme import ThemeSpec  # noqa: E402

_THEMES = Path(__file__).resolve().parent.parent / "themes"
THEME_SECTION = theme_section(ThemeSpec.load(_THEMES / "pricing_power.yaml"))

from conftest import make_corpus, write_price_cache

THEMES_DIR = Path(__file__).resolve().parent.parent / "themes"

CONFIG = {
    "input": {"source_parquet": "corpus.parquet", "irreproducible_source": True},
    "subset": {"metric": "sim_top", "top_fraction": 0.8, "per_content_type": True},
    "sample": {"bins": 4, "bin_mode": "quantile", "fraction_per_bin": 0.5, "seed": 17},
    "experiment": {"granularity": "chunk", "topology": "combined"},
    "split": {"heldout_fraction": 0.2, "seed": 17},
    "training": {"epochs": 5, "evaluate_epochs": [3, 5], "keep_merged": [3, 5]},
    "teacher": {"workers": 4, "health_attempts": 0},
    "student": {"workers": 4, "health_attempts": 0},
}


@pytest.fixture
def pipeline(tmp_path: Path):
    run_dir = tmp_path / "PricingPower_2026-08-24"
    run_dir.mkdir()
    make_corpus(240).to_parquet(run_dir / "corpus.parquet", index=False)
    config = DistillConfig.from_dict(CONFIG, base=run_dir,
                                     theme=THEME_SECTION).require_valid()
    config.backtest = BacktestConfig(signal_mode="class_sign", chart=False)
    theme = ThemeSpec.load(THEMES_DIR / "pricing_power.yaml").require_valid()
    paths = RunPaths.for_run(run_dir, dry_run=True)
    return config, theme, paths


def _manifest(paths: RunPaths, artifact: Path) -> dict:
    path = paths.manifest_for(artifact)
    assert path.exists(), f"no manifest beside {artifact}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_whole_pipeline_runs_dry_and_every_artifact_is_provenanced(pipeline, capsys):
    config, theme, paths = pipeline

    result = subset.run(config, theme, paths)
    assert paths.subset_parquet.exists()
    subset_frame = pd.read_parquet(paths.subset_parquet)
    assert len(subset_frame) == result["rows"]
    source = pd.read_parquet(paths.run_dir / "corpus.parquet")
    for content_type in ("news", "transcripts"):
        available = int((source["content_type"] == content_type).sum())
        kept = int((subset_frame["content_type"] == content_type).sum())
        assert 0.7 * available <= kept <= 0.9 * available, (content_type, available, kept)
    manifest = _manifest(paths, paths.subset_parquet)
    assert manifest["inputs"]["source"]["sha256"], "an input is identified by content, not name"
    assert manifest["config_hash"]
    assert any("irreproducible" in note for note in manifest["notes"]), (
        "a source no script can rebuild has to say so in every manifest"
    )

    sample.run(config, theme, paths)
    sampled = pd.read_parquet(paths.sample_parquet)
    assert {"bin", "bin_lo", "bin_hi"} <= set(sampled.columns)
    assert 0 < len(sampled) < len(subset_frame)
    sample.run(config, theme, paths, force=True)
    pd.testing.assert_frame_equal(pd.read_parquet(paths.sample_parquet), sampled)

    teacher.run(config, theme, paths, dry_run=True, assume_yes=True)
    labels = pd.read_parquet(paths.teacher_labels("chunk"))
    assert len(labels) == len(sampled)
    assert labels["ok"].all()
    assert set(labels["classification"]) <= set(theme.class_names)
    assert (labels["text"] == labels["chunk_text"]).all(), (
        "at chunk level the text that went on the wire IS the chunk"
    )
    prompts_dir = paths.teacher_prompts_dir("chunk")
    assert (prompts_dir / "system.chunk.news.txt").exists()
    assert (prompts_dir / "system.chunk.transcripts.txt").exists()
    assert (prompts_dir / "stub.chunk.txt").read_text().startswith("You are a pricing-power")
    fingerprint = (prompts_dir / "fingerprint.txt").read_text().strip()
    teacher_manifest = _manifest(paths, paths.teacher_labels("chunk"))
    assert teacher_manifest["fingerprints"]["prompt"] == fingerprint

    before = paths.teacher_checkpoint("chunk", "news").read_text()
    teacher.run(config, theme, paths, dry_run=True, assume_yes=True, force=True)
    assert paths.teacher_checkpoint("chunk", "news").read_text() == before

    split = teacher.run_split(config, theme, paths)
    held = pd.read_parquet(paths.heldout_split("chunk"))
    train = pd.read_parquet(paths.train_split("chunk"))
    assert split["train"] + split["heldout"] == len(labels)
    assert not set(held["row_key"]) & set(train["row_key"])
    assert paths.heldout_keys_parquet.exists(), "the portable key list, for another granularity"
    with pytest.raises(SystemExit, match="--new-split refused"):
        teacher.run_split(config, theme, paths, new_split=True)
    assert teacher.run_split(config, theme, paths)["reused"] is True

    exported = export.run(config, theme, paths)
    jsonl = paths.train_jsonl("chunk", "combined", "both")
    assert jsonl.exists()
    records = [json.loads(line) for line in jsonl.read_text().splitlines()]
    assert len(records) == len(train)
    first = records[0]
    assert [m["role"] for m in first["request"]["messages"]] == ["system", "user"]
    assert first["request"]["messages"][0]["content"].startswith("You are a pricing-power")
    assert first["workload_id"] == "pricing_power"
    assert first["prompt_fingerprint"] == exported["prompt_fingerprint"]
    target = json.loads(first["response"]["choices"][0]["message"]["content"])
    assert set(target) == {"classification", "reasoning"}, "the label and the rationale, and no third field"
    assert list(target) == ["classification", "reasoning"], "the rationale is emitted LAST"

    for epoch in config.training.evaluate_epochs:
        scored = evaluate.run(config, theme, paths, epoch=epoch, dry_run=True)
        assert scored["rows"] == len(held)
        assert Path(scored["metrics"]).exists()
    tags = sorted(p.name for p in paths.evaluations_dir("chunk", "combined")
                  .glob("*.metrics.json"))
    assert tags == ["chunk.combined.both.e3.metrics.json",
                    "chunk.combined.both.e5.metrics.json"], (
        "the tag is derived from the experiment and the epoch, never typed"
    )
    comparison = paths.eval_comparison("chunk", "combined").read_text()
    assert "e3" in comparison and "e5" in comparison
    assert "does not choose" in comparison
    text = paths.eval_metrics_txt("chunk", "combined", "chunk.combined.both.e3").read_text()
    for block in ("OVERALL", "NEWS", "TRANSCRIPTS", "calibration", "MACRO-F1", "confusion"):
        assert block in text, block

    labelled = corpus.run(config, theme, paths, dry_run=True, assume_yes=True)
    corpus_frame = pd.read_parquet(paths.corpus_parquet("chunk", "combined"))
    assert len(corpus_frame) == len(subset_frame), (
        "the corpus pass labels the SUBSET, not the sample"
    )
    assert labelled["signal_share"] >= 0
    corpus_manifest = _manifest(paths, paths.corpus_parquet("chunk", "combined"))
    assert corpus_manifest["fingerprints"]["prompt"] == exported["prompt_fingerprint"]
    assert corpus_manifest["fingerprints"]["prompt"] != fingerprint
    assert corpus_manifest["inputs"]["subset"]["sha256"]
    assert corpus_manifest["outputs"]["checkpoint.news"]["travels"] is False

    config.backtest.price_cache = str(write_price_cache(
        paths.run_dir.parent / "prices_cache", subset_frame["entity_id"],
        "2023-11-01", "2025-04-30"))
    config.backtest.half_life = 1
    books = backtest.run(config, theme, paths)
    assert paths.backtest_comparison.exists()
    assert "chunk.combined" in books["books"]
    assert books["priced"] is True, "a cache that covers the window prices every book"
    signals = pd.read_parquet(paths.backtest_book_dir("chunk.combined") / "signals.parquet")
    assert list(signals.columns) == ["date", "entity_id", "signal"]
    assert "sentiment_only" in books["books"], "the floor is mandatory, not optional"
    assert "reference" in books["books"], "so is the reference label set, where one exists"
    assert {"chunk.combined.h1", "reference.h1"} <= set(books["books"]), (
        "a half-life puts a smoothed book beside every labelled one")
    assert "sentiment_only.h1" not in books["books"], "the floor stays raw"

    escaped = [p for p in paths.run_dir.iterdir()
               if p.name not in {"_dry_run", "corpus.parquet"}]
    assert escaped == [], f"a dry run wrote outside _dry_run/: {escaped}"


def test_the_same_pipeline_runs_at_sentence_level_without_a_second_code_path(pipeline):
    """The granularity fork is gone: same stages, same records, a different strategy."""
    config, theme, paths = pipeline
    config.experiment.granularity = "sentence"
    subset.run(config, theme, paths)
    sample.run(config, theme, paths)
    teacher.run(config, theme, paths, dry_run=True, assume_yes=True)
    labels = pd.read_parquet(paths.teacher_labels("sentence"))
    assert {"n_candidates", "n_calls", "sentence_sim", "reasoning_from"} <= set(labels.columns)
    assert (labels["n_calls"] <= labels["n_candidates"].clip(lower=1)).all()
    stopped = labels.loc[labels["reasoning_from"] == "stop"]
    if len(stopped):
        assert (stopped["text"].str.len() > 0).all()
        assert stopped["text"].isin(
            [s for row in pd.read_parquet(paths.sample_parquet)["extracted_sentences"]
             for s in row]
        ).all()
    teacher.run_split(config, theme, paths)
    export.run(config, theme, paths)
    jsonl = paths.train_jsonl("sentence", "combined", "both")
    record = json.loads(jsonl.read_text().splitlines()[0])
    system = record["request"]["messages"][0]["content"]
    assert "label the sentence" in system, "the sentence-level stub, derived from the granularity"
    assert "SENTENCE:" in record["request"]["messages"][1]["content"]


def test_a_second_granularity_can_reuse_the_first_one_s_heldout_keys(pipeline):
    """The only reason a chunk-level and a sentence-level student are comparable at all."""
    config, theme, paths = pipeline
    subset.run(config, theme, paths)
    sample.run(config, theme, paths)
    teacher.run(config, theme, paths, dry_run=True, assume_yes=True)
    teacher.run_split(config, theme, paths)
    chunk_keys = set(pd.read_parquet(paths.heldout_keys_parquet)["row_key"])

    config.experiment.granularity = "sentence"
    config.split.heldout_keys_from = str(paths.heldout_keys_parquet)
    teacher.run(config, theme, paths, dry_run=True, assume_yes=True)
    teacher.run_split(config, theme, paths)
    sentence_held = set(pd.read_parquet(paths.heldout_split("sentence"))["row_key"])
    assert sentence_held == chunk_keys


def test_the_specialist_topology_labels_each_half_with_its_own_model(pipeline):
    config, theme, paths = pipeline
    config.experiment.topology = "specialist"
    config.student_overrides = {
        "news": {"model": "news-e5", "url": "http://localhost:8001/v1/chat/completions"},
        "transcripts": {"model": "trans-e5", "url": "http://localhost:8002/v1/chat/completions"},
    }
    subset.run(config, theme, paths)
    sample.run(config, theme, paths)
    teacher.run(config, theme, paths, dry_run=True, assume_yes=True)
    teacher.run_split(config, theme, paths)
    exported = export.run(config, theme, paths)
    assert set(exported["datasets"]) == {"news", "transcripts"}, (
        "a specialist topology needs one training set per content type"
    )
    corpus.run(config, theme, paths, content_type="news", dry_run=True, assume_yes=True)
    result = corpus.run(config, theme, paths, content_type="transcripts", dry_run=True,
                        assume_yes=True)
    assert result["rows"] == len(pd.read_parquet(paths.subset_parquet))
    manifest = _manifest(paths, paths.corpus_parquet("chunk", "specialist"))
    per_type = manifest["model"]["per_content_type"]
    assert per_type["transcripts"]["model"] == "trans-e5"
    assert paths.corpus_parquet("chunk", "specialist") != \
        paths.corpus_parquet("chunk", "combined")


def test_two_label_sets_cross_tabulate_one_to_one(pipeline):
    config, theme, paths = pipeline
    subset.run(config, theme, paths)
    sample.run(config, theme, paths)
    teacher.run(config, theme, paths, dry_run=True, assume_yes=True)
    teacher.run_split(config, theme, paths)
    export.run(config, theme, paths)
    corpus.run(config, theme, paths, dry_run=True, assume_yes=True)
    combined = pd.read_parquet(paths.corpus_parquet("chunk", "combined"))
    table = corpus.cross_tabulate(combined, combined, label_field="classification",
                                 names=("a", "b"))
    total = int(table.loc["All", "All"])
    assert total == int(combined["ok"].sum()) or total == len(combined)
    with pytest.raises(Exception):
        corpus.cross_tabulate(pd.concat([combined, combined]), combined,
                              label_field="classification", names=("a", "b"))


def test_the_input_assertion_stops_the_run_before_any_call(pipeline):
    """On the unrepaired corpus this fires on 130,071 of 484,982 rows."""
    config, theme, paths = pipeline
    make_corpus(240, blank_rows=30).to_parquet(paths.run_dir / "corpus.parquet", index=False)
    subset.run(config, theme, paths)
    sample.run(config, theme, paths)
    with pytest.raises(SystemExit, match="null or blank"):
        teacher.run(config, theme, paths, dry_run=True, assume_yes=True)
    assert not paths.teacher_labels("chunk").exists()


def test_a_fingerprint_mismatch_stops_the_corpus_pass(pipeline):
    config, theme, paths = pipeline
    subset.run(config, theme, paths)
    sample.run(config, theme, paths)
    teacher.run(config, theme, paths, dry_run=True, assume_yes=True)
    teacher.run_split(config, theme, paths)
    export.run(config, theme, paths)
    jsonl = paths.train_jsonl("chunk", "combined", "both")
    lines = jsonl.read_text().splitlines()
    record = json.loads(lines[0])
    record["prompt_fingerprint"] = "0000000000000000"
    lines[0] = json.dumps(record)
    jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="fingerprint mismatch"):
        corpus.run(config, theme, paths, dry_run=True, assume_yes=True)


def test_epoch_zero_skips_the_fingerprint_gate_and_a_trained_epoch_does_not(pipeline, capsys):
    """The gate asks whether a model is served the prompt it was trained on. The base model
    (epoch 0) was trained on nothing, so it is neither stopped nor warned; a checkpoint is."""
    config, theme, paths = pipeline
    subset.run(config, theme, paths)
    sample.run(config, theme, paths)
    teacher.run(config, theme, paths, dry_run=True, assume_yes=True)
    teacher.run_split(config, theme, paths)
    export.run(config, theme, paths)
    jsonl = paths.train_jsonl("chunk", "combined", "both")
    lines = jsonl.read_text().splitlines()
    record = json.loads(lines[0])
    record["prompt_fingerprint"] = "0000000000000000"
    lines[0] = json.dumps(record)
    jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")

    capsys.readouterr()
    evaluate.run(config, theme, paths, epoch=0, prompt_style="full", model_label="base",
                 dry_run=True)
    assert "fingerprint mismatch" not in capsys.readouterr().out
    with pytest.raises(SystemExit, match="fingerprint mismatch"):
        evaluate.run(config, theme, paths, epoch=3, dry_run=True)


def test_a_drifted_system_message_is_caught_byte_for_byte(pipeline):
    """The check the fingerprint cannot do on its own: what the WEIGHTS were trained with."""
    config, theme, paths = pipeline
    subset.run(config, theme, paths)
    sample.run(config, theme, paths)
    teacher.run(config, theme, paths, dry_run=True, assume_yes=True)
    teacher.run_split(config, theme, paths)
    export.run(config, theme, paths)
    jsonl = paths.train_jsonl("chunk", "combined", "both")
    lines = jsonl.read_text().splitlines()
    record = json.loads(lines[0])
    record["request"]["messages"][0]["content"] += " And one more instruction."
    lines[0] = json.dumps(record)
    jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="NOT the one the student was trained on"):
        corpus.run(config, theme, paths, dry_run=True, assume_yes=True)


def test_granularity_both_runs_the_whole_chain_twice(pipeline):
    """`granularity: both` labels the SAME sample with two teachers reading different text."""
    config, theme, paths = pipeline
    config.experiment.granularity = "both"
    assert config.granularities() == ("chunk", "sentence")

    subset.run(config, theme, paths)
    sample.run(config, theme, paths)
    teacher.run(config, theme, paths, dry_run=True, assume_yes=True)
    teacher.run_split(config, theme, paths)

    chunk_keys = set(pd.read_parquet(paths.heldout_split("chunk"))["row_key"])
    sentence_keys = set(pd.read_parquet(paths.heldout_split("sentence"))["row_key"])
    assert chunk_keys and chunk_keys == sentence_keys, "same rows, two rulers"

    for granularity in ("chunk", "sentence"):
        labels = pd.read_parquet(paths.teacher_labels(granularity))
        assert len(labels) == len(pd.read_parquet(paths.sample_parquet))
        export.run(config, theme, paths, granularity=granularity)
        assert paths.train_jsonl(granularity, "combined", "both").exists()

    fingerprints = set()
    for granularity in ("chunk", "sentence"):
        first = json.loads(
            paths.train_jsonl(granularity, "combined", "both").read_text().splitlines()[0])
        fingerprints.add(first["prompt_fingerprint"])
        assert first["granularity"] == granularity
    assert len(fingerprints) == 2, "two granularities, two prompts, two fingerprints"


def test_the_heldout_keys_file_is_written_once_per_run(pipeline):
    config, theme, paths = pipeline
    config.experiment.granularity = "both"
    subset.run(config, theme, paths)
    sample.run(config, theme, paths)
    teacher.run(config, theme, paths, dry_run=True, assume_yes=True)
    teacher.run_split(config, theme, paths)
    before = paths.heldout_keys_parquet.read_bytes()
    teacher.run_split(config, theme, paths)
    assert paths.heldout_keys_parquet.read_bytes() == before


def test_the_subset_metric_is_derived_from_the_per_sentence_sims(tmp_path, capsys):
    """A step-7 artifact carries no scalar similarity at all -- only the per-sentence list.
    The pricing-power run had `sim_top` because its input was an assembled corpus, so every
    run adopted straight off the box hits this."""
    from distill.config import DistillConfig
    from distill.paths import RunPaths
    from distill.stages import subset
    from distill.theme import ThemeSpec

    run_dir = tmp_path / "ESG_2026-08-13"
    run_dir.mkdir()
    frame = make_corpus(240).drop(columns=["sim_top"])
    frame.to_parquet(run_dir / "corpus.parquet", index=False)

    config = DistillConfig.from_dict({
        "input": {"source_parquet": "corpus.parquet", "reference_label_column": None},
        "subset": {"metric": "sim_top", "top_fraction": 0.5, "per_content_type": True},
        "experiment": {"granularity": "chunk", "topology": "combined"},
    }, base=run_dir, theme=THEME_SECTION).require_valid()
    theme = ThemeSpec.from_dict(THEME_SECTION).require_valid()
    paths = RunPaths.for_run(run_dir)

    subset.run(config, theme, paths)
    import json
    manifest = json.loads(
        Path(str(paths.subset_parquet) + ".manifest.json").read_text(encoding="utf-8"))
    assert any("derived as the row maximum of extracted_sentence_sims" in note
               for note in manifest["notes"]), manifest["notes"]

    import pandas as pd
    written = pd.read_parquet(paths.subset_parquet)
    assert "sim_top" in written.columns
    assert written["sim_top"].notna().all()
    for sims, value in zip(written["extracted_sentence_sims"], written["sim_top"]):
        assert value == max(sims)


def test_a_missing_metric_with_no_sims_to_derive_it_from_still_blocks(tmp_path):
    """The derivation is not a way to make any missing column optional."""
    from distill.config import DistillConfig
    from distill.paths import RunPaths
    from distill.preflight import check

    run_dir = tmp_path / "output"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir.parent / "config").mkdir(parents=True, exist_ok=True)
    make_corpus(60).drop(columns=["sim_top", "extracted_sentence_sims"]).to_parquet(
        run_dir / "corpus.parquet", index=False)
    config = DistillConfig.from_dict({
        "input": {"source_parquet": "corpus.parquet", "reference_label_column": None},
        "subset": {"metric": "sim_top"},
    }, base=run_dir, theme=THEME_SECTION)
    flight, _ = check(config, RunPaths.for_run(run_dir), stage="subset")
    assert not flight.passed
    assert any("sim_top" in e for e in flight.errors)


def _subset_run(tmp_path, rows: int, **subset):
    from distill.config import DistillConfig
    from distill.paths import RunPaths
    from distill.theme import ThemeSpec

    run_dir = tmp_path / "R_2026-01-01"
    run_dir.mkdir()
    make_corpus(rows).to_parquet(run_dir / "corpus.parquet", index=False)
    settings = {"metric": "sim_top", "top_fraction": 1.0, "per_content_type": True}
    settings.update(subset)
    config = DistillConfig.from_dict({
        "input": {"source_parquet": "corpus.parquet", "reference_label_column": None},
        "subset": settings,
    }, base=run_dir, theme=THEME_SECTION).require_valid()
    return config, ThemeSpec.from_dict(THEME_SECTION).require_valid(), RunPaths.for_run(run_dir)


def test_the_subset_ceiling_keeps_the_top_rows_not_a_sample(tmp_path):
    """A ceiling that sampled would throw away the property the cut was chosen for. Over the
    ceiling the cut is TIGHTENED: what survives is the rows nearest the theme."""
    import pandas as pd

    from distill.stages import subset

    config, theme, paths = _subset_run(tmp_path, 300, max_rows=100)
    subset.run(config, theme, paths)
    written = pd.read_parquet(paths.subset_parquet)
    assert len(written) == 100

    source = pd.read_parquet(paths.run_dir / "corpus.parquet")
    for content_type, group in written.groupby("content_type"):
        floor = group["sim_top"].min()
        available = source[source["content_type"] == content_type]
        assert (available["sim_top"] >= floor).sum() >= len(group)


def test_the_subset_ceiling_holds_the_content_type_mix(tmp_path):
    """A global cut would quietly change the news/transcripts mix, which is the one thing
    `per_content_type` exists to hold steady."""
    import pandas as pd

    from distill.stages import subset

    config, theme, paths = _subset_run(tmp_path, 300, max_rows=100)
    before = pd.read_parquet(paths.run_dir / "corpus.parquet")["content_type"].value_counts(
        normalize=True)
    subset.run(config, theme, paths)
    after = pd.read_parquet(paths.subset_parquet)["content_type"].value_counts(normalize=True)
    for content_type, share in before.items():
        assert abs(after.get(content_type, 0) - share) < 0.02, (content_type, after)


def test_no_ceiling_leaves_the_subset_alone(tmp_path):
    import pandas as pd

    from distill.stages import subset

    config, theme, paths = _subset_run(tmp_path, 120, max_rows=None)
    subset.run(config, theme, paths)
    assert len(pd.read_parquet(paths.subset_parquet)) == 120


def test_a_ceiling_above_the_cut_changes_nothing(tmp_path):
    import pandas as pd

    from distill.stages import subset

    config, theme, paths = _subset_run(tmp_path, 120, max_rows=1_000_000)
    subset.run(config, theme, paths)
    assert len(pd.read_parquet(paths.subset_parquet)) == 120


def test_the_heldout_ceiling_lowers_the_fraction_instead_of_truncating():
    """Truncating the result would empty a thin (content_type, label) cell. Lowering the
    fraction shrinks every cell by the same factor."""
    from distill.config import DistillConfig
    from distill.stages.teacher import heldout_fraction

    config = DistillConfig.from_dict(
        {"split": {"heldout_fraction": 0.10, "max_heldout": 50_000}},
        base=Path("."), theme=THEME_SECTION)
    assert heldout_fraction(config, 100_000) == 0.10
    assert heldout_fraction(config, 500_000) == 0.10
    assert heldout_fraction(config, 1_000_000) == 0.05
    assert heldout_fraction(config, 2_000_000) == 0.025


def test_no_heldout_ceiling_leaves_the_fraction_alone():
    from distill.config import DistillConfig
    from distill.stages.teacher import heldout_fraction

    config = DistillConfig.from_dict(
        {"split": {"heldout_fraction": 0.10, "max_heldout": None}},
        base=Path("."), theme=THEME_SECTION)
    assert heldout_fraction(config, 10_000_000) == 0.10


def test_only_the_heldout_is_capped_by_default():
    """The subset ceiling is opt-in -- `top_fraction` is the instrument for deciding how much
    corpus the signal is built on. The held-out ceiling is on, because that set is scored once
    per checkpoint per student."""
    from distill.config import DistillConfig

    config = DistillConfig.from_dict({}, base=Path("."), theme=THEME_SECTION)
    assert config.subset.max_rows is None
    assert config.split.max_heldout == 50_000


def test_the_heldout_is_spread_across_the_metric_bins(tmp_path):
    """min(10%, 50k), taken evenly from every bin: the ruler has to cover the whole similarity
    range, not just where the rows pile up."""
    import numpy as np
    import pandas as pd

    from distill.config import DistillConfig
    from distill.paths import RunPaths
    from distill.stages import teacher
    from distill.theme import ThemeSpec

    run_dir = tmp_path / "R_2026-01-01"
    labels_dir = run_dir / "teacher_labels" / "chunk"
    labels_dir.mkdir(parents=True)
    rng = np.random.default_rng(3)
    rows = 4_000
    pd.DataFrame({
        "row_key": [f"K{i:06d}" for i in range(rows)],
        "content_type": rng.choice(["news", "transcripts"], rows),
        "bin": rng.integers(0, 20, rows),
        "classification": rng.choice(["gainer", "irrelevant", "loser"], rows,
                                     p=[0.10, 0.86, 0.04]),
        "ok": True,
    }).to_parquet(labels_dir / "labels.parquet", index=False)

    config = DistillConfig.from_dict({
        "input": {"source_parquet": "corpus.parquet"},
        "split": {"heldout_fraction": 0.10, "seed": 17,
                  "stratify_on": ["content_type", "label", "bin"]},
    }, base=run_dir, theme=THEME_SECTION)
    theme = ThemeSpec.from_dict(THEME_SECTION).require_valid()
    paths = RunPaths.for_run(run_dir)
    teacher.run_split(config, theme, paths, granularities=("chunk",))

    held = pd.read_parquet(paths.heldout_split("chunk"))
    assert 0.08 * rows <= len(held) <= 0.12 * rows, len(held)
    covered = held["bin"].nunique()
    assert covered >= 18, f"only {covered} of 20 bins in the held-out"
    share = held["bin"].value_counts(normalize=True)
    assert share.max() < 0.15, share.head()


def test_teacher_limit_draws_per_content_type_and_keeps_the_column(pipeline):
    """`--limit` is the flag you reach for to test a run cheaply, and it was broken.

    From pandas 2.2 `groupby(...).apply(...)` no longer passes the grouping column to the
    block, so the frame that came back had lost `content_type` and `assert_labelable` died on
    the next line with a KeyError. It fired only WITH the flag, which is why nothing caught it
    until the first real smoke run -- against a 120B that had just spent three minutes loading.
    """
    config, theme, paths = pipeline
    subset.run(config, theme, paths)
    sample.run(config, theme, paths)

    teacher.run(config, theme, paths, dry_run=True, assume_yes=True, limit=2)

    labels = pd.read_parquet(paths.teacher_labels("chunk"))
    column = config.input.content_type_column
    per_type = labels[column].value_counts().to_dict()
    assert set(per_type) == {"news", "transcripts"}, per_type
    assert all(count <= 2 for count in per_type.values()), f"per content type, not overall: {per_type}"


def test_export_refuses_a_training_set_written_from_different_rows(pipeline):
    """Reuse tested only that the file existed, and that cost a training run.

    A run relabelled at a new coverage kept the JSONL from the old one: `export` reported
    "already written (100 rows)" while the split held 1,520, and `train` spent a GPU on the
    stale file. The prompt fingerprint gate cannot catch it -- the prompt had not changed, only
    the rows had.
    """
    config, theme, paths = pipeline
    subset.run(config, theme, paths)
    sample.run(config, theme, paths)
    teacher.run(config, theme, paths, dry_run=True, assume_yes=True)
    teacher.run_split(config, theme, paths)
    export.run(config, theme, paths)

    jsonl = paths.train_jsonl("chunk", "combined", "both")
    lines = jsonl.read_text(encoding="utf-8").splitlines(keepends=True)
    assert len(lines) > 2
    jsonl.write_text("".join(lines[:2]), encoding="utf-8")

    with pytest.raises(SystemExit, match="written from different data"):
        export.run(config, theme, paths)

    export.run(config, theme, paths, force=True)
    assert sum(1 for _ in jsonl.open(encoding="utf-8")) == len(lines)
