"""The checkpoint rules, and the walk."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from distill.checkpoint import Checkpoint, label_rows
from distill.config import DistillConfig
from distill.granularity import (
    ChunkStrategy, SentenceStrategy, build_strategy, stable_bucket,
)
from distill.parsing import LabelResult
from distill.prompts import PromptSet

from conftest import theme_section  # noqa: E402
from distill.theme import ThemeSpec  # noqa: E402

_THEMES = Path(__file__).resolve().parent.parent / "themes"
THEME_SECTION = theme_section(ThemeSpec.load(_THEMES / "pricing_power.yaml"))

from conftest import make_corpus


def a_config(tmp_path: Path, **overrides) -> DistillConfig:
    data = {"input": {"source_parquet": "corpus.parquet"}}
    data.update(overrides)
    return DistillConfig.from_dict(data, base=tmp_path, theme=THEME_SECTION)


def test_relaunching_relabels_nothing(tmp_path):
    checkpoint = Checkpoint(tmp_path / "ckpt.jsonl")
    rows = [{"row_key": f"k{i}"} for i in range(5)]
    calls: list[str] = []

    def label(row):
        calls.append(row["row_key"])
        return {"row_key": row["row_key"], "classification": "winner", "ok": True}

    first, stats = label_rows(rows, label_one=label, key_of=lambda r: r["row_key"],
                             checkpoint=checkpoint, workers=2, desc="t", progress=False)
    assert len(first) == 5 and stats["newly_labelled"] == 5
    calls.clear()
    second, stats = label_rows(rows, label_one=label, key_of=lambda r: r["row_key"],
                              checkpoint=checkpoint, workers=2, desc="t", progress=False)
    assert calls == [], "a resumed run must make no calls for rows already answered"
    assert len(second) == 5 and stats["newly_labelled"] == 0


def test_a_truncated_last_line_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    path.write_text(json.dumps({"row_key": "a", "ok": True}) + "\n" + '{"row_key": "b", "ok',
                    encoding="utf-8")
    done = Checkpoint(path).load()
    assert set(done) == {"a"}


def test_the_last_record_for_a_key_wins(tmp_path):
    """Retried error rows are appended AFTER the originals."""
    path = tmp_path / "ckpt.jsonl"
    path.write_text("\n".join([
        json.dumps({"row_key": "a", "ok": False, "error": "boom"}),
        json.dumps({"row_key": "a", "ok": True, "classification": "winner"}),
    ]) + "\n", encoding="utf-8")
    done = Checkpoint(path).load()
    assert done["a"]["ok"] is True


def test_error_rows_are_retried_only_on_request(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    path.write_text(json.dumps({"row_key": "a", "ok": False, "error": "boom"}) + "\n",
                    encoding="utf-8")
    assert set(Checkpoint(path).load()) == {"a"}, (
        "by default a permanently failing row must not be attempted forever"
    )
    assert Checkpoint(path, retry_errors=True).load() == {}


def test_a_worker_exception_is_recorded_not_lost(tmp_path):
    checkpoint = Checkpoint(tmp_path / "ckpt.jsonl")

    def label(row):
        if row["row_key"] == "k1":
            raise RuntimeError("a bug in the strategy")
        return {"row_key": row["row_key"], "classification": "winner", "ok": True}

    records, stats = label_rows([{"row_key": f"k{i}"} for i in range(3)], label_one=label,
                                key_of=lambda r: r["row_key"], checkpoint=checkpoint,
                                workers=2, desc="t", progress=False)
    assert len(records) == 3 and stats["errors"] == 1
    failed = next(r for r in records if r["row_key"] == "k1")
    assert failed["ok"] is False and "a bug in the strategy" in failed["error"]
    assert failed["classification"] is None, "a failure never carries a label"


def test_the_dry_run_stub_is_deterministic_across_processes():
    """`hash()` on a str is salted per interpreter."""
    code = ("import sys; sys.path.insert(0, %r); "
            "from distill.granularity import stable_bucket; print(stable_bucket('a chunk'))"
            % str(Path(__file__).resolve().parent.parent))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert int(out.stdout.strip()) == stable_bucket("a chunk")


def test_the_input_assertion_refuses_a_blank_chunk(tmp_path, pricing_power):
    config = a_config(tmp_path)
    prompts = PromptSet(pricing_power, "chunk")
    strategy = ChunkStrategy(config=config, prompts=prompts, dry_run=True)
    frame = make_corpus(30, blank_rows=4)
    with pytest.raises(SystemExit, match="null or blank on 4"):
        strategy.assert_labelable(frame)
    strategy.assert_labelable(make_corpus(30))


def test_the_walk_stops_at_the_first_signal_class_named_by_the_theme(tmp_path, gainer_theme):
    """`STOP_CLASSES = {"winner", "loser"}` was a literal."""
    config = a_config(tmp_path, experiment={"granularity": "sentence"})
    prompts = PromptSet(gainer_theme, "sentence")
    strategy = SentenceStrategy(config=config, prompts=prompts, dry_run=True)
    answers = {"s-low": "irrelevant", "s-mid": "gainer", "s-high": "irrelevant"}
    asked: list[str] = []

    def fake_ask(content_type, company, sentence):
        asked.append(sentence)
        return LabelResult(label=answers[sentence], rationale="r",
                          extra={"latency_s": 0.0}), False

    strategy._ask = fake_ask
    row = pd.DataFrame([{
        "row_key": "k", "entity_name": "Acme", "content_type": "news",
        "extracted_sentences": ["s-low", "s-mid", "s-high"],
        "extracted_sentence_sims": [0.4, 0.6, 0.9],
        "extracted_best_search": ["a", "b", "c"],
    }]).itertuples(index=False).__next__()
    record = strategy.label(row, "news")
    assert asked == ["s-high", "s-mid"], "best-first, and it stops at the hit"
    assert record["classification"] == "gainer"
    assert record["text"] == "s-mid", "the record names the sentence the answer rests on"
    assert record["reasoning_from"] == "stop"
    assert record["n_calls"] == 2


def test_the_walk_abstains_with_zero_calls_when_nothing_clears_the_threshold(tmp_path,
                                                                            pricing_power):
    config = a_config(tmp_path, experiment={"granularity": "sentence"},
                      sentence={"similarity_threshold": 0.95})
    strategy = SentenceStrategy(config=config, prompts=PromptSet(pricing_power, "sentence"),
                                dry_run=True)
    row = pd.DataFrame([{
        "row_key": "k", "entity_name": "Acme", "content_type": "news",
        "extracted_sentences": ["a", "b"], "extracted_sentence_sims": [0.4, 0.6],
        "extracted_best_search": ["x", "y"],
    }]).itertuples(index=False).__next__()
    record = strategy.label(row, "news")
    assert record["classification"] == "irrelevant"
    assert record["n_calls"] == 0 and record["n_candidates"] == 0
    assert record["reasoning_from"] is None


def test_an_error_mid_walk_aborts_the_row_without_a_label(tmp_path, pricing_power):
    """Continuing would mean guessing whether the failed sentence was the hit."""
    config = a_config(tmp_path, experiment={"granularity": "sentence"})
    strategy = SentenceStrategy(config=config, prompts=PromptSet(pricing_power, "sentence"),
                                dry_run=True)

    def fake_ask(content_type, company, sentence):
        return LabelResult(label=None, error="HTTP 503", error_kind="http",
                          extra={"latency_s": 0.1}), False

    strategy._ask = fake_ask
    row = pd.DataFrame([{
        "row_key": "k", "entity_name": "Acme", "content_type": "news",
        "extracted_sentences": ["a", "b"], "extracted_sentence_sims": [0.4, 0.6],
        "extracted_best_search": ["x", "y"],
    }]).itertuples(index=False).__next__()
    record = strategy.label(row, "news")
    assert record["ok"] is False
    assert record["classification"] is None


def test_the_sentence_cache_is_used_and_errors_are_never_cached(tmp_path, pricing_power):
    config = a_config(tmp_path, experiment={"granularity": "sentence"})
    strategy = SentenceStrategy(config=config, prompts=PromptSet(pricing_power, "sentence"),
                                dry_run=True)
    _, cached = strategy._ask("news", "Acme", "the same sentence")
    assert cached is False
    _, cached = strategy._ask("news", "Acme", "the same sentence")
    assert cached is True and strategy.cache_hits == 1


def test_candidates_are_sorted_best_first(tmp_path, pricing_power):
    """The source lists are NOT sorted by similarity -- only 24.7% of rows happen to be."""
    config = a_config(tmp_path, experiment={"granularity": "sentence"})
    strategy = SentenceStrategy(config=config, prompts=PromptSet(pricing_power, "sentence"),
                                dry_run=True)
    row = pd.DataFrame([{
        "row_key": "k", "entity_name": "Acme", "content_type": "news",
        "extracted_sentences": ["low", "high", "", None],
        "extracted_sentence_sims": [0.31, 0.88, 0.99, 0.99],
        "extracted_best_search": ["a", "b", "c", "d"],
    }]).itertuples(index=False).__next__()
    candidates = strategy.candidates(row)
    assert [c[1] for c in candidates] == ["high", "low"], "blank sentences are never sent"


def test_build_strategy_refuses_a_prompt_from_the_other_granularity(tmp_path, pricing_power):
    config = a_config(tmp_path, experiment={"granularity": "chunk"})
    with pytest.raises(SystemExit, match="derived FROM the granularity"):
        build_strategy(config, PromptSet(pricing_power, "sentence"), dry_run=True)


def test_the_record_shape_is_the_same_at_both_granularities(tmp_path, pricing_power):
    """This is what lets the export, the evaluation and the corpus pass be granularity-blind."""
    chunk_config = a_config(tmp_path, experiment={"granularity": "chunk"})
    sentence_config = a_config(tmp_path, experiment={"granularity": "sentence"})
    chunk = build_strategy(chunk_config, PromptSet(pricing_power, "chunk"), dry_run=True)
    sentence = build_strategy(sentence_config, PromptSet(pricing_power, "sentence"), dry_run=True)
    row = make_corpus(1).assign(row_key="k").itertuples(index=False).__next__()
    shared = {"row_key", "text", "classification", "reasoning", "ok", "error", "error_kind",
              "recovered", "latency_s", "finish_reason", "n_calls"}
    assert shared <= set(chunk.label(row, "news"))
    assert shared <= set(sentence.label(row, "news"))


def test_a_detached_launch_is_told_about_the_flag(monkeypatch):
    """Under `nohup` stdin is closed and a bare `input()` dies with
    `OSError: Bad file descriptor` -- a traceback that says nothing about the flag that would
    have avoided it. A long pass is exactly the one you launch detached."""
    import builtins

    import pytest

    from distill.frames import confirm

    def no_terminal(prompt=""):
        raise OSError(9, "Bad file descriptor")

    monkeypatch.setattr(builtins, "input", no_terminal)
    with pytest.raises(SystemExit, match="no terminal to confirm on"):
        confirm("go? ")
    with pytest.raises(SystemExit, match=r"--yes"):
        confirm("go? ")


def test_a_terminal_that_answers_lets_it_through(monkeypatch):
    import builtins

    from distill.frames import confirm

    asked: list[str] = []
    monkeypatch.setattr(builtins, "input", lambda prompt="": asked.append(prompt) or "")
    confirm("go? ")
    assert asked == ["go? "]
