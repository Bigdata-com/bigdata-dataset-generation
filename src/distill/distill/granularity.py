"""Granularity as a strategy, not as a second copy of the file."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Any, Protocol

from .client import LabelClient
from .config import DistillConfig
from .parsing import LabelResult
from .prompts import PromptSet

BASE_RECORD_FIELDS = ("row_key", "text", "ok", "error", "error_kind", "recovered",
                      "latency_s", "finish_reason")


def stable_bucket(*parts: Any, buckets: int = 100) -> int:
    """A hash that is the same in every process."""
    blob = " ".join("" if p is None else str(p) for p in parts)
    digest = hashlib.md5(blob.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


class Strategy(Protocol):
    granularity: str

    def required_columns(self) -> tuple[str, ...]: ...
    def assert_labelable(self, frame) -> None: ...
    def label(self, row: Any, content_type: str) -> dict[str, Any]: ...
    def probe_row(self, row: Any) -> dict[str, Any] | None: ...
    def record_fields(self) -> tuple[str, ...]: ...


def _record(prompts: PromptSet, row_key: str, text: str, result: LabelResult,
            **extra: Any) -> dict[str, Any]:
    """The one record shape."""
    record: dict[str, Any] = {
        "row_key": row_key,
        "text": text,
        prompts.label_field: result.label,
        "ok": result.ok,
        "error": result.error,
        "error_kind": result.error_kind,
        "recovered": bool(result.recovered),
        "latency_s": result.extra.get("latency_s"),
        "finish_reason": result.extra.get("finish_reason"),
    }
    if prompts.rationale_field:
        record[prompts.rationale_field] = result.rationale
    record.update(extra)
    missing = [f for f in BASE_RECORD_FIELDS if f not in record]
    if missing:
        raise AssertionError(f"the record shape lost {missing}; every stage downstream reads it")
    return record


@dataclass
class ChunkStrategy:
    """One call per row, the whole passage, no walk and no threshold."""

    config: DistillConfig
    prompts: PromptSet
    client: LabelClient | None = None
    dry_run: bool = False
    granularity: str = "chunk"

    def required_columns(self) -> tuple[str, ...]:
        columns = [self.config.input.text_column, self.config.input.entity_name_column]
        if "transcripts" in self.config.subset.content_types:
            columns.append(self.config.input.headline_column)
        return tuple(dict.fromkeys(columns))

    def assert_labelable(self, frame) -> None:
        """THE INPUT ASSERTION."""
        content_column = self.config.input.content_type_column
        columns = [self.config.input.text_column]
        if "transcripts" in set(frame[content_column].unique()):
            columns.append(self.config.input.headline_column)
        for column in columns:
            if column not in frame.columns:
                raise SystemExit(f"the input has no {column!r} column, and the chunk-level prompt "
                                 f"cannot be built without it")
            blank = frame[column].isna() | (frame[column].fillna("").astype(str).str.strip() == "")
            count = int(blank.sum())
            if count:
                by_type = frame.loc[blank, content_column].value_counts().to_dict()
                raise SystemExit(
                    f"{column} is null or blank on {count:,} of {len(frame):,} rows ({by_type}).\n"
                    f"Labelling a blank {column} yields a confident, meaningless "
                    f"'{self.prompts.abstain_class}'. Fix the input, do not label around it -- and "
                    f"note that for pricing power the repaired corpus is SOURCE: no script "
                    f"reproduces the recovery of its 130,071 missing chunk texts."
                )

    def prompt_row(self, row: Any) -> dict[str, Any]:
        return {
            "company_name": getattr(row, self.config.input.entity_name_column),
            "text": getattr(row, self.config.input.text_column),
            "headline": getattr(row, self.config.input.headline_column, "") or "",
        }

    def probe_row(self, row: Any) -> dict[str, Any] | None:
        """One row shaped for a single call, for the pre-flight template probe."""
        return self.prompt_row(row)

    def label(self, row: Any, content_type: str) -> dict[str, Any]:
        text = getattr(row, self.config.input.text_column)
        if self.dry_run:
            result = _stub_answer(self.prompts, text)
        else:
            assert self.client is not None, "no client: this strategy was built for a dry run"
            result = self.client.label(self.prompt_row(row), content_type)
        return _record(self.prompts, row.row_key, text, result, n_calls=1)

    def record_fields(self) -> tuple[str, ...]:
        return ("n_calls",)


@dataclass
class SentenceStrategy:
    """Walk a row's extracted sentences best-first; stop at the first non-abstain answer."""

    config: DistillConfig
    prompts: PromptSet
    client: LabelClient | None = None
    dry_run: bool = False
    granularity: str = "sentence"

    def __post_init__(self) -> None:
        self._cache: dict[tuple, LabelResult] = {}
        self._cache_lock = threading.Lock()
        self.cache_hits = 0
        self.stop_classes = set(self.prompts.theme.signal_classes)

    def required_columns(self) -> tuple[str, ...]:
        return (self.config.sentence.sentences_column, self.config.sentence.sims_column,
                self.config.input.entity_name_column)

    def assert_labelable(self, frame) -> None:
        """The input assertion at sentence level: a row with no usable sentence at all."""
        for column in (self.config.sentence.sentences_column, self.config.sentence.sims_column):
            if column not in frame.columns:
                raise SystemExit(
                    f"the input has no {column!r} column. Sentence-level labelling needs Part A's "
                    f"extracted sentences and their similarities; run at chunk level, or point "
                    f"input at 'similarity/combinations_with_similarity.parquet'."
                )
        sizes = frame[self.config.sentence.sentences_column].map(
            lambda v: 0 if v is None else len(v)
        )
        empty = int((sizes == 0).sum())
        if empty == len(frame):
            raise SystemExit(
                f"every one of {len(frame):,} rows has an empty "
                f"{self.config.sentence.sentences_column!r}. There is nothing to walk."
            )
        if empty:
            print(f"{empty:,} of {len(frame):,} rows have no sentence to send and will abstain")

    def candidates(self, row: Any) -> list[tuple[float, str, int, str | None]]:
        """[(similarity, sentence, index, seed statement)] best-first, above the threshold."""
        sentences = getattr(row, self.config.sentence.sentences_column, None)
        sims = getattr(row, self.config.sentence.sims_column, None)
        sentences = [] if sentences is None else list(sentences)
        sims = [] if sims is None else list(sims)
        seeds = getattr(row, self.config.sentence.seeds_column, None)
        seeds = None if seeds is None else list(seeds)
        threshold = self.config.sentence.similarity_threshold
        out: list[tuple[float, str, int, str | None]] = []
        for index, (sentence, sim) in enumerate(zip(sentences, sims)):
            if not isinstance(sentence, str) or not sentence.strip():
                continue
            value = float(sim)
            if value != value or value < threshold:
                continue
            seed = None
            if seeds is not None and index < len(seeds):
                seed = str(seeds[index])
            out.append((value, sentence, index, seed))
        out.sort(key=lambda c: c[0], reverse=True)
        return out

    def _ask(self, content_type: str, company: str, sentence: str) -> tuple[LabelResult, bool]:
        """One call, through the (content_type, company, sentence) cache."""
        key = (content_type, company, sentence)
        if self.config.sentence.cache:
            with self._cache_lock:
                hit = self._cache.get(key)
            if hit is not None:
                with self._cache_lock:
                    self.cache_hits += 1
                return hit, True
        if self.dry_run:
            result = _stub_answer(self.prompts, sentence)
        else:
            assert self.client is not None, "no client: this strategy was built for a dry run"
            result = self.client.label(
                {"company_name": company, "text": sentence, "headline": ""}, content_type
            )
        if self.config.sentence.cache and result.ok:
            with self._cache_lock:
                self._cache[key] = result
        return result, False

    def probe_row(self, row: Any) -> dict[str, Any] | None:
        """The best candidate sentence, shaped exactly as `_ask` shapes it. None when the row
        has no candidate at all, in which case the caller should probe a different row."""
        candidates = self.candidates(row)
        if not candidates:
            return None
        return {"company_name": getattr(row, self.config.input.entity_name_column),
                "text": candidates[0][1], "headline": ""}

    def label(self, row: Any, content_type: str) -> dict[str, Any]:
        """Visit the candidates best-first and stop at the first non-abstain answer."""
        company = getattr(row, self.config.input.entity_name_column)
        candidates = self.candidates(row)
        extras: dict[str, Any] = {
            "n_candidates": len(candidates), "n_calls": 0, "n_cached": 0,
            "sentence_sim": None, "sentence_index": None, "best_search": None,
            "reasoning_from": None,
        }
        if not candidates:
            abstain = LabelResult(label=self.prompts.abstain_class, rationale="",
                                  extra={"latency_s": 0.0, "finish_reason": "no_candidates"})
            return _record(self.prompts, row.row_key, "", abstain, **extras)

        first: tuple[LabelResult, float, str, int, str | None] | None = None
        for sim, sentence, index, seed in candidates:
            result, cached = self._ask(content_type, company, sentence)
            extras["n_calls"] += 1
            extras["n_cached"] += int(cached)
            if not result.ok:
                extras.update(sentence_sim=sim, sentence_index=index, best_search=seed)
                return _record(self.prompts, row.row_key, sentence, result, **extras)
            if first is None:
                first = (result, sim, sentence, index, seed)
            if result.label in self.stop_classes:
                extras.update(sentence_sim=sim, sentence_index=index, best_search=seed,
                              reasoning_from="stop")
                return _record(self.prompts, row.row_key, sentence, result, **extras)

        result, sim, sentence, index, seed = first
        extras.update(sentence_sim=sim, sentence_index=index, best_search=seed,
                      reasoning_from="top")
        abstained = LabelResult(label=self.prompts.abstain_class, rationale=result.rationale,
                                extra=dict(result.extra))
        return _record(self.prompts, row.row_key, sentence, abstained, **extras)

    def record_fields(self) -> tuple[str, ...]:
        return ("n_candidates", "n_calls", "n_cached", "sentence_sim", "sentence_index",
                "best_search", "reasoning_from")


def _stub_answer(prompts: PromptSet, text: str) -> LabelResult:
    """The dry-run answer: no call at all, deterministic on the text, corpus-shaped."""
    signal = list(prompts.theme.signal_classes)
    abstain = prompts.abstain_class or (signal[0] if signal else "unknown")
    bucket = stable_bucket(text)
    label = abstain
    if signal:
        if bucket < 18:
            label = signal[0]
        elif len(signal) > 1 and bucket < 29:
            label = signal[1]
    return LabelResult(label=label, rationale=f"stub reasoning ({label})",
                       extra={"latency_s": 0.0, "finish_reason": "stub"})


def build_strategy(config: DistillConfig, prompts: PromptSet, *,
                   client: LabelClient | None = None, dry_run: bool = False) -> Strategy:
    """The only place a granularity becomes an object."""
    if prompts.granularity not in config.granularities():
        raise SystemExit(
            f"prompt set is {prompts.granularity!r} but this run covers "
            f"{list(config.granularities())}. The prompt is derived FROM the granularity and "
            f"must never be chosen independently of it."
        )
    if prompts.granularity == "chunk":
        return ChunkStrategy(config=config, prompts=prompts, client=client, dry_run=dry_run)
    return SentenceStrategy(config=config, prompts=prompts, client=client, dry_run=dry_run)
