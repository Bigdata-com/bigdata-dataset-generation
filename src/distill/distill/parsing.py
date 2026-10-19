"""Response parsing — the repair cascade, and what a failure is allowed to become."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_RE = THINK_RE


class LabelContract(Protocol):
    """What parsing needs to know about the taxonomy."""

    @property
    def label_field(self) -> str: ...
    @property
    def class_names(self) -> tuple[str, ...]: ...
    @property
    def rationale_field(self) -> str | None: ...


@dataclass
class LabelResult:
    """One answer."""

    label: str | None = None
    rationale: str = ""
    reasoning_trace: str = ""
    raw_response: str | None = None
    error: str | None = None
    error_kind: str | None = None
    recovered: bool = False
    extra: dict[str, Any] = dc_field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.label is not None


def error_result(msg: str, *, kind: str, raw: str = "") -> LabelResult:
    """A failure carries NO label."""
    return LabelResult(label=None, raw_response=raw or None, error=msg, error_kind=kind)


def _last_balanced_object(text: str) -> str | None:
    """The last top-level `{...}`, matched by brace balance, scanning backwards."""
    depth = 0
    end: int | None = None
    in_string = False
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if ch == '"':
            backslashes = 0
            j = i - 1
            while j >= 0 and text[j] == "\\":
                backslashes += 1
                j -= 1
            if backslashes % 2 == 0:
                in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "}":
            if depth == 0:
                end = i
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0 and end is not None:
                return text[i:end + 1]
    return None


def _quote_bare_values(text: str, fields: tuple[str, ...]) -> str:
    """Quote string values the model left unquoted."""
    not_already_valid = r'(?!\s*["\{\[]|\s*-?\d|\s*true\b|\s*false\b|\s*null\b)'

    def _quoted(raw_value: str) -> str:
        value = raw_value.strip().rstrip(",").strip()
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

    for name in fields:
        key = r'("' + re.escape(name) + r'"\s*:\s*)'
        text = re.sub(
            key + not_already_valid + r"(.*)\s*\}\s*$",
            lambda m: m.group(1) + _quoted(m.group(2)) + "}",
            text,
            flags=re.DOTALL,
        )
        text = re.sub(
            key + not_already_valid + r'([^\n]*?)\s*(?=,\s*"\w+"\s*:)',
            lambda m: m.group(1) + _quoted(m.group(2)),
            text,
        )
    return text


def split_reasoning(raw: str) -> tuple[str, str]:
    """Separate a reasoning trace from the JSON answer."""
    trace_parts = _THINK_RE.findall(raw)
    trace = " ".join(t[7:-8].strip() for t in trace_parts) if trace_parts else ""
    body = _THINK_RE.sub("", raw).strip()

    if "</think>" in body:
        head, _, tail = body.rpartition("</think>")
        trace = (trace + " " + head).strip() if trace else head.strip()
        body = tail.strip()

    if not trace:
        brace = body.find("{")
        if brace > 0:
            prefix = body[:brace].strip()
            if prefix and not prefix.startswith("```"):
                trace = prefix
    return trace, body


def _coerce(value: Any, allowed: tuple[str, ...]) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for candidate in allowed:
        if text.lower() == candidate.lower():
            return candidate
    return None


def _regex_label(text: str, contract: LabelContract) -> str | None:
    """Stage 6."""
    pattern = re.compile(
        r'"' + re.escape(contract.label_field) + r'"\s*:\s*"([^"\\]{1,64})"'
    )
    for match in pattern.finditer(text):
        label = _coerce(match.group(1), tuple(contract.class_names))
        if label is not None:
            return label
    return None


def parse(raw: str, contract: LabelContract) -> LabelResult:
    """Parse a raw model response against the run's label contract."""
    if not raw:
        return error_result("empty response from the endpoint", kind="empty", raw=raw)

    label_field = contract.label_field
    classes = tuple(contract.class_names)
    rationale_field = contract.rationale_field

    trace, clean = split_reasoning(raw.strip())
    if clean.startswith("```"):
        clean = "\n".join(
            line for line in clean.splitlines() if not line.strip().startswith("```")
        ).strip()

    string_fields = tuple(f for f in (rationale_field,) if f)

    def _attempt(text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    parsed = _attempt(clean)
    if parsed is None:
        candidate = _last_balanced_object(clean)
        if candidate is not None:
            parsed = _attempt(candidate)
            if parsed is None and string_fields:
                parsed = _attempt(_quote_bare_values(candidate, string_fields))
        if parsed is None:
            start, end = clean.find("{"), clean.rfind("}")
            if start != -1 and end > start:
                parsed = _attempt(clean[start:end + 1])
        if parsed is None and string_fields:
            parsed = _attempt(_quote_bare_values(clean, string_fields))
        if parsed is None:
            parsed = _attempt(clean.rstrip().rstrip(",") + "}")
        if parsed is None:
            recovered = _regex_label(clean, contract)
            if recovered is not None:
                return LabelResult(
                    label=recovered, rationale="", reasoning_trace=trace,
                    raw_response=raw, recovered=True,
                )
            return error_result(f"JSON parse error. Raw: {raw[:200]}", kind="parse", raw=raw)

    if not isinstance(parsed, dict):
        return error_result(
            f"expected a JSON object, got {type(parsed).__name__}", kind="schema", raw=raw
        )

    if label_field not in parsed:
        return error_result(
            f"response has no {label_field!r} field. Raw: {raw[:200]}", kind="schema", raw=raw
        )

    label = _coerce(parsed.get(label_field), classes)
    if label is None:
        aliases = getattr(contract, "alias_map", None)
        if callable(aliases):
            label = aliases().get(str(parsed.get(label_field, "")).strip().lower())
    if label is None:
        return error_result(
            f"unknown {label_field}: {parsed.get(label_field)!r}", kind="schema", raw=raw
        )

    rationale = ""
    if rationale_field:
        rationale = str(parsed.get(rationale_field, "") or "")

    return LabelResult(label=label, rationale=rationale,
                       reasoning_trace=trace, raw_response=raw)
