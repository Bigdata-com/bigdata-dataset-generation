"""Prompt assembly: a fixed header, the user's body, a fixed contract."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .theme import CONTENT_TYPES, GRANULARITIES, ThemeSpec

HEADER: dict[str, str] = {
    "chunk.news": (
        "You are a financial analyst building a {signal} signal for an equity strategy.\n"
        "\n"
        "TASK\n"
        "You are given a news chunk retrieved for a specific company. Determine whether this "
        "excerpt\ncontains a meaningful signal about that company's {SIGNAL}."
    ),
    "chunk.transcripts": (
        "You are a financial analyst building a {signal} signal for an equity strategy.\n"
        "\n"
        "TASK\n"
        "You are given a transcript excerpt from an earnings call or similar event retrieved for "
        "a specific company. Determine whether this excerpt\ncontains a meaningful signal about "
        "that company's {SIGNAL}."
    ),
    "sentence.news": (
        "You are a financial analyst assessing a SPECIFIC company's {SIGNAL} from a single "
        "sentence of news or an earnings call.\n"
        "\n"
        "You are given the COMPANY and one SENTENCE extracted from text about the company. "
        "Judge only from the sentence; no prior knowledge about the company."
    ),
    "sentence.transcripts": (
        "You are a financial analyst assessing a SPECIFIC company's {SIGNAL} from a single "
        "sentence of an earnings-call transcript.\n"
        "\n"
        "You are given the COMPANY and one SENTENCE from the transcript. Judge only from the "
        "sentence; no prior knowledge about the company."
    ),
}

TRANSCRIPT_RULE: dict[str, str] = {
    "chunk": """TRANSCRIPT — HEADLINE AND TARGET COMPANY
The HEADLINE names the company the transcript event is primarily about (typically before the
first colon). Judge the TARGET COMPANY only.
- Same firm (allow suffixes, punctuation, Inc/Corp/Ltd, Holdings, minor variants): treat the
  excerpt as about the TARGET COMPANY, and read "we / our" as the TARGET COMPANY even where
  the name is absent.
- Different firm: the event is about the headline company, but you still judge the TARGET
  COMPANY. Do not assume "we / our" refers to it. Use explicit mentions of it, its products,
  its competitors, or dynamics that affect it.
If unsure whether two names are the same firm, do not attribute "we / our" to the TARGET
COMPANY, but still use any direct statement about it.""",
    "sentence": """TRANSCRIPT CONTEXT — WHO IS THE SUBJECT
The sentence is spoken by a company's management, which refers to itself as "we / our / us /
the company". The COMPANY you must judge may be that speaker, OR a different company the
speaker names — a supplier, customer, competitor, partner or peer.
- If the COMPANY is the speaker, judge the "we / our" facts.
- If the COMPANY is a different named company, judge ONLY what is said ABOUT it. A "we / our"
  fact then belongs to the SPEAKER, not to the COMPANY. So does a fact about the speaker's
  own competitors, customers or suppliers, unless the COMPANY is that party.""",
}

USER_TEMPLATE: dict[str, str] = {
    "chunk.news": "TARGET COMPANY: {company_name}\n\nCHUNK:\n{text}",
    "chunk.transcripts": (
        "HEADLINE: {headline}\n\nTARGET COMPANY: {company_name}\n\n"
        "The headline identifies the company the transcript event is primarily about.\n"
        "Classify for the TARGET COMPANY above (not necessarily the headline company).\n\n"
        "TRANSCRIPT EXCERPT:\n{text}"
    ),
    "sentence.news": "COMPANY: {company_name}\nSENTENCE: {text}",
    "sentence.transcripts": "COMPANY: {company_name}\nSENTENCE: {text}",
}

CONTRACT_HEADER = "OUTPUT\nRespond only with a valid JSON object, no text outside it:"
STUB_NOUN = {"chunk": "excerpt", "sentence": "sentence"}
STUB_RATIONALE = "one sentence"


def _pair(granularity: str, content_type: str) -> str:
    if granularity not in GRANULARITIES:
        raise ValueError(f"granularity must be one of {list(GRANULARITIES)}, got {granularity!r}")
    if content_type not in CONTENT_TYPES:
        raise ValueError(f"content_type must be one of {list(CONTENT_TYPES)}, got {content_type!r}")
    return f"{granularity}.{content_type}"


def output_fields(theme: ThemeSpec) -> tuple[str, ...]:
    """The label and the rationale, in the order the model must emit them."""
    return (theme.label_field, theme.rationale_field)


def render_contract(theme: ThemeSpec, *, inline: bool = False) -> str:
    """The JSON shape, built from the class names so it cannot disagree with them."""
    if inline:
        options = "|".join(f'"{name}"' for name in theme.class_names)
        return ('{"' + theme.label_field + '": ' + options + ', "' + theme.rationale_field
                + '": "<' + STUB_RATIONALE + '>"}')
    options = " | ".join(f'"{name}"' for name in theme.class_names)
    body = (f'"{theme.label_field}": {options}',
            f'"{theme.rationale_field}": "<{theme.rationale_instruction}>"')
    return "{\n" + ",\n".join(f"  {part}" for part in body) + "\n}"


def render_system_prompt(theme: ThemeSpec, granularity: str, content_type: str) -> str:
    """Header, then the user's body verbatim, then the contract."""
    key = _pair(granularity, content_type)
    sections = [
        HEADER[key].format(signal=theme.title, SIGNAL=theme.title.upper()),
        theme.body.strip(),
    ]
    if content_type == "transcripts":
        sections.append(TRANSCRIPT_RULE[granularity])
    sections.append(CONTRACT_HEADER + "\n\n" + render_contract(theme))
    return "\n\n".join(sections)


def render_user_template(theme: ThemeSpec, granularity: str, content_type: str) -> str:
    return USER_TEMPLATE[_pair(granularity, content_type)]


def render_stub(theme: ThemeSpec, granularity: str) -> str:
    """The short prompt the student is trained AND served with."""
    if granularity not in GRANULARITIES:
        raise ValueError(f"granularity must be one of {list(GRANULARITIES)}")
    return (
        f"You are a {theme.slug} classifier. For the target company, label the "
        f"{STUB_NOUN[granularity]} and respond with only compact JSON: "
        + render_contract(theme, inline=True) + "."
    )


def template_fields(template: str) -> tuple[str, ...]:
    """The `{placeholder}` names a template needs, in order of first appearance."""
    seen: list[str] = []
    for match in re.finditer(r"\{(\w+)\}", template):
        if match.group(1) not in seen:
            seen.append(match.group(1))
    return tuple(seen)


class BlankInput(ValueError):
    """The row has no text to send."""


@dataclass(frozen=True)
class PromptSet:
    """Every prompt one experiment needs, and nothing it does not."""

    theme: ThemeSpec
    granularity: str
    prompt_style: str = "full"

    def __post_init__(self) -> None:
        if self.granularity not in GRANULARITIES:
            raise ValueError(f"granularity must be one of {list(GRANULARITIES)}, "
                             f"got {self.granularity!r}")
        if self.prompt_style not in ("full", "stub"):
            raise ValueError(f"prompt_style must be full|stub, got {self.prompt_style!r}")

    @property
    def label_field(self) -> str:
        return self.theme.label_field

    @property
    def class_names(self) -> tuple[str, ...]:
        return self.theme.class_names

    @property
    def abstain_class(self) -> str | None:
        return self.theme.abstain_class

    @property
    def rationale_field(self) -> str | None:
        return self.theme.rationale_field

    def output_fields(self) -> tuple[str, ...]:
        return output_fields(self.theme)

    def alias_map(self) -> dict[str, str]:
        return self.theme.alias_map()

    def system_prompt(self, content_type: str) -> str:
        if self.prompt_style == "stub":
            return render_stub(self.theme, self.granularity)
        return render_system_prompt(self.theme, self.granularity, content_type)

    def user_template(self, content_type: str) -> str:
        return render_user_template(self.theme, self.granularity, content_type)

    def required_fields(self, content_type: str) -> tuple[str, ...]:
        return template_fields(self.user_template(content_type))

    def stub(self) -> str:
        return render_stub(self.theme, self.granularity)

    def user_content(self, row: Mapping[str, Any], content_type: str) -> str:
        """Fill the template from a row."""
        template = self.user_template(content_type)
        needed = template_fields(template)
        values = dict(row)
        text = values.get("text")
        if isinstance(text, str) and len(text) > self.theme.max_text_chars:
            values["text"] = text[: self.theme.max_text_chars]
        missing = [f for f in needed if not str(values.get(f, "") or "").strip()]
        if missing:
            raise BlankInput(
                f"{self.granularity}/{content_type}: the prompt needs {list(needed)} and these "
                f"are missing or blank: {missing}"
            )
        return template.format(**{f: values.get(f, "") for f in needed})

    def messages(self, row: Mapping[str, Any], content_type: str) -> list[dict[str, str]]:
        """The single message factory, used by every stage."""
        return [
            {"role": "system", "content": self.system_prompt(content_type)},
            {"role": "user", "content": self.user_content(row, content_type)},
        ]

    def fingerprint(self) -> str:
        """Hash of everything that goes on the wire, the stub included."""
        payload = {
            "theme": self.theme.fingerprint(),
            "granularity": self.granularity,
            "prompt_style": self.prompt_style,
            "classes": list(self.class_names),
            "output_fields": list(self.output_fields()),
            "system": {ct: render_system_prompt(self.theme, self.granularity, ct)
                       for ct in CONTENT_TYPES},
            "user_template": {ct: self.user_template(ct) for ct in CONTENT_TYPES},
            "stub": self.stub(),
            "max_text_chars": self.theme.max_text_chars,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def as_stub(self) -> PromptSet:
        from dataclasses import replace
        return replace(self, prompt_style="stub")

    def as_full(self) -> PromptSet:
        from dataclasses import replace
        return replace(self, prompt_style="full")

    def describe(self) -> str:
        """Every prompt this set can send, printable before a run spends anything."""
        lines = [
            f"theme        : {self.theme.id} ({self.theme.title})   definition "
            f"{self.theme.fingerprint()}",
            f"granularity  : {self.granularity}",
            f"classes      : {list(self.class_names)}   abstain={self.abstain_class}   "
            f"signs={self.theme.signs()}",
            f"output fields: {list(self.output_fields())}",
            f"prompt style : {self.prompt_style}",
            f"fingerprint  : {self.fingerprint()}",
            f"body         : {len(self.theme.body.strip()):,} chars, written by the user and "
            f"sent verbatim",
        ]
        for content_type in CONTENT_TYPES:
            system = render_system_prompt(self.theme, self.granularity, content_type)
            lines += [
                "",
                "=" * 72,
                f"[{self.granularity}/{content_type}] system {len(system):,} chars, "
                f"~{len(system) // 3.7:.0f} tokens",
                "=" * 72,
                system,
                "",
                f"--- user template  (requires {list(self.required_fields(content_type))}) ---",
                self.user_template(content_type),
            ]
        stub = self.stub()
        lines += ["", "=" * 72, f"[serving stub, {self.granularity}] {len(stub):,} chars",
                  "=" * 72, stub]
        return "\n".join(lines)
