"""Guided-decoding schema and early-stop sequences, both built from the prompt set."""

from __future__ import annotations

import json
from typing import Any

from .prompts import PromptSet


def json_schema(prompts: PromptSet) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name in prompts.output_fields():
        if name == prompts.label_field:
            properties[name] = {"type": "string", "enum": list(prompts.class_names)}
        else:
            properties[name] = {"type": "string"}
        required.append(name)
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": required}


def response_format(prompts: PromptSet, *, name: str | None = None) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name or f"{prompts.theme.id}_label",
            "strict": True,
            "schema": json_schema(prompts),
        },
    }


def early_stop_sequences(prompts: PromptSet) -> list[str]:
    """Stop strings that end generation once the label is on the wire (+40%).

    The separator has to come from `json.dumps`, never from a literal. The student is trained
    on `stages.export.target_content`, which serialises with json's default `", "`, so what it
    emits is `, "reasoning"` -- WITH the space. A hand-written `,"reasoning"` matches nothing,
    vLLM never stops, and the flag then buys no throughput at all, silently: no error, no
    warning, just the old rate. That is how it shipped, unmeasured, until 2026-08-28.

    `output_fields` emits the label first, so cutting at the rationale key always leaves the
    label complete.
    """
    rationale = prompts.rationale_field
    if not rationale:
        return []
    key = f'"{rationale}"'
    rendered = json.dumps({prompts.label_field: "", rationale: ""}, ensure_ascii=False)
    before = rendered[: rendered.index(key)]
    return [before[before.rindex(",") :] + key]
