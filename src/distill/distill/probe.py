"""One call before a metered pass: is the right chat template loaded?

The failure this catches. A student is trained to emit `<think></think>` and then the JSON,
and is served with `ops/n34b_nothink_template.jinja`, whose `enable_thinking` defaults to
False. The MERGED model directory carries the base model's template instead, which opens a
`<think>` block on `add_generation_prompt` whatever `enable_thinking` says -- so forgetting
`--chat-template` gets you a model writing prose it was never trained to produce. Measured:
2.26x fewer calls/s and accuracy 94% -> 33.5%, with nothing in the output saying so.

Why not `chat_template_kwargs: {enable_thinking: false}`, which the handover proposed as a
belt to those braces: it cannot work. Our own template already defaults the flag to False, so
passing it is redundant where the template is right; and the base template ignores it, so it is
useless where it would be needed. The only thing that flag can actually do is turn reasoning ON
if someone ever passes `true`. The answer itself is the signal, and this reads it.
"""

from __future__ import annotations

from typing import Any

from .client import LabelClient


def probe_template(client: LabelClient, strategy: Any, frame: Any, content_type: str,
                   *, attempts: int = 5) -> None:
    """Send one real row and refuse the pass if the answer carries a reasoning trace.

    Tries up to `attempts` rows because a sentence-level strategy cannot shape a probe row
    for a chunk that yielded no candidate sentence. Costs one call, and runs before any
    checkpoint exists -- so a refusal leaves nothing half-labelled to resume from.
    """
    if client.prompts.prompt_style != "stub":
        return
    for row in frame.head(attempts).itertuples(index=False):
        probe = strategy.probe_row(row)
        if probe is None:
            continue
        client.require_no_stray_reasoning(probe, content_type)
        return
    print(f"template probe: no probe row in the first {attempts} rows -- SKIPPED. The chat "
          f"template is unverified for this pass.")
