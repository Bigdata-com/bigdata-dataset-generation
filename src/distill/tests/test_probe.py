"""The pre-flight template probe: does the served student write a reasoning trace?"""

from __future__ import annotations

from pathlib import Path

import pytest

from distill.client import EndpointConfig, LabelClient
from distill.prompts import PromptSet
from distill.theme import ThemeSpec

THEMES_DIR = Path(__file__).resolve().parent.parent / "themes"

CLEAN = '{"classification": "winner", "reasoning": "prices rose"}'
BASE_TEMPLATE = ('<think>\nThe company raised prices across the portfolio, so this is a\n'
                 'winner. Let me double-check the margin comment.\n</think>\n'
                 '{"classification": "winner", "reasoning": "prices rose"}')
PROSE_FIRST = ('Looking at this chunk, the company clearly raised prices. '
               '{"classification": "winner", "reasoning": "prices rose"}')


def client_answering(raw: str, *, style: str = "stub") -> LabelClient:
    theme = ThemeSpec.load(THEMES_DIR / "pricing_power.yaml")
    prompts = PromptSet(theme, "chunk", prompt_style=style)
    client = LabelClient(prompts, EndpointConfig(url="http://x/v1", model="student"))
    client._post = lambda payload: (raw, "stop", None)
    return client


ROW = {"company_name": "Acme", "text": "Acme raised prices 8% and held volume.", "headline": ""}


def test_a_clean_answer_passes():
    """Silence is the pass. The gate's whole product is the SystemExit it does not raise."""
    client_answering(CLEAN).require_no_stray_reasoning(ROW, "news")


def test_the_base_template_is_caught_and_the_pass_refused():
    """The real failure: --chat-template omitted, so the base template opens <think>."""
    client = client_answering(BASE_TEMPLATE)
    severity, message = client.stray_reasoning(ROW, "news")
    assert severity == "error"
    assert "double-check the margin" in message, "the message must quote what came back"
    assert "--chat-template" in message
    assert "enable_thinking" in message and "will NOT fix this" in message

    with pytest.raises(SystemExit) as exc:
        client.require_no_stray_reasoning(ROW, "news")
    assert "94% -> 33.5%" in str(exc.value)


def test_prose_before_the_json_warns_but_does_not_block(capsys):
    """No <think>, so the template is probably right and it may be one row."""
    client = client_answering(PROSE_FIRST)
    severity, _ = client.stray_reasoning(ROW, "news")
    assert severity == "warn"
    client.require_no_stray_reasoning(ROW, "news")
    assert "prose before the JSON" in capsys.readouterr().out


def test_a_teacher_is_never_probed():
    """It reasons on purpose: turning that off cost 92.8% -> 62.7%."""
    client = client_answering(BASE_TEMPLATE, style="full")
    assert client.stray_reasoning(ROW, "news") is None
    client.require_no_stray_reasoning(ROW, "news")


def test_the_probe_costs_one_call():
    client = client_answering(CLEAN)
    calls = {"n": 0}
    original = client._post
    def counting(payload):
        calls["n"] += 1
        return original(payload)
    client._post = counting
    client.require_no_stray_reasoning(ROW, "news")
    assert calls["n"] == 1


def test_probe_template_walks_past_rows_with_no_probe_row(capsys):
    """A sentence strategy cannot shape a probe row for a chunk with no candidate sentence."""
    import pandas as pd

    from distill.probe import probe_template

    class Strategy:
        def __init__(self):
            self.seen = 0
        def probe_row(self, row):
            self.seen += 1
            return None if self.seen < 3 else dict(ROW)

    strategy = Strategy()
    frame = pd.DataFrame({"row_key": [f"k{i}" for i in range(5)]})
    probe_template(client_answering(CLEAN), strategy, frame, "news")
    assert strategy.seen == 3


def test_probe_template_says_so_when_it_could_not_probe(capsys):
    import pandas as pd

    from distill.probe import probe_template

    class Strategy:
        def probe_row(self, row):
            return None

    frame = pd.DataFrame({"row_key": [f"k{i}" for i in range(9)]})
    probe_template(client_answering(CLEAN), Strategy(), frame, "news", attempts=4)
    out = capsys.readouterr().out
    assert "SKIPPED" in out and "unverified" in out
