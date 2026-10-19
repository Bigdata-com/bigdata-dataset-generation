"""The theme file, and the three things the framework adds to the body."""

from __future__ import annotations

import dataclasses

import pytest

from distill.prompts import (
    CONTRACT_HEADER, HEADER, TRANSCRIPT_RULE, BlankInput, PromptSet, render_contract,
    render_stub, render_system_prompt, render_user_template, template_fields,
)
from distill.theme import ThemeClass, ThemeSpec

LEGACY_CHUNK_STUB = (
    'You are a pricing-power classifier. For the target company, label the excerpt and '
    'respond with only compact JSON: {"classification": "winner"|"loser"|"irrelevant", '
    '"reasoning": "<one sentence>"}.'
)


def test_a_theme_is_four_fields(pricing_power):
    assert pricing_power.id == "pricing_power"
    assert pricing_power.title == "pricing power"
    assert pricing_power.class_names == ("winner", "loser", "irrelevant")
    assert pricing_power.abstain_class == "irrelevant"
    assert pricing_power.signs() == {"winner": 1.0, "loser": -1.0, "irrelevant": 0.0}
    assert len(pricing_power.body.strip()) > 1000


def test_the_three_facts_prose_cannot_carry_are_declared(gainer_theme):
    """The parser needs the class list."""
    assert gainer_theme.signal_classes == ("gainer", "loser")
    assert gainer_theme.abstain_class == "irrelevant"
    assert gainer_theme.signs()["gainer"] == 1.0


def test_a_non_abstain_class_without_a_sign_is_refused(exposure_theme):
    broken = dataclasses.replace(
        exposure_theme,
        classes=tuple(dataclasses.replace(c, sign=None) if c.name == "exposed" else c
                      for c in exposure_theme.classes),
    )
    assert any("has no `sign`" in issue for issue in broken.validate())


def test_exactly_one_abstain_class(pricing_power):
    two = dataclasses.replace(
        pricing_power,
        classes=tuple(dataclasses.replace(c, abstain=True) for c in pricing_power.classes),
    )
    assert any("exactly one class must be the abstain class" in i for i in two.validate())


def test_a_stub_body_is_refused(pricing_power):
    thin = dataclasses.replace(pricing_power, body="Pricing power is about prices.")
    assert any("the body is" in i for i in thin.validate())


def test_a_body_that_never_mentions_a_class_it_declares_is_refused(pricing_power):
    """The contract would ask for a label the prompt never explains."""
    renamed = dataclasses.replace(
        pricing_power,
        classes=(ThemeClass(name="beneficiary", sign=1.0),
                 ThemeClass(name="victim", sign=-1.0),
                 ThemeClass(name="irrelevant", abstain=True)),
    )
    issues = renamed.validate()
    assert any("never mentions" in i for i in issues), issues


def test_a_sign_of_zero_survives_a_save_and_load(exposure_theme, tmp_path):
    """`not_exposed` is worth 0 and that IS the decision."""
    path = exposure_theme.save(tmp_path / "market_exposure.yaml")
    reloaded = ThemeSpec.load(path)
    assert reloaded.signs() == {"exposed": 1.0, "not_exposed": 0.0, "silent": 0.0}
    assert reloaded.fingerprint() == exposure_theme.fingerprint()


def test_an_unknown_key_is_refused(pricing_power, tmp_path):
    """The prose all goes in `body`."""
    import yaml

    data = pricing_power.to_dict()
    data["framing_question"] = "who has the upper hand?"
    path = tmp_path / "typo.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(SystemExit, match="unknown key"):
        ThemeSpec.load(path)


def test_classes_may_be_written_as_a_mapping(tmp_path):
    """`name: sign`, with no sign meaning the abstain class."""
    path = tmp_path / "t.yaml"
    path.write_text(
        "id: t\n"
        "title: a thing\n"
        "classes:\n"
        "  up: 1\n"
        "  down: -1\n"
        "  nothing:\n"
        "body: |\n"
        "  WHAT IS A THING?\n"
        "\n"
        "  UP: it goes up, clearly and repeatedly, in a way the text states outright.\n"
        "\n"
        "  DOWN: it goes down, clearly and repeatedly, in a way the text states outright.\n"
        "\n"
        "  NOTHING: the text says nothing about the thing. Not adjacent topics, not vague\n"
        "  gestures at it. When in doubt, choose NOTHING and be strict about it.\n",
        encoding="utf-8")
    theme = ThemeSpec.load(path).require_valid()
    assert theme.class_names == ("up", "down", "nothing")
    assert theme.abstain_class == "nothing"
    assert theme.signs() == {"up": 1.0, "down": -1.0, "nothing": 0.0}


def test_the_body_is_sent_verbatim_in_all_four_prompts(pricing_power):
    """Nothing reformats it: no reflowing, no substitution, no rewriting."""
    body = pricing_power.body.strip()
    for granularity in ("chunk", "sentence"):
        for content_type in ("news", "transcripts"):
            text = render_system_prompt(pricing_power, granularity, content_type)
            assert body in text, f"{granularity}/{content_type} altered the body"


def test_the_four_prompts_differ_only_in_the_header_and_the_transcript_rule(pricing_power):
    rendered = {
        f"{g}.{ct}": render_system_prompt(pricing_power, g, ct)
        for g in ("chunk", "sentence") for ct in ("news", "transcripts")
    }
    assert len(set(rendered.values())) == 4
    for key, text in rendered.items():
        granularity, content_type = key.split(".")
        assert text.startswith(HEADER[key].format(signal="pricing power",
                                                  SIGNAL="PRICING POWER"))
        assert text.endswith(CONTRACT_HEADER + "\n\n" + render_contract(pricing_power))
        remainder = text
        remainder = remainder.replace(HEADER[key].format(signal="pricing power",
                                                        SIGNAL="PRICING POWER"), "")
        remainder = remainder.replace(CONTRACT_HEADER + "\n\n"
                                      + render_contract(pricing_power), "")
        if content_type == "transcripts":
            remainder = remainder.replace(TRANSCRIPT_RULE[granularity], "")
        assert remainder.strip() == pricing_power.body.strip()


def test_the_transcript_rule_appears_only_for_transcripts(pricing_power):
    for granularity in ("chunk", "sentence"):
        news = render_system_prompt(pricing_power, granularity, "news")
        transcripts = render_system_prompt(pricing_power, granularity, "transcripts")
        assert TRANSCRIPT_RULE[granularity] not in news
        assert TRANSCRIPT_RULE[granularity] in transcripts


def test_the_transcript_rule_is_theme_independent(pricing_power, exposure_theme):
    """A transcript is retrieved for a company that may not be the one talking."""
    for granularity in ("chunk", "sentence"):
        a = render_system_prompt(pricing_power, granularity, "transcripts")
        b = render_system_prompt(exposure_theme, granularity, "transcripts")
        assert TRANSCRIPT_RULE[granularity] in a
        assert TRANSCRIPT_RULE[granularity] in b


def test_the_header_carries_the_signal_name_and_the_granularity(pricing_power):
    chunk = render_system_prompt(pricing_power, "chunk", "news")
    sentence = render_system_prompt(pricing_power, "sentence", "news")
    assert "building a pricing power signal for an equity strategy" in chunk
    assert "a news chunk retrieved for a specific company" in chunk
    assert "PRICING POWER from a single sentence" in sentence
    assert "one SENTENCE extracted from text about the company" in sentence


def test_the_contract_is_built_from_the_class_names(gainer_theme, exposure_theme):
    """So a prompt and the code cannot disagree about the taxonomy."""
    text = render_system_prompt(gainer_theme, "chunk", "news")
    assert '"classification": "gainer" | "loser" | "irrelevant"' in text
    assert '"winner"' not in text
    exposure = render_system_prompt(exposure_theme, "chunk", "news")
    assert '"exposed" | "not_exposed" | "silent"' in exposure


def test_the_stub_is_generated_and_matches_what_the_students_were_served(pricing_power):
    assert render_stub(pricing_power, "chunk") == LEGACY_CHUNK_STUB
    assert len(LEGACY_CHUNK_STUB) == 195
    sentence = render_stub(pricing_power, "sentence")
    assert "label the sentence" in sentence and "label the excerpt" not in sentence


def test_the_user_templates_are_the_frameworks(pricing_power):
    assert render_user_template(pricing_power, "chunk", "news") == \
        "TARGET COMPANY: {company_name}\n\nCHUNK:\n{text}"
    assert "HEADLINE" in render_user_template(pricing_power, "chunk", "transcripts")
    assert render_user_template(pricing_power, "sentence", "news") == \
        "COMPANY: {company_name}\nSENTENCE: {text}"


def test_required_fields_are_derived_from_the_template(pricing_power):
    prompts = PromptSet(pricing_power, "chunk")
    assert set(prompts.required_fields("news")) == {"company_name", "text"}
    assert "headline" in prompts.required_fields("transcripts")
    assert template_fields("a {x} and {y} and {x}") == ("x", "y")


def test_a_prompt_cannot_be_chosen_independently_of_the_granularity(pricing_power):
    chunk = PromptSet(pricing_power, "chunk")
    sentence = PromptSet(pricing_power, "sentence")
    assert chunk.system_prompt("news") != sentence.system_prompt("news")
    assert chunk.fingerprint() != sentence.fingerprint()
    with pytest.raises(ValueError):
        PromptSet(pricing_power, "paragraph")


def test_blank_input_is_refused_rather_than_sent(pricing_power):
    """On the unrepaired corpus this fires on 130,071 of 484,982 rows."""
    prompts = PromptSet(pricing_power, "chunk")
    with pytest.raises(BlankInput):
        prompts.user_content({"company_name": "Acme", "text": "   "}, "news")
    with pytest.raises(BlankInput):
        prompts.user_content({"company_name": "Acme", "text": "hello", "headline": ""},
                             "transcripts")
    ok = prompts.user_content({"company_name": "Acme", "text": "hello", "headline": "H"},
                              "transcripts")
    assert "Acme" in ok and "hello" in ok


def test_text_is_truncated_at_the_theme_cap(pricing_power):
    prompts = PromptSet(pricing_power, "chunk")
    long_text = "x" * (pricing_power.max_text_chars + 500)
    content = prompts.user_content({"company_name": "Acme", "text": long_text}, "news")
    assert content.count("x") == pricing_power.max_text_chars


def test_one_message_factory_for_every_stage(pricing_power):
    prompts = PromptSet(pricing_power, "chunk")
    row = {"company_name": "Acme", "text": "Prices rose.", "headline": "Acme: Q1"}
    messages = prompts.messages(row, "transcripts")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == prompts.system_prompt("transcripts")
    stub_messages = prompts.as_stub().messages(row, "transcripts")
    assert stub_messages[0]["content"] == prompts.stub()
    assert stub_messages[1]["content"] == messages[1]["content"], (
        "the user message must not depend on the prompt style: the student sees the same text "
        "the teacher saw"
    )


def test_the_fingerprint_covers_the_body_and_the_stub(pricing_power):
    base = PromptSet(pricing_power, "chunk")
    assert base.fingerprint() == PromptSet(pricing_power, "chunk").fingerprint()
    reworded = PromptSet(dataclasses.replace(
        pricing_power, body=pricing_power.body + "\nOne more line."), "chunk")
    assert base.fingerprint() != reworded.fingerprint()
    renamed = PromptSet(dataclasses.replace(pricing_power, title="pricing-power v2"), "chunk")
    assert base.fingerprint() != renamed.fingerprint(), "the title reaches the header and the stub"


def test_class_names_appear_nowhere_in_the_code_path(gainer_theme):
    """A theme that renames its classes needs no code change, at any layer."""
    prompts = PromptSet(gainer_theme, "sentence")
    text = prompts.system_prompt("news")
    assert "gainer" in text and "winner" not in text
    assert '"gainer"' in prompts.stub()
    assert prompts.class_names == ("gainer", "loser", "irrelevant")


def test_describe_prints_every_prompt_and_both_the_body_and_the_stub(pricing_power):
    described = PromptSet(pricing_power, "chunk").describe()
    for fragment in ("[chunk/news]", "[chunk/transcripts]", "[serving stub, chunk]",
                     "written by the user and sent verbatim", "--- user template"):
        assert fragment in described


def test_show_prompts_covers_both_granularities(tmp_path, capsys):
    """`granularity: both` used to reach PromptSet as the literal string 'both'."""
    import argparse
    from pathlib import Path

    import yaml

    from distill.cli import cmd_show_prompts
    from conftest import theme_section

    run = tmp_path / "output"
    run.mkdir(parents=True, exist_ok=True)
    (run.parent / "config").mkdir(parents=True, exist_ok=True)
    theme = ThemeSpec.load(Path(__file__).resolve().parent.parent / "themes"
                           / "pricing_power.yaml")
    (run.parent / "config" / "data_extraction.yaml").write_text(
        yaml.safe_dump({"theme": theme_section(theme)}), encoding="utf-8")
    (run.parent / "config" / "distill.yaml").write_text(
        yaml.safe_dump({"input": {"source_parquet": "corpus.parquet"},
                        "experiment": {"granularity": "both", "topology": "combined"}}),
        encoding="utf-8")
    (run / "corpus.parquet").touch()

    args = argparse.Namespace(run=str(run), config=None, dry_run=False, theme=None,
                              granularity=None, stage_name="config")
    assert cmd_show_prompts(args) == 0
    out = capsys.readouterr().out
    assert "chunk" in out and "sentence" in out
