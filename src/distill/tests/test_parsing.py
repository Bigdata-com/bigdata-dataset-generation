"""The parse cascade."""

from __future__ import annotations

import dataclasses

from distill.parsing import parse, split_reasoning
from distill.prompts import PromptSet
from distill.schema import early_stop_sequences
from distill.stages.export import target_content


def contract(theme, granularity="chunk"):
    return PromptSet(theme, granularity)


def test_clean_json(pricing_power):
    result = parse('{"classification": "winner", "reasoning": "prices rose"}',
                   contract(pricing_power))
    assert result.ok and result.label == "winner" and result.rationale == "prices rose"
    assert not result.recovered


def test_a_missing_label_field_is_an_error_not_a_silent_abstain(pricing_power):
    """The original read `Classification(parsed.get("classification"."""
    result = parse('{"reasoning": "no idea"}', contract(pricing_power))
    assert not result.ok
    assert result.label is None
    assert result.error_kind == "schema"


def test_an_unknown_label_is_an_error(pricing_power):
    result = parse('{"classification": "maybe"}', contract(pricing_power))
    assert not result.ok and result.label is None


def test_case_and_whitespace_are_coerced(pricing_power):
    assert parse('{"classification": " WINNER "}', contract(pricing_power)).label == "winner"


def test_reasoning_traces_are_stripped(pricing_power):
    raw = '<think>Let me consider the margins.</think>{"classification": "loser"}'
    result = parse(raw, contract(pricing_power))
    assert result.label == "loser"
    assert "margins" in result.reasoning_trace


def test_a_closing_tag_with_no_opening_one(pricing_power):
    """vLLM's chat template opens the block, so the opening tag never reaches the completion."""
    raw = 'thinking out loud about pricing</think>{"classification": "winner"}'
    result = parse(raw, contract(pricing_power))
    assert result.label == "winner"
    assert "thinking out loud" in result.reasoning_trace


def test_a_draft_object_inside_the_reasoning_does_not_poison_the_slice(pricing_power):
    """first-brace-to-last-brace spans the draft and the answer and parses as nothing."""
    raw = ('I might say {"classification": "loser"} but on reflection\n'
           '{"classification": "winner", "reasoning": "raised prices"}')
    result = parse(raw, contract(pricing_power))
    assert result.label == "winner"


def test_code_fences(pricing_power):
    raw = '```json\n{"classification": "irrelevant", "reasoning": "nothing"}\n```'
    assert parse(raw, contract(pricing_power)).label == "irrelevant"


def test_an_unquoted_rationale_is_repaired(pricing_power):
    """Observed repeatedly in reasoning mode."""
    raw = ('{"classification": "irrelevant", "reasoning": The excerpt focuses on the '
           'buyback, not on pricing power.}')
    result = parse(raw, contract(pricing_power))
    assert result.ok and result.label == "irrelevant"
    assert "buyback" in result.rationale


def test_a_greedy_repair_survives_quotes_inside_the_value(pricing_power):
    raw = ('{"classification": "winner", "reasoning": The filing says "increased pricing," '
           'which held margins.}')
    assert parse(raw, contract(pricing_power)).label == "winner"


def test_truncated_json_from_early_stop_serving_is_closed(pricing_power):
    raw = '{"classification": "winner"'
    assert parse(raw, contract(pricing_power)).label == "winner"


def test_the_early_stop_string_is_text_the_student_actually_emits(pricing_power):
    """The half that `test_truncated_json_from_early_stop_serving_is_closed` cannot see.

    The stop string and the training target are written by two different modules. They
    disagreed by one space -- `,"reasoning"` against json's `, "reasoning"` -- so vLLM never
    stopped and `early_stop: true` bought nothing, with no error to say so. Assert the
    agreement rather than the string, because a literal here is the bug.
    """
    prompts = contract(pricing_power)
    target = target_content(prompts, {"classification": "winner", "reasoning": "prices rose"})
    (stop,) = early_stop_sequences(prompts)

    assert stop in target, f"vLLM would never stop: {stop!r} is not in {target!r}"
    assert target[: target.index(stop)] == '{"classification": "winner"', (
        "the cut has to leave the label complete and nothing of the rationale"
    )


def test_early_stop_then_parse_is_a_round_trip(pricing_power):
    """Cut the target where the server would cut it, and the parser must still read the label."""
    prompts = contract(pricing_power)
    target = target_content(prompts, {"classification": "loser", "reasoning": "discounting"})
    (stop,) = early_stop_sequences(prompts)

    served = target[: target.index(stop)]
    result = parse(served, prompts)
    assert result.ok and result.label == "loser"
    assert result.rationale == "", "the rationale is gone, and must not be guessed at"
    assert not result.recovered, "a deliberate truncation is not a salvaged parse"


def test_the_stop_string_follows_a_renamed_rationale_field(pricing_power):
    """It is derived from the theme, not from the word `reasoning`."""
    theme = dataclasses.replace(pricing_power, rationale_field="justification")
    prompts = PromptSet(theme, "chunk")
    (stop,) = early_stop_sequences(prompts)
    assert stop == ', "justification"'
    assert stop in target_content(prompts, {"classification": "winner", "justification": "x"})


def test_the_regex_fallback_recovers_the_two_corpus_failures(pricing_power):
    """Both were deterministic, and only the trailing rationale broke the JSON."""
    toys = ('{"classification": "irrelevant", "reasoning": "The chunk is about Toys"R"Us '
            'store closures, not pricing power."}')
    result = parse(toys, contract(pricing_power))
    assert result.ok and result.label == "irrelevant"
    assert result.recovered, "a recovered row must be marked, never look like a clean parse"
    assert result.rationale == "", "the rationale is dropped, not guessed at"

    turn_leak = ('{"classification": "winner", "reasoning": "Pricing has been'
                 'assistant\n<think></think>')
    leaked = parse(turn_leak, contract(pricing_power))
    assert leaked.ok and leaked.label == "winner" and leaked.recovered


def test_the_fallback_cannot_invent_a_label(pricing_power):
    assert not parse('{"classification": "excellent", "reasoning": broken', contract(pricing_power)).ok
    assert not parse("total nonsense, no json at all", contract(pricing_power)).ok


def test_an_empty_response(pricing_power):
    result = parse("", contract(pricing_power))
    assert not result.ok and result.error_kind == "empty"


def test_a_json_array_is_a_schema_error(pricing_power):
    result = parse('["winner"]', contract(pricing_power))
    assert not result.ok and result.error_kind == "schema"


def test_a_field_nobody_asked_for_is_ignored(pricing_power):
    """An older checkpoint may emit a field the contract no longer has."""
    raw = '{"classification": "winner", "confidence": "high", "reasoning": "x"}'
    result = parse(raw, contract(pricing_power))
    assert result.ok and result.label == "winner" and result.rationale == "x"


def test_class_names_come_from_the_theme(gainer_theme):
    result = parse('{"classification": "gainer", "reasoning": "took share"}',
                   contract(gainer_theme))
    assert result.ok and result.label == "gainer"
    assert not parse('{"classification": "winner"}', contract(gainer_theme)).ok


def test_split_reasoning_on_untagged_prose(pricing_power):
    """The 120B in reasoning mode emits no tags at all."""
    raw = 'The company clearly raised prices.\n{"classification": "winner"}'
    trace, body = split_reasoning(raw)
    assert "raised prices" in trace
    assert body == raw
    assert parse(raw, contract(pricing_power)).label == "winner"
