# Writing a theme body

A theme is four fields. Three are trivial; the fourth is the `body`, and it is the whole
prompt except the header and the mechanics:

```yaml
id: pricing_power
title: pricing power        # goes into the header and the stub, lowercase prose
classes:
  winner: 1
  loser: -1
  irrelevant:               # no sign = this is the abstain class
body: |
  ...
```

The framework wraps the body in a header that says what kind of text is coming
(chunk or sentence, news or transcript), adds the transcript speaker rule when the text is a
transcript, and appends the JSON contract built from your class names. **Nothing reformats
the body.** What you write is what the model reads.

This file is here so a body can be drafted for a new theme — by hand or by a model — without
reverse-engineering an existing one.

## What the body has to contain

Four things, in this order. No headings are required; the ones below are what the
pricing-power body happens to use.

**1. What the concept is.** Two or three sentences. Say what it means, then what the strong
end and the weak end look like. Not a definition from a textbook — the distinction the model
has to draw.

> Pricing power reflects how much control a company has over the prices it charges. A company
> with strong pricing power can set or raise prices without losing customers — because its
> product is differentiated, essential, or has no good substitute. A company with weak pricing
> power is forced to accept the prices the market dictates.

**2. One line that turns the concept into a decision.** This is the sentence the model leans
on when the text is ambiguous, and it is worth more care than its length suggests.

> Think of it as a question: who has the upper hand — the company or its customers?

**3. One paragraph per class, naming the MARKERS and not just the direction.** Start each
with the class name in capitals, exactly as it appears in `classes`. Say what the text looks
like when that label applies — the words and situations, not a restatement of the label.

> WINNER: the company has the upper hand on pricing. The company sets prices, raises them,
> passes through cost increases, or commands a premium. Customers accept it. Margins hold or
> expand. Pricing is a source of strength.

**4. The abstain class, and the near-misses that specifically do not count.** This is the
paragraph that earns its keep. The abstain class is the *majority* class — 55-71% of every
corpus measured — so it is the one the model most has to get right, and a vague abstain list
produces a teacher that fires on anything merely topical. Name the near-misses explicitly.

> IRRELEVANT: the excerpt does not meaningfully speak to the target company's pricing power.
> This includes: insider transactions, stock price movements, analyst upgrades/downgrades
> without pricing content, M&A news, macroeconomic commentary, or excerpts where the target
> company is mentioned only peripherally. When in doubt, classify as IRRELEVANT. Be strict —
> a weak or vague signal is not a signal.

End with the strictness instruction. It is one sentence and it is what keeps precision up.

## Two shapes of class list, and they are not the same problem

**A polar pair** — two opposite directions on one axis, plus abstain. `winner`/`loser`,
`gainer`/`loser`, `beneficiary`/`victim`. One class is +1, the other −1. The signal sums them
per company per day. This is the shape everything downstream was designed around.

**A presence pair** — has it / does not have it, plus abstain. `high_exposure`/`low_exposure`,
`exposed`/`not_exposed`. Two questions the body has to answer that a polar pair does not:

- **Is the negative side −1, or 0?** For pricing power, `loser` is real negative information.
  For exposure, "the text says the company is **not** exposed" may be worth −1 in a long/short
  book or may be worth nothing. Put the answer in `classes`; the code will not guess it.
- **What is left for the abstain class?** This is the trap. `not_exposed` and `irrelevant` are
  easy to confuse: one is *the text asserts absence*, the other is *the text is silent*. If the
  body does not draw that line in as many words, the teacher collapses the two and the rare
  class disappears. Say it explicitly in the abstain paragraph.

## What not to put in the body

- **The class names as JSON.** The contract is appended for you, built from `classes`, so the
  two cannot disagree.
- **Anything about chunks versus sentences.** The header says which one is coming. Write the
  body so it reads correctly either way — "the excerpt" and "the text" work for both.
- **The transcript speaker rule.** It is added automatically for transcripts, identically for
  every theme, because a transcript is retrieved for a company that may not be the one talking
  and getting it wrong mislabels a whole class of rows.
- **Output-format instructions, JSON examples, or "respond only with".** All appended.

## Checks that run on it

`distill check-config` refuses a theme whose body is under 200 characters, whose body never
mentions a class it declares, that has no abstain class or more than one, or whose non-abstain
classes are missing a sign. `distill show-prompts` prints all four assembled prompts and both
stubs, so the exact text can be read before a run spends anything.
