#!/usr/bin/env python3
"""Extract, from one chunk of text, the sentences that mention the target entity."""

import html
import re

ABBREV = {
    "inc", "corp", "co", "cos", "ltd", "plc", "llc", "lp", "llp", "bros",
    "mfg", "pty", "spa",
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "messrs", "hon",
    "gen", "sen", "rep", "gov", "lt", "col", "capt", "sgt", "adm", "pres",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
    "vs", "etc", "al", "approx", "no", "nos", "fig", "figs", "vol", "vols",
    "pp", "pg", "ca", "cf", "viz", "dept", "est", "incl", "esp",
    "ave", "blvd", "rd", "ste", "mt", "ft", "ext", "min", "max",
}

_DOT = ""
_ELL = ""

_ABBREV_RE = re.compile(
    r'\b(' + '|'.join(sorted((re.escape(a) for a in ABBREV), key=len, reverse=True)) + r')\.',
    re.IGNORECASE,
)

_BOUNDARY = re.compile(r'[.!?]["\'”’)\]]*(\s+)(?=["“‘(\[]?[A-Z0-9])')


def _protect(text: str) -> str:
    """Neutralize "fake" periods."""
    text = re.sub(r'\.\.\.+', lambda m: _ELL * len(m.group()), text)
    text = re.sub(r'(?<=\d)\.(?=\d)', _DOT, text)
    text = re.sub(r'(?<![A-Za-z])([A-Za-z])\.', r'\1' + _DOT, text)
    text = _ABBREV_RE.sub(lambda m: m.group(1) + _DOT, text)
    return text


def sentence_spans(raw: str):
    """[(start, end, text), ...] with indices referring to `raw` (RAW text)."""
    if not isinstance(raw, str):
        return []
    protected = _protect(raw)
    spans = []
    start = 0
    for m in _BOUNDARY.finditer(protected):
        end = m.start(1)
        if end > start:
            spans.append((start, end, raw[start:end]))
        start = m.end(1)
    if start < len(raw) and raw[start:].strip():
        spans.append((start, len(raw), raw[start:]))
    return spans


def parse_entity_offsets(value):
    """Extract the (start, end) pairs."""
    if value is None:
        return []
    if not isinstance(value, str):
        try:
            return [(int(a), int(b)) for a, b in value]
        except Exception:
            value = str(value)
    return [(int(a), int(b)) for a, b in re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', value)]


def _first_word_pos(text, a, b):
    """First "content" index in [a, b): skip leading whitespace and punctuation."""
    n = len(text)
    for p in range(a, min(b, n)):
        if text[p].isalnum():
            return p
    return a


def extract_target_sentences(chunk_text, entity_offsets):
    """The deduplicated, ordered sentences holding at least one target offset."""
    offsets = parse_entity_offsets(entity_offsets)
    spans = sentence_spans(chunk_text)
    hit_idx, seen = [], set()
    for (a, b) in offsets:
        a = _first_word_pos(chunk_text, a, b)
        for i, (s, e, _t) in enumerate(spans):
            if s <= a < e:
                if i not in seen:
                    seen.add(i)
                    hit_idx.append(i)
                break
    out = []
    for i in hit_idx:
        s = re.sub(r'\s+', ' ', html.unescape(spans[i][2])).strip()
        out.append(s)
    return out
