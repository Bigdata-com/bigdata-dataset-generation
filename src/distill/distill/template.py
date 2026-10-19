"""The three documented config files a new run starts from."""

from __future__ import annotations

import os
import re
from pathlib import Path

from .paths import DISTILL_CONFIG_NAME, EXTRACTION_CONFIG_NAME, config_dir
from .theme import ThemeSpec

DEFAULT_RATIONALE = "one sentence, specific to this excerpt and this company"

SCAFFOLD = """\
  id: my_theme
  title: my theme                 # goes into the header and the stub, lowercase prose
  classes:
    positive: 1
    negative: -1
    irrelevant:                   # no sign = this is the abstain class
  body: |
    FILL THIS IN. Four things, in this order -- see prompt_guidelines.md:
      1. what the concept is, and what its strong and weak ends look like
      2. one line that turns it into a decision
      3. one paragraph per class, in CAPITALS, naming the markers and not just the direction
      4. the abstain class and the near-misses that specifically do NOT count, then
         "When in doubt, classify as IRRELEVANT. Be strict -- a weak or vague signal is not
         a signal."
"""



def theme_block(theme: ThemeSpec | None) -> str:
    """The theme section, indented into the config file."""
    if theme is None:
        return SCAFFOLD.rstrip("\n")
    lines = [f"  id: {theme.id}", f"  title: {theme.title}", "  classes:"]
    for klass in theme.classes:
        lines.append(f"    {klass.name}:" if klass.abstain or klass.sign is None
                     else f"    {klass.name}: {klass.sign:g}")
    lines.append("  body: |")
    for line in theme.body.strip().splitlines():
        lines.append(f"    {line}" if line.strip() else "")
    if theme.rationale_instruction != DEFAULT_RATIONALE:
        lines.append(f"  rationale_instruction: {theme.rationale_instruction}")
    if theme.max_text_chars != 6000:
        lines.append(f"  max_text_chars: {theme.max_text_chars}")
    return "\n".join(lines)

def _read_config(name: str) -> str:
    return (config_dir() / name).read_text(encoding="utf-8")


def _replace_block(text: str, key: str, block: str) -> str:
    """Swap one top-level block, keeping every comment around it."""
    lines = text.splitlines(keepends=True)
    start = next((i for i, l in enumerate(lines) if l.startswith(f"{key}:")), None)
    if start is None:
        raise SystemExit(f"`{key}:` is not in the checked-in {DISTILL_CONFIG_NAME}")
    end = start + 1
    while end < len(lines) and not re.match(r"^[a-z_]+:", lines[end]):
        end += 1
    tail = lines[end:]
    gap = "" if not tail or tail[0].strip() == "" else "\n"
    return "".join(lines[:start]) + f"{key}:\n{block}\n{gap}" + "".join(tail)


def write_extraction_template(path: str | Path, *, theme: ThemeSpec | None,
                              run_dir: Path | None = None) -> Path:
    """Rewrite the `theme:` section of `data_extraction.yaml`, leaving the rest untouched."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base = path.read_text(encoding="utf-8") if path.exists() \
        else _read_config(EXTRACTION_CONFIG_NAME)
    path.write_text(_replace_block(base, "theme", theme_block(theme)), encoding="utf-8")
    return path
