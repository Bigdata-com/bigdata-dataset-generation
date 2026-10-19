"""ThemeSpec — four fields, and the body is one of them."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Any

import yaml

THEME_SPEC_VERSION = "2"

GRANULARITIES: tuple[str, ...] = ("chunk", "sentence")
TOPOLOGIES: tuple[str, ...] = ("combined", "specialist", "both")
CONTENT_TYPES: tuple[str, ...] = ("news", "transcripts")

THEME_KEYS = frozenset({
    "id", "title", "body", "classes", "label_field", "rationale_field",
    "rationale_instruction", "max_text_chars", "owner", "version", "notes",
})
CLASS_KEYS = frozenset({"name", "sign", "abstain", "aliases"})

MIN_BODY_CHARS = 200


@dataclass(frozen=True)
class ThemeClass:
    name: str
    sign: float | None = None
    abstain: bool = False
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ThemeSpec:
    id: str
    title: str
    body: str
    classes: tuple[ThemeClass, ...]
    label_field: str = "classification"
    rationale_field: str = "reasoning"
    rationale_instruction: str = "one sentence, specific to this excerpt and this company"
    max_text_chars: int = 6000
    owner: str = ""
    version: str = "1"
    notes: str = ""

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.classes)

    @property
    def abstain_class(self) -> str | None:
        for c in self.classes:
            if c.abstain:
                return c.name
        return None

    @property
    def signal_classes(self) -> tuple[str, ...]:
        """The non-abstain classes: the walk's stopping rule, taken from the theme."""
        return tuple(c.name for c in self.classes if not c.abstain)

    @property
    def slug(self) -> str:
        """`pricing-power`, for the stub's "You are a <slug> classifier"."""
        return self.title.replace(" ", "-")

    def klass(self, name: str) -> ThemeClass:
        for c in self.classes:
            if c.name == name:
                return c
        raise KeyError(f"{self.id}: no class named {name!r} (have {list(self.class_names)})")

    def signs(self) -> dict[str, float]:
        """name -> signal sign."""
        out: dict[str, float] = {}
        for c in self.classes:
            if c.sign is not None:
                out[c.name] = float(c.sign)
            elif c.abstain:
                out[c.name] = 0.0
        return out

    def alias_map(self) -> dict[str, str]:
        return {a.lower(): c.name for c in self.classes for a in c.aliases}

    def validate(self) -> list[str]:
        issues: list[str] = []
        if not self.id.strip():
            issues.append("no id")
        if not self.title.strip():
            issues.append("no title — it is what the header and the stub call the signal")
        if len(self.body.strip()) < MIN_BODY_CHARS:
            issues.append(
                f"the body is {len(self.body.strip())} characters, under {MIN_BODY_CHARS}. It has "
                f"to say what the concept is, what each class looks like, and what does not "
                f"count — see prompt_guidelines.md."
            )
        if not self.classes:
            issues.append("no classes")
        names = [c.name for c in self.classes]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            issues.append(f"duplicate class names: {dupes}")
        abstains = [c.name for c in self.classes if c.abstain]
        if len(abstains) != 1:
            issues.append(
                f"exactly one class must be the abstain class, found {abstains or 'none'} — it is "
                f"what a model picks when nothing applies, and what a failed parse must never "
                f"silently become"
            )
        for c in self.classes:
            if not c.abstain and c.sign is None:
                issues.append(
                    f"class {c.name!r} has no `sign` — the backtest cannot build a signal without "
                    f"one, and guessing it is the bug this field exists to prevent"
                )
        signal_signs = {c.name: c.sign for c in self.classes if not c.abstain}
        if signal_signs and all((s or 0.0) == 0.0 for s in signal_signs.values()):
            issues.append(f"every non-abstain class has sign 0: the signal is identically zero "
                          f"({signal_signs})")
        body_upper = self.body.upper()
        unexplained = [c.name for c in self.classes if c.name.upper() not in body_upper]
        if unexplained:
            issues.append(
                f"the body never mentions {unexplained}. The contract will ask for those labels "
                f"and nothing in the prompt says what they mean."
            )
        if self.rationale_field == self.label_field:
            issues.append("rationale_field collides with label_field")
        return issues

    def require_valid(self) -> ThemeSpec:
        issues = self.validate()
        if issues:
            raise SystemExit(f"theme {self.id!r} has problems:\n  - " + "\n  - ".join(issues))
        return self

    def fingerprint(self) -> str:
        payload = {
            "spec_version": THEME_SPEC_VERSION,
            "id": self.id,
            "title": self.title,
            "body": self.body.strip(),
            "label_field": self.label_field,
            "rationale_field": self.rationale_field,
            "rationale_instruction": self.rationale_instruction,
            "max_text_chars": self.max_text_chars,
            "classes": [{"name": c.name, "sign": c.sign, "abstain": c.abstain,
                         "aliases": list(c.aliases)} for c in self.classes],
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThemeSpec:
        unknown = sorted(set(data) - THEME_KEYS)
        if unknown:
            raise SystemExit(
                f"theme {data.get('id')!r} has unknown key(s) {unknown}.\n"
                f"Known keys: {sorted(THEME_KEYS)}\n"
                f"Refused rather than ignored: a key this loader does not read would look like it "
                f"had been honoured. The prose all goes in `body`."
            )
        raw_classes = data.get("classes") or []
        classes: list[ThemeClass] = []
        if isinstance(raw_classes, dict):
            for name, sign in raw_classes.items():
                classes.append(ThemeClass(name=str(name),
                                          sign=None if sign is None else float(sign),
                                          abstain=(sign is None)))
        else:
            for entry in raw_classes:
                if not isinstance(entry, dict):
                    raise SystemExit(f"theme {data.get('id')!r}: each class must be a mapping, "
                                     f"got {entry!r}")
                bad = sorted(set(entry) - CLASS_KEYS)
                if bad:
                    raise SystemExit(f"theme {data.get('id')!r}, class {entry.get('name')!r}: "
                                     f"unknown key(s) {bad}. Known: {sorted(CLASS_KEYS)}")
                classes.append(ThemeClass(
                    name=entry["name"],
                    sign=None if entry.get("sign") is None else float(entry["sign"]),
                    abstain=bool(entry.get("abstain", False)),
                    aliases=tuple(entry.get("aliases") or ()),
                ))
        return cls(
            id=data["id"],
            title=data.get("title", data["id"]),
            body=data.get("body", ""),
            classes=tuple(classes),
            label_field=data.get("label_field", "classification"),
            rationale_field=data.get("rationale_field", "reasoning"),
            rationale_instruction=data.get(
                "rationale_instruction", "one sentence, specific to this excerpt and this company"),
            max_text_chars=int(data.get("max_text_chars", 6000)),
            owner=data.get("owner", ""),
            version=str(data.get("version", "1")),
            notes=data.get("notes", ""),
        )

    @classmethod
    def load(cls, path: str | Path) -> ThemeSpec:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"theme file not found: {path}")
        with path.open(encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh))

    @classmethod
    def resolve(cls, name_or_path: str | Path, *, themes_dir: Path | None = None) -> ThemeSpec:
        """Accept a path, or a bare theme id resolved under `themes_dir`."""
        candidate = Path(name_or_path)
        if candidate.suffix in (".yaml", ".yml") or candidate.exists():
            return cls.load(candidate)
        root = themes_dir or (Path(__file__).resolve().parent.parent / "themes")
        for suffix in (".yaml", ".yml"):
            probe = root / f"{name_or_path}{suffix}"
            if probe.exists():
                return cls.load(probe)
        available = sorted(p.stem for p in root.glob("*.y*ml")) if root.exists() else []
        raise FileNotFoundError(
            f"no theme {name_or_path!r} under {root} (available: {available or '(none)'}).\n"
            f"A new theme is four fields: id, title, body, classes. See "
            f"{root.parent / 'distill' / 'prompt_guidelines.md'}."
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "title": self.title}
        if self.owner:
            out["owner"] = self.owner
        out["version"] = self.version
        out["classes"] = [
            {k: v for k, v in {"name": c.name, "sign": c.sign, "abstain": c.abstain,
                               "aliases": list(c.aliases)}.items()
             if not (v is None or v is False or v == [])}
            for c in self.classes
        ]
        out["body"] = self.body
        if self.label_field != "classification":
            out["label_field"] = self.label_field
        if self.rationale_field != "reasoning":
            out["rationale_field"] = self.rationale_field
        out["rationale_instruction"] = self.rationale_instruction
        out["max_text_chars"] = self.max_text_chars
        if self.notes:
            out["notes"] = self.notes
        return out

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, allow_unicode=True, width=100)
        return path
