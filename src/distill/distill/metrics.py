"""Metrics -- the file a person reads to choose a checkpoint, so it is a deliverable."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

FAIL_TOKEN = "__fail__"


@dataclass
class ClassMetrics:
    name: str
    support: int = 0
    predicted: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"support": self.support, "predicted": self.predicted,
                "precision": round(self.precision, 4), "recall": round(self.recall, 4),
                "f1": round(self.f1, 4)}


@dataclass
class Report:
    """One block: overall, or one content type, or one intersection."""

    title: str
    classes: tuple[str, ...]
    rows: int = 0
    unparseable: int = 0
    accuracy: float = 0.0
    macro_f1: float = 0.0
    abstain_baseline_accuracy: float = 0.0
    abstain_baseline_macro_f1: float = 0.0
    per_class: dict[str, ClassMetrics] = field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "rows": self.rows,
            "unparseable": self.unparseable,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "abstain_baseline_accuracy": round(self.abstain_baseline_accuracy, 4),
            "abstain_baseline_macro_f1": round(self.abstain_baseline_macro_f1, 4),
            "per_class": {k: v.to_dict() for k, v in self.per_class.items()},
            "predicted_counts": {k: v.predicted for k, v in self.per_class.items()},
            "gold_counts": {k: v.support for k, v in self.per_class.items()},
            "confusion": self.confusion,
        }

    def render(self, *, abstain: str | None = None) -> str:
        width = max([len(c) for c in self.classes] + [len(FAIL_TOKEN), 8]) + 2
        lines = [f"--- {self.title}  ({self.rows:,} rows) ---",
                 f"{'class':<{width}}{'support':>9}{'predicted':>11}{'precision':>11}"
                 f"{'recall':>9}{'f1':>8}"]
        for name in self.classes:
            m = self.per_class[name]
            lines.append(
                f"{name:<{width}}{m.support:>9,}{m.predicted:>11,}"
                f"{100 * m.precision:>10.1f}%{100 * m.recall:>8.1f}%{m.f1:>8.3f}"
            )
        lines.append(
            f"accuracy {100 * self.accuracy:.2f}%   MACRO-F1 {self.macro_f1:.4f}   "
            f"unparseable {self.unparseable:,}"
        )
        if abstain:
            lines.append(
                f"  all-{abstain} baseline would score {100 * self.abstain_baseline_accuracy:.1f}% "
                f"accuracy and a macro-F1 of {self.abstain_baseline_macro_f1:.3f} -- which is why "
                f"macro-F1 is the headline and accuracy is not a metric here"
            )
        lines.append("confusion (rows = teacher, columns = model):")
        header_cols = [*self.classes, FAIL_TOKEN]
        cell = max(9, max(len(c) for c in header_cols) + 1)
        lines.append(" " * width + "".join(f"{c[:cell - 1]:>{cell}}" for c in header_cols))
        for truth in self.classes:
            row = self.confusion.get(truth, {})
            lines.append(f"{truth:<{width}}" + "".join(f"{row.get(p, 0):>{cell},}" for p in header_cols))
        lines.append("calibration -- the model's class counts against the teacher's:")
        for name in self.classes:
            m = self.per_class[name]
            drift = m.predicted - m.support
            share = (100 * drift / m.support) if m.support else 0.0
            lines.append(f"  {name:<{width}} teacher {m.support:>8,}   model {m.predicted:>8,}"
                         f"   {drift:+8,}  ({share:+6.1f}%)")
        return "\n".join(lines)


def _empty_report(title: str, classes: Sequence[str]) -> Report:
    report = Report(title=title, classes=tuple(classes))
    report.per_class = {c: ClassMetrics(name=c) for c in classes}
    report.confusion = {t: {p: 0 for p in (*classes, FAIL_TOKEN)} for t in classes}
    return report


def score(
    pairs: Iterable[tuple[str, str | None]],
    classes: Sequence[str],
    *,
    title: str = "OVERALL",
    abstain: str | None = None,
) -> Report:
    """Score (gold, predicted) pairs."""
    classes = tuple(classes)
    report = _empty_report(title, classes)
    gold_counts = {c: 0 for c in classes}
    pred_counts = {c: 0 for c in classes}
    correct = 0
    total = 0
    failed = 0

    for gold, predicted in pairs:
        if gold not in gold_counts:
            continue
        total += 1
        gold_counts[gold] += 1
        effective = predicted if predicted in pred_counts else FAIL_TOKEN
        if effective == FAIL_TOKEN:
            failed += 1
        else:
            pred_counts[effective] += 1
            if effective == gold:
                correct += 1
        report.confusion[gold][effective] = report.confusion[gold].get(effective, 0) + 1

    report.rows = total
    report.unparseable = failed
    report.accuracy = correct / total if total else 0.0
    f1s: list[float] = []
    for name in classes:
        tp = report.confusion[name].get(name, 0)
        fn = gold_counts[name] - tp
        fp = pred_counts[name] - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / gold_counts[name] if gold_counts[name] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        report.per_class[name] = ClassMetrics(
            name=name, support=gold_counts[name], predicted=pred_counts[name],
            precision=precision, recall=recall, f1=f1,
        )
        if gold_counts[name]:
            f1s.append(f1)
    report.macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    if abstain and total:
        share = gold_counts.get(abstain, 0) / total
        report.abstain_baseline_accuracy = share
        used = sum(1 for name in classes if gold_counts[name])
        baseline_f1 = (2 * share / (share + 1)) if share else 0.0
        report.abstain_baseline_macro_f1 = baseline_f1 / used if used else 0.0
    return report


@dataclass
class Evaluation:
    """Every report for one checkpoint, plus what produced it."""

    tag: str
    classes: tuple[str, ...]
    abstain: str | None
    model: str = ""
    checkpoint: str = ""
    prompt_fingerprint: str = ""
    heldout: str = ""
    rows: int = 0
    unparseable: int = 0
    reports: dict[str, Report] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def overall(self) -> Report:
        return self.reports["overall"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "model": self.model,
            "checkpoint": self.checkpoint,
            "prompt_fingerprint": self.prompt_fingerprint,
            "heldout": self.heldout,
            "classes": list(self.classes),
            "abstain": self.abstain,
            "rows": self.rows,
            "unparseable": self.unparseable,
            "macro_f1": round(self.overall.macro_f1, 4) if "overall" in self.reports else None,
            "accuracy": round(self.overall.accuracy, 4) if "overall" in self.reports else None,
            "reports": {k: v.to_dict() for k, v in self.reports.items()},
            **self.extra,
        }

    def render(self) -> str:
        head = [
            "=" * 72,
            f"EVALUATION  {self.tag}",
            "=" * 72,
            f"model            : {self.model}",
            f"checkpoint       : {self.checkpoint or '(not stated)'}",
            f"held-out         : {self.heldout}",
            f"prompt           : {self.prompt_fingerprint}",
            f"classes          : {list(self.classes)}  abstain={self.abstain}",
            f"rows             : {self.rows:,}   unparseable {self.unparseable:,} "
            f"({100 * self.unparseable / max(self.rows, 1):.2f}%)",
            "",
        ]
        blocks = [self.reports[key].render(abstain=self.abstain)
                  for key in ("overall", *sorted(k for k in self.reports if k != "overall"))]
        return "\n".join(head + ["\n\n".join(blocks), ""])

    def write(self, json_path: str | Path, text_path: str | Path) -> None:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        Path(text_path).write_text(self.render(), encoding="utf-8")


def comparison_table(evaluations: Sequence[Evaluation], *, keys: Sequence[str] | None = None) -> str:
    """One table across checkpoints, so the epoch curve is readable from disk."""
    if not evaluations:
        return "(no evaluations)"
    first = evaluations[0]
    blocks = keys or ["overall", *sorted(k for k in first.reports if k != "overall")]
    name_width = max(len(e.tag) for e in evaluations) + 2

    lines = ["=" * 72, "COMPARISON ACROSS CHECKPOINTS", "=" * 72,
             "The pipeline does not choose. Three defensible criteria picked three different",
             "checkpoints on one measured run: aggregate macro-F1, rare-class recall, and",
             "calibration against the teacher's proportions. Read all three columns.", ""]

    header = f"{'checkpoint':<{name_width}}" + "".join(f"{('macroF1 ' + b)[:14]:>15}" for b in blocks)
    lines += [header, "-" * len(header)]
    for evaluation in evaluations:
        row = f"{evaluation.tag:<{name_width}}"
        for block in blocks:
            report = evaluation.reports.get(block)
            row += f"{report.macro_f1:>15.4f}" if report else f"{'-':>15}"
        lines.append(row)

    lines += ["", f"{'checkpoint':<{name_width}}{'accuracy':>11}{'unparseable':>13}"]
    for evaluation in evaluations:
        lines.append(f"{evaluation.tag:<{name_width}}"
                     f"{100 * evaluation.overall.accuracy:>10.2f}%{evaluation.unparseable:>13,}")

    for name in first.classes:
        lines += ["", f"class {name!r}: per-checkpoint precision / recall / f1, and the model's "
                      f"count against the teacher's"]
        lines.append(f"{'checkpoint':<{name_width}}{'prec':>9}{'recall':>9}{'f1':>9}"
                     f"{'teacher':>10}{'model':>9}{'drift':>9}")
        for evaluation in evaluations:
            m = evaluation.overall.per_class.get(name)
            if m is None:
                continue
            lines.append(
                f"{evaluation.tag:<{name_width}}{100 * m.precision:>8.1f}%{100 * m.recall:>8.1f}%"
                f"{m.f1:>9.3f}{m.support:>10,}{m.predicted:>9,}{m.predicted - m.support:>+9,}"
            )
    return "\n".join(lines) + "\n"


def load_evaluation(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
