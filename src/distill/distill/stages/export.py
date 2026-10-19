"""Stage 11 -- the training JSONL, and the commands that turn it into a student."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .. import frames, serving
from ..config import DistillConfig
from ..manifest import Manifest
from ..paths import RunPaths
from ..prompts import BlankInput, PromptSet
from ..theme import ThemeSpec
from .teacher import build_prompts


def target_content(prompts: PromptSet, row: dict[str, Any]) -> str:
    """The assistant message a student is trained to emit, in canonical field order."""
    payload: dict[str, Any] = {}
    for name in prompts.output_fields():
        if name == prompts.label_field:
            payload[name] = row.get(prompts.label_field)
        else:
            payload[name] = str(row.get(name, "") or "")
    return json.dumps(payload, ensure_ascii=False)


def write_jsonl(prompts: PromptSet, rows: list[dict[str, Any]], out_path: Path,
                *, content_column: str, client_id: str) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    valid = set(prompts.class_names)
    written = skipped_unlabelled = skipped_invalid = 0
    per_class: dict[str, int] = {}
    invalid_examples: list[str] = []

    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            label = row.get(prompts.label_field)
            if label is None or (isinstance(label, float) and label != label):
                skipped_unlabelled += 1
                continue
            if label not in valid:
                skipped_invalid += 1
                invalid_examples.append(f"label {label!r}")
                continue
            content_type = str(row.get(content_column))
            try:
                messages = prompts.messages(
                    {"company_name": row.get("company_name"),
                     "text": row.get("text"),
                     "headline": row.get("headline") or ""},
                    content_type,
                )
            except BlankInput as exc:
                skipped_invalid += 1
                invalid_examples.append(str(exc))
                continue
            record = {
                "request": {"messages": messages},
                "response": {"choices": [{"message": {
                    "role": "assistant", "content": target_content(prompts, row)}}]},
                "workload_id": prompts.theme.id,
                "client_id": client_id,
                "prompt_fingerprint": prompts.fingerprint(),
                "granularity": prompts.granularity,
                "row_key": row.get("row_key"),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            per_class[label] = per_class.get(label, 0) + 1

    return {"written": written, "skipped_unlabelled": skipped_unlabelled,
            "skipped_invalid": skipped_invalid, "per_class": per_class,
            "invalid_examples": invalid_examples[:5]}


def validate_jsonl(prompts: PromptSet, path: Path) -> list[str]:
    """Check a training file before spending GPU hours on it."""
    problems: list[str] = []
    fingerprints: set[str] = set()
    per_class: dict[str, int] = {}
    lines = 0
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            lines += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f"line {number}: not valid JSON")
                continue
            messages = record.get("request", {}).get("messages", [])
            if len(messages) < 2 or messages[0].get("role") != "system" \
                    or messages[1].get("role") != "user":
                problems.append(f"line {number}: expected a system message then a user message")
                continue
            if not str(messages[1].get("content", "")).strip():
                problems.append(f"line {number}: empty user content")
            try:
                target = json.loads(record["response"]["choices"][0]["message"]["content"])
            except Exception:
                problems.append(f"line {number}: assistant content is not parseable JSON")
                continue
            label = target.get(prompts.label_field)
            if label not in set(prompts.class_names):
                problems.append(f"line {number}: target label {label!r} is not in the taxonomy")
            else:
                per_class[label] = per_class.get(label, 0) + 1
            if record.get("workload_id") != prompts.theme.id:
                problems.append(f"line {number}: workload_id {record.get('workload_id')!r} != "
                                f"{prompts.theme.id!r}")
            if record.get("prompt_fingerprint"):
                fingerprints.add(record["prompt_fingerprint"])
    if lines == 0:
        problems.append("the file is empty")
    if len(fingerprints) > 1:
        problems.append(
            f"rows were produced under {len(fingerprints)} different prompts ({sorted(fingerprints)})"
            " -- the definitions changed mid-run, so some labels are stale"
        )
    elif fingerprints and prompts.fingerprint() not in fingerprints:
        problems.append(
            f"the file was built under prompt {fingerprints.pop()} but this run renders "
            f"{prompts.fingerprint()} -- re-export, or the student is trained on stale definitions"
        )
    missing = set(prompts.class_names) - set(per_class) - {prompts.abstain_class or ""}
    if missing:
        problems.append(f"no training examples for: {sorted(missing)}")
    return problems


def stub_matches_training(train_jsonl: Path, system_sent: str) -> None:
    """THE BYTE-FOR-BYTE CHECK, and it is not redundant with the fingerprint."""
    if not train_jsonl.exists():
        print(f"stub check: {train_jsonl.name} not found -- SKIPPED. Point --train-jsonl at the "
              f"file the student was trained on, or say --no-stub-check on purpose.")
        return
    with train_jsonl.open(encoding="utf-8") as fh:
        first = fh.readline()
    if not first.strip():
        print(f"stub check: {train_jsonl.name} is empty -- SKIPPED")
        return
    trained_on = json.loads(first)["request"]["messages"][0]["content"]
    if trained_on != system_sent:
        raise SystemExit(
            f"the system message this run would send is NOT the one the student was trained on "
            f"({train_jsonl}).\n"
            f"  trained on ({len(trained_on)} chars): {trained_on!r}\n"
            f"  would send ({len(system_sent)} chars): {system_sent!r}\n"
            f"The measured cost of that mismatch is 2.4 points. Fix the theme, or serve the "
            f"checkpoint that matches."
        )


def run(config: DistillConfig, theme: ThemeSpec, paths: RunPaths, *,
        granularity: str | None = None,
        force: bool = False) -> dict[str, Any]:
    granularity = granularity or config.granularities()[0]
    topology = config.experiment.topology
    prompts = build_prompts(config, theme, granularity, style="stub")

    train_path = frames.require_existing(paths.train_split(granularity), "the train split",
                                         "distill split")
    manifest = Manifest(stage="export", experiment=f"{granularity}.{topology}")
    manifest.add_input("train_split", train_path)
    train = pd.read_parquet(train_path)
    content_column = config.input.content_type_column


    export_columns = ["row_key", "text", "headline", "company_name", content_column,
                      prompts.label_field]
    if prompts.rationale_field:
        export_columns.append(prompts.rationale_field)
    work = train.copy()
    if "company_name" not in work.columns:
        work["company_name"] = work[config.input.entity_name_column]
    if "headline" not in work.columns:
        work["headline"] = work.get(config.input.headline_column, "")
    if prompts.rationale_field and prompts.rationale_field in work.columns:
        work[prompts.rationale_field] = work[prompts.rationale_field].fillna("")

    written: dict[str, dict[str, Any]] = {}
    for dataset in config.datasets_for_topology():
        types = config.content_type_for_dataset(dataset)
        subset = work.loc[work[content_column].isin(types)]
        out_path = paths.train_jsonl(granularity, topology, dataset)
        if subset.empty:
            print(f"[{dataset}] no rows -- skipping")
            continue
        if out_path.exists() and not force:
            existing = sum(1 for _ in out_path.open(encoding="utf-8"))
            if existing != len(subset):
                raise SystemExit(
                    f"[{dataset}] {out_path.name} holds {existing:,} rows but the split now has "
                    f"{len(subset):,}. It was written from different data and reusing it would "
                    f"train on the old rows under the new run's name. Pass --force to rebuild "
                    f"it, which is almost certainly what you want.")
            print(f"[{dataset}] already written: {out_path.name} ({existing:,} rows). "
                  f"Pass --force to rebuild.")
            written[dataset] = {"path": str(out_path), "written": existing, "reused": True}
            continue
        summary = write_jsonl(prompts, frames.rows_as_dicts(subset, export_columns), out_path,
                             content_column=content_column,
                             client_id=theme.id)
        problems = validate_jsonl(prompts, out_path)
        if problems:
            raise SystemExit(f"[{dataset}] the training file has problems:\n  - "
                             + "\n  - ".join(problems))
        if summary["skipped_invalid"]:
            raise SystemExit(
                f"[{dataset}] {summary['skipped_invalid']} rows were skipped as invalid "
                f"({summary['invalid_examples']}). Do not train on a silently shortened set -- "
                f"find out which rows and why."
            )
        share = {k: f"{v:,} ({v / max(summary['written'], 1):.1%})"
                 for k, v in sorted(summary["per_class"].items())}
        print(f"training set: {summary['written']:,} rows, {share}")
        written[dataset] = {"path": str(out_path), **summary}

    for dataset, summary in written.items():
        rows = summary["written"]
        estimate = serving.training_estimate(rows, config.training)

    manifest.config = {
        "experiment": {"granularity": granularity, "topology": topology},
        "training": config.training.__dict__,
        "datasets": list(written),
    }
    manifest.fingerprints = {"theme": theme.fingerprint(), "prompt": prompts.fingerprint(),
                             "stub_chars": len(prompts.stub())}
    manifest.counts = {dataset: {"written": s.get("written"), "per_class": s.get("per_class")}
                       for dataset, s in written.items()}
    manifest.dry_run = paths.dry_run
    for dataset, summary in written.items():
        manifest.add_output(f"train.{dataset}", Path(summary["path"]))
    manifest.note("adapters and merged models are box-only derived artifacts: the adapter is "
                  "the thing worth keeping, merged models can be rebuilt from it")
    manifest.write(paths.manifest_for(paths.training_dir(granularity, topology, create=True)))
    return {"datasets": written, "prompt_fingerprint": prompts.fingerprint()}
