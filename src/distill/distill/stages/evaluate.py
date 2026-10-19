"""Stage 12 -- score a model on the held-out rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .. import frames, metrics as metrics_mod
from ..checkpoint import Checkpoint, label_rows
from ..client import LabelClient
from ..config import DistillConfig
from ..granularity import _stub_answer
from ..manifest import Manifest
from ..paths import RunPaths
from ..prompts import PromptSet
from ..theme import ThemeSpec
from .export import stub_matches_training
from .teacher import build_prompts, endpoint_from


def default_tag(config: DistillConfig, *, dataset: str, epoch: int | None,
                model_label: str | None) -> str:
    """The tag is DERIVED from the experiment's identity, never typed."""
    parts = [config.experiment.granularity, config.experiment.topology, dataset]
    if epoch is not None:
        parts.append(f"e{epoch}")
    if model_label:
        parts.append(model_label.replace("/", "_"))
    return ".".join(parts)


def run(config: DistillConfig, theme: ThemeSpec, paths: RunPaths, *,
        granularity: str | None = None,
        dataset: str = "both", epoch: int | None = None, role: str = "student",
        model_label: str | None = None, prompt_style: str | None = None,
        train_jsonl: str | None = None, no_stub_check: bool = False,
        expect_fingerprint: str | None = None, allow_fingerprint_mismatch: bool = False,
        dry_run: bool = False, limit: int | None = None,
        content_type: str | None = None) -> dict[str, Any]:
    granularity = granularity or config.granularities()[0]
    topology = config.experiment.topology
    settings = getattr(config, role, None)
    if settings is None:
        raise SystemExit(f"no `{role}` section in the config")

    hosted = settings.price_in_per_m is not None
    style = prompt_style or ("full" if role == "teacher" or hosted else "stub")
    reason = ("named on the command line" if prompt_style else
              "the teacher's own prompt" if role == "teacher" else
              "a hosted model gets the full prompt, not someone else's stub" if hosted else
              "the stub the student was trained on")
    prompts = build_prompts(config, theme, granularity, style=style)
    tag = default_tag(config, dataset=dataset, epoch=epoch,
                      model_label=model_label or (settings.model if style == "full" else None))

    heldout_path = frames.require_existing(paths.heldout_split(granularity), "the held-out split",
                                          "distill split")
    manifest = Manifest(stage="evaluate", experiment=f"{granularity}.{topology}", dry_run=dry_run)
    manifest.add_input("heldout", heldout_path)
    held = pd.read_parquet(heldout_path)
    content_column = config.input.content_type_column

    types = (content_type,) if content_type else config.content_type_for_dataset(dataset)
    held = held.loc[held[content_column].isin(types)]
    if limit is not None:
        held = held.head(limit)
    if held.empty:
        raise SystemExit(f"nothing to score in {heldout_path.name} for content types {list(types)}")

    print(f"\nevaluating {settings.model} on {len(held):,} held-out rows")
    # The gate asks whether the model is served the prompt it was trained on. Epoch 0 is the
    # base model, trained on nothing, so there is no such prompt and nothing to check.
    expected = None if epoch == 0 else (expect_fingerprint or _fingerprint_from_training(
        _training_path(config, paths, granularity, topology, dataset, train_jsonl)))
    if expected and expected != prompts.fingerprint():
        message = (f"prompt fingerprint mismatch: this run would score with "
                   f"{prompts.fingerprint()}, the training set was exported with {expected}. "
                   f"The student would be served a prompt it was not trained on -- the measured "
                   f"cost of that mismatch is 2.4 points.")
        if not allow_fingerprint_mismatch:
            raise SystemExit(message + "\nPass --allow-fingerprint-mismatch only if you know why "
                                       "they differ.")
        print("WARNING: " + message)

    if style == "stub" and not no_stub_check:
        stub_matches_training(
            _training_path(config, paths, granularity, topology, dataset, train_jsonl),
            prompts.system_prompt(types[0]),
        )


    client: LabelClient | None = None
    if not dry_run:
        client = LabelClient(prompts, endpoint_from(settings, default_model=role))
        client.wait_for_endpoint()
    checkpoint = Checkpoint(
        paths.evaluations_dir(granularity, topology, create=True) / f"{tag}.ckpt.jsonl"
    )
    rows = held.to_dict("records")

    def score_one(row: dict[str, Any]) -> dict[str, Any]:
        content = str(row[content_column])
        if dry_run:
            result = _stub_answer(prompts, str(row.get("text") or ""))
        else:
            assert client is not None
            result = client.label(_model_input(row, config), content)
        return {
            "row_key": row["row_key"],
            content_column: content,
            "bin": row.get("bin"),
            "gold": row[prompts.label_field],
            "pred": result.label,
            "pred_rationale": result.rationale,
            "recovered": bool(result.recovered),
            "ok": result.ok,
            "error": result.error,
            "error_kind": result.error_kind,
            "latency_s": result.extra.get("latency_s"),
        }

    records, run_stats = label_rows(
        rows, label_one=score_one, key_of=lambda row: row["row_key"], checkpoint=checkpoint,
        workers=settings.workers, desc=tag, label_field="pred",
    )
    predictions = pd.DataFrame.from_records(records)

    evaluation = build_evaluation(
        predictions, prompts, theme, tag=tag, model=settings.model or role,
        checkpoint=str(epoch) if epoch is not None else "",
        heldout=heldout_path.name, content_column=content_column,
    )
    overall = evaluation.reports.get("overall")
    if overall is not None:
        print(f"  macro-F1 {overall.macro_f1:.4f}   accuracy {overall.accuracy:.2%}"
              + (f"   unparseable {evaluation.unparseable}" if evaluation.unparseable else ""))

    cost = client.cost() if client is not None else None
    if cost:
        evaluation.extra["cost"] = cost
        print(f"cost of this run, from the API's own usage field: ${cost['usd']:.4f} "
              f"(${cost['usd_per_call'] or 0:.6f} per call)")
    evaluation.extra["throughput"] = run_stats

    json_path = paths.eval_metrics_json(granularity, topology, tag)
    text_path = paths.eval_metrics_txt(granularity, topology, tag)
    evaluation.write(json_path, text_path)
    predictions_path = paths.eval_predictions(granularity, topology, tag)
    frames.write_parquet(predictions, predictions_path)

    write_comparison(config, theme, paths, granularity)

    manifest.config = {
        "experiment": {"granularity": granularity, "topology": topology, "dataset": dataset},
        "evaluation": {"tag": tag, "prompt_style": style, "epoch": epoch, "role": role,
                       "limit": limit, "content_types": list(types)},
        role: {"url": settings.url, "model": settings.model, "workers": settings.workers,
               "guided_json": settings.guided_json,
               "max_output_tokens": settings.max_output_tokens},
        "serve": (settings.serve.__dict__ if settings.serve else None),
    }
    manifest.fingerprints = {"theme": theme.fingerprint(), "prompt": prompts.fingerprint(),
                             "expected": expected}
    manifest.model = {"role": role, "model": settings.model, "epoch": epoch, "cost": cost}
    manifest.counts = {name: block for name, block in
                       evaluation.to_dict()["reports"].items()}
    manifest.add_output("metrics", json_path).add_output("predictions", predictions_path)
    manifest.write(paths.manifest_for(json_path))
    return {"tag": tag, "macro_f1": evaluation.overall.macro_f1,
            "accuracy": evaluation.overall.accuracy, "rows": evaluation.rows,
            "unparseable": evaluation.unparseable, "metrics": str(text_path)}


def build_evaluation(predictions: pd.DataFrame, prompts: PromptSet, theme: ThemeSpec, *,
                     tag: str, model: str, checkpoint: str, heldout: str,
                     content_column: str) -> metrics_mod.Evaluation:
    classes = tuple(prompts.class_names)
    evaluation = metrics_mod.Evaluation(
        tag=tag, classes=classes, abstain=prompts.abstain_class, model=model,
        checkpoint=checkpoint, prompt_fingerprint=prompts.fingerprint(), heldout=heldout,
        rows=len(predictions), unparseable=int(predictions["pred"].isna().sum()),
    )
    pairs = list(zip(predictions["gold"], predictions["pred"]))
    evaluation.reports["overall"] = metrics_mod.score(
        pairs, classes, title="OVERALL", abstain=prompts.abstain_class
    )
    for value in sorted(predictions[content_column].dropna().unique()):
        group = predictions.loc[predictions[content_column] == value]
        evaluation.reports[str(value)] = metrics_mod.score(
            list(zip(group["gold"], group["pred"])), classes, title=str(value).upper(),
            abstain=prompts.abstain_class,
        )
    return evaluation


def write_comparison(config: DistillConfig, theme: ThemeSpec, paths: RunPaths,
                     granularity: str | None = None) -> Path | None:
    """One table across every checkpoint scored for this experiment, read from disk."""
    granularity = granularity or config.granularities()[0]
    topology = config.experiment.topology
    directory = paths.evaluations_dir(granularity, topology)
    if not directory.exists():
        return None
    evaluations: list[metrics_mod.Evaluation] = []
    for path in sorted(directory.glob("*.metrics.json")):
        data = metrics_mod.load_evaluation(path)
        evaluation = metrics_mod.Evaluation(
            tag=data["tag"], classes=tuple(data["classes"]), abstain=data.get("abstain"),
            model=data.get("model", ""), checkpoint=data.get("checkpoint", ""),
            prompt_fingerprint=data.get("prompt_fingerprint", ""),
            heldout=data.get("heldout", ""), rows=data.get("rows", 0),
            unparseable=data.get("unparseable", 0),
        )
        for name, block in data.get("reports", {}).items():
            report = metrics_mod.Report(title=block["title"], classes=tuple(data["classes"]),
                                        rows=block["rows"], unparseable=block["unparseable"],
                                        accuracy=block["accuracy"], macro_f1=block["macro_f1"],
                                        abstain_baseline_accuracy=block
                                        .get("abstain_baseline_accuracy", 0.0),
                                        abstain_baseline_macro_f1=block
                                        .get("abstain_baseline_macro_f1", 0.0),
                                        confusion=block.get("confusion", {}))
            report.per_class = {
                key: metrics_mod.ClassMetrics(
                    name=key, support=value["support"], predicted=value.get("predicted", 0),
                    precision=value["precision"], recall=value["recall"], f1=value["f1"])
                for key, value in block["per_class"].items()
            }
            evaluation.reports[name] = report
        evaluations.append(evaluation)
    if not evaluations:
        return None
    heldouts = {e.heldout for e in evaluations}
    if len(heldouts) > 1:
        print(f"WARNING: {directory} holds evaluations against different held-out files "
              f"({sorted(heldouts)}). They are not column-by-column comparable.")
    path = paths.eval_comparison(granularity, topology)
    path.write_text(metrics_mod.comparison_table(evaluations), encoding="utf-8")
    return path


def _model_input(row: dict[str, Any], config: DistillConfig) -> dict[str, Any]:
    """Exactly the fields the export used, so the messages are byte-identical."""
    return {
        "company_name": row.get("company_name") or row.get(config.input.entity_name_column),
        "text": row.get("text"),
        "headline": row.get("headline") or "",
    }


def _training_path(config: DistillConfig, paths: RunPaths, granularity: str, topology: str,
                   dataset: str, override: str | None) -> Path:
    if override:
        resolved = config.path(override)
        assert resolved is not None
        return resolved
    return paths.train_jsonl(granularity, topology, dataset)


def _fingerprint_from_training(path: Path) -> str | None:
    if not path.exists():
        return None
    import json
    with path.open(encoding="utf-8") as fh:
        line = fh.readline()
    if not line.strip():
        return None
    try:
        return json.loads(line).get("prompt_fingerprint")
    except json.JSONDecodeError:
        return None


def _bin_table(predictions: pd.DataFrame, prompts: PromptSet, content_column: str) -> str:
    """Macro-F1 by metric bin -- the reason the sample was stratified in the first place."""
    if "bin" not in predictions.columns or predictions["bin"].isna().all():
        return ""
    lines = ["--- macro-F1 by metric bin ---"]
    classes = tuple(prompts.class_names)
    for value in sorted(predictions[content_column].dropna().unique()):
        group = predictions.loc[predictions[content_column] == value]
        rows = []
        for bin_index, bucket in group.groupby("bin"):
            report = metrics_mod.score(list(zip(bucket["gold"], bucket["pred"])), classes)
            rows.append({"bin": bin_index, "rows": len(bucket),
                         "macro_f1": round(report.macro_f1, 3),
                         "accuracy": round(report.accuracy, 3)})
        lines.append(f"\n[{value}]")
        lines.append(pd.DataFrame(rows).to_string(index=False))
    return "\n".join(lines)
