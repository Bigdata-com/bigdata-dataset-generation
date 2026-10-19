"""Stage 13 -- the trained student labels the whole SUBSET."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .. import frames
from ..checkpoint import Checkpoint, label_rows
from ..client import LabelClient
from ..config import REFERENCE_COLUMN, DistillConfig
from ..granularity import build_strategy
from ..manifest import Manifest
from ..paths import RunPaths
from ..probe import probe_template
from ..serving import require_matching_granularity
from ..theme import ThemeSpec
from .export import stub_matches_training
from .evaluate import _fingerprint_from_training, _training_path
from .teacher import build_prompts, endpoint_from


def run(config: DistillConfig, theme: ThemeSpec, paths: RunPaths, *,
        granularity: str | None = None,
        content_type: str | None = None, dataset: str | None = None,
        dry_run: bool = False, assume_yes: bool = False, retry_errors: bool = False,
        no_stub_check: bool = False, allow_fingerprint_mismatch: bool = False,
        limit: int | None = None, assemble_only: bool = False) -> dict[str, Any]:
    granularity = granularity or config.granularities()[0]
    topology = config.experiment.topology
    prompts = build_prompts(config, theme, granularity, style="stub")
    content_column = config.input.content_type_column

    subset_path = frames.require_existing(paths.subset_parquet, "the subset", "distill subset")
    manifest = Manifest(stage="corpus", experiment=f"{granularity}.{topology}", dry_run=dry_run)
    manifest.add_input("subset", subset_path)
    subset = pd.read_parquet(subset_path)

    types = (content_type,) if content_type else tuple(config.subset.content_types)
    print(f"\ncorpus pass, {granularity}.{topology}")

    strategy_frame = subset.loc[subset[content_column].isin(types)]
    if limit is not None:
        taken = [group.head(limit) for _, group in strategy_frame.groupby(content_column)]
        strategy_frame = (pd.concat(taken).reset_index(drop=True) if taken
                          else strategy_frame.iloc[:0])
        print(f"--limit {limit} per content type -> {len(strategy_frame):,} rows")

    stats: dict[str, Any] = {}
    if not assemble_only:
        for one_type in types:
            group = strategy_frame.loc[strategy_frame[content_column] == one_type]
            if group.empty:
                print(f"[{one_type}] no rows -- skipping")
                continue
            settings = config.student_for(one_type)
            which_dataset = dataset or (one_type if topology == "specialist" else "both")
            training_path = _training_path(config, paths, granularity, topology,
                                           which_dataset, None)
            expected = _fingerprint_from_training(training_path)
            if expected and expected != prompts.fingerprint():
                message = (f"[{one_type}] prompt fingerprint mismatch: this pass would ask with "
                           f"{prompts.fingerprint()}, the student was trained under {expected}. "
                           f"Refusing to label a corpus through the wrong prompt -- the measured "
                           f"cost of a train/serve mismatch is 2.4 points.")
                if not allow_fingerprint_mismatch:
                    raise SystemExit(message)
                print("WARNING: " + message)
            if not no_stub_check:
                stub_matches_training(training_path, prompts.system_prompt(one_type))

            require_matching_granularity(settings, granularity)

            client: LabelClient | None = None
            if not dry_run:
                client = LabelClient(prompts, endpoint_from(settings, default_model="student"))
                client.wait_for_endpoint()
            strategy = build_strategy(config, prompts, client=client, dry_run=dry_run)
            strategy.assert_labelable(group)
            if client is not None:
                probe_template(client, strategy, group, one_type)

            if not assume_yes and not dry_run:
                frames.confirm(f">>> about to label {len(group):,} {one_type} rows. Press "
                               f"Enter, or Ctrl-C to stop. ")

            checkpoint = Checkpoint(paths.corpus_checkpoint(granularity, topology, one_type),
                                    retry_errors=retry_errors)
            rows = list(group.itertuples(index=False))
            _, run_stats = label_rows(
                rows,
                label_one=lambda row, ct=one_type: strategy.label(row, ct),
                key_of=lambda row: row.row_key,
                checkpoint=checkpoint,
                workers=settings.workers,
                desc=f"corpus[{one_type}]",
                label_field=prompts.label_field,
            )
            run_stats["model"] = settings.model
            run_stats["url"] = settings.url
            if client is not None and client.cost():
                run_stats["cost"] = client.cost()
            stats[one_type] = run_stats

    parts: list[pd.DataFrame] = []
    for one_type in tuple(config.subset.content_types):
        path = paths.corpus_checkpoint(granularity, topology, one_type)
        if not path.exists():
            print(f"[{one_type}] no checkpoint at {path.name} -- not part of this corpus yet")
            continue
        records = list(Checkpoint(path).load().values())
        if not records:
            continue
        labelled = pd.DataFrame.from_records(records)
        group = subset.loc[subset[content_column] == one_type]
        merged = frames.merge_labels(group, labelled, how="inner")
        parts.append(merged)

    if not parts:
        raise SystemExit("nothing to assemble: no checkpoint holds any record")
    corpus = pd.concat(parts, ignore_index=True)
    out_path = paths.corpus_parquet(granularity, topology)
    frames.write_parquet(corpus, out_path)

    summary = report(corpus, prompts, config, theme)

    manifest.config = {
        "experiment": {"granularity": granularity, "topology": topology},
        "student": {"url": config.student.url, "model": config.student.model,
                    "workers": config.student.workers,
                    "max_output_tokens": config.student.max_output_tokens},
        "student_overrides": config.student_overrides,
        "serve": (config.student.serve.__dict__ if config.student.serve else None),
        "limit": limit,
    }
    manifest.fingerprints = {"theme": theme.fingerprint(), "prompt": prompts.fingerprint()}
    manifest.model = {"role": "student", "per_content_type": stats}
    manifest.counts = {"rows": len(corpus), **summary}
    manifest.add_output("corpus", out_path)
    for one_type in tuple(config.subset.content_types):
        path = paths.corpus_checkpoint(granularity, topology, one_type)
        if path.exists():
            manifest.add_output(f"checkpoint.{one_type}", path, travels=False)
    manifest.note("the checkpoints are box-only and safe to delete once the parquet is verified "
                  "on both machines (row count, error count, per-content-type crosstab, row_key "
                  "uniqueness)")
    manifest.write(paths.manifest_for(out_path))
    return {"rows": len(corpus), "path": str(out_path), **summary}


def report(corpus: pd.DataFrame, prompts, config: DistillConfig,
           theme: ThemeSpec) -> dict[str, Any]:
    label_field = prompts.label_field
    content_column = config.input.content_type_column
    errors = int((~corpus["ok"]).sum())
    recovered = int(corpus.get("recovered", pd.Series(dtype=bool)).sum())
    print(f"\nrows {len(corpus):,}   unparseable {errors:,} "
          f"({100 * errors / max(len(corpus), 1):.4f}%)")
    if errors:
        print("  anything downstream should drop these: they carry no label, by design")
    ok = corpus.loc[corpus["ok"]]

    signal = set(theme.signal_classes)
    share = float(ok[label_field].isin(signal).mean()) if not ok.empty else 0.0

    out: dict[str, Any] = {
        "errors": errors, "recovered": recovered,
        "class_counts": frames.class_counts(ok, label_field),
        "signal_share": round(share, 4),
        "class_counts_by_content_type": {
            str(ct): frames.class_counts(ok.loc[ok[content_column] == ct], label_field)
            for ct in sorted(ok[content_column].unique())
        },
    }
    reference = REFERENCE_COLUMN if config.input.reference_label_column else None
    if reference and reference in ok.columns:
        agreement = float((ok[label_field] == ok[reference]).mean())
        print(f"\nagreement with the label set the corpus arrived carrying: "
              f"{100 * agreement:.2f}%")
        out["reference_agreement"] = round(agreement, 4)
    return out


def cross_tabulate(left: pd.DataFrame, right: pd.DataFrame, *, label_field: str,
                   names: tuple[str, str] = ("left", "right")) -> pd.DataFrame:
    """Two label sets, row by row, on `row_key`."""
    merged = left[["row_key", label_field]].merge(
        right[["row_key", label_field]], on="row_key", how="inner",
        suffixes=(f"_{names[0]}", f"_{names[1]}"), validate="one_to_one",
    )
    return pd.crosstab(merged[f"{label_field}_{names[0]}"], merged[f"{label_field}_{names[1]}"],
                       margins=True)
