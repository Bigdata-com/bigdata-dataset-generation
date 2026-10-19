"""Stage 10 -- the teacher labels the sample, and the split is written once."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .. import frames
from ..checkpoint import Checkpoint, label_rows
from ..client import EndpointConfig, LabelClient
from ..config import REFERENCE_COLUMN, DistillConfig, EndpointSettings
from ..granularity import build_strategy
from ..manifest import Manifest
from ..paths import RunPaths
from ..prompts import CONTENT_TYPES, PromptSet
from ..theme import ThemeSpec


def endpoint_from(settings: EndpointSettings, *, default_model: str) -> EndpointConfig:
    import os
    api_key = settings.api_key
    if settings.api_key_env:
        api_key = os.environ.get(settings.api_key_env, "").strip()
        if not api_key:
            raise SystemExit(
                f"the config asks for the API key in ${settings.api_key_env} and it is empty. "
                f"Export it, or set api_key: EMPTY for a local vLLM."
            )
    return EndpointConfig(
        url=settings.url,
        model=settings.model or default_model,
        api_key=api_key or "EMPTY",
        workers=settings.workers,
        max_output_tokens=settings.max_output_tokens,
        timeout=settings.timeout,
        retries=settings.retries,
        temperature=settings.temperature,
        send_temperature=settings.send_temperature,
        guided_json=settings.guided_json,
        early_stop=settings.early_stop,
        extra_body=dict(settings.extra_body),
        health_attempts=settings.health_attempts,
        health_delay=settings.health_delay,
        price_in_per_m=settings.price_in_per_m,
        price_out_per_m=settings.price_out_per_m,
    )


def build_prompts(config: DistillConfig, theme: ThemeSpec, granularity: str | None = None,
                  *, style: str = "full") -> PromptSet:
    """The prompt set for one granularity."""
    if granularity is None:
        covered = config.granularities()
        if len(covered) != 1:
            raise SystemExit(
                f"this run covers {list(covered)}, so the granularity has to be named. This is "
                f"a bug in the caller, not in the config."
            )
        granularity = covered[0]
    return PromptSet(theme=theme, granularity=granularity, prompt_style=style)


def write_prompts(prompts: PromptSet, directory) -> list[str]:
    """Every prompt this run sends, on disk beside the labels it produced."""
    directory.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for content_type in CONTENT_TYPES:
        path = directory / f"system.{prompts.granularity}.{content_type}.txt"
        path.write_text(prompts.system_prompt(content_type), encoding="utf-8")
        written.append(str(path))
        path = directory / f"user_template.{prompts.granularity}.{content_type}.txt"
        path.write_text(prompts.user_template(content_type), encoding="utf-8")
        written.append(str(path))
    path = directory / f"stub.{prompts.granularity}.txt"
    path.write_text(prompts.stub(), encoding="utf-8")
    written.append(str(path))
    path = directory / "fingerprint.txt"
    path.write_text(prompts.fingerprint() + "\n", encoding="utf-8")
    written.append(str(path))
    return written


def run(config: DistillConfig, theme: ThemeSpec, paths: RunPaths, *,
        dry_run: bool = False, assume_yes: bool = False, retry_errors: bool = False,
        force: bool = False, limit: int | None = None,
        granularities: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Label the sample, once per granularity asked for.

    `granularities` narrows what the config covers, so the caller decides how much of the menu
    to run without editing the config to do it.
    """
    covered = granularities or config.granularities()
    if len(covered) > 1:
        out: dict[str, Any] = {}
        for one in covered:
            print("\n" + "#" * 100)
            print(f"# GRANULARITY {one.upper()}  ({covered.index(one) + 1} of {len(covered)})")
            print("#" * 100)
            out[one] = _run_one(config, theme, paths, one, dry_run=dry_run,
                               assume_yes=assume_yes, retry_errors=retry_errors, force=force,
                               limit=limit)
        return out
    return _run_one(config, theme, paths, covered[0], dry_run=dry_run, assume_yes=assume_yes,
                    retry_errors=retry_errors, force=force, limit=limit)


def _run_one(config: DistillConfig, theme: ThemeSpec, paths: RunPaths, granularity: str, *,
             dry_run: bool = False, assume_yes: bool = False, retry_errors: bool = False,
             force: bool = False, limit: int | None = None) -> dict[str, Any]:
    prompts = build_prompts(config, theme, granularity)
    manifest = Manifest(stage="teacher", experiment=granularity, dry_run=dry_run)

    if dry_run:
        print(prompts.describe())
        print()

    sample_path = frames.require_existing(paths.sample_parquet, "the sample", "distill sample")
    manifest.add_input("sample", sample_path)
    sample = pd.read_parquet(sample_path)
    if limit is not None:
        drawn = [group.sample(n=min(limit, len(group)), random_state=config.sample.seed)
                 for _, group in sample.groupby(config.input.content_type_column)]
        sample = pd.concat(drawn).reset_index(drop=True) if drawn else sample.iloc[:0]
        print(f"--limit {limit} per content type -> {len(sample):,} rows")

    labels_path = paths.teacher_labels(granularity)
    if labels_path.exists() and not force and not retry_errors:
        existing = pd.read_parquet(labels_path, columns=["row_key"])
        print(f"labels already written: {labels_path} ({len(existing):,} rows). "
              f"The checkpoint makes a rerun cheap; pass --force to reassemble.")

    client: LabelClient | None = None
    if not dry_run:
        client = LabelClient(prompts, endpoint_from(config.teacher, default_model="teacher"),
                            harvest=config.teacher.harvest)
        client.wait_for_endpoint()
    strategy = build_strategy(config, prompts, client=client, dry_run=dry_run)
    strategy.assert_labelable(sample)

    prompts_dir = paths.teacher_prompts_dir(granularity, create=True)
    write_prompts(prompts, prompts_dir)

    if not assume_yes and not dry_run:
        frames.confirm(f"\n>>> about to label {len(sample):,} rows. Press Enter, or Ctrl-C to "
                       f"stop here. ")

    content_column = config.input.content_type_column
    parts: list[pd.DataFrame] = []
    stats: dict[str, Any] = {}
    for content_type in config.subset.content_types:
        group = sample.loc[sample[content_column] == content_type]
        if group.empty:
            continue
        checkpoint = Checkpoint(paths.teacher_checkpoint(granularity, content_type),
                               retry_errors=retry_errors)
        rows = list(group.itertuples(index=False))
        records, run_stats = label_rows(
            rows,
            label_one=lambda row, ct=content_type: strategy.label(row, ct),
            key_of=lambda row: row.row_key,
            checkpoint=checkpoint,
            workers=config.teacher.workers,
            desc=f"label[{content_type}]",
            label_field=prompts.label_field,
        )
        stats[content_type] = run_stats
        labelled = pd.DataFrame.from_records(records)
        parts.append(frames.merge_labels(group, labelled))

    if not parts:
        raise SystemExit("nothing labelled")
    labels = pd.concat(parts, ignore_index=True)
    frames.write_parquet(labels, labels_path)

    summary = report(labels, prompts, config, theme)
    cost = client.cost() if client is not None else None
    if cost:
        print(f"\ncost of this pass, from the API's own usage field: ${cost['usd']:.4f} "
              f"(${cost['usd_per_call'] or 0:.6f} per call)")

    manifest.config = {
        "experiment": {"granularity": granularity},
        "teacher": {"url": config.teacher.url, "model": config.teacher.model,
                    "workers": config.teacher.workers,
                    "max_output_tokens": config.teacher.max_output_tokens,
                    "extra_body": config.teacher.extra_body,
                    "guided_json": config.teacher.guided_json},
        "serve": (config.teacher.serve.__dict__ if config.teacher.serve else None),
        "sentence": (config.sentence.__dict__ if granularity == "sentence" else None),
        "limit": limit,
    }
    manifest.fingerprints = {"theme": theme.fingerprint(), "theme_id": theme.id,
                             "prompt": prompts.fingerprint()}
    manifest.model = {"role": "teacher", "model": config.teacher.model, "url": config.teacher.url,
                      "cost": cost}
    manifest.counts = {"rows": len(labels), **summary, "per_content_type_run": stats}
    manifest.add_output("labels", labels_path)
    for content_type in config.subset.content_types:
        checkpoint_path = paths.teacher_checkpoint(granularity, content_type)
        if checkpoint_path.exists():
            manifest.add_output(f"checkpoint.{content_type}", checkpoint_path, travels=False)
    manifest.note("prompts/ carries the exact system messages this pass sent")
    manifest.write(paths.manifest_for(labels_path))
    return {"rows": len(labels), "path": str(labels_path), **summary}


def report(labels: pd.DataFrame, prompts: PromptSet, config: DistillConfig,
           theme: ThemeSpec) -> dict[str, Any]:
    label_field = prompts.label_field
    content_column = config.input.content_type_column
    ok = labels.loc[labels["ok"]]
    errors = int((~labels["ok"]).sum())
    print(f"\nlabelled: {len(ok):,}   errors: {errors:,}   "
          f"recovered by the regex fallback: {int(labels.get('recovered', pd.Series(dtype=bool)).sum()):,}")
    if errors:
        print(labels.loc[~labels["ok"], "error_kind"].value_counts().to_string())
        print("  'budget' means the reasoning trace ran past max_output_tokens -- raise it")

    out: dict[str, Any] = {"errors": errors, "class_counts": {}}
    if ok.empty:
        return out

    print(f"\n{label_field}, % of the {len(ok):,} rows:")
    print(ok[label_field].value_counts(normalize=True).mul(100).round(2)
          .rename("pct").to_frame().to_string())
    out["class_counts"] = frames.class_counts(ok, label_field)
    out["class_counts_by_content_type"] = {
        str(ct): frames.class_counts(ok.loc[ok[content_column] == ct], label_field)
        for ct in sorted(ok[content_column].unique())
    }

    reference = REFERENCE_COLUMN if config.input.reference_label_column else None
    if reference and reference in ok.columns:
        agreement = float((ok[label_field] == ok[reference]).mean())
        print(f"\nagreement with the label set the corpus arrived carrying ({reference}): "
              f"{100 * agreement:.2f}%")
        print(pd.crosstab(ok[reference], ok[label_field], margins=True).to_string())
        signal = set(theme.signal_classes)
        both = ok.loc[ok[label_field].isin(signal) & ok[reference].isin(signal)]
        if not both.empty:
            flips = int((both[label_field] != both[reference]).sum())
            print(f"  polarity flips (both rulers say signal, opposite direction): {flips:,} of "
                  f"{len(both):,} = {100 * flips / len(both):.2f}%")
            out["polarity_flips"] = {"flips": flips, "of": len(both)}
        out["reference_agreement"] = round(agreement, 4)

    return out


def heldout_fraction(config: DistillConfig, rows: int) -> float:
    """The fraction to hold out, lowered if `split.max_heldout` would be exceeded.

    Lowering the FRACTION rather than trimming the result afterwards is what keeps the
    stratification intact: every (content_type, label) cell shrinks by the same factor, so the
    thin cells stay proportional instead of one of them being emptied by a truncation.

    Every checkpoint of every student is scored on this set, so its size is a cost paid once
    per evaluation and many times per run.
    """
    wanted = config.split.heldout_fraction
    ceiling = config.split.max_heldout
    if not ceiling or not rows:
        return wanted
    if rows * wanted <= ceiling:
        return wanted
    lowered = ceiling / rows
    print(f"split.max_heldout: {wanted:.0%} of {rows:,} would hold out "
          f"{int(rows * wanted):,}, over the {ceiling:,} ceiling -- holding out "
          f"{lowered:.2%} instead, every stratum scaled by the same factor")
    return lowered


def run_split(config: DistillConfig, theme: ThemeSpec, paths: RunPaths, *,
              new_split: bool = False,
              granularities: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Write the split, once per granularity asked for."""
    covered = granularities or config.granularities()
    if len(covered) == 1:
        return _split_one(config, theme, paths, covered[0], new_split=new_split,
                          heldout_keys_from=(str(paths.heldout_keys_parquet)
                                             if paths.heldout_keys_parquet.exists() else None))
    out: dict[str, Any] = {}
    for index, one in enumerate(covered):
        print("\n" + "#" * 72)
        print(f"# SPLIT, GRANULARITY {one.upper()}  ({index + 1} of {len(covered)})")
        print("#" * 72)
        reuse = None
        if paths.heldout_keys_parquet.exists():
            reuse = str(paths.heldout_keys_parquet)
            print("reusing the held-out keys already on disk: same rows, two rulers, which is "
                  "what makes the granularities comparable")
        out[one] = _split_one(config, theme, paths, one, heldout_keys_from=reuse,
                              new_split=new_split)
    return out


def _split_one(config: DistillConfig, theme: ThemeSpec, paths: RunPaths, granularity: str, *,
               heldout_keys_from: str | None = None,
               new_split: bool = False) -> dict[str, Any]:
    """Hold out a fraction of the labelled sample -- once, and never again."""
    label_field = theme.label_field
    train_path = paths.train_split(granularity)
    heldout_path = paths.heldout_split(granularity)

    if train_path.exists() and heldout_path.exists():
        if new_split:
            raise SystemExit(
                f"--new-split refused: {train_path.name} and {heldout_path.name} already exist. "
                f"Every evaluation measured so far scores those held-out rows, and a new split "
                f"invalidates all of them. Delete them by hand if that is really what you want."
            )
        train = pd.read_parquet(train_path)
        held = pd.read_parquet(heldout_path)
        print(f"reusing the existing split: {len(train):,} train / {len(held):,} held out")
        return {"train": len(train), "heldout": len(held), "reused": True}

    labels_path = frames.require_existing(paths.teacher_labels(granularity), "the teacher labels",
                                         "distill teacher")
    manifest = Manifest(stage="split", experiment=granularity).add_input("labels", labels_path)
    labels = pd.read_parquet(labels_path)

    bad = int((~labels["ok"]).sum())
    usable = labels.loc[labels["ok"] & labels[label_field].notna()].copy()
    if bad:
        print(f"{bad:,} of {len(labels):,} rows dropped: the teacher failed on them, and a "
              f"student must never be trained on 'nothing parseable' as if it were a label.")

    content_column = config.input.content_type_column
    keys_from = heldout_keys_from or config.split.heldout_keys_from
    if keys_from:
        source = config.path(keys_from)
        if source is None or not source.exists():
            raise SystemExit(f"held-out keys not found: {source}")
        wanted = set(pd.read_parquet(source, columns=["row_key"])["row_key"])
        held = usable.loc[usable["row_key"].isin(wanted)]
        train = usable.drop(index=held.index)
        print(f"held-out taken from {source.name}: {len(wanted):,} keys there, {len(held):,} of "
              f"them are in this sample ({100 * len(held) / max(len(wanted), 1):.1f}%)")
        expected = heldout_fraction(config, len(usable))
        if len(held) < 0.5 * expected * len(usable):
            print(f"  WARNING: far below the {expected:.2%} a fresh split "
                  f"would hold ({int(expected * len(usable)):,} rows). The "
                  f"per-class cells may be too thin to measure. Check that both runs used the "
                  f"same seed and the same bins.")
    else:
        rng = np.random.default_rng(config.split.seed)
        keys = [content_column if s == "content_type" else (label_field if s == "label" else s)
                for s in config.split.stratify_on]
        missing = [k for k in keys if k not in usable.columns]
        if missing:
            raise SystemExit(f"split.stratify_on refers to columns the labels do not have: {missing}")
        fraction = heldout_fraction(config, len(usable))
        indices: list[Any] = []
        cells = empty = empty_rows = 0
        for _, group in usable.groupby(keys):
            cells += 1
            take = int(round(len(group) * fraction))
            if take:
                indices.extend(rng.choice(group.index.to_numpy(), size=take, replace=False).tolist())
            else:
                empty += 1
                empty_rows += len(group)
        held = usable.loc[indices]
        train = usable.drop(index=held.index)
        print(f"new split, stratified on {keys}, seed {config.split.seed}, "
              f"{fraction:.2%} held out -> {len(held):,} rows across {cells:,} strata")
        if empty:
            print(f"  {empty:,} of {cells:,} strata were too thin to hold anything out at "
                  f"{fraction:.2%} ({empty_rows:,} rows, all in train). Drop `bin` from "
                  f"split.stratify_on for a coarser table, or raise split.max_heldout.")

    frames.write_parquet(train, train_path)
    frames.write_parquet(held, heldout_path)
    if not paths.heldout_keys_parquet.exists():
        keys_frame = held[["row_key", content_column, label_field]].copy()
        frames.write_parquet(keys_frame, paths.heldout_keys_parquet)

    manifest.config = {"split": {"heldout_fraction": config.split.heldout_fraction,
                                 "max_heldout": config.split.max_heldout,
                                 "seed": config.split.seed,
                                 "stratify_on": list(config.split.stratify_on),
                                 "heldout_keys_from": keys_from},
                       "granularity": granularity}
    manifest.fingerprints = {"theme": theme.fingerprint()}
    manifest.counts = {
        "train": len(train), "heldout": len(held), "dropped_teacher_failures": bad,
        "heldout_class_counts": frames.class_counts(held, label_field),
        "heldout_by_content_type": {
            str(ct): frames.class_counts(held.loc[held[content_column] == ct], label_field)
            for ct in sorted(held[content_column].unique())
        },
    }
    manifest.dry_run = paths.dry_run
    manifest.add_output("train", train_path).add_output("heldout", heldout_path)
    manifest.add_output("heldout_keys", paths.heldout_keys_parquet)
    manifest.note("the held-out key list is portable: point another granularity's "
                  "split.heldout_keys_from at it to score the same rows")
    manifest.write(paths.manifest_for(heldout_path))
    return {"train": len(train), "heldout": len(held)}
