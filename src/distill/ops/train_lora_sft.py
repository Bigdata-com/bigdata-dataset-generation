#!/usr/bin/env python3
"""LoRA SFT from a distillation JSONL."""

from __future__ import annotations

import os

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

GPTOSS_EXPERT_LAYERS = (
    "7.mlp.experts.gate_up_proj",
    "7.mlp.experts.down_proj",
    "15.mlp.experts.gate_up_proj",
    "15.mlp.experts.down_proj",
    "23.mlp.experts.gate_up_proj",
    "23.mlp.experts.down_proj",
)


def model_family(model: str) -> str:
    m = model.lower()
    if "gpt-oss" in m or "gpt_oss" in m:
        return "gpt-oss"
    if "nemotron-3" in m or "nemotron_3" in m or "nemotron3" in m:
        return "nemotron3"
    if "nemotron" in m or "mamba" in m:
        return "nemotron"
    return "default"


def chat_template_kwargs(family: str) -> dict | None:
    if family in ("nemotron", "nemotron3"):
        return {"enable_thinking": False}
    if family == "gpt-oss":
        return {"reasoning_effort": "low"}
    return None


def load_base_model(model: str, family: str, revision: str | None = None):
    if family == "gpt-oss":
        from transformers import Mxfp4Config

        return AutoModelForCausalLM.from_pretrained(
            model,
            revision=revision,
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            quantization_config=Mxfp4Config(dequantize=True),
            use_cache=False,
            device_map={"": 0},
        )
    if family == "nemotron3":
        return AutoModelForCausalLM.from_pretrained(
            model,
            revision=revision,
            torch_dtype=torch.bfloat16,
            use_cache=False,
            device_map={"": 0},
        )
    return AutoModelForCausalLM.from_pretrained(
        model,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )


def build_lora_config(model: str, family: str, r: int) -> LoraConfig:
    if family == "gpt-oss":
        return LoraConfig(
            r=r,
            lora_alpha=r * 2,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
            target_parameters=list(GPTOSS_EXPERT_LAYERS),
        )
    if family == "nemotron3":
        tp = [t for t in os.environ.get("N3_TARGET_PARAMETERS", "").split(",") if t.strip()]
        extra = {"target_parameters": tp, "lora_dropout": 0.0} if tp else {"lora_dropout": 0.05}
        return LoraConfig(
            r=r,
            lora_alpha=r * 2,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "in_proj"],
            **extra,
        )
    if family == "nemotron":
        return LoraConfig(
            r=r,
            lora_alpha=r * 2,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                "linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj",
            ],
        )
    return LoraConfig(
        r=r,
        lora_alpha=r * 2,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def extract_classification(record: dict) -> str:
    content = record["messages"][-1]["content"]
    return json.loads(content)["classification"].strip().lower()


def dataset_labels(records: list[dict]) -> tuple[str, ...]:
    """The classes the data actually contains, most common first."""
    counts: dict[str, int] = defaultdict(int)
    for rec in records:
        counts[extract_classification(rec)] += 1
    return tuple(sorted(counts, key=lambda k: (-counts[k], k)))


def parse_ratios(spec: str, labels: tuple[str, ...]) -> dict[str, float]:
    """`name=value` pairs, or bare values positionally against `labels`."""
    parts = [x.strip() for x in spec.split(",") if x.strip()]
    if not parts:
        raise ValueError(f"no values in {spec!r}")
    if all("=" in part for part in parts):
        out = {}
        for part in parts:
            name, _, value = part.partition("=")
            out[name.strip()] = float(value)
    else:
        if len(parts) != len(labels):
            raise ValueError(
                f"got {len(parts)} positional values for {len(labels)} classes {list(labels)}. "
                f"Use name=value pairs, e.g. '{labels[0]}=35,{labels[-1]}=50'."
            )
        out = {name: float(value) for name, value in zip(labels, parts)}
    unknown = sorted(set(out) - set(labels))
    if unknown:
        raise ValueError(
            f"{unknown} are not classes in this dataset (it has {list(labels)}). This is the "
            f"check that used to be missing: an unknown class name meant every row of the "
            f"REAL class was dropped without a word."
        )
    if sum(out.values()) <= 0:
        raise ValueError(f"ratios/weights must sum to > 0, got {spec!r}")
    return out


def balance_records(
    records: list[dict],
    ratios: dict[str, float],
    seed: int,
) -> list[dict]:
    """Resample to target class ratios."""
    rng = random.Random(seed)
    by_cls: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_cls[extract_classification(rec)].append(rec)

    named = [(cls, len(by_cls.get(cls, []))) for cls in ratios if ratios.get(cls, 0) > 0]
    named = [(cls, n) for cls, n in named if n > 0]
    if not named:
        return records
    anchor, anchor_n = min(named, key=lambda item: item[1])
    anchor_ratio = ratios[anchor]
    total = int(anchor_n / anchor_ratio)
    targets = {cls: max(1, int(total * ratios[cls])) for cls in ratios}

    balanced: list[dict] = []
    for cls, target in targets.items():
        pool = by_cls.get(cls, [])
        if not pool:
            continue
        if len(pool) >= target:
            balanced.extend(rng.sample(pool, target))
        else:
            balanced.extend(rng.choices(pool, k=target))

    rng.shuffle(balanced)
    counts = defaultdict(int)
    for rec in balanced:
        counts[extract_classification(rec)] += 1
    print(
        f"balanced sampling (anchor {anchor}={anchor_n:,}):",
        " ".join(f"{k}={v:,}" for k, v in sorted(counts.items())),
        f"total={len(balanced):,}",
    )
    return balanced


def apply_class_weights(
    records: list[dict],
    weights: dict[str, float],
    seed: int,
) -> list[dict]:
    """Duplicate or subsample records to approximate per-class weights."""
    rng = random.Random(seed)
    by_cls: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_cls[extract_classification(rec)].append(rec)

    weighted: list[dict] = []
    for cls, pool in by_cls.items():
        w = weights.get(cls, 1.0)
        if w <= 0:
            continue
        if w >= 1.0:
            weighted.extend(pool)
            extra = int((w - 1.0) * len(pool))
            if extra:
                weighted.extend(rng.choices(pool, k=extra))
        else:
            keep = max(1, int(len(pool) * w))
            weighted.extend(rng.sample(pool, min(keep, len(pool))))

    rng.shuffle(weighted)
    counts = defaultdict(int)
    for rec in weighted:
        counts[extract_classification(rec)] += 1
    print("class weights:", " ".join(f"{k}={v:,}" for k, v in sorted(counts.items())),
          f"total={len(weighted):,}")
    return weighted


def load_jsonl_messages(path: Path, seed: int, val_ratio: float) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    skipped = 0
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                skipped += 1
                print(f"WARNING: skip line {line_no}: {exc}")
                continue
            msgs = rec["request"]["messages"]
            assistant = rec["response"]["choices"][0]["message"]["content"]
            records.append({"messages": msgs + [{"role": "assistant", "content": assistant}]})
    if skipped:
        print(f"WARNING: skipped {skipped} malformed JSONL line(s)")

    rng = random.Random(seed)
    rng.shuffle(records)
    n_val = int(len(records) * val_ratio)
    return records[n_val:], records[:n_val]


def format_chat(example: dict, tokenizer, template_kwargs: dict | None) -> dict:
    kwargs = template_kwargs or {}
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
        chat_template_kwargs=kwargs or None,
    )
    return {"text": text}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="nvidia/NVIDIA-Nemotron-Nano-9B-v2")
    p.add_argument("--revision", default=None,
                   help="the Hugging Face commit of --model; default: the latest")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.0,
                   help="FIX 2: was 0.1, and the split it made never reached the trainer, so "
                        "it silently discarded 10%% of the data. 0.0 unless you say otherwise.")
    p.add_argument("--allow-discarded-val", action="store_true",
                   help="permit --val-ratio > 0, knowing the held-back rows are simply unused")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--grad-checkpointing", choices=("auto", "on", "off"), default="auto",
                   help="FIX 4: 'auto' checkpoints on the families that used to need it. "
                        "'off' measured 1.50x on a 4B that was using 15 GB of 141 -- "
                        "recomputing activations to save memory you are not short of is "
                        "calculation paid for nothing. TRAIN_GRAD_CKPT=0 does the same.")
    p.add_argument("--no-resume", action="store_true",
                   help="FIX 5: this script resumes from output-dir/checkpoint-* on its own, "
                        "so pointing it at a used directory silently continues an old run. "
                        "TRAIN_NO_RESUME=1 does the same.")
    p.add_argument("--balance-classes", action="store_true",
                   help="resample train set to --balance-ratios")
    p.add_argument("--balance-ratios", default="",
                   help="percent targets as name=value pairs, e.g. 'winner=35,loser=15,"
                        "irrelevant=50'. NOTE: capping the majority class was measured to "
                        "distort the prior and cost precision across every class. Unused by "
                        "`distill`, and left here only because removing a flag is not a fix.")
    p.add_argument("--class-weights", default="",
                   help="duplication weights as name=value pairs, e.g. 'winner=1.5,loser=0.5'")
    args = p.parse_args()

    family = model_family(args.model)
    template_kwargs = chat_template_kwargs(family)
    print(f"model={args.model} family={family} template_kwargs={template_kwargs}")

    if args.val_ratio > 0 and not args.allow_discarded_val:
        raise SystemExit(
            f"--val-ratio {args.val_ratio} would hold back "
            f"{args.val_ratio:.0%} of the rows and NEVER give them to the trainer -- the "
            f"validation split is computed and then unused. Pass --val-ratio 0.0 (the "
            f"default), or --allow-discarded-val if discarding them is what you mean."
        )
    train_rows, val_rows = load_jsonl_messages(args.dataset, args.seed, args.val_ratio)
    print(f"train={len(train_rows):,} val={len(val_rows):,} (val is NOT used for training "
          f"and is NOT evaluated)")

    labels = dataset_labels(train_rows)
    print(f"classes in the data: {list(labels)}")
    if args.balance_classes:
        if not args.balance_ratios:
            raise SystemExit("--balance-classes needs --balance-ratios, and there is no safe "
                             "default: the old 35/15/50 was a stronger prior distortion than "
                             "the one it was meant to fix.")
        ratios = parse_ratios(args.balance_ratios, labels)
        ratio_sum = sum(ratios.values())
        ratios = {k: v / ratio_sum for k, v in ratios.items()}
        train_rows = balance_records(train_rows, ratios, args.seed + 1)
    if args.class_weights:
        weights = parse_ratios(args.class_weights, labels)
        train_rows = apply_class_weights(train_rows, weights, args.seed + 2)
        print(f"train after weighting={len(train_rows):,}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_base_model(args.model, family, args.revision)
    if family == "gpt-oss" and args.grad_checkpointing != "off" \
            and os.environ.get("TRAIN_GRAD_CKPT") != "0":
        model.gradient_checkpointing_enable()

    lora = build_lora_config(args.model, family, args.lora_r)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = Dataset.from_list([format_chat(r, tokenizer, template_kwargs) for r in train_rows])

    world = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    target = 16 if family == "gpt-oss" else 64
    grad_accum = max(1, target // (args.batch_size * world))
    effective = args.batch_size * grad_accum * world
    if world > 1:
        print(f"distributed: {world} processes, so accumulation is {grad_accum} rather than "
              f"{max(1, target // args.batch_size)} -- effective batch stays {effective}")
    if effective != target:
        print(f"WARNING: effective batch is {effective}, not the {target} this recipe was "
              f"measured at. {args.batch_size} x {world} does not divide {target}; pick a "
              f"batch size that does, or the numbers are not comparable with the ones on "
              f"record.")
    if args.grad_checkpointing == "auto":
        grad_checkpointing = family in ("gpt-oss", "nemotron3")
        if os.environ.get("TRAIN_GRAD_CKPT") == "0":
            grad_checkpointing = False
            print("TRAIN_GRAD_CKPT=0: gradient checkpointing off")
    else:
        grad_checkpointing = args.grad_checkpointing == "on"
    print(f"gradient checkpointing: {'on' if grad_checkpointing else 'off'}"
          + ("" if grad_checkpointing else " -- read nvidia-smi at steps 18-20, not at 2-5"))

    sft_kwargs: dict = dict(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if family == "gpt-oss" else "epoch",
        save_steps=250 if family == "gpt-oss" else None,
        save_total_limit=3 if family == "gpt-oss" else None,
        report_to="none",
        max_length=args.max_seq_length,
        dataset_text_field="text",
        gradient_checkpointing=grad_checkpointing,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    if world > 1 or family == "gpt-oss":
        sft_kwargs["ddp_find_unused_parameters"] = False
    steps_per_epoch = max(1, -(-len(train_rows) // (args.batch_size * grad_accum * world)))
    print(f"effective batch {effective} "
          f"({args.batch_size} x {grad_accum} accumulation"
          + (f" x {world} processes" if world > 1 else "")
          + f"), {steps_per_epoch:,} steps/epoch, {steps_per_epoch * args.epochs:,} steps total")
    print("checkpoints this run will write: "
          + ", ".join(f"epoch {e} -> checkpoint-{steps_per_epoch * e:,}".replace(",", "")
                      for e in range(1, args.epochs + 1)))
    training_args = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    output_dir = args.output_dir.resolve()
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]),
    )
    resume: str | bool = False
    refuse = args.no_resume or os.environ.get("TRAIN_NO_RESUME") == "1"
    if checkpoints and refuse:
        raise SystemExit(
            f"--no-resume, and {args.output_dir} already holds "
            f"{[c.name for c in checkpoints]}. This script resumes from the newest checkpoint "
            f"on its own, so continuing here would extend an old run under a new name. Point "
            f"--output-dir at a fresh directory."
        )
    if checkpoints:
        resume = str(checkpoints[-1])
        print(f"resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))
    print(f"saved adapter → {args.output_dir / 'final'}")

if __name__ == "__main__":
    main()
