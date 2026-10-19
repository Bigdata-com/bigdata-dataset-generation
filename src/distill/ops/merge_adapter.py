#!/usr/bin/env python3
"""Fold a LoRA adapter into the base weights, at tensor level."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

MIN_PLAUSIBLE_BYTES = 1 * 1024 ** 3


def find_snapshot(base: str, revision: str | None = None) -> Path:
    """Resolve a base model to a directory holding a config.json.

    With a revision, only that snapshot will do: the cache can hold several.
    """
    candidate = Path(base).expanduser()
    if (candidate / "config.json").exists():
        return candidate
    import os

    roots = [Path(p) for p in (os.environ.get("HF_HUB_CACHE"), os.environ.get("HF_HOME")) if p]
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    stem = "models--" + base.replace("/", "--")
    for root in roots:
        for probe in (root / stem / "snapshots", root / "hub" / stem / "snapshots", root):
            if not probe.exists():
                continue
            for directory in sorted(probe.iterdir()):
                if revision and directory.name != revision:
                    continue
                if (directory / "config.json").exists():
                    return directory
    raise SystemExit(
        f"cannot resolve base model {base!r}"
        + (f" at revision {revision}" if revision else "")
        + " to a directory with a config.json.\n"
        f"Looked under: {[str(r) for r in roots]}\n"
        f"Pass --base as a path, or set HF_HUB_CACHE (NOT HF_HOME -- the layout is flat)."
    )


def build_deltas(adapter_dir: Path) -> tuple[dict, float, dict]:
    from safetensors.torch import load_file

    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise SystemExit(f"{adapter_dir} has no adapter_config.json -- is this a checkpoint dir?")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    scale = config["lora_alpha"] / config["r"]

    weights_path = adapter_dir / "adapter_model.safetensors"
    if not weights_path.exists():
        raise SystemExit(f"{adapter_dir} has no adapter_model.safetensors")
    adapter = load_file(str(weights_path))

    deltas: dict = {}
    for key in adapter:
        if not key.endswith(".lora_A.weight"):
            continue
        stem = key[: -len(".lora_A.weight")]
        a = adapter[key].float()
        b = adapter[stem + ".lora_B.weight"].float()
        base_key = stem.replace("base_model.model.", "").replace(".base_layer", "") + ".weight"
        if base_key.startswith("model."):
            base_key = "backbone." + base_key[len("model."):]
        deltas[base_key] = (b @ a) * scale
    if not deltas:
        raise SystemExit(f"no LoRA pairs found in {weights_path}")
    return deltas, scale, config


def checkpoint_keys(snapshot: Path) -> tuple[set[str], list[str]]:
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        return set(index["weight_map"]), sorted(set(index["weight_map"].values()))
    from safetensors import safe_open

    with safe_open(str(snapshot / "model.safetensors"), framework="pt") as handle:
        return set(handle.keys()), ["model.safetensors"]


def merge(base: str, adapter_dir: Path, out_dir: Path, *, force: bool,
          revision: str | None = None) -> int:
    from safetensors.torch import load_file, save_file

    snapshot = find_snapshot(base, revision)
    deltas, _, _ = build_deltas(adapter_dir)
    known, shards = checkpoint_keys(snapshot)

    missing = sorted(k for k in deltas if k not in known)
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(deltas)} delta keys are not in the checkpoint, e.g. "
            f"{missing[:3]}.\nThe key-prefix remap is wrong for this model. Refusing to write "
            f"a merge that applies nothing."
        )

    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise SystemExit(f"{out_dir} is not empty. Pass --force to overwrite it.")
    out_dir.mkdir(parents=True, exist_ok=True)

    applied = 0
    for shard in shards:
        tensors = load_file(str(snapshot / shard))
        hits = 0
        for name in list(tensors):
            if name in deltas:
                tensors[name] = (tensors[name].float() + deltas[name]).to(tensors[name].dtype)
                hits += 1
        save_file(tensors, str(out_dir / shard), metadata={"format": "pt"})
        applied += hits
        del tensors
    if applied != len(deltas):
        raise SystemExit(
            f"applied {applied} deltas but built {len(deltas)}. The merge is incomplete; do "
            f"not serve {out_dir}."
        )

    for name in ("config.json", "generation_config.json", "model.safetensors.index.json",
                 "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "tokenizer.model", "chat_template.jinja", "preprocessor_config.json"):
        source = snapshot / name
        if source.exists():
            shutil.copy2(source, out_dir / name)

    size = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    print(f"merged -> {out_dir}  ({size / 1024 ** 3:.2f} GiB, {applied} deltas)")
    if size < MIN_PLAUSIBLE_BYTES:
        raise SystemExit(
            f"the merged directory is only {size / 1024 ** 2:.1f} MiB. A real 4B merge is "
            f"~7.5 G. Do not serve this."
        )
    return size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", required=True, help="base model: a path or an HF id")
    parser.add_argument("--adapter", required=True, type=Path,
                        help="the checkpoint directory, e.g. .../adapters/both/checkpoint-2049")
    parser.add_argument("--out", required=True, type=Path, help="where the merged model goes")
    parser.add_argument("--force", action="store_true", help="overwrite a non-empty --out")
    parser.add_argument("--revision", default=None,
                        help="the Hugging Face commit of --base; default: the first snapshot")
    args = parser.parse_args()
    merge(args.base, args.adapter, args.out, force=args.force, revision=args.revision)
    return 0

if __name__ == "__main__":
    sys.exit(main())
