#!/usr/bin/env python3
"""The embedding model, loaded on every visible card and called in length-sorted batches.

The contract is the one the rest of the step relies on, and nothing here may weaken it:

    encode(texts) -> (n, d) float32, L2-normalised, ORDER PRESERVED

Embedding a corpus is throughput work and no model here is too big for one card, so the
cards are used the way the labelling stage uses them: **one independent replica per card,
each doing a share of the work** -- data parallel. Nothing is split across cards and nothing
crosses the bus.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    revision: str | None = None           # the Hugging Face commit; None takes the latest
    dtype: str = "bfloat16"
    max_seq_length: int | None = None     # None leaves the model's own default in place
    trust_remote_code: bool = False
    batch_tokens: int = 32_768
    max_batch_sentences: int = 256


# BF16, never the NVFP4 variants: those need Blackwell, and this runs on Hopper.
MODELS: dict[str, ModelSpec] = {
    "nemotron-1b": ModelSpec(
        key="nemotron-1b",
        hf_id="nvidia/Nemotron-3-Embed-1B-BF16",
        revision="c0c9fea93ea424587517f2c59e20db9f1d6bf615",
        trust_remote_code=True,           # model_type ministral3 / Ministral3Model
        batch_tokens=32_768,
        max_batch_sentences=256,
    ),
    # Four times the weights for a ranking that is not four times better, but it is the one
    # to reach for if 2048 dimensions ever turn out to be the limit.
    "nemotron-8b": ModelSpec(
        key="nemotron-8b",
        hf_id="nvidia/llama-embed-nemotron-8b",
        trust_remote_code=True,           # model_type llama_bidirec / LlamaBidirectionalModel
        batch_tokens=16_384,
        max_batch_sentences=128,
    ),
}


def clean(texts) -> list[str]:
    """Blank and whitespace-only inputs become a single space.

    It matters more than it looks: dropping a blank would shorten the vector list, and every
    downstream list in this step is positional, so the similarities would silently attach to
    the wrong sentences.
    """
    return [t if (isinstance(t, str) and t.strip()) else " " for t in texts]


def token_budget_batches(texts: list[str], batch_tokens: int,
                         max_sentences: int) -> list[list[int]]:
    """Group indices into batches bounded by the PADDED token cost, longest-first.

    A fixed sentence count is the wrong unit here. Sentences run from ~143 characters at the
    median to ~8,500 at the tail, so a batch sized on the median is far heavier when it
    happens to catch the tail. Sorting by length also keeps padding near zero.
    """
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for index in order:
        # ~4 characters per token is close enough to size a batch.
        approx = max(1, len(texts[index]) // 4)
        new_max = max(current_max, approx)
        if current and (len(current) + 1 > max_sentences
                        or new_max * (len(current) + 1) > batch_tokens):
            batches.append(current)
            current, current_max = [index], approx
        else:
            current.append(index)
            current_max = new_max
    if current:
        batches.append(current)
    return batches


def visible_devices() -> list[str]:
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]


class Encoder:
    """One replica of `spec` per card, called in parallel over a shared batch list."""

    def __init__(self, spec: ModelSpec, devices: list[str] | None = None):
        import torch
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        self.torch = torch
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[spec.dtype]

        # Every visible card, unless the caller named some. A quiet fallback to one card
        # would halve the throughput and say nothing.
        targets = list(devices) if devices else (visible_devices() or [None])

        self.models = []
        for target in targets:
            kwargs = {"trust_remote_code": spec.trust_remote_code, "revision": spec.revision}
            if target:
                kwargs["device"] = target
            # transformers 5 renamed `torch_dtype` to `dtype` and rejects the old name; 4.x
            # only knows the old one. Try the new spelling first and fall back, so the same
            # file runs against either. The dtype is not optional to get right -- silently
            # loading a BF16 model in float32 would double the memory and the time.
            try:
                model = SentenceTransformer(spec.hf_id, model_kwargs={"dtype": dtype}, **kwargs)
            except (TypeError, ValueError) as exc:
                if "dtype" not in str(exc):
                    raise
                model = SentenceTransformer(spec.hf_id, model_kwargs={"torch_dtype": dtype},
                                            **kwargs)
            if spec.max_seq_length:
                model.max_seq_length = spec.max_seq_length
            model.eval()
            self.models.append(model)
        self.model = self.models[0]
        self.devices = [str(m.device) for m in self.models]

        # Seed statements and extracted sentences are the same kind of text, so they carry
        # the SAME marker -- the model's own query prefix, on both sides of the dot product.
        # A repo that applied a prompt of its own on top would put the two sides into
        # different sub-spaces, so refuse rather than encode something nobody chose.
        default_prompt = getattr(self.model, "default_prompt_name", None)
        if default_prompt:
            raise SystemExit(
                f"{spec.hf_id} applies the prompt {default_prompt!r} by default "
                f"({getattr(self.model, 'prompts', {}).get(default_prompt)!r}), on top of the "
                f"prefix this step applies itself. Handle it before encoding anything."
            )
        self.prefix = (getattr(self.model, "prompts", {}) or {}).get("query", "")

    @property
    def dim(self) -> int:
        # sentence-transformers 6 renamed this; 3.x/4.x only has the old name.
        getter = (getattr(self.model, "get_embedding_dimension", None)
                  or self.model.get_sentence_embedding_dimension)
        return int(getter())

    @property
    def max_seq_length(self) -> int:
        return int(self.model.max_seq_length)

    def encode(self, texts, show_progress: bool = False) -> np.ndarray:
        """(n, d) float32, L2-normalised, in the order given."""
        texts = [self.prefix + t for t in clean(list(texts))]
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        if not texts:
            return out
        batches = token_budget_batches(texts, self.spec.batch_tokens,
                                       self.spec.max_batch_sentences)
        bar = None
        if show_progress:
            try:
                from tqdm.auto import tqdm
                bar = tqdm(total=len(texts), unit="sent", unit_scale=True, desc="encode")
            except ImportError:
                bar = None
        lock = threading.Lock()

        def work(model, own: list[list[int]]) -> None:
            with self.torch.inference_mode():
                for indices in own:
                    block = model.encode([texts[i] for i in indices],
                                         batch_size=len(indices),
                                         normalize_embeddings=True,
                                         convert_to_numpy=True,
                                         show_progress_bar=False)
                    out[np.asarray(indices)] = np.asarray(block, dtype=np.float32)
                    if bar is not None:
                        with lock:
                            bar.update(len(indices))

        if len(self.models) == 1:
            work(self.model, batches)
        else:
            # Round-robin, not a split down the middle: the batches are ordered longest
            # first, so halving the list would hand one card every heavy batch.
            shares = [batches[i::len(self.models)] for i in range(len(self.models))]
            with ThreadPoolExecutor(max_workers=len(self.models)) as pool:
                for future in [pool.submit(work, model, share)
                               for model, share in zip(self.models, shares)]:
                    future.result()

        if bar is not None:
            bar.close()
        if self.torch.cuda.is_available():
            for index in range(self.torch.cuda.device_count()):
                self.torch.cuda.synchronize(index)
        return out

    def close(self, verbose: bool = True) -> None:
        """Give the cards back, the way `distill` takes a server down when its stage is over.

        Called from a `finally`: the labelling stage wants both cards whole, and a 2 GB
        replica per card left behind by a step that finished half an hour ago is the kind of
        thing nobody looks for. Never raises -- a teardown that fails must not mask the error
        that sent us here.
        """
        import gc

        self.models = []
        self.model = None
        gc.collect()
        if not self.torch.cuda.is_available():
            return
        try:
            self.torch.cuda.empty_cache()
            self.torch.cuda.ipc_collect()
        except Exception:                                      # noqa: BLE001
            return
        if verbose:
            cards = []
            for index in range(self.torch.cuda.device_count()):
                free, total = self.torch.cuda.mem_get_info(index)
                cards.append(f"GPU{index} {(total - free) // 2**20} MiB")
            print("encoder released: " + ", ".join(cards))
