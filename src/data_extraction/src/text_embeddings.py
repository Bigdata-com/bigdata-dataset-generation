#!/usr/bin/env python3
"""Embed text: on this machine's cards, or with the OpenAI embeddings API.

`embed_model` in the run's config picks one. The two are called identically and return the
same thing -- (n, d) float32, L2-normalised, order preserved -- because the similarity step
is a bare `np.dot`, which is only cosine on normalised vectors, and every list downstream is
positional.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

DEFAULT_MODEL = "nemotron-1b"
OPENAI_MODEL = "text-embedding-3-small"

OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"
BATCH_SIZE = 1000

MAX_CHARS_PER_REQUEST = 500_000
MAX_CHARS_PER_INPUT = 28_000


def _make_batches(texts: list, max_inputs: int) -> list[list]:
    """Split into requests bounded by BOTH the input count and the total character budget."""
    batches: list[list] = []
    current: list = []
    current_chars = 0
    for text in texts:
        size = len(text) if isinstance(text, str) else 0
        if current and (len(current) >= max_inputs or current_chars + size > MAX_CHARS_PER_REQUEST):
            batches.append(current)
            current, current_chars = [], 0
        current.append(text)
        current_chars += size
    if current:
        batches.append(current)
    return batches


def _embed_batch(batch: list[str], model: str, api_key: str,
                 timeout: int, max_retries: int) -> list[list[float]]:
    """Embed one batch, retrying transient errors."""
    payload_input = [
        (t[:MAX_CHARS_PER_INPUT] if (isinstance(t, str) and t.strip()) else " ")
        for t in batch
    ]
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {"model": model, "input": payload_input}
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(OPENAI_EMBED_URL, headers=headers, json=payload,
                                     timeout=timeout)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            if response.status_code >= 400:
                lengths = sorted(len(t) for t in payload_input)
                raise requests.exceptions.HTTPError(
                    f"{response.status_code} from {OPENAI_EMBED_URL}: {response.text[:600]} "
                    f"[batch of {len(payload_input)} inputs, longest {lengths[-1]} chars, "
                    f"{sum(lengths)} chars total]", response=response)
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda d: d["index"])
            return [d["embedding"] for d in data]
        except requests.exceptions.RequestException:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def embed_texts(
    texts,
    model: str = OPENAI_MODEL,
    batch_size: int = BATCH_SIZE,
    api_key: str | None = None,
    timeout: int = 60,
    max_retries: int = 5,
    normalize: bool = True,
    show_progress: bool = True,
    max_workers: int = 1,
) -> np.ndarray:
    """Embed a list of strings -> (n, d) float32, L2-normalised by default, order preserved."""
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set (environment or .env), and no api_key= passed.")

    texts = list(texts)
    batches = _make_batches(texts, batch_size)

    progress = None
    if show_progress:
        try:
            from tqdm.auto import tqdm
            progress = tqdm(total=len(texts), desc="embedding", unit="txt")
        except ImportError:
            progress = None

    results: list[list[list[float]]] = [None] * len(batches)  # type: ignore[list-item]
    if max_workers and max_workers > 1 and len(batches) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_embed_batch, b, model, api_key, timeout, max_retries): i
                for i, b in enumerate(batches)
            }
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()
                if progress is not None:
                    progress.update(len(batches[index]))
    else:
        for index, batch in enumerate(batches):
            results[index] = _embed_batch(batch, model, api_key, timeout, max_retries)
            if progress is not None:
                progress.update(len(batch))
    if progress is not None:
        progress.close()

    vectors = np.asarray([v for batch_result in results for v in batch_result], dtype=np.float32)
    if normalize and vectors.size:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
    return vectors


def resolve_model(cfg: dict) -> tuple[str, str]:
    """(provider, model name) from the run's config."""
    from encoders import MODELS

    name = str((cfg or {}).get("embed_model") or DEFAULT_MODEL).strip()
    if name in MODELS:
        return "local", name
    if name == "openai":
        return "openai", OPENAI_MODEL
    if name.startswith("text-embedding"):
        return "openai", name
    raise SystemExit(
        f"embed_model {name!r} is not a model this step can run. Use {DEFAULT_MODEL!r}."
    )


class LocalEmbedder:
    """The model on this machine's cards. Weights load here, not at import."""

    provider = "local"

    def __init__(self, name: str, models_dir: Path | None = None):
        from encoders import MODELS, Encoder, visible_devices

        # Weights live in `models/`, beside `config/` and `output/`, where the notebook's
        # download cell puts them. Without this they would be fetched a second time into
        # ~/.cache and nothing would say why the step stalled on a 2 GB download.
        if models_dir is not None:
            os.environ.setdefault("HF_HUB_CACHE", str(models_dir))
        self.name = name
        self.spec = MODELS[name]
        self.encoder = Encoder(self.spec)
        self.cards = self.encoder.devices
        self._all_cards = len(visible_devices())

    def describe(self) -> str:
        where = ", ".join(self.cards) if self.cards else "cpu"
        return (f"{self.name} ({self.spec.hf_id}), dim {self.encoder.dim}, "
                f"max_seq {self.encoder.max_seq_length}, on {where}")

    def warnings(self) -> list[str]:
        """Whether this run will actually use the machine it is on."""
        if not self._all_cards:
            return ["no CUDA device visible: this encodes on the CPU, which is hours rather "
                    "than minutes. Check `python3 -c \'import torch; "
                    "print(torch.cuda.is_available())\'`."]
        if self._all_cards > len(self.cards):
            return [f"only {len(self.cards)} of {self._all_cards} cards in use: one replica "
                    f"per card is what makes this pass take minutes, and the rest of the "
                    f"machine is sitting idle."]
        return []

    def encode(self, texts) -> np.ndarray:
        return self.encoder.encode(texts)

    def close(self) -> None:
        self.encoder.close()


class OpenAIEmbedder:
    """The OpenAI embeddings API, over plain `requests`."""

    provider = "openai"

    def __init__(self, name: str, max_workers: int = 4):
        self.name = name
        self.max_workers = max_workers

    def describe(self) -> str:
        return f"{self.name}, over the OpenAI API with {self.max_workers} worker(s)"

    def warnings(self) -> list[str]:
        return []

    def encode(self, texts) -> np.ndarray:
        return embed_texts(texts, model=self.name, max_workers=self.max_workers,
                           show_progress=False)

    def close(self) -> None:
        return None


def make_embedder(cfg: dict, models_dir: Path | None = None):
    """The embedder this run's config asks for, ready to encode."""
    provider, name = resolve_model(cfg)
    if provider == "openai":
        return OpenAIEmbedder(name, max_workers=int((cfg or {}).get("embed_max_workers", 4)))
    return LocalEmbedder(name, models_dir=models_dir)
