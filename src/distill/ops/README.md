# `distill/ops` — the scripts training and serving depend on

`distill` has to own everything it needs, so the two scripts and one template that training
and serving depend on live here rather than wherever the training host happens to keep them.
That matters for three reasons:

* the training host's copy of `train_lora_sft.py` is a **host** file the training container
  executes. Editing it there changes what runs for every job on that host.
* the merge script actually used was `manual_merge_ours.py` — NVIDIA's `manual_merge.py`
  with three paths hand-edited at the top — and it exists in no repository.
* the chat template is not optional, and a run whose template is "wherever it was last
  copied to" cannot be reproduced.

| file | what | changed from the original |
|---|---|---|
| `train_lora_sft.py` | LoRA SFT, from the platform's training host | three fixes, listed below |
| `merge_adapter.py` | tensor-level LoRA merge, from `code/manual_merge.py` | paths are arguments, not module constants |
| `n34b_nothink_template.jinja` | the serving chat template, byte-identical | none |

## The three trainer fixes

1. **The class list comes from the data.** `labels = ("winner", "loser", "irrelevant")` was
   a literal, so `--balance-classes` on a theme whose classes are `gainer`/`loser` found
   `by_cls.get("winner")` empty and dropped **every** `gainer` row without a word. Ratios
   and weights are now `name=value`, and naming a class the dataset does not contain is an
   error.
2. **`--val-ratio` no longer lies.** It defaulted to 0.1, and the split it produced was
   computed and then never handed to the trainer — silently discarding 10% of the data.
   The default is 0.0 and a non-zero value is refused unless `--allow-discarded-val`.
3. **The epoch → checkpoint mapping is printed up front.** Checkpoints are
   `checkpoint-<step>`, not `checkpoint-<epoch>`: 43,671 rows at an effective batch of 64
   is 683 steps per epoch, so "epoch 3" is `checkpoint-2049`. `distill train` resolves it
   by listing the directory; the trainer states it so a hand-run merge cannot guess wrong.

`--balance-classes` and `--class-weights` are kept and unused. Capping the majority class
was measured to distort the prior and cost precision across every class, and the old
35/15/50 default was a stronger distortion than the problem it was meant to fix.

## Two rules about how these are run

**Detached, inside the container, and stopped inside the container.**

    docker exec -d tx-train bash -c "python3 .../train_lora_sft.py … >> <log> 2>&1"
    docker exec tx-train ps aux | grep -c train_lora_sft     # 6: wrapper + main + 4 loaders
    docker exec tx-train pkill -f train_lora_sft.py          # the only way to stop it

A foreground `docker exec` is a trap: killing the local client does not signal the process
inside the container. Three "restarts" once stacked four concurrent trainers on one card —
20 processes, 138 GB of 140 in use, 21 s/step, two of them writing into the same
`--output-dir` — and every timing taken after the first launch was measuring a shared GPU.
A step time that is a clean multiple of the expected one usually means N trainers.

**Training happens inside `tx-train`, never on the host.** The host venv's torch has no
fused Mamba kernels: 64.9 s/step and an OOM at step 21. `tx-train:kernels` cannot serve
vLLM either (`mamba-ssm` pins `tilelang==0.1.8` over the 0.1.9 vLLM needs) — train in that
image, serve from a clean `vllm/vllm-openai`.
