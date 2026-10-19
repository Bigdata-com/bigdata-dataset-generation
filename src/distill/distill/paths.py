"""The run folder — every artifact path derived from the config, never from a `--tag`."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

STEP_NAMES: dict[str, str] = {
    "subset": "subset",
    "sample": "sample",
    "teacher": "teacher_labels",
    "training": "training",
    "evaluations": "evaluations",
    "corpus": "corpus_labels",
    "ic": "ic",
    "backtest": "backtest",
}

CONFIG_DIR_NAME = "config"
EXTRACTION_CONFIG_NAME = "data_extraction.yaml"
DISTILL_CONFIG_NAME = "distill.yaml"
CONFIG_NAMES = (EXTRACTION_CONFIG_NAME, DISTILL_CONFIG_NAME)
PART_A_SIMILARITY = "similarity/combinations_with_similarity.parquet"
DRY_RUN_DIR = "_dry_run"


CONFIG_FILENAMES = (EXTRACTION_CONFIG_NAME, DISTILL_CONFIG_NAME)


def config_dir() -> Path:
    """The repository's `config/` — the one place these three files exist.

    They are checked in, not generated: you edit them and every stage reads them from here.
    `THEME_PIPELINE_CONFIG` overrides the location for an installed package that sits
    outside the repository.
    """
    override = os.environ.get("THEME_PIPELINE_CONFIG")
    if override:
        found = Path(override).expanduser().resolve()
        if not all((found / name).is_file() for name in CONFIG_FILENAMES):
            raise SystemExit(f"THEME_PIPELINE_CONFIG={found} does not hold "
                             f"{', '.join(CONFIG_FILENAMES)}")
        return found
    for parent in Path(__file__).resolve().parents:
        found = parent / "config"
        if all((found / name).is_file() for name in CONFIG_FILENAMES):
            return found
    raise SystemExit(
        "the repository's `config/` was not found. It holds the three files this pipeline "
        f"reads ({', '.join(CONFIG_FILENAMES)}) and they are checked in, not generated. "
        "Point THEME_PIPELINE_CONFIG at it if you installed the package outside the repo.")


OUTPUT_DIR_NAME = "output"


def default_output_dir() -> Path:
    """`output/` beside `config/` — where every stage writes. One workflow, one folder.

    `THEME_PIPELINE_OUTPUT` overrides it; `--run` still takes precedence over both.
    """
    override = os.environ.get("THEME_PIPELINE_OUTPUT")
    root = Path(override).expanduser() if override else config_dir().parent / OUTPUT_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()

ARTIFACT_SUFFIXES = frozenset({".parquet", ".json", ".jsonl", ".csv", ".txt", ".png", ".ipynb",
                               ".log", ".yaml", ".yml"})

def experiment_key(granularity: str, topology: str) -> str:
    """The identity of one experiment, and the only thing artifact paths fan out on."""
    return f"{granularity}.{topology}"


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    dry_run: bool = False

    @classmethod
    def for_run(cls, run_dir: str | Path, *, dry_run: bool = False) -> RunPaths:
        return cls(Path(run_dir).resolve(), dry_run=dry_run)

    @property
    def theme_run_name(self) -> str:
        return self.run_dir.name

    @property
    def root(self) -> Path:
        """Where Part B writes."""
        return self.run_dir / DRY_RUN_DIR if self.dry_run else self.run_dir

    def step_dir(self, key: str, *parts: str, create: bool = False) -> Path:
        path = self.root / STEP_NAMES[key]
        for part in parts:
            path = path / part
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def config_dir(self) -> Path:
        """`config/` BESIDE the output folder — one place, edited by hand, never copied.

        Strictly the sibling, with no fall-back to a repository-wide location: a run pointed
        somewhere else must not silently read, or write, the checked-in config.
        """
        return self.run_dir.parent / CONFIG_DIR_NAME

    @property
    def extraction_config(self) -> Path:
        """The theme, and Part A's settings."""
        return self.config_dir / EXTRACTION_CONFIG_NAME

    @property
    def distill_config(self) -> Path:
        return self.config_dir / DISTILL_CONFIG_NAME

    @property
    def part_a_similarity(self) -> Path:
        return self.run_dir / PART_A_SIMILARITY

    @property
    def subset_parquet(self) -> Path:
        return self.step_dir("subset") / "subset.parquet"

    @property
    def sample_parquet(self) -> Path:
        return self.step_dir("sample") / "sample.parquet"

    @property
    def heldout_keys_parquet(self) -> Path:
        return self.step_dir("sample") / "heldout_keys.parquet"

    def teacher_dir(self, granularity: str, *, create: bool = False) -> Path:
        return self.step_dir("teacher", granularity, create=create)

    def teacher_labels(self, granularity: str) -> Path:
        return self.teacher_dir(granularity) / "labels.parquet"

    def teacher_checkpoint(self, granularity: str, content_type: str) -> Path:
        return self.teacher_dir(granularity) / f"labels.{content_type}.ckpt.jsonl"

    def teacher_prompts_dir(self, granularity: str, *, create: bool = False) -> Path:
        path = self.teacher_dir(granularity) / "prompts"
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def train_split(self, granularity: str) -> Path:
        return self.teacher_dir(granularity) / "train.parquet"

    def heldout_split(self, granularity: str) -> Path:
        return self.teacher_dir(granularity) / "heldout.parquet"

    def training_dir(self, granularity: str, topology: str, *, create: bool = False) -> Path:
        return self.step_dir("training", experiment_key(granularity, topology), create=create)

    def train_jsonl(self, granularity: str, topology: str, dataset: str) -> Path:
        """`dataset` is the half a student is trained on: news, transcripts or both."""
        return self.training_dir(granularity, topology) / f"train.{dataset}.jsonl"

    def adapter_dir(self, granularity: str, topology: str, dataset: str) -> Path:
        return self.training_dir(granularity, topology) / "adapters" / dataset

    def merged_dir(self, granularity: str, topology: str, dataset: str, epoch: int) -> Path:
        return self.training_dir(granularity, topology) / "merged" / dataset / f"epoch{epoch}"

    def train_log(self, granularity: str, topology: str, dataset: str) -> Path:
        return self.training_dir(granularity, topology) / f"train.{dataset}.log"

    def evaluations_dir(self, granularity: str, topology: str, *, create: bool = False) -> Path:
        return self.step_dir("evaluations", experiment_key(granularity, topology), create=create)

    def eval_metrics_json(self, granularity: str, topology: str, tag: str) -> Path:
        return self.evaluations_dir(granularity, topology) / f"{tag}.metrics.json"

    def eval_metrics_txt(self, granularity: str, topology: str, tag: str) -> Path:
        return self.evaluations_dir(granularity, topology) / f"{tag}.metrics.txt"

    def eval_predictions(self, granularity: str, topology: str, tag: str) -> Path:
        return self.evaluations_dir(granularity, topology) / f"{tag}.predictions.parquet"

    def eval_comparison(self, granularity: str, topology: str) -> Path:
        return self.evaluations_dir(granularity, topology) / "comparison.txt"

    def corpus_dir(self, granularity: str, topology: str, *, create: bool = False) -> Path:
        return self.step_dir("corpus", experiment_key(granularity, topology), create=create)

    def corpus_parquet(self, granularity: str, topology: str) -> Path:
        return self.corpus_dir(granularity, topology) / "corpus.parquet"

    def corpus_checkpoint(self, granularity: str, topology: str, content_type: str) -> Path:
        return self.corpus_dir(granularity, topology) / f"corpus.{content_type}.ckpt.jsonl"

    def backtest_dir(self, *parts: str, create: bool = False) -> Path:
        return self.step_dir("backtest", *parts, create=create)

    def backtest_book_dir(self, name: str, *, create: bool = False) -> Path:
        return self.backtest_dir(name, create=create)

    @property
    def backtest_comparison(self) -> Path:
        return self.backtest_dir() / "comparison.csv"

    @property
    def backtest_chart(self) -> Path:
        return self.backtest_dir() / "chart.png"

    def manifest_for(self, artifact: Path) -> Path:
        """A manifest sits beside its artifact, named after it."""
        if artifact.suffix.lower() in ARTIFACT_SUFFIXES:
            return artifact.with_suffix(artifact.suffix + ".manifest.json")
        return artifact / "manifest.json"

    def describe(self) -> str:
        lines = [f"run    : {self.run_dir}"]
        if self.dry_run:
            lines.append(f"DRY RUN: everything under {self.root} — no real artifact can be named")
        return "\n".join(lines)
