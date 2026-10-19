"""`distill.yaml` — labelling, distillation and fine-tuning, in the repository's `config/`."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from .paths import DISTILL_CONFIG_NAME, EXTRACTION_CONFIG_NAME
from .theme import CONTENT_TYPES, GRANULARITIES, TOPOLOGIES

TRAINING_GPU_MODES = ("one", "ddp", "per_granularity")


def read_theme(path: str | Path) -> dict[str, Any] | str | None:
    """The `theme:` section of a `data_extraction.yaml`, or None if there is nothing to read."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    theme = data.get("theme")
    if isinstance(theme, (dict, str)):
        return theme
    return None

REFERENCE_COLUMN = "reference_label"


def _resolve(path: str | Path | None, base: Path) -> Path | None:
    if path in (None, ""):
        return None
    candidate = Path(str(path)).expanduser()
    return candidate if candidate.is_absolute() else (base / candidate)


@dataclass
class InputConfig:
    """Where the rows come from."""

    similarity_parquet: str | None = None
    source_parquet: str | None = None
    irreproducible_source: bool = False
    key_fields: tuple[str, ...] = ("doc_id", "chunk_index", "entity_id")
    text_column: str = "chunk_text"
    headline_column: str = "headline"
    entity_name_column: str = "entity_name"
    content_type_column: str = "content_type"
    reference_label_column: str | None = "classification"
    date_column: str = "date"
    timestamp_column: str = "timestamp"
    relevance_column: str = "chunk_relevance"
    sentiment_column: str = "chunk_sentiment"


@dataclass
class SubsetConfig:
    metric: str = "sim_top"
    top_fraction: float = 1.0
    max_rows: int | None = None
    per_content_type: bool = True
    content_types: tuple[str, ...] = CONTENT_TYPES


@dataclass
class SampleConfig:
    bins: int = 20
    bin_mode: str = "quantile"
    fraction_per_bin: float | None = 0.10
    count_per_bin: int | None = None
    max_rows: int | None = None
    seed: int = 17
    all_rows: bool = False


@dataclass
class ExperimentConfig:
    granularity: str = "both"
    topology: str = "combined"


@dataclass
class SentenceConfig:
    """Only read when granularity == sentence."""

    similarity_threshold: float = 0.30
    sentences_column: str = "extracted_sentences"
    sims_column: str = "extracted_sentence_sims"
    seeds_column: str = "extracted_best_search"
    cache: bool = True


@dataclass
class ServeConfig:
    """The docker line for one model, so it stops being copied between documents."""

    image: str = "vllm/vllm-openai:v0.20.0"
    container: str = ""
    gpus: str = "all"
    port: int = 8000
    model_path: str = ""
    revision: str | None = None
    served_model_name: str = ""
    chat_template: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    volumes: tuple[str, ...] = ()
    detach: bool = True


@dataclass
class EndpointSettings:
    url: str = "http://localhost:8000/v1/chat/completions"
    model: str = ""
    api_key: str = "EMPTY"
    api_key_env: str | None = None
    workers: int = 24
    max_output_tokens: int = 3000
    timeout: int = 300
    retries: int = 3
    temperature: float = 0.0
    send_temperature: bool | None = None
    guided_json: bool = False
    early_stop: bool = False
    harvest: bool = False
    extra_body: dict[str, Any] = field(default_factory=dict)
    health_attempts: int = 20
    health_delay: float = 15.0
    price_in_per_m: float | None = None
    price_out_per_m: float | None = None
    serve: ServeConfig | None = None


@dataclass
class SplitConfig:
    heldout_fraction: float = 0.10
    max_heldout: int | None = 50_000
    seed: int = 17
    stratify_on: tuple[str, ...] = ("content_type", "label", "bin")
    heldout_keys_from: str | None = None


@dataclass
class TrainingConfig:
    base_model: str = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
    base_revision: str | None = None
    lora_r: int = 64
    epochs: int = 5
    evaluate_epochs: tuple[int, ...] = (3, 5)
    keep_merged: tuple[int, ...] = (3, 5)
    val_ratio: float = 0.0
    batch_size: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 4096
    container: str = "tx-train"
    trainer_script: str = "ops/train_lora_sft.py"
    merge_script: str = "ops/merge_adapter.py"
    chat_template: str = "ops/n34b_nothink_template.jinja"
    container_mount: str | None = None
    link_root: str | None = None
    adapters_root: str | None = None
    merged_root: str | None = None
    seconds_per_step: tuple[float, float] = (4.3, 7.0)
    effective_batch: int = 64
    gpus: str = "one"
    devices: tuple[int, ...] = (0, 1)
    gradient_checkpointing: bool | None = None

    def processes(self) -> int:
        """How many trainer processes ONE training uses. Only `ddp` uses more than one."""
        return len(self.devices) if self.gpus == "ddp" else 1

    def cards_per_training(self) -> int:
        """How many cards one training occupies."""
        return len(self.devices) if self.gpus == "ddp" else 1

    def parallel_trainings(self) -> int:
        """How many trainings run at once."""
        return len(self.devices) if self.gpus == "per_granularity" else 1


@dataclass
class BacktestConfig:
    enabled: bool = True
    library_path: str | None = None
    price_cache: str | None = None
    universe_code: str = "US_ML_SRP2"
    leg_size: int | None = 160
    max_position: float = 0.05
    long_weight: float = 0.5
    short_weight: float = 0.5
    cutoff_ny: str = "15:30"
    half_life: int = 0
    """Calendar days a label keeps half its weight. 0 is the raw signal: it fires for one day.

    Applied to every labelled book the stage builds -- the student's and the reference's --
    beside its raw self. The sentiment floor stays raw: it already runs nearly fully deployed,
    so smoothing only decays it, and a smoothed floor is a lower bar than the raw one.
    """
    end: str | None = None
    """The last signal date the IC and the backtest use, `YYYY-MM-DD`. Null is the last signal.

    Set it with `--end` on `distill ic` and `distill backtest`: distill.yaml has no
    `backtest:` section. Signals after it are left out, and nothing upstream is touched. The
    backtest needs prices up to it; the IC's one-month horizon needs about a month past it.
    """
    signal_mode: str = "class_sign"
    reference_name: str = "reference"
    sentiment_floor: bool = True
    chart: bool = True


@dataclass
class DistillConfig:
    theme: dict[str, Any] = field(default_factory=dict)
    input: InputConfig = field(default_factory=InputConfig)
    subset: SubsetConfig = field(default_factory=SubsetConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    sentence: SentenceConfig = field(default_factory=SentenceConfig)
    teacher: EndpointSettings = field(default_factory=EndpointSettings)
    student: EndpointSettings = field(default_factory=EndpointSettings)
    student_overrides: dict[str, Any] = field(default_factory=dict)
    split: SplitConfig = field(default_factory=SplitConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    _base: Path = field(default_factory=Path.cwd, repr=False)
    _source: Path | None = field(default=None, repr=False)

    RETIRED_KEYS = frozenset({"run_dir", "paths"})

    TOP_LEVEL_KEYS = frozenset({
        "input", "subset", "sample", "experiment", "sentence", "teacher",
        "student", "student_overrides", "split", "training",
    })

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, base: Path, source: Path | None = None,
                  theme: dict[str, Any] | None = None) -> DistillConfig:
        data = dict(data or {})
        for retired in sorted(cls.RETIRED_KEYS & set(data)):
            data.pop(retired)
        unknown = sorted(set(data) - cls.TOP_LEVEL_KEYS - cls.RETIRED_KEYS)
        if unknown:
            hint = ""
            if "backtest" in unknown:
                hint = (" `backtest` is not a config section: the strategy defaults live in "
                        "`BacktestConfig`, and the one a run actually chooses is "
                        "`distill backtest --price-cache`.")
            if "theme" in unknown:
                hint = (f" The theme moved to {EXTRACTION_CONFIG_NAME}, where the search reads "
                        f"it: one definition for what is retrieved and what is labelled. Move "
                        f"the whole `theme:` block into that file.")
            raise SystemExit(
                f"{DISTILL_CONFIG_NAME}: unknown top-level key(s) {unknown}.{hint}\n"
                f"Known: {sorted(cls.TOP_LEVEL_KEYS)}"
            )
        def section(name: str, kind):
            raw = data.get(name) or {}
            if not isinstance(raw, dict):
                raise SystemExit(f"distill.yaml: `{name}` must be a mapping, got {type(raw).__name__}")
            return _build(kind, raw, name)
        if isinstance(theme, str):
            raise SystemExit(
                f"`theme` in {EXTRACTION_CONFIG_NAME} is a theme id ({theme!r}), but it has to "
                f"be the theme itself: id, title, classes and body, inline in that file. Seed "
                f"it with:\n"
                f"  copy the block from src/distill/themes/{theme}.yaml"
            )
        config = cls(
            theme=dict(theme or {}),
            input=section("input", InputConfig),
            subset=section("subset", SubsetConfig),
            sample=section("sample", SampleConfig),
            experiment=section("experiment", ExperimentConfig),
            sentence=section("sentence", SentenceConfig),
            teacher=_endpoint(data.get("teacher") or {}, "teacher"),
            student=_endpoint(data.get("student") or {}, "student"),
            student_overrides=dict(data.get("student_overrides") or {}),
            split=section("split", SplitConfig),
            training=section("training", TrainingConfig),
        )
        config._base = base
        config._source = source
        return config

    @classmethod
    def load_run(cls, paths) -> DistillConfig:
        """Load a run's configuration.

        The backtest used to be a third YAML file holding two lines. Its strategy settings are
        defaults that no run ever changed, so they live in `BacktestConfig`; the one a run does
        choose -- where the price cache is -- is a flag on `distill backtest`, so it is named
        by the command instead of edited into a file.
        """
        return cls.load(paths.distill_config, base=paths.run_dir,
                        extraction=paths.extraction_config)

    @classmethod
    def load(cls, path: str | Path, *, base: Path | None = None,
             extraction: Path | None = None) -> DistillConfig:
        """Load one run's configuration."""
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"no {path.name} at {path}. The three config files are checked in under "
                f"`config/`, beside the output folder -- they are edited, not generated."
            )
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if base is None:
            base = path.parent.parent if path.parent.name == "config" else path.parent
        if extraction is None:
            extraction = path.parent / EXTRACTION_CONFIG_NAME
        return cls.from_dict(data, base=base, source=path, theme=read_theme(extraction))

    @property
    def base(self) -> Path:
        return self._base

    def path(self, value: str | Path | None) -> Path | None:
        """Resolve one configured path relative to the config file's own directory."""
        return _resolve(value, self._base)

    def repo_path(self, value: str | Path | None) -> Path | None:
        """Resolve a path that names something in the REPOSITORY rather than in the run.

        `input.similarity_parquet` is written BY the run and is relative to it, which is why
        `load_run` bases everything on the run folder. A price cache and the backtest library
        are not: they sit beside `config/`, and resolving them the same way looks for them
        inside `output/`, where they are not.
        """
        if value is None:
            return None
        candidate = Path(str(value)).expanduser()
        if candidate.is_absolute():
            return candidate
        root = self._source.parent.parent if self._source else self._base
        return (root / candidate).resolve()

    def source_frame_path(self) -> Path:
        """The one input Part B may assume."""
        if self.input.source_parquet:
            resolved = self.path(self.input.source_parquet)
        elif self.input.similarity_parquet:
            resolved = self.path(self.input.similarity_parquet)
        else:
            raise SystemExit(
                "distill.yaml: set input.similarity_parquet (Part A's "
                "'similarity/combinations_with_similarity.parquet') or input.source_parquet."
            )
        assert resolved is not None
        if not resolved.exists():
            raise SystemExit(f"input not found: {resolved}")
        return resolved

    def granularities(self) -> tuple[str, ...]:
        """The granularities this run covers."""
        if self.experiment.granularity == "both":
            return GRANULARITIES
        return (self.experiment.granularity,)

    def student_for(self, content_type: str | None = None) -> EndpointSettings:
        """The student endpoint for one half of the corpus."""
        if not content_type or content_type not in self.student_overrides:
            return self.student
        merged = {k: v for k, v in asdict(self.student).items() if k != "serve"}
        merged.update(dict(self.student_overrides[content_type]))
        serve_raw = merged.pop("serve", None)
        settings = _build(EndpointSettings, merged, f"student_overrides.{content_type}")
        if serve_raw:
            settings.serve = _build(ServeConfig, dict(serve_raw),
                                    f"student_overrides.{content_type}.serve")
        elif self.student.serve is not None:
            settings.serve = self.student.serve
        return settings

    def datasets_for_topology(self) -> tuple[str, ...]:
        """Which training sets this topology needs."""
        topology = self.experiment.topology
        if topology == "combined":
            return ("both",)
        if topology == "specialist":
            return tuple(self.subset.content_types)
        return (*self.subset.content_types, "both")

    def content_type_for_dataset(self, dataset: str) -> tuple[str, ...]:
        return tuple(self.subset.content_types) if dataset == "both" else (dataset,)

    def theme_spec(self):
        """The theme this run labels with, read out of `data_extraction.yaml`."""
        from .theme import ThemeSpec

        if not self.theme:
            raise SystemExit(
                f"no `theme:` section in {EXTRACTION_CONFIG_NAME}, so there is nothing to label "
                f"with. It needs id, title, classes and body -- see "
                f"distill/distill/prompt_guidelines.md, or seed one with "
                f"copy one from src/distill/themes/."
            )
        return ThemeSpec.from_dict(self.theme).require_valid()

    @property
    def theme_id(self) -> str:
        return str(self.theme.get("id", "")) if self.theme else ""

    def payload(self) -> dict[str, Any]:
        """What the config hash covers."""
        out = asdict(self)
        for private in ("_base", "_source"):
            out.pop(private, None)
        return out

    def to_yaml(self) -> str:
        """Serialise the parts that belong to `distill.yaml`."""
        payload = {k: v for k, v in self.payload().items() if k not in ("backtest", "theme")}
        return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml(), encoding="utf-8")
        return path

    def validate(self) -> list[str]:
        issues: list[str] = []
        if not self.theme:
            issues.append(
                f"no `theme:` section in {EXTRACTION_CONFIG_NAME} — the first config file has "
                f"to carry the theme the whole run means: id, title, classes and body"
            )
        else:
            from .theme import ThemeSpec
            try:
                issues.extend(f"theme: {issue}"
                              for issue in ThemeSpec.from_dict(self.theme).validate())
            except SystemExit as exc:
                issues.append(f"theme: {exc}")
        if self.experiment.granularity not in (*GRANULARITIES, "both"):
            issues.append(f"experiment.granularity must be one of "
                          f"{[*GRANULARITIES, 'both']}, got {self.experiment.granularity!r}")
        if self.experiment.topology not in TOPOLOGIES:
            issues.append(f"experiment.topology must be one of {list(TOPOLOGIES)}, "
                          f"got {self.experiment.topology!r}")
        if self.sample.bin_mode not in ("quantile", "width"):
            issues.append(f"sample.bin_mode must be quantile|width, got {self.sample.bin_mode!r}")
        if self.sample.bins < 1:
            issues.append("sample.bins must be >= 1")
        if not self.sample.all_rows and self.sample.count_per_bin is None:
            fraction = self.sample.fraction_per_bin
            if fraction is None or not (0 < fraction <= 1):
                issues.append("sample.fraction_per_bin must be in (0, 1] unless count_per_bin "
                              "or all_rows is set")
        if not (0 < self.subset.top_fraction <= 1):
            issues.append("subset.top_fraction must be in (0, 1]")
        if not (0 <= self.split.heldout_fraction < 1):
            issues.append("split.heldout_fraction must be in [0, 1)")
        if self.subset.max_rows is not None and self.subset.max_rows < 1:
            issues.append(f"subset.max_rows must be >= 1 or null, got {self.subset.max_rows}")
        if self.split.max_heldout is not None and self.split.max_heldout < 1:
            issues.append(f"split.max_heldout must be >= 1 or null, got "
                          f"{self.split.max_heldout}")
        if self.training.gpus not in TRAINING_GPU_MODES:
            issues.append(f"training.gpus must be one of {list(TRAINING_GPU_MODES)}, got "
                          f"{self.training.gpus!r}")
        elif self.training.gpus != "one" and len(self.training.devices) < 2:
            issues.append(f"training.gpus is {self.training.gpus!r} but training.devices is "
                          f"{list(self.training.devices)} -- it needs at least two cards")
        if self.training.gpus == "per_granularity" and len(self.granularities()) < 2:
            issues.append(
                "training.gpus is 'per_granularity' but this run covers one granularity, so "
                "there is only one dataset to place. Set experiment.granularity to both, or "
                "use 'ddp' to put both cards on the single training."
            )
        world = self.training.processes()
        product = self.training.batch_size * world
        if self.training.effective_batch % product:
            issues.append(
                f"training.batch_size {self.training.batch_size} x {world} process(es) does "
                f"not divide effective_batch {self.training.effective_batch}. The trainer "
                f"computes accumulation as effective // (batch x processes), so these would "
                f"disagree and the epoch-to-checkpoint mapping would be wrong."
            )
        bad_overrides = sorted(set(self.student_overrides) - set(CONTENT_TYPES))
        if bad_overrides:
            issues.append(f"student_overrides keys must be content types, got {bad_overrides}")
        unknown = sorted(set(self.subset.content_types) - set(CONTENT_TYPES))
        if unknown:
            issues.append(f"subset.content_types has unknown values: {unknown}")
        if self.training.epochs < 1:
            issues.append("training.epochs must be >= 1")
        beyond = [e for e in self.training.evaluate_epochs if e > self.training.epochs or e < 1]
        if beyond:
            issues.append(f"training.evaluate_epochs {beyond} are outside 1..{self.training.epochs} "
                          "— those checkpoints will not exist")
        merged_beyond = [e for e in self.training.keep_merged
                         if e not in tuple(self.training.evaluate_epochs)]
        if merged_beyond:
            issues.append(f"training.keep_merged {merged_beyond} are not in evaluate_epochs, so "
                          "nothing would merge them")
        if "sentence" in self.granularities() and not (0 <= self.sentence.similarity_threshold <= 1):
            issues.append("sentence.similarity_threshold must be in [0, 1]")
        if self.backtest.half_life < 0:
            issues.append(f"backtest.half_life must be >= 0, got {self.backtest.half_life}")
        if self.backtest.end is not None:
            try:
                date.fromisoformat(str(self.backtest.end))
            except ValueError:
                issues.append(f"backtest.end must be a YYYY-MM-DD date or null, got "
                              f"{self.backtest.end!r}")
        if self.backtest.enabled and self.backtest.signal_mode not in (
            "class_sign", "relevance_class_sign"
        ):
            issues.append(
                f"backtest.signal_mode must be class_sign|relevance_class_sign, got "
                f"{self.backtest.signal_mode!r}."
            )
        return issues

    def require_valid(self) -> DistillConfig:
        issues = self.validate()
        if issues:
            where = f" ({self._source})" if self._source else ""
            raise SystemExit(f"distill.yaml{where} has problems:\n  - " + "\n  - ".join(issues))
        return self


def _build(kind, raw: dict[str, Any], where: str):
    """Instantiate a dataclass from a mapping, refusing unknown keys."""
    fields = {f.name: f for f in kind.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise SystemExit(
            f"distill.yaml: `{where}` has unknown key(s) {unknown}. "
            f"Known keys: {sorted(fields)}"
        )
    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        annotation = str(fields[name].type)
        if "tuple" in annotation and isinstance(value, list):
            value = tuple(value)
        kwargs[name] = value
    return kind(**kwargs)


def _endpoint(raw: dict[str, Any], where: str) -> EndpointSettings:
    raw = dict(raw)
    serve_raw = raw.pop("serve", None)
    settings = _build(EndpointSettings, raw, where)
    if serve_raw:
        settings.serve = _build(ServeConfig, dict(serve_raw), f"{where}.serve")
    return settings


def search_end(paths) -> str | None:
    """`search.end_date` from `data_extraction.yaml`, or None if it is not set."""
    try:
        data = yaml.safe_load(paths.extraction_config.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return None
    end = (data.get("search") or {}).get("end_date")
    return str(end) if end else None
