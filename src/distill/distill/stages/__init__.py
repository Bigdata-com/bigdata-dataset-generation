"""The stages, in the order they run."""

from . import (  # noqa: F401
    backtest, corpus, evaluate, export, ic, sample, subset, teacher, train,
)

ORDER = ("subset", "sample", "teacher", "export", "train", "evaluate", "corpus",
         "ic", "backtest")
