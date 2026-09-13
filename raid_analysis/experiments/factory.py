"""Factory functions to construct selectors and evaluators from config."""

from __future__ import annotations

from .config import ExperimentConfig
from ..evaluation.ablation import AblationEvaluator
from ..evaluation.confound import ConfoundEvaluator
from ..evaluation.patching import PatchingEvaluator
from ..evaluation.probe_accuracy import ProbeAccuracyEvaluator
from ..evaluation.protocol import Evaluator
from ..selection.auc import AUCSelector
from ..selection.protocol import NeuronSelector
from ..selection.sparse_probe import SparseProbeSelector


_SELECTOR_REGISTRY: dict[str, type] = {
    "sparse_probe": SparseProbeSelector,
    "auc": AUCSelector,
}

_EVALUATOR_REGISTRY: dict[str, type] = {
    "probe_accuracy": ProbeAccuracyEvaluator,
    "ablation": AblationEvaluator,
    "patching": PatchingEvaluator,
    "confound": ConfoundEvaluator,
}


def build_selector(config: ExperimentConfig) -> NeuronSelector:
    """Construct a selector from ``config.selector`` + ``config.selector_params``."""
    name = config.selector
    if name not in _SELECTOR_REGISTRY:
        raise ValueError(
            f"Unknown selector {name!r}. "
            f"Available: {sorted(_SELECTOR_REGISTRY)}"
        )
    cls = _SELECTOR_REGISTRY[name]
    return cls(**config.selector_params)


def build_evaluator(config: ExperimentConfig) -> Evaluator:
    """Construct an evaluator from ``config.evaluator`` + ``config.evaluator_params``."""
    name = config.evaluator
    if name not in _EVALUATOR_REGISTRY:
        raise ValueError(
            f"Unknown evaluator {name!r}. "
            f"Available: {sorted(_EVALUATOR_REGISTRY)}"
        )
    cls = _EVALUATOR_REGISTRY[name]
    return cls(**config.evaluator_params)
