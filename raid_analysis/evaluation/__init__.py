"""Method-agnostic evaluation protocols, L2 probe factory, and implementations."""

from .ablation import AblationEvaluator
from .causal import ablate_neurons, patch_neurons
from .confound import ConfoundEvaluator
from .patching import PatchingEvaluator
from .probe_accuracy import ProbeAccuracyEvaluator
from .probe_factory import (
    EvalProbe,
    load_eval_probe,
    save_eval_probe,
    train_eval_probe,
)
from .protocol import Evaluator, FoldResult, load_fold_result, save_fold_result

__all__ = [
    "AblationEvaluator",
    "ConfoundEvaluator",
    "Evaluator",
    "EvalProbe",
    "FoldResult",
    "PatchingEvaluator",
    "ProbeAccuracyEvaluator",
    "ablate_neurons",
    "load_eval_probe",
    "load_fold_result",
    "patch_neurons",
    "save_eval_probe",
    "save_fold_result",
    "train_eval_probe",
]
