"""ProbeAccuracyEvaluator — baseline accuracy of the L2 evaluation probe.

The simplest evaluator: scores the eval probe on unmodified test data.
Reports accuracy, AUC-ROC, and F1.  Used standalone in the sparse probe
sweep (Experiment 1) and as the baseline measurement in ablation/patching.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from ..data.metadata import SampleMetadata
from ..selection.protocol import SelectionResult
from .probe_factory import EvalProbe
from .protocol import Evaluator


class ProbeAccuracyEvaluator(Evaluator):
    """Evaluate L2 probe accuracy on the test fold."""

    def __repr__(self) -> str:
        return "ProbeAccuracyEvaluator()"

    def evaluate(
        self,
        test_activations: np.ndarray,
        test_labels: np.ndarray,
        test_metadata: SampleMetadata,
        selection: SelectionResult,
        eval_probe: EvalProbe,
    ) -> dict[str, Any]:
        """Score the eval probe on unmodified test activations.

        Returns:
            Dict with ``accuracy``, ``auc_roc``, ``f1``, ``n_test``,
            and ``n_selected`` (number of neurons in the selection set).
        """
        accuracy = eval_probe.score(test_activations, test_labels)

        probas = eval_probe.predict_proba(test_activations)[:, 1]
        try:
            auc = float(roc_auc_score(test_labels, probas))
        except ValueError:
            auc = float("nan")

        preds = eval_probe.predict(test_activations)
        f1 = float(f1_score(test_labels, preds, zero_division=0.0))

        return {
            "accuracy": accuracy,
            "auc_roc": auc,
            "f1": f1,
            "n_test": len(test_labels),
            "n_selected": selection.n_selected,
        }
