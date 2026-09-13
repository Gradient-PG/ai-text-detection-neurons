"""Evaluator protocol and FoldResult dataclass.

Every evaluation method (ablation, patching, confound, probe accuracy)
implements the :class:`Evaluator` protocol.  It receives test-fold data plus
artifacts from the selection step and the L2 evaluation probe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..data.metadata import SampleMetadata
from ..selection.protocol import SelectionResult


@dataclass
class FoldResult:
    """Metrics and artifacts produced by one evaluator on one fold.

    Attributes:
        fold_idx: Fold number within the CV run.
        seed: Random seed that generated this fold partition.
        neuron_indices: The neuron set that was evaluated (copied from
            :class:`SelectionResult` for self-contained serialization).
        metrics: Evaluator-specific metric dict (accuracy, dose-response
            data, flip rates, correlation tables, etc.).
    """

    fold_idx: int
    seed: int
    neuron_indices: set[tuple[int, int]]
    metrics: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        keys = list(self.metrics.keys())[:5]
        return (
            f"FoldResult(fold={self.fold_idx}, seed={self.seed}, "
            f"n_neurons={len(self.neuron_indices)}, "
            f"metrics={keys}{'...' if len(self.metrics) > 5 else ''})"
        )


@runtime_checkable
class Evaluator(Protocol):
    """Protocol that every evaluation method must implement.

    Implementations receive test-fold data, the selection result from Step 1,
    and the L2 evaluation probe from Step 2.  The caller (the experiment
    runner) guarantees that *test_activations* and *test_labels* are from the
    held-out fold that was not used for selection or probe training.
    """

    def evaluate(
        self,
        test_activations: np.ndarray,
        test_labels: np.ndarray,
        test_metadata: SampleMetadata,
        selection: SelectionResult,
        eval_probe: Any,
    ) -> dict[str, Any]:
        """Evaluate the selected neurons on held-out test data.

        Args:
            test_activations: ``(N_test, D)`` concatenated activations.
            test_labels: ``(N_test,)`` binary labels.
            test_metadata: Per-sample metadata for the test fold.
            selection: Output of the selector (Step 1) on the train fold.
            eval_probe: L2 evaluation probe trained on the train fold (Step 2).

        Returns:
            Metric dict. Keys and value types are evaluator-specific.
        """
        ...


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def save_fold_result(result: FoldResult, directory: str | Path) -> None:
    """Save a :class:`FoldResult` to *directory* as JSON.

    Neuron indices are stored as a list of ``[layer, neuron_idx]`` pairs.
    Metric values must be JSON-serializable (use ``float()``, ``.tolist()``
    on numpy objects before storing).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    payload = {
        "fold_idx": result.fold_idx,
        "seed": result.seed,
        "n_neurons": len(result.neuron_indices),
        "neuron_indices": sorted([list(t) for t in result.neuron_indices]),
        "metrics": result.metrics,
    }
    with open(directory / "eval_metrics.json", "w") as f:
        json.dump(payload, f, indent=2)


def load_fold_result(directory: str | Path) -> FoldResult:
    """Load a :class:`FoldResult` from *directory*."""
    directory = Path(directory)
    with open(directory / "eval_metrics.json") as f:
        payload = json.load(f)

    neuron_indices = {tuple(pair) for pair in payload["neuron_indices"]}

    return FoldResult(
        fold_idx=payload["fold_idx"],
        seed=payload["seed"],
        neuron_indices=neuron_indices,
        metrics=payload["metrics"],
    )
