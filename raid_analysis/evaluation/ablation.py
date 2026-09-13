"""AblationEvaluator — group ablation sweep with random-k baseline.

Tests *necessity*: does ablating the selected neurons cause a disproportionate
accuracy drop compared to ablating random neurons of the same count?

Produces ablation sweep data: for each k in ``k_values``, records the accuracy
and accuracy drop when ablating the top-k selected neurons vs random-k neurons.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..data.activations import layer_neuron_to_global
from ..data.metadata import SampleMetadata
from .causal import ablate_neurons
from ..selection.protocol import SelectionResult
from .probe_factory import EvalProbe
from .protocol import Evaluator


class AblationEvaluator(Evaluator):
    """Evaluate necessity via group ablation at multiple k values.

    For each k, ablates the top-k neurons (by selection ranking) using
    mean ablation and measures the L2 eval probe accuracy drop.  A random-k
    baseline is computed over multiple random draws for comparison.

    Args:
        k_values: List of neuron counts to ablate. Use ``-1`` as a sentinel
            to ablate the full selected set (``k = len(selection.ranking)``
            per fold). Duplicates and values exceeding the selected-set size
            are dropped; valid values preserve their YAML order.
        method: Ablation method (``"mean"`` or ``"zero"``).
        n_random_seeds: Number of random draws for the random-k baseline.
    """

    def __init__(
        self,
        k_values: list[int] | None = None,
        method: str = "mean",
        n_random_seeds: int = 20,
    ) -> None:
        self.k_values = k_values or [1, 2, 5, 10, 20, 50, 100, 200]
        self.method = method
        self.n_random_seeds = n_random_seeds

    def __repr__(self) -> str:
        return (
            f"AblationEvaluator(k_values={self.k_values}, "
            f"method={self.method!r}, n_random_seeds={self.n_random_seeds})"
        )

    def evaluate(
        self,
        test_activations: np.ndarray,
        test_labels: np.ndarray,
        test_metadata: SampleMetadata,
        selection: SelectionResult,
        eval_probe: EvalProbe,
    ) -> dict[str, Any]:
        baseline_acc = eval_probe.score(test_activations, test_labels)

        ranked_global = [
            layer_neuron_to_global(layer, neuron)
            for layer, neuron in selection.ranking
        ]
        n_features = test_activations.shape[1]
        train_mean = selection.train_mean

        resolved_ks: list[int] = []
        seen: set[int] = set()
        for raw_k in self.k_values:
            k = len(ranked_global) if raw_k == -1 else raw_k
            if k <= 0 or k > len(ranked_global) or k in seen:
                continue
            seen.add(k)
            resolved_ks.append(k)

        ablation_sweep: list[dict[str, Any]] = []

        for k in resolved_ks:

            # Selected top-k ablation
            top_k_indices = ranked_global[:k]
            ablated = ablate_neurons(
                test_activations, top_k_indices,
                method=self.method, dataset_mean=train_mean,
            )
            selected_acc = eval_probe.score(ablated, test_labels)
            selected_drop = baseline_acc - selected_acc

            # Random-k baseline
            rng = np.random.RandomState(42)
            random_accs: list[float] = []
            for _ in range(self.n_random_seeds):
                random_indices = rng.choice(
                    n_features, size=k, replace=False,
                ).tolist()
                ablated_rand = ablate_neurons(
                    test_activations, random_indices,
                    method=self.method, dataset_mean=train_mean,
                )
                random_accs.append(eval_probe.score(ablated_rand, test_labels))

            random_drops = [baseline_acc - a for a in random_accs]

            ablation_sweep.append({
                "k": k,
                "selected_acc": float(selected_acc),
                "selected_drop": float(selected_drop),
                "random_acc_mean": float(np.mean(random_accs)),
                "random_acc_std": float(np.std(random_accs)),
                "random_drop_mean": float(np.mean(random_drops)),
                "random_drop_std": float(np.std(random_drops)),
            })

        return {
            "baseline_accuracy": float(baseline_acc),
            "ablation_sweep": ablation_sweep,
            "method": self.method,
            "n_random_seeds": self.n_random_seeds,
            "n_selected": selection.n_selected,
            "n_test": len(test_labels),
        }
