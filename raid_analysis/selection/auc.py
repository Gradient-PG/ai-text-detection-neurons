"""AUCSelector — neuron selection via Mann-Whitney U test + AUC threshold.

Wraps the existing :func:`compute_neuron_statistics` and
:func:`identify_discriminative_neurons` pipeline into the
:class:`NeuronSelector` protocol.
"""

from __future__ import annotations

import numpy as np

from ..data.activations import global_to_layer_neuron
from ..data.neuron_stats import (
    compute_neuron_statistics,
    identify_discriminative_neurons,
)
from .protocol import NeuronSelector, SelectionResult


class AUCSelector(NeuronSelector):
    """Select neurons whose per-neuron AUC exceeds a threshold.

    Runs Mann-Whitney U test per neuron, applies Bonferroni correction,
    and selects neurons that are both significant and have strong AUC
    effect size.

    Args:
        alpha: Family-wise error rate before Bonferroni correction.
        auc_threshold: AUC boundary (neurons with AUC > threshold or
            AUC < 1-threshold are considered strong).
    """

    def __init__(
        self,
        alpha: float = 0.001,
        auc_threshold: float = 0.7,
    ) -> None:
        self.alpha = alpha
        self.auc_threshold = auc_threshold

    def select(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        *,
        random_state: int | None = None,
    ) -> SelectionResult:
        n_features = activations.shape[1]

        stats_df = compute_neuron_statistics(activations, labels)
        stats_df, corrected_alpha = identify_discriminative_neurons(
            stats_df,
            alpha=self.alpha,
            auc_threshold=self.auc_threshold,
            n_total_tests=n_features,
        )

        disc = stats_df[stats_df["discriminative"]].copy()
        disc = disc.sort_values("auc_deviation", ascending=False)

        neuron_indices: set[tuple[int, int]] = set()
        ranking: list[tuple[int, int]] = []

        for _, row in disc.iterrows():
            global_idx = int(row["neuron_idx"])
            pair = global_to_layer_neuron(global_idx)
            neuron_indices.add(pair)
            ranking.append(pair)

        train_mean = activations.mean(axis=0)

        return SelectionResult(
            neuron_indices=neuron_indices,
            ranking=ranking,
            probe=None,
            train_mean=train_mean,
            metadata={
                "alpha": self.alpha,
                "auc_threshold": self.auc_threshold,
                "corrected_alpha": float(corrected_alpha),
                "n_discriminative": len(neuron_indices),
                "n_significant": int(stats_df["significant"].sum()),
                "n_strong_effect": int(stats_df["strong_effect"].sum()),
            },
        )
