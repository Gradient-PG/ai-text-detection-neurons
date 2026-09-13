"""Experiment 6 — AUC vs sparse probe comparison.

Runs AUCSelector on the same folds as Experiment 1, then compares:
- Overlap (Jaccard, top-k intersection) between AUC-selected and L1-selected sets
- Comparative ablation drop using the same L2 eval probe
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..data.activations import neuron_set_to_global_indices
from ..data.metadata import SampleMetadata
from ..data.splits import CVSplit
from ..evaluation.probe_factory import train_eval_probe
from ..evaluation.causal import ablate_neurons
from ..utils.set_ops import jaccard_similarity
from ..selection.auc import AUCSelector
from .config import ExperimentConfig, save_config
from .source_loader import load_source_selections


def run_auc_comparison(
    activations: np.ndarray,
    labels: np.ndarray,
    metadata: SampleMetadata,
    splits_by_seed: dict[int, list[CVSplit]],
    config: ExperimentConfig,
    source_dir: Path,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run AUC comparison against Experiment 1's sparse probe selections.

    Args:
        activations: ``(N, D)`` concatenated activations.
        labels: ``(N,)`` binary labels.
        metadata: Per-sample metadata.
        splits_by_seed: Must match Experiment 1's splits.
        config: Experiment configuration with ``selector_params``.
        source_dir: Root of the sparse probe sweep output.
        output_dir: If provided, persist results here.

    Returns:
        Dict with per-fold and aggregate comparison metrics.
    """
    l1_selections = load_source_selections(source_dir, splits_by_seed)

    auc_selector = AUCSelector(
        alpha=config.selector_params.get("alpha", 0.001),
        auc_threshold=config.selector_params.get("auc_threshold", 0.7),
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, output_dir / "config.yaml")

    print(f"AUC comparison: source={source_dir}")

    fold_results: list[dict[str, Any]] = []

    for seed, splits in sorted(splits_by_seed.items()):
        for fold in splits:
            train_acts = activations[fold.train_idx]
            train_labels = labels[fold.train_idx]
            test_acts = activations[fold.test_idx]
            test_labels = labels[fold.test_idx]

            auc_selection = auc_selector.select(
                train_acts, train_labels, random_state=seed,
            )
            l1_selection = l1_selections[(seed, fold.fold_idx)]

            # Overlap metrics
            jaccard = jaccard_similarity(
                auc_selection.neuron_indices, l1_selection.neuron_indices
            )
            intersection = auc_selection.neuron_indices & l1_selection.neuron_indices

            # Train shared L2 eval probe
            eval_probe = train_eval_probe(
                train_acts, train_labels,
                C=config.eval_probe_C,
                max_iter=config.eval_probe_max_iter,
                random_state=seed,
            )

            baseline_acc = eval_probe.score(test_acts, test_labels)

            # Ablate AUC-selected
            auc_global = neuron_set_to_global_indices(
                auc_selection.neuron_indices
            ).tolist()
            ablated_auc = ablate_neurons(
                test_acts, auc_global,
                method="mean", dataset_mean=auc_selection.train_mean,
            )
            auc_acc = eval_probe.score(ablated_auc, test_labels)

            # Ablate L1-selected
            l1_global = neuron_set_to_global_indices(
                l1_selection.neuron_indices
            ).tolist()
            ablated_l1 = ablate_neurons(
                test_acts, l1_global,
                method="mean", dataset_mean=l1_selection.train_mean,
            )
            l1_acc = eval_probe.score(ablated_l1, test_labels)

            fold_results.append({
                "seed": seed,
                "fold_idx": fold.fold_idx,
                "jaccard": float(jaccard),
                "intersection_size": len(intersection),
                "auc_n_selected": auc_selection.n_selected,
                "l1_n_selected": l1_selection.n_selected,
                "baseline_accuracy": float(baseline_acc),
                "auc_ablation_acc": float(auc_acc),
                "auc_ablation_drop": float(baseline_acc - auc_acc),
                "l1_ablation_acc": float(l1_acc),
                "l1_ablation_drop": float(baseline_acc - l1_acc),
            })

    # Aggregate
    aggregate = _aggregate_comparison(fold_results)

    result = {
        "fold_results": fold_results,
        "aggregate": aggregate,
    }

    if output_dir is not None:
        with open(output_dir / "comparison_results.json", "w") as f:
            json.dump(result, f, indent=2)

    return result


def _aggregate_comparison(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean ± std for all numeric comparison metrics."""
    if not fold_results:
        raise ValueError("_aggregate_comparison: No fold results to aggregate")

    agg: dict[str, Any] = {}
    numeric_keys = [
        "jaccard", "intersection_size", "auc_n_selected", "l1_n_selected",
        "baseline_accuracy", "auc_ablation_acc", "auc_ablation_drop",
        "l1_ablation_acc", "l1_ablation_drop",
    ]
    for key in numeric_keys:
        values = [fr[key] for fr in fold_results]
        agg[f"{key}_mean"] = float(np.mean(values))
        agg[f"{key}_std"] = float(np.std(values))

    agg["n_folds"] = len(fold_results)
    return agg
