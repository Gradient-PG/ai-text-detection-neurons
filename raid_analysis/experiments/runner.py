"""Experiment runner — the method-agnostic CV fold loop.

Enforces train/test separation for every fold:
  Step 1: selector.select(train_acts, train_labels)  → SelectionResult
  Step 2: train_eval_probe(train_acts, train_labels)  → EvalProbe
  Step 3: evaluator.evaluate(test_acts, ..., selection, eval_probe) → metrics

The runner never exposes train data to the evaluator or test data to the
selector.  All results are saved per fold and aggregated across folds × seeds.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..data.metadata import SampleMetadata
from ..data.splits import CVSplit
from ..evaluation.probe_factory import (
    save_eval_probe,
    train_eval_probe,
)
from ..evaluation.protocol import Evaluator, FoldResult, save_fold_result
from ..selection.protocol import NeuronSelector, SelectionResult, save_selection
from .config import ExperimentConfig


@dataclass
class ExperimentResult:
    """Aggregated output of a full experiment run across all folds × seeds.

    Attributes:
        fold_results: Every individual fold result.
        aggregate_metrics: Averaged/aggregated metrics across folds.
        neuron_stability: ``{(layer, neuron_idx): fraction_of_folds}`` — how
            often each neuron appeared in the selection across all runs.
        stable_neurons: Neurons appearing in ≥ *stability_threshold* of runs.
        config: Snapshot of experiment configuration.
    """

    fold_results: list[FoldResult]
    aggregate_metrics: dict[str, Any]
    neuron_stability: dict[tuple[int, int], float]
    stable_neurons: set[tuple[int, int]]
    config: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ExperimentResult(n_folds={len(self.fold_results)}, "
            f"stable_neurons={len(self.stable_neurons)}, "
            f"metrics={list(self.aggregate_metrics.keys())})"
        )


def run_experiment(
    activations: np.ndarray,
    labels: np.ndarray,
    metadata: SampleMetadata,
    splits_by_seed: dict[int, list[CVSplit]],
    evaluator: Evaluator,
    config: ExperimentConfig,
    *,
    selector: NeuronSelector | None = None,
    precomputed_selections: dict[tuple[int, int], SelectionResult] | None = None,
    output_dir: Path | None = None,
) -> ExperimentResult:
    """Run the full CV experiment loop.

    Either *selector* or *precomputed_selections* must be provided:

    - **selector**: runs Step 1 (neuron selection on train data) each fold.
    - **precomputed_selections**: ``{(seed, fold_idx): SelectionResult}`` loaded
      from a prior experiment.  Skips Step 1.  Used by dependent experiments
      (ablation, patching, confound) that reuse Experiment 1's neuron sets.

    Args:
        activations: ``(N, D)`` concatenated activations (all samples).
        labels: ``(N,)`` binary labels.
        metadata: Per-sample metadata aligned with activations.
        splits_by_seed: ``{seed: [CVSplit, ...]}`` from :func:`generate_multi_seed_splits`.
        evaluator: An :class:`Evaluator` implementation.
        config: Experiment configuration (carries stability_threshold,
            eval_probe_C, eval_probe_max_iter, etc.).
        selector: A :class:`NeuronSelector` implementation (Step 1).
            The runner passes ``random_state=seed`` to ``select()`` each fold.
        precomputed_selections: Pre-loaded selections keyed by ``(seed, fold_idx)``.
        output_dir: If provided, persist per-fold results and aggregate here.

    Returns:
        :class:`ExperimentResult` with per-fold results and aggregate.
    """
    if selector is None and precomputed_selections is None:
        raise ValueError("Either 'selector' or 'precomputed_selections' must be provided")

    fold_results: list[FoldResult] = []
    all_neuron_sets: list[set[tuple[int, int]]] = []

    total_folds = sum(len(folds) for folds in splits_by_seed.values())
    fold_counter = 0

    for seed, splits in sorted(splits_by_seed.items()):
        for fold in splits:
            fold_counter += 1
            t0 = time.time()

            train_acts = activations[fold.train_idx]
            train_labels = labels[fold.train_idx]
            test_acts = activations[fold.test_idx]
            test_labels = labels[fold.test_idx]
            test_meta = metadata[fold.test_idx]

            # Step 1: Selection (train data only, or precomputed)
            if precomputed_selections is not None:
                key = (seed, fold.fold_idx)
                if key not in precomputed_selections:
                    raise KeyError(
                        f"No precomputed selection for seed={seed}, "
                        f"fold={fold.fold_idx}"
                    )
                selection = precomputed_selections[key]
            else:
                assert selector is not None
                selection = selector.select(
                    train_acts, train_labels, random_state=seed,
                )

            # Step 2: Evaluation probe (train data only)
            eval_probe = train_eval_probe(
                train_acts,
                train_labels,
                C=config.eval_probe_C,
                max_iter=config.eval_probe_max_iter,
                random_state=seed,
            )

            # Step 2b: Score selector's own probe on test data (if available)
            if selection.probe is not None and selection.scaler is not None:
                X_test_scaled = selection.scaler.transform(test_acts)
                selector_test_acc = float(selection.probe.score(X_test_scaled, test_labels))
            else:
                selector_test_acc = None

            # Step 3: Evaluation (test data only)
            metrics = evaluator.evaluate(
                test_acts, test_labels, test_meta, selection, eval_probe,
            )

            if selector_test_acc is not None:
                metrics["selector_test_accuracy"] = selector_test_acc

            fold_result = FoldResult(
                fold_idx=fold.fold_idx,
                seed=seed,
                neuron_indices=selection.neuron_indices,
                metrics=metrics,
            )
            fold_results.append(fold_result)
            all_neuron_sets.append(selection.neuron_indices)

            elapsed = time.time() - t0
            n_sel = selection.n_selected
            acc = metrics.get("accuracy", None)
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            print(
                f"  [{fold_counter}/{total_folds}] "
                f"seed={seed} fold={fold.fold_idx}  "
                f"selected={n_sel}  acc={acc_str}  "
                f"({elapsed:.1f}s)"
            )

            if output_dir is not None:
                fold_dir = output_dir / f"seed_{seed}" / f"fold_{fold.fold_idx}"
                if precomputed_selections is None:
                    save_selection(selection, fold_dir)
                save_eval_probe(eval_probe, fold_dir)
                save_fold_result(fold_result, fold_dir)

    # Aggregate
    aggregate = _aggregate_metrics(fold_results)

    if precomputed_selections is not None:
        stability: dict[tuple[int, int], float] = {}
        stable: set[tuple[int, int]] = set()
    else:
        stability = _compute_neuron_stability(all_neuron_sets)
        stable = {
            n for n, frac in stability.items()
            if frac >= config.stability_threshold
        }

    result = ExperimentResult(
        fold_results=fold_results,
        aggregate_metrics=aggregate,
        neuron_stability=stability,
        stable_neurons=stable,
        config={},
    )

    if output_dir is not None:
        _save_aggregate(result, output_dir)

    return result


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_metrics(fold_results: list[FoldResult]) -> dict[str, Any]:
    """Compute mean ± std for every numeric metric across folds."""
    if not fold_results:
        return {}

    all_keys: set[str] = set()
    for fr in fold_results:
        all_keys.update(fr.metrics.keys())

    agg: dict[str, Any] = {}
    for key in sorted(all_keys):
        values = []
        for fr in fold_results:
            v = fr.metrics.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                values.append(float(v))

        if values:
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
            agg[f"{key}_min"] = float(np.min(values))
            agg[f"{key}_max"] = float(np.max(values))

    agg["n_folds"] = len(fold_results)
    return agg


def _compute_neuron_stability(
    neuron_sets: list[set[tuple[int, int]]],
) -> dict[tuple[int, int], float]:
    """Fraction of runs each neuron appears in."""
    if not neuron_sets:
        return {}

    counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    for ns in neuron_sets:
        for n in ns:
            counts[n] += 1

    total = len(neuron_sets)
    return {n: cnt / total for n, cnt in counts.items()}


def _save_aggregate(result: ExperimentResult, output_dir: Path) -> None:
    """Save aggregate.json with cross-fold summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    stability_serializable = {
        f"{layer}_{neuron}": frac
        for (layer, neuron), frac in sorted(result.neuron_stability.items())
    }
    stable_serializable = sorted([list(t) for t in result.stable_neurons])

    payload = {
        "aggregate_metrics": result.aggregate_metrics,
        "n_stable_neurons": len(result.stable_neurons),
        "stable_neurons": stable_serializable,
        "neuron_stability": stability_serializable,
    }
    with open(output_dir / "aggregate.json", "w") as f:
        json.dump(payload, f, indent=2)
