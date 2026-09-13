"""Experiment 7 — Smaller-model / ablate-complement test (§5.1 (B)).

Tests the paper's *"smaller model is sufficient"* claim by restricting the
evaluation to the selected neurons in two mathematically near-equivalent ways
and comparing both against the all-features baseline.

For each fold (reusing the source experiment's selections):

1. **Baseline** — L2 probe fit on the train fold with all 9,216 features,
   scored on the unmodified test fold.
2. **Smaller model** — L2 probe fit on the train fold using *only* the
   selected feature columns, scored on the test fold's selected columns.
3. **Ablate-complement** — the baseline full L2 probe is scored on test
   activations where every neuron **NOT** in the selected set has been
   mean-ablated (using the train-fold mean from the source selection). This
   is the dual of ``AblationEvaluator``: instead of "what happens if we
   remove the selected set?" it asks "what happens if we remove everything
   *except* the selected set?".

Both framings should give very similar accuracy; large differences would
indicate that the selected-feature subspace captures genuinely different
geometry when the probe is re-trained on it vs. when it is inherited from
the full model.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from ..data.activations import neuron_set_to_global_indices
from ..data.metadata import SampleMetadata
from ..data.splits import CVSplit
from ..evaluation.causal import ablate_neurons
from ..evaluation.probe_factory import train_eval_probe
from .config import ExperimentConfig, save_config
from .source_loader import load_source_selections


def run_restricted_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    metadata: SampleMetadata,
    splits_by_seed: dict[int, list[CVSplit]],
    config: ExperimentConfig,
    source_dir: Path,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the smaller-model / ablate-complement experiment.

    Args:
        activations: ``(N, D)`` concatenated activations.
        labels: ``(N,)`` binary labels.
        metadata: Per-sample metadata (unused here, kept for symmetry).
        splits_by_seed: Must match the source experiment's splits.
        config: Experiment configuration. Reads ``eval_probe_C``,
            ``eval_probe_max_iter``.
        source_dir: Root of the source experiment's per-fold selections.
        output_dir: If provided, persist results here.

    Returns:
        Dict with ``fold_results`` and ``aggregate`` metrics.
    """
    del metadata  # not used; kept for interface symmetry

    selections = load_source_selections(source_dir, splits_by_seed)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, output_dir / "config.yaml")

    n_total_features = activations.shape[1]
    print(
        f"Restricted-probe experiment: source={source_dir} "
        f"(D={n_total_features})"
    )

    total_folds = sum(len(f) for f in splits_by_seed.values())
    fold_results: list[dict[str, Any]] = []
    counter = 0

    for seed, splits in sorted(splits_by_seed.items()):
        for fold in splits:
            counter += 1
            fold_metrics = _run_fold(
                activations=activations,
                labels=labels,
                fold=fold,
                seed=seed,
                selection=selections[(seed, fold.fold_idx)],
                config=config,
                n_total_features=n_total_features,
            )
            fold_results.append(fold_metrics)

            print(
                f"  [{counter}/{total_folds}] "
                f"seed={seed} fold={fold.fold_idx}  "
                f"k={fold_metrics['n_selected']}  "
                f"base={fold_metrics['baseline_accuracy']:.4f}  "
                f"small={fold_metrics['smaller_model_accuracy']:.4f}  "
                f"ablC={fold_metrics['ablate_complement_accuracy']:.4f}"
            )

            if output_dir is not None:
                fold_dir = (
                    output_dir / f"seed_{seed}" / f"fold_{fold.fold_idx}"
                )
                fold_dir.mkdir(parents=True, exist_ok=True)
                with open(fold_dir / "eval_metrics.json", "w") as f:
                    json.dump(fold_metrics, f, indent=2)

    aggregate = _aggregate(fold_results)

    result = {
        "fold_results": fold_results,
        "aggregate": aggregate,
    }

    if output_dir is not None:
        with open(output_dir / "restricted_probe_results.json", "w") as f:
            json.dump(result, f, indent=2)
        with open(output_dir / "aggregate.json", "w") as f:
            json.dump({"aggregate_metrics": aggregate}, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# Per-fold logic
# ---------------------------------------------------------------------------

def _run_fold(
    *,
    activations: np.ndarray,
    labels: np.ndarray,
    fold: CVSplit,
    seed: int,
    selection,
    config: ExperimentConfig,
    n_total_features: int,
) -> dict[str, Any]:
    """Compute baseline / smaller-model / ablate-complement metrics for one fold."""
    train_acts = activations[fold.train_idx]
    train_labels = labels[fold.train_idx]
    test_acts = activations[fold.test_idx]
    test_labels = labels[fold.test_idx]

    selected_global = neuron_set_to_global_indices(
        selection.neuron_indices
    ).astype(np.int64)
    n_selected = int(selected_global.size)

    # ---- (1) Baseline: full L2 probe on all features --------------------
    full_probe = train_eval_probe(
        train_acts,
        train_labels,
        C=config.eval_probe_C,
        max_iter=config.eval_probe_max_iter,
        random_state=seed,
    )
    baseline_acc = full_probe.score(test_acts, test_labels)
    baseline_auc, baseline_f1 = _auc_f1(full_probe, test_acts, test_labels)

    # ---- (2) Smaller model: new L2 probe on selected features only ------
    if n_selected == 0:
        smaller_acc = float("nan")
        smaller_auc = float("nan")
        smaller_f1 = float("nan")
    else:
        train_acts_sel = train_acts[:, selected_global]
        test_acts_sel = test_acts[:, selected_global]
        smaller_probe = train_eval_probe(
            train_acts_sel,
            train_labels,
            C=config.eval_probe_C,
            max_iter=config.eval_probe_max_iter,
            random_state=seed,
        )
        smaller_acc = smaller_probe.score(test_acts_sel, test_labels)
        smaller_auc, smaller_f1 = _auc_f1(
            smaller_probe, test_acts_sel, test_labels
        )

    # ---- (3) Ablate-complement: full probe, mean-ablate non-selected ----
    if selection.train_mean is None:
        # Source selection was saved without a train_mean; fall back to the
        # train fold's empirical mean (identical in practice — the source
        # pipeline uses the same train fold).
        train_mean = train_acts.mean(axis=0)
    else:
        train_mean = selection.train_mean

    selected_mask = np.zeros(n_total_features, dtype=bool)
    selected_mask[selected_global] = True
    complement_indices = np.nonzero(~selected_mask)[0].tolist()

    if n_selected == 0:
        # Nothing is selected → complement is everything → probe sees only
        # training means and collapses to the majority-class baseline.
        ablate_complement_acc = float("nan")
        ablate_complement_auc = float("nan")
        ablate_complement_f1 = float("nan")
    else:
        ablated_test = ablate_neurons(
            test_acts,
            complement_indices,
            method="mean",
            dataset_mean=train_mean,
        )
        ablate_complement_acc = full_probe.score(ablated_test, test_labels)
        ablate_complement_auc, ablate_complement_f1 = _auc_f1(
            full_probe, ablated_test, test_labels
        )

    compression_ratio = (
        n_selected / n_total_features if n_total_features else float("nan")
    )

    return {
        "seed": int(seed),
        "fold_idx": int(fold.fold_idx),
        "n_selected": n_selected,
        "n_total_features": int(n_total_features),
        "compression_ratio": float(compression_ratio),
        # Baseline (full probe, unmodified test activations)
        "baseline_accuracy": float(baseline_acc),
        "baseline_auc_roc": float(baseline_auc),
        "baseline_f1": float(baseline_f1),
        # Smaller-model framing
        "smaller_model_accuracy": float(smaller_acc),
        "smaller_model_auc_roc": float(smaller_auc),
        "smaller_model_f1": float(smaller_f1),
        "smaller_model_drop": float(baseline_acc - smaller_acc),
        # Ablate-complement framing
        "ablate_complement_accuracy": float(ablate_complement_acc),
        "ablate_complement_auc_roc": float(ablate_complement_auc),
        "ablate_complement_f1": float(ablate_complement_f1),
        "ablate_complement_drop": float(baseline_acc - ablate_complement_acc),
        # Agreement between the two framings
        "smaller_vs_ablate_complement_diff": float(
            smaller_acc - ablate_complement_acc
        ),
        "n_train": int(len(train_labels)),
        "n_test": int(len(test_labels)),
    }


def _auc_f1(probe, activations: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Compute ROC-AUC and F1 for a fitted :class:`EvalProbe`."""
    try:
        probas = probe.predict_proba(activations)[:, 1]
        auc = float(roc_auc_score(labels, probas))
    except ValueError:
        auc = float("nan")
    preds = probe.predict(activations)
    f1 = float(f1_score(labels, preds, zero_division=0.0))
    return auc, f1


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_NUMERIC_KEYS = (
    "n_selected",
    "compression_ratio",
    "baseline_accuracy",
    "baseline_auc_roc",
    "baseline_f1",
    "smaller_model_accuracy",
    "smaller_model_auc_roc",
    "smaller_model_f1",
    "smaller_model_drop",
    "ablate_complement_accuracy",
    "ablate_complement_auc_roc",
    "ablate_complement_f1",
    "ablate_complement_drop",
    "smaller_vs_ablate_complement_diff",
)


def _aggregate(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean / std / min / max for each numeric key across folds."""
    if not fold_results:
        raise ValueError("_aggregate: No fold results to aggregate")

    agg: dict[str, Any] = {}
    for key in _NUMERIC_KEYS:
        values = [
            fr[key] for fr in fold_results
            if key in fr and not _is_nan(fr[key])
        ]
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        agg[f"{key}_mean"] = float(arr.mean())
        agg[f"{key}_std"] = float(arr.std())
        agg[f"{key}_min"] = float(arr.min())
        agg[f"{key}_max"] = float(arr.max())

    agg["n_folds"] = len(fold_results)
    return agg


def _is_nan(x: Any) -> bool:
    try:
        return bool(np.isnan(x))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Dataclass serialisation helper (kept for symmetry with exp_auc_comparison)
# ---------------------------------------------------------------------------

def _config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return asdict(config)
