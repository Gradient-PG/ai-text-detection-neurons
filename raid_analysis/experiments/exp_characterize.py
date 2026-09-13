"""Experiment 5 — Characterize the validated neuron set.

No fold loop — operates on the aggregate stable neuron sets from Experiment 1.
Computes layer distribution, cross-generator Jaccard overlap, core neuron set,
and linear direction alignment (CAV) with multi-seed variance estimation.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.set_ops import core_neurons, jaccard_matrix
from ..utils.linear import lr_weight_vector, mean_difference_vector
from .config import ExperimentConfig, save_config

_DEFAULT_CAV_SEEDS = [42, 123, 456]


def run_characterize_experiment(
    stable_sets: dict[str, set[tuple[int, int]]],
    activations_by_generator: dict[str, np.ndarray],
    labels_by_generator: dict[str, np.ndarray],
    config: ExperimentConfig,
    *,
    output_dir: Path | None = None,
    cav_seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Characterize validated neuron sets across generators.

    Args:
        stable_sets: ``{generator: {(layer, neuron_idx), ...}}`` stable neuron
            sets from Experiment 1, one per generator.
        activations_by_generator: ``{generator: (N, D)}`` concatenated
            activations for computing linear directions.
        labels_by_generator: ``{generator: (N,)}`` binary labels.
        config: Experiment configuration.
        output_dir: If provided, persist results here.
        cav_seeds: Random seeds for the LR train/test split used in CAV
            alignment scoring.  Defaults to ``[42, 123, 456]``.  Each seed
            produces one independent estimate; results are reported as
            mean ± std across seeds.

    Returns:
        Dict with layer distributions, Jaccard matrix, core neurons, and
        CAV alignment scores (mean and std over seeds).
    """
    if cav_seeds is None:
        cav_seeds = _DEFAULT_CAV_SEEDS

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, output_dir / "config.yaml")

    generators = sorted(stable_sets.keys())
    print(f"Characterize experiment: {len(generators)} generators")
    print(f"CAV seeds: {cav_seeds}")

    # Layer distribution per generator
    layer_distributions: dict[str, dict[int, int]] = {}
    for gen in generators:
        counts: Counter[int] = Counter()
        for layer, _ in stable_sets[gen]:
            counts[layer] += 1
        layer_distributions[gen] = dict(sorted(counts.items()))

    # Cross-generator Jaccard matrix
    jaccard_df, names = jaccard_matrix(stable_sets)
    jaccard_data = {
        "matrix": jaccard_df.values.tolist(),
        "generators": names,
    }

    # Core neurons
    min_k = max(2, len(generators) // 2)
    core_set = core_neurons(stable_sets, min_generators=min_k)
    core_list = sorted([list(t) for t in core_set])

    # Linear direction alignment (CAV) — multi-seed variance estimation
    # mean_diff is fully deterministic (no randomness); only lr_weight_vector
    # varies with the random train/test split and LR initialisation.
    cav_scores: dict[str, dict[str, Any]] = {}
    for gen in generators:
        if gen not in activations_by_generator:
            continue
        acts = activations_by_generator[gen]
        lbls = labels_by_generator[gen]

        # Deterministic: computed once, same for every seed
        mean_diff = mean_difference_vector(acts, lbls)
        mean_diff_norm = float(np.linalg.norm(mean_diff))

        seed_cosines: list[float] = []
        seed_lr_accs: list[float] = []
        seed_lr_norms: list[float] = []

        for seed in cav_seeds:
            lr_weights, lr_acc = lr_weight_vector(acts, lbls, random_state=seed)
            lr_norm = float(np.linalg.norm(lr_weights))

            if mean_diff_norm > 0 and lr_norm > 0:
                cosine = float(
                    np.dot(mean_diff, lr_weights) / (mean_diff_norm * lr_norm)
                )
            else:
                cosine = 0.0

            seed_cosines.append(cosine)
            seed_lr_accs.append(float(lr_acc))
            seed_lr_norms.append(lr_norm)

        cav_scores[gen] = {
            # Per-seed raw values for reproducibility
            "seed_cosine_similarities": seed_cosines,
            "seed_lr_accuracies": seed_lr_accs,
            "seed_lr_norms": seed_lr_norms,
            # Aggregated statistics
            "cosine_similarity_mean": float(np.mean(seed_cosines)),
            "cosine_similarity_std": float(np.std(seed_cosines, ddof=1) if len(seed_cosines) > 1 else 0.0),
            "lr_accuracy_mean": float(np.mean(seed_lr_accs)),
            "lr_accuracy_std": float(np.std(seed_lr_accs, ddof=1) if len(seed_lr_accs) > 1 else 0.0),
            "lr_norm_mean": float(np.mean(seed_lr_norms)),
            "lr_norm_std": float(np.std(seed_lr_norms, ddof=1) if len(seed_lr_norms) > 1 else 0.0),
            # mean_diff_norm is deterministic — no variance
            "mean_diff_norm": mean_diff_norm,
            "n_cav_seeds": len(cav_seeds),
            # Backward-compatible aliases (mean values)
            "cosine_similarity": float(np.mean(seed_cosines)),
            "lr_accuracy": float(np.mean(seed_lr_accs)),
            "lr_norm": float(np.mean(seed_lr_norms)),
        }

    result = {
        "layer_distributions": layer_distributions,
        "jaccard": jaccard_data,
        "core_neurons": core_list,
        "core_min_generators": min_k,
        "n_core": len(core_set),
        "cav_scores": cav_scores,
        "cav_seeds": cav_seeds,
        "n_generators": len(generators),
        "per_generator_n_stable": {
            gen: len(stable_sets[gen]) for gen in generators
        },
    }

    if output_dir is not None:
        with open(output_dir / "characterize_results.json", "w") as f:
            json.dump(result, f, indent=2)

    # Print a concise summary of CAV scores
    print("\nCAV alignment summary (mean ± std across seeds):")
    for gen in generators:
        if gen in cav_scores:
            s = cav_scores[gen]
            print(
                f"  {gen:15s}  cosine={s['cosine_similarity_mean']:.3f}±{s['cosine_similarity_std']:.3f}"
                f"  lr_acc={s['lr_accuracy_mean']:.3f}±{s['lr_accuracy_std']:.3f}"
                f"  ‖Δμ‖={s['mean_diff_norm']:.2f}"
                f"  ‖w‖={s['lr_norm_mean']:.2f}±{s['lr_norm_std']:.2f}"
            )

    return result
