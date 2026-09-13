"""High-level data loading: activations + labels + metadata + subsampling.

Centralises the load-and-subsample pattern so it can be reused by CLI
scripts, notebooks, and tests without duplicating the logic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..experiments.config import ExperimentConfig
from .activations import concat_all_layers, load_activations
from .metadata import SampleMetadata, load_metadata


def load_experiment_data(
    config: ExperimentConfig,
    generator: str,
    *,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, SampleMetadata]:
    """Load activations, labels, and metadata for a generator.

    Applies balanced subsampling to ``config.max_samples`` if set.

    Args:
        config: Experiment configuration (uses ``activations_root``,
            ``max_samples``).
        generator: Generator slug (e.g. ``"gpt4"``).
        verbose: Print subsampling info.

    Returns:
        ``(activations, labels, metadata)`` tuple.
    """
    from raid_pipeline.raid_loader import slug

    results_root = Path(config.activations_root)
    results_path = results_root / f"activations_raid_{slug(generator)}"

    acts_dict, labels, _meta_json = load_activations(results_path)
    activations = concat_all_layers(acts_dict)
    metadata = load_metadata(results_path)

    n_total = len(labels)
    max_n = config.max_samples
    if max_n and max_n < n_total:
        activations, labels, metadata = _balanced_subsample(
            activations, labels, metadata, max_n,
        )
        if verbose:
            print(f"Subsampled {n_total} -> {len(labels)} samples (balanced)")

    return activations, labels, metadata


def _balanced_subsample(
    activations: np.ndarray,
    labels: np.ndarray,
    metadata: SampleMetadata,
    max_n: int,
) -> tuple[np.ndarray, np.ndarray, SampleMetadata]:
    """Subsample to *max_n* with equal class balance."""
    rng = np.random.RandomState(42)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    per_class = max_n // 2

    chosen_pos = rng.choice(pos_idx, size=min(per_class, len(pos_idx)), replace=False)
    chosen_neg = rng.choice(neg_idx, size=min(per_class, len(neg_idx)), replace=False)
    keep = np.sort(np.concatenate([chosen_pos, chosen_neg]))

    return activations[keep], labels[keep], metadata[keep]
