"""Causal validation primitives: ablation and activation patching."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np


def ablate_neurons(
    activations: np.ndarray,
    neuron_indices: Sequence[int],
    method: Literal["zero", "mean"] = "zero",
    dataset_mean: np.ndarray | None = None,
) -> np.ndarray:
    """
    Ablate specific neurons in an activation matrix (tests *necessity*).

    Args:
        activations: (N_samples, N_neurons) array.
        neuron_indices: Column indices to ablate.
        method: ``"zero"`` sets to 0; ``"mean"`` replaces with per-neuron dataset mean.
        dataset_mean: Precomputed (N_neurons,) mean vector.  **Required** when
            ``method="mean"`` — must be computed from the *training* fold to
            avoid test-set leakage.

    Returns:
        Modified copy of ``activations``.
    """
    out = activations.copy()
    idx = list(neuron_indices)

    if method == "zero":
        out[:, idx] = 0.0
    elif method == "mean":
        if dataset_mean is None:
            raise ValueError(
                "dataset_mean must be provided when method='mean'. "
                "Use selection.train_mean to avoid test-set leakage."
            )
        out[:, idx] = dataset_mean[idx]
    else:
        raise ValueError(f"Unknown ablation method: {method!r}; use 'zero' or 'mean'")

    return out


def patch_neurons(
    base_activations: np.ndarray,
    donor_activations: np.ndarray,
    neuron_indices: Sequence[int],
) -> np.ndarray:
    """
    Activation patching: swap specific neuron columns from donor into base (tests *sufficiency*).

    Both arrays must have the same shape ``(N_samples, N_neurons)``.

    Returns:
        Modified copy of ``base_activations`` with patched columns.
    """
    if base_activations.shape != donor_activations.shape:
        raise ValueError(
            f"Shape mismatch: base {base_activations.shape} vs donor {donor_activations.shape}"
        )
    out = base_activations.copy()
    idx = list(neuron_indices)
    out[:, idx] = donor_activations[:, idx]
    return out
