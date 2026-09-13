"""Load raw activations from .npy files produced by the extraction pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..constants import ALL_LAYERS, N_NEURONS_PER_LAYER


def load_activations(
    results_path: str | Path,
) -> tuple[dict[int, np.ndarray], np.ndarray, dict[str, Any]]:
    """
    Load all per-layer activation arrays, labels, and metadata from disk.

    Returns:
        (activations_dict, labels, metadata) where activations_dict maps
        layer index -> (N_samples, 768) array.
    """
    results_path = Path(results_path)

    with open(results_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    activations = {}
    for layer in metadata["layers"]:
        layer_file = results_path / f"layer_{layer}_activations.npy"
        activations[layer] = np.load(layer_file)

    labels = np.load(results_path / "labels.npy")
    return activations, labels, metadata


def load_activations_for_model(
    model: str,
    results_root: str | Path,
) -> tuple[dict[int, np.ndarray], np.ndarray, dict[str, Any]]:
    """
    Convenience wrapper that resolves the ``activations_raid_{slug}`` path convention.

    Args:
        model: RAID model name (e.g. ``"gpt4"``, ``"mistral-chat"``).
        results_root: Parent directory containing ``activations_raid_*`` folders.
    """
    from raid_pipeline.raid_loader import slug

    results_root = Path(results_root)
    results_path = results_root / f"activations_raid_{slug(model)}"
    return load_activations(results_path)


def load_activation_column(
    results_path: Path, layer: int, neuron_idx: int
) -> np.ndarray:
    """Load a single neuron's activation vector across all samples."""
    acts = np.load(results_path / f"layer_{layer}_activations.npy")
    return acts[:, neuron_idx]


def build_full_neuron_matrix(
    neurons_df: pd.DataFrame, results_path: Path
) -> np.ndarray:
    """
    Build (n_all_neurons, n_samples) activation matrix for all 9,216 neurons.

    Row order matches neurons_df row order: layer 1 neurons 0-767, then layer 2, etc.
    """
    frames = []
    for layer in ALL_LAYERS:
        layer_acts = np.load(results_path / f"layer_{layer}_activations.npy")
        frames.append(layer_acts.T)
    return np.vstack(frames)


# ---------------------------------------------------------------------------
# Concatenated activations for experiment pipeline
# ---------------------------------------------------------------------------


def concat_all_layers(
    activations: dict[int, np.ndarray],
    layers: list[int] | None = None,
) -> np.ndarray:
    """Concatenate per-layer activations into a single ``(N, D)`` array.

    Layers are concatenated in ascending order.  With all 12 BERT layers the
    result has shape ``(N, 9216)``.  The column mapping is deterministic:

    * columns ``0..767``     → layer 1
    * columns ``768..1535``  → layer 2
    * ...
    * columns ``8448..9215`` → layer 12

    Args:
        activations: ``{layer: (N, 768)}`` dict from :func:`load_activations`.
        layers: Subset of layers to include (default: all available, sorted).

    Returns:
        ``(N, n_layers * 768)`` array.
    """
    if layers is None:
        layers = sorted(activations.keys())
    return np.concatenate([activations[l] for l in layers], axis=1)


def global_to_layer_neuron(global_idx: int) -> tuple[int, int]:
    """Convert a flat index in the concatenated array to ``(layer, neuron_idx)``.

    Layer numbering is 1-based (matching BERT convention and :data:`ALL_LAYERS`).

    >>> global_to_layer_neuron(0)
    (1, 0)
    >>> global_to_layer_neuron(768)
    (2, 0)
    >>> global_to_layer_neuron(9215)
    (12, 767)
    """
    layer_0based, neuron_idx = divmod(global_idx, N_NEURONS_PER_LAYER)
    return layer_0based + 1, neuron_idx


def layer_neuron_to_global(layer: int, neuron_idx: int) -> int:
    """Convert ``(layer, neuron_idx)`` to a flat index in the concatenated array.

    Inverse of :func:`global_to_layer_neuron`.

    >>> layer_neuron_to_global(1, 0)
    0
    >>> layer_neuron_to_global(12, 767)
    9215
    """
    return (layer - 1) * N_NEURONS_PER_LAYER + neuron_idx


def neuron_set_to_global_indices(
    neuron_set: set[tuple[int, int]],
) -> np.ndarray:
    """Convert a set of ``(layer, neuron_idx)`` tuples to sorted flat indices."""
    return np.array(
        sorted(layer_neuron_to_global(l, n) for l, n in neuron_set),
        dtype=np.int64,
    )


def global_indices_to_neuron_set(
    global_indices: np.ndarray,
) -> set[tuple[int, int]]:
    """Convert flat indices back to ``{(layer, neuron_idx), ...}`` set."""
    return {global_to_layer_neuron(int(idx)) for idx in global_indices}
