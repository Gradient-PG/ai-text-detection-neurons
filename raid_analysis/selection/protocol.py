"""NeuronSelector protocol and SelectionResult dataclass.

Every neuron selection method (sparse probe, AUC, IG, ...) implements the
:class:`NeuronSelector` protocol.  It receives train-fold data and returns a
:class:`SelectionResult`.  It never sees test data and never knows about folds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..data.activations import (
    global_indices_to_neuron_set,
    global_to_layer_neuron,
    layer_neuron_to_global,
    neuron_set_to_global_indices,
)


@dataclass
class SelectionResult:
    """Output of a neuron selection method.

    Attributes:
        neuron_indices: Canonical ``{(layer, neuron_idx), ...}`` set.
        ranking: Neurons ordered by importance (most important first).
            Each entry is ``(layer, neuron_idx)``.
        probe: The trained selection model, if the method produces one
            (e.g. the L1 LogReg).  Saved for diagnostics only — evaluation
            uses a separate L2 probe.
        scaler: The fitted scaler associated with the probe (e.g.
            ``StandardScaler``).  Needed to score the probe on new data.
        train_mean: ``(D,)`` mean activation vector computed on the training
            fold.  Used as the replacement value for mean ablation.
        metadata: Method-specific information (C value, AUC scores, etc.).
    """

    neuron_indices: set[tuple[int, int]]
    ranking: list[tuple[int, int]]
    probe: Any | None = None
    scaler: Any | None = None
    train_mean: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_selected(self) -> int:
        return len(self.neuron_indices)

    def __repr__(self) -> str:
        return (
            f"SelectionResult(n_selected={self.n_selected}, "
            f"ranked={len(self.ranking)}, "
            f"has_probe={self.probe is not None})"
        )


@runtime_checkable
class NeuronSelector(Protocol):
    """Protocol that every neuron selection method must implement.

    Implementations receive train-fold activations and labels and return
    a :class:`SelectionResult`.  The caller (the experiment runner) is
    responsible for providing only the train partition.
    """

    def select(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        *,
        random_state: int | None = None,
    ) -> SelectionResult:
        """Select neurons from *activations* ``(N_train, D)``.

        Args:
            activations: Concatenated activation array (all layers).
            labels: ``(N_train,)`` binary labels.
            random_state: Seed for reproducibility.  When called from the
                experiment runner, this is set to the current CV seed so that
                each fold gets an independently seeded selection.

        Returns:
            :class:`SelectionResult` with at least ``neuron_indices`` populated.
        """
        ...


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_selection(result: SelectionResult, directory: str | Path) -> None:
    """Persist a :class:`SelectionResult` to *directory*.

    Saves:
      * ``selection.json`` — neuron indices, ranking, metadata (JSON-safe)
      * ``train_mean.npy``  — if present
      * ``probe_weights.npy`` — if the probe exposes ``coef_`` (sklearn)
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    global_indices = neuron_set_to_global_indices(result.neuron_indices)
    ranking_global = [
        layer_neuron_to_global(l, n) for l, n in result.ranking
    ]

    payload = {
        "neuron_indices_global": global_indices.tolist(),
        "ranking_global": ranking_global,
        "n_selected": result.n_selected,
        "metadata": result.metadata,
    }
    with open(directory / "selection.json", "w") as f:
        json.dump(payload, f, indent=2)

    if result.train_mean is not None:
        np.save(directory / "train_mean.npy", result.train_mean)

    if result.probe is not None and hasattr(result.probe, "coef_"):
        np.save(directory / "probe_weights.npy", result.probe.coef_)


def load_selection(directory: str | Path) -> SelectionResult:
    """Load a :class:`SelectionResult` from *directory*.

    The ``probe`` field is not restored (it is method-specific and only saved
    for diagnostics via ``probe_weights.npy``).
    """
    directory = Path(directory)

    with open(directory / "selection.json") as f:
        payload = json.load(f)

    neuron_indices = global_indices_to_neuron_set(
        np.array(payload["neuron_indices_global"], dtype=np.int64)
    )

    ranking = [
        global_to_layer_neuron(g) for g in payload["ranking_global"]
    ]

    train_mean = None
    train_mean_path = directory / "train_mean.npy"
    if train_mean_path.exists():
        train_mean = np.load(train_mean_path)

    return SelectionResult(
        neuron_indices=neuron_indices,
        ranking=ranking,
        probe=None,
        train_mean=train_mean,
        metadata=payload.get("metadata", {}),
    )
