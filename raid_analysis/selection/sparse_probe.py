"""SparseProbeSelector — L1 logistic regression for neuron selection.

When given multiple C values, performs an internal sweep with knee detection
to find the optimal regularisation strength.  A single-element list skips
the sweep and uses that C directly.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from ..data.activations import global_to_layer_neuron
from .protocol import NeuronSelector, SelectionResult


class SparseProbeSelector(NeuronSelector):
    """Select neurons via L1-penalised logistic regression.

    Args:
        C_values: C values to evaluate.  Multiple values trigger an internal
            train/val split, sweep, and knee detection.  A single value is
            used directly.
        solver: Sklearn solver supporting L1 (``"saga"`` or ``"liblinear"``).
        max_iter: Solver iteration limit.
        random_state: Seed for solver and internal split reproducibility.
            Overridden by the runner to the current CV seed.
        inner_val_fraction: Fraction held out for the internal validation
            split during the C sweep (default 0.2).
    """

    def __init__(
        self,
        *,
        C_values: list[float],
        solver: str = "liblinear",
        max_iter: int = 1000,
        random_state: int = 42,
        inner_val_fraction: float = 0.2,
    ) -> None:
        self.C_values = C_values
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.inner_val_fraction = inner_val_fraction

    def select(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        *,
        random_state: int | None = None,
    ) -> SelectionResult:
        seed = random_state if random_state is not None else self.random_state

        if len(self.C_values) > 1:
            best_C, sweep_curve, knee_idx = self._pick_best_C(
                activations, labels, seed,
            )
        else:
            best_C = self.C_values[0]
            sweep_curve = None
            knee_idx = None

        result = self._fit(activations, labels, best_C, seed)

        if sweep_curve is not None:
            result.metadata["sweep_curve"] = sweep_curve
            result.metadata["knee_idx"] = knee_idx
            result.metadata["knee_C"] = best_C
            result.metadata["C_values"] = self.C_values

        return result

    # ------------------------------------------------------------------

    def _fit(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        C: float,
        seed: int,
    ) -> SelectionResult:
        """Fit one L1 probe at the given C on the full training data."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(activations)

        lr = LogisticRegression(
            C=C,
            l1_ratio=1.0,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=seed,
        )
        lr.fit(X_scaled, labels)

        coef = lr.coef_[0]
        abs_coef = np.abs(coef)

        nonzero_global = np.where(abs_coef > 0)[0]
        neuron_indices: set[tuple[int, int]] = {
            global_to_layer_neuron(int(g)) for g in nonzero_global
        }

        ranked_global = np.argsort(-abs_coef)
        ranking: list[tuple[int, int]] = [
            global_to_layer_neuron(int(g))
            for g in ranked_global
            if abs_coef[g] > 0
        ]

        return SelectionResult(
            neuron_indices=neuron_indices,
            ranking=ranking,
            probe=lr,
            scaler=scaler,
            train_mean=activations.mean(axis=0),
            metadata={
                "C": C,
                "solver": self.solver,
                "n_nonzero": len(neuron_indices),
                "n_features": activations.shape[1],
                "train_accuracy": float(lr.score(X_scaled, labels)),
            },
        )

    def _pick_best_C(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        seed: int,
    ) -> tuple[float, list[dict], int]:
        """Sweep C values on an inner split and return (best_C, curve, knee_idx)."""
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.inner_val_fraction,
            random_state=seed,
        )
        train_idx, val_idx = next(splitter.split(activations, labels))

        scaler = StandardScaler()
        X_train = scaler.fit_transform(activations[train_idx])
        X_val = scaler.transform(activations[val_idx])
        y_train = labels[train_idx]
        y_val = labels[val_idx]

        sweep_curve: list[dict] = []
        for C in self.C_values:
            lr = LogisticRegression(
                C=C,
                l1_ratio=1.0,
                solver=self.solver,
                max_iter=self.max_iter,
                random_state=seed,
            )
            lr.fit(X_train, y_train)
            sweep_curve.append({
                "C": C,
                "val_accuracy": float(lr.score(X_val, y_val)),
                "n_nonzero": int(np.sum(np.abs(lr.coef_[0]) > 0)),
            })

        knee_idx = _find_knee(sweep_curve)
        return sweep_curve[knee_idx]["C"], sweep_curve, knee_idx

    def __repr__(self) -> str:
        return (
            f"SparseProbeSelector(C_values={self.C_values}, "
            f"solver={self.solver!r}, max_iter={self.max_iter})"
        )


# ---------------------------------------------------------------------------
# Knee detection
# ---------------------------------------------------------------------------

def _find_knee(sweep_curve: list[dict]) -> int:
    """Find the elbow in the accuracy-vs-sparsity curve.

    Uses the maximum-distance-to-line method: draw a line from the most
    sparse point to the least sparse point, and find the point with the
    greatest perpendicular distance from that line.

    Falls back to the midpoint if all points are collinear or there are
    fewer than 3 points.
    """
    if len(sweep_curve) < 3:
        return len(sweep_curve) // 2

    xs = np.array([p["n_nonzero"] for p in sweep_curve], dtype=np.float64)
    ys = np.array([p["val_accuracy"] for p in sweep_curve], dtype=np.float64)

    order = np.argsort(xs)
    xs_sorted = xs[order]
    ys_sorted = ys[order]

    p1 = np.array([xs_sorted[0], ys_sorted[0]])
    p2 = np.array([xs_sorted[-1], ys_sorted[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-12:
        return len(sweep_curve) // 2

    line_unit = line_vec / line_len

    distances = np.zeros(len(xs_sorted))
    for i in range(len(xs_sorted)):
        point = np.array([xs_sorted[i], ys_sorted[i]])
        proj = np.dot(point - p1, line_unit)
        closest = p1 + proj * line_unit
        distances[i] = np.linalg.norm(point - closest)

    best_sorted_idx = int(np.argmax(distances))
    return int(order[best_sorted_idx])
