"""L2 evaluation probe factory (Step 2 of the fold loop).

The evaluation probe is an L2-regularised logistic regression trained on ALL
features.  It serves as a method-independent measurement instrument: it does
not know which neurons the selector chose.  Ablation / patching evaluators
score this probe on modified test activations to quantify causal impact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class EvalProbe:
    """A fitted L2 evaluation probe with its scaler.

    Use :meth:`score` for accuracy or :meth:`predict` / :meth:`predict_proba`
    for finer-grained evaluation.
    """

    lr: LogisticRegression
    scaler: StandardScaler

    def score(self, activations: np.ndarray, labels: np.ndarray) -> float:
        """Classification accuracy on (possibly ablated) activations."""
        X = self.scaler.transform(activations)
        return float(self.lr.score(X, labels))

    def predict(self, activations: np.ndarray) -> np.ndarray:
        """Predicted class labels."""
        X = self.scaler.transform(activations)
        return self.lr.predict(X)

    def predict_proba(self, activations: np.ndarray) -> np.ndarray:
        """Class probability estimates ``(N, 2)``."""
        X = self.scaler.transform(activations)
        return self.lr.predict_proba(X)


def train_eval_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    *,
    max_iter: int = 5000,
    C: float = 1.0,
    random_state: int = 42,
) -> EvalProbe:
    """Train an L2 logistic regression on all features.

    This is Step 2 of the experiment fold loop.  The probe is trained on the
    **train fold** and later scored on the **test fold** by evaluators.

    Args:
        activations: ``(N_train, D)`` — typically ``D = 9216`` (all layers).
        labels: ``(N_train,)`` binary labels.
        max_iter: Solver iteration limit.
        C: Inverse regularisation strength (L2).
        random_state: Seed for solver reproducibility.

    Returns:
        :class:`EvalProbe` ready for scoring.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(activations)

    lr = LogisticRegression(
        C=C,
        l1_ratio=0.0,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
    )
    lr.fit(X_scaled, labels)

    return EvalProbe(lr=lr, scaler=scaler)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_eval_probe(probe: EvalProbe, directory: str | Path) -> None:
    """Save the L2 probe weights and scaler statistics to *directory*.

    Saves ``l2_probe_weights.npy``, ``l2_scaler_mean.npy``,
    ``l2_scaler_scale.npy``.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "l2_probe_weights.npy", probe.lr.coef_)
    np.save(directory / "l2_probe_intercept.npy", probe.lr.intercept_)
    np.save(directory / "l2_scaler_mean.npy", probe.scaler.mean_)
    np.save(directory / "l2_scaler_scale.npy", probe.scaler.scale_)


def load_eval_probe(directory: str | Path) -> EvalProbe:
    """Reconstruct an :class:`EvalProbe` from saved weights.

    The restored probe can score activations identically to the original.
    """
    directory = Path(directory)

    coef = np.load(directory / "l2_probe_weights.npy")
    intercept = np.load(directory / "l2_probe_intercept.npy")
    mean = np.load(directory / "l2_scaler_mean.npy")
    scale = np.load(directory / "l2_scaler_scale.npy")

    n_features = coef.shape[1]

    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = scale
    scaler.var_ = scale ** 2
    scaler.n_features_in_ = n_features

    lr = LogisticRegression()
    lr.coef_ = coef
    lr.intercept_ = intercept
    lr.classes_ = np.array([0, 1])
    lr.n_features_in_ = n_features

    return EvalProbe(lr=lr, scaler=scaler)
