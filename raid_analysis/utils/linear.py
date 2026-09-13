"""Linear representation validation: mean difference vector and LR weight alignment."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def mean_difference_vector(
    activations: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Compute mu_AI - mu_human across all neurons."""
    ai_mask = labels == 1
    human_mask = labels == 0
    return activations[ai_mask].mean(axis=0) - activations[human_mask].mean(axis=0)


def lr_weight_vector(
    activations: np.ndarray,
    labels: np.ndarray,
    *,
    train_size: float = 0.8,
    max_iter: int = 1000,
    random_state: int = 42,
) -> tuple[np.ndarray, float]:
    """Train logistic regression and return (weight_vector, test_accuracy).

    The weight vector is un-standardised so it lives in the original
    activation space and is directly comparable to
    :func:`mean_difference_vector`.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        activations, labels,
        train_size=train_size,
        random_state=random_state,
        stratify=labels,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )
    lr.fit(X_train_scaled, y_train)
    test_accuracy = float(lr.score(X_test_scaled, y_test))

    weights_scaled = lr.coef_[0]
    weights_original = weights_scaled / scaler.scale_

    return weights_original, test_accuracy
