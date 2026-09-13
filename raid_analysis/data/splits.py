"""CV split generation with label × domain stratification.

Splits are index arrays into the activation/label arrays. They are generated
once per experiment configuration and serialized so downstream experiments can
reuse the exact same partitions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold


@dataclass
class CVSplit:
    """A single train/test partition within a CV run."""

    fold_idx: int
    seed: int
    train_idx: np.ndarray
    test_idx: np.ndarray

    def __repr__(self) -> str:
        return (
            f"CVSplit(fold={self.fold_idx}, seed={self.seed}, "
            f"train={len(self.train_idx)}, test={len(self.test_idx)})"
        )


def _make_stratification_key(
    labels: np.ndarray,
    domain_ids: np.ndarray | None,
) -> np.ndarray:
    """Combine label and domain into a single stratification key.

    If *domain_ids* is provided, each unique (label, domain) pair becomes its
    own stratum.  This guarantees every fold has balanced domain representation
    within each class — required for same-domain patching pairs and confound
    checks.  Falls back to label-only stratification when domains are absent.
    """
    if domain_ids is None:
        return labels
    return np.array(
        [f"{int(l)}_{int(d)}" for l, d in zip(labels, domain_ids)]
    )


def generate_cv_splits(
    labels: np.ndarray,
    *,
    n_folds: int = 5,
    seed: int = 42,
    domain_ids: np.ndarray | None = None,
) -> list[CVSplit]:
    """Generate stratified K-fold splits.

    Stratification is by ``label × domain`` when *domain_ids* is provided,
    otherwise by label only.

    Args:
        labels: ``(N,)`` binary labels.
        n_folds: Number of CV folds.
        seed: Random seed for the fold assignment.
        domain_ids: ``(N,)`` integer-encoded domain labels (optional).

    Returns:
        List of *n_folds* :class:`CVSplit` objects.
    """
    strat_key = _make_stratification_key(labels, domain_ids)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    splits: list[CVSplit] = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(labels, strat_key)):
        splits.append(
            CVSplit(
                fold_idx=fold_idx,
                seed=seed,
                train_idx=train_idx.astype(np.int64),
                test_idx=test_idx.astype(np.int64),
            )
        )
    return splits


def generate_multi_seed_splits(
    labels: np.ndarray,
    *,
    n_folds: int = 5,
    seeds: Sequence[int] = (42, 123, 456, 789, 1024),
    domain_ids: np.ndarray | None = None,
) -> dict[int, list[CVSplit]]:
    """Generate CV splits for multiple seeds.

    Returns:
        ``{seed: [CVSplit, ...]}`` mapping.
    """
    return {
        s: generate_cv_splits(
            labels, n_folds=n_folds, seed=s, domain_ids=domain_ids
        )
        for s in seeds
    }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_splits(
    splits_by_seed: dict[int, list[CVSplit]],
    path: str | Path,
) -> None:
    """Serialize splits to a JSON file.

    Index arrays are stored as lists of ints for portability.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {}
    for seed, folds in splits_by_seed.items():
        payload[str(seed)] = [
            {
                "fold_idx": s.fold_idx,
                "seed": s.seed,
                "train_idx": s.train_idx.tolist(),
                "test_idx": s.test_idx.tolist(),
            }
            for s in folds
        ]

    with open(path, "w") as f:
        json.dump(payload, f)


def load_splits(path: str | Path) -> dict[int, list[CVSplit]]:
    """Deserialize splits from a JSON file produced by :func:`save_splits`."""
    path = Path(path)
    with open(path) as f:
        payload = json.load(f)

    result: dict[int, list[CVSplit]] = {}
    for seed_str, folds in payload.items():
        seed = int(seed_str)
        result[seed] = [
            CVSplit(
                fold_idx=d["fold_idx"],
                seed=d["seed"],
                train_idx=np.array(d["train_idx"], dtype=np.int64),
                test_idx=np.array(d["test_idx"], dtype=np.int64),
            )
            for d in folds
        ]
    return result
