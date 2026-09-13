"""Set operations: Jaccard similarity, core-neuron analysis."""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard index: |A n B| / |A u B|.  Returns 0 if both sets are empty."""
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def jaccard_matrix(
    neuron_sets: dict[str, set],
) -> tuple[pd.DataFrame, list[str]]:
    """Pairwise Jaccard similarity for all generator pairs."""
    names = sorted(neuron_sets.keys())
    n = len(names)
    mat = np.eye(n, dtype=np.float64)

    for i, j in combinations(range(n), 2):
        val = jaccard_similarity(neuron_sets[names[i]], neuron_sets[names[j]])
        mat[i, j] = val
        mat[j, i] = val

    return pd.DataFrame(mat, index=names, columns=names), names


def core_neurons(
    neuron_sets: dict[str, set],
    min_generators: int,
) -> set:
    """Neurons appearing in at least ``min_generators`` discriminative sets."""
    counts: Counter = Counter()
    for s in neuron_sets.values():
        counts.update(s)
    return {neuron for neuron, cnt in counts.items() if cnt >= min_generators}
