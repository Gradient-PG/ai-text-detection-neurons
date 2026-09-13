"""Load selections from a completed source experiment.

Dependent experiments (ablation, patching, confound) reuse a prior
experiment's neuron selections.  This module loads them from the
per-fold directory layout produced by :func:`run_experiment`.
"""

from __future__ import annotations

from pathlib import Path

from ..data.splits import CVSplit
from ..selection.protocol import SelectionResult, load_selection


def load_source_selections(
    source_dir: Path,
    splits_by_seed: dict[int, list[CVSplit]],
) -> dict[tuple[int, int], SelectionResult]:
    """Load per-fold selections from a source experiment directory.

    Expects the directory layout produced by :func:`run_experiment`::

        source_dir/
          seed_{s}/
            fold_{f}/
              selection.json
              train_mean.npy

    Args:
        source_dir: Root of the source experiment output.
        splits_by_seed: The splits to iterate — only loads folds that exist.

    Returns:
        ``{(seed, fold_idx): SelectionResult}`` mapping.
    """
    selections: dict[tuple[int, int], SelectionResult] = {}

    for seed, folds in sorted(splits_by_seed.items()):
        for fold in folds:
            fold_dir = source_dir / f"seed_{seed}" / f"fold_{fold.fold_idx}"
            if not (fold_dir / "selection.json").exists():
                raise FileNotFoundError(
                    f"Selection not found at {fold_dir}. "
                    f"Did the source experiment run with seed={seed}, fold={fold.fold_idx}?"
                )
            selections[(seed, fold.fold_idx)] = load_selection(fold_dir)

    return selections
