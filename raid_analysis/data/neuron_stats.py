"""Core statistical pipeline: per-neuron Mann-Whitney U test + AUC effect size."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def compute_neuron_statistics(
    activations: np.ndarray, labels: np.ndarray
) -> pd.DataFrame:
    """
    Compute per-neuron Mann-Whitney U test and AUC effect size.

    Args:
        activations: (N_samples, N_neurons) array.
        labels: (N_samples,) array of 0/1 labels.

    Returns:
        DataFrame with columns: neuron_idx, ai_median, ai_q25, ai_q75,
        human_median, human_q25, human_q75, u_statistic, p_value, auc,
        auc_deviation, direction.
    """
    ai_mask = labels == 1
    human_mask = labels == 0
    n_neurons = activations.shape[1]

    results = []

    for neuron_idx in tqdm(range(n_neurons), desc="Testing neurons"):
        ai_acts = activations[ai_mask, neuron_idx]
        human_acts = activations[human_mask, neuron_idx]

        u_stat, p_value = stats.mannwhitneyu(
            ai_acts, human_acts, alternative="two-sided"
        )

        y_true = np.concatenate([np.ones(len(ai_acts)), np.zeros(len(human_acts))])
        y_scores = np.concatenate([ai_acts, human_acts])

        try:
            auc = roc_auc_score(y_true, y_scores)
        except Exception:
            auc = 0.5

        results.append(
            {
                "neuron_idx": neuron_idx,
                "ai_median": np.median(ai_acts),
                "ai_q25": np.percentile(ai_acts, 25),
                "ai_q75": np.percentile(ai_acts, 75),
                "human_median": np.median(human_acts),
                "human_q25": np.percentile(human_acts, 25),
                "human_q75": np.percentile(human_acts, 75),
                "u_statistic": u_stat,
                "p_value": p_value,
                "auc": auc,
                "auc_deviation": abs(auc - 0.5),
                "direction": "AI-preferring" if auc > 0.5 else "Human-preferring",
            }
        )

    return pd.DataFrame(results)


def identify_discriminative_neurons(
    stats_df: pd.DataFrame,
    alpha: float = 0.001,
    auc_threshold: float = 0.7,
    n_total_tests: int | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    Mark discriminative neurons via Bonferroni-corrected significance + AUC threshold.

    Adds columns ``significant``, ``strong_effect``, ``discriminative`` in-place.

    Args:
        stats_df: Per-neuron statistics from :func:`compute_neuron_statistics`.
        alpha: Family-wise error rate before correction (default 0.001).
        auc_threshold: AUC boundary (default 0.7 → AUC > 0.7 or AUC < 0.3).
        n_total_tests: Total number of hypothesis tests for Bonferroni correction.
            Pass ``N_BERT_NEURONS`` (9,216) to apply a single global correction
            across all 12 layers simultaneously, which is the statistically correct
            approach for BERT-base-uncased.  If ``None``, falls back to
            ``len(stats_df)`` (per-layer correction, less conservative).

    Returns:
        (stats_df, corrected_alpha).
    """
    n_tests = n_total_tests if n_total_tests is not None else len(stats_df)
    corrected_alpha = alpha / n_tests

    stats_df["significant"] = stats_df["p_value"] < corrected_alpha
    stats_df["strong_effect"] = (stats_df["auc"] > auc_threshold) | (
        stats_df["auc"] < (1 - auc_threshold)
    )
    stats_df["discriminative"] = stats_df["significant"] & stats_df["strong_effect"]

    return stats_df, corrected_alpha
