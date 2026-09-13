"""ConfoundEvaluator — confound correlation checks on held-out test data.

For each selected neuron, measures whether its activation values correlate
with known confounds (text length, domain) on the test fold.  Neurons whose
activation is better explained by a confound than by AI/human class should
be flagged for exclusion.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from ..data.metadata import SampleMetadata
from ..selection.protocol import SelectionResult
from .probe_factory import EvalProbe
from .protocol import Evaluator


class ConfoundEvaluator(Evaluator):
    """Evaluate confound correlations for selected neurons.

    Computes:
    - Spearman correlation between each selected neuron's activation and
      text length on the test fold.
    - One-way ANOVA F-statistic between each selected neuron's activation
      and domain membership on the test fold.

    Args:
        confounds: Which confound checks to run.  Subset of
            ``["text_length", "domain"]``.
        significance_threshold: p-value threshold for flagging a neuron.
    """

    def __init__(
        self,
        confounds: list[str] | None = None,
        significance_threshold: float = 0.05,
    ) -> None:
        self.confounds = confounds or ["text_length", "domain"]
        self.significance_threshold = significance_threshold

    def __repr__(self) -> str:
        return (
            f"ConfoundEvaluator(confounds={self.confounds}, "
            f"significance_threshold={self.significance_threshold})"
        )

    def evaluate(
        self,
        test_activations: np.ndarray,
        test_labels: np.ndarray,
        test_metadata: SampleMetadata,
        selection: SelectionResult,
        eval_probe: EvalProbe,
    ) -> dict[str, Any]:
        global_indices = {
            (layer - 1) * 768 + neuron: (layer, neuron)
            for layer, neuron in selection.neuron_indices
        }

        neuron_results: list[dict[str, Any]] = []
        n_flagged_length = 0
        n_flagged_domain = 0

        for g_idx, (layer, neuron) in sorted(global_indices.items()):
            acts = test_activations[:, g_idx]
            entry: dict[str, Any] = {
                "layer": layer,
                "neuron": neuron,
                "global_idx": g_idx,
            }

            flagged = False

            if "text_length" in self.confounds:
                rho, p = stats.spearmanr(acts, test_metadata.text_lengths)
                entry["text_length_rho"] = float(rho) if not np.isnan(rho) else 0.0
                entry["text_length_p"] = float(p) if not np.isnan(p) else 1.0
                if entry["text_length_p"] < self.significance_threshold:
                    n_flagged_length += 1
                    flagged = True

            if "domain" in self.confounds:
                unique_domains = np.unique(test_metadata.domain_ids)
                if len(unique_domains) > 1:
                    groups = [
                        acts[test_metadata.domain_ids == d]
                        for d in unique_domains
                    ]
                    groups = [g for g in groups if len(g) > 0]
                    if len(groups) > 1:
                        f_stat, p = stats.f_oneway(*groups)
                        entry["domain_f"] = float(f_stat) if not np.isnan(f_stat) else 0.0
                        entry["domain_p"] = float(p) if not np.isnan(p) else 1.0
                    else:
                        entry["domain_f"] = 0.0
                        entry["domain_p"] = 1.0
                else:
                    entry["domain_f"] = 0.0
                    entry["domain_p"] = 1.0

                if entry["domain_p"] < self.significance_threshold:
                    n_flagged_domain += 1
                    flagged = True

            entry["flagged"] = flagged
            neuron_results.append(entry)

        n_flagged = sum(1 for nr in neuron_results if nr["flagged"])

        return {
            "per_neuron": neuron_results,
            "n_selected": selection.n_selected,
            "n_flagged": n_flagged,
            "n_flagged_text_length": n_flagged_length,
            "n_flagged_domain": n_flagged_domain,
            "significance_threshold": self.significance_threshold,
            "n_test": len(test_labels),
        }
