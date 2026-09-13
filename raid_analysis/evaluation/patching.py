"""PatchingEvaluator — activation patching for sufficiency / necessity testing.

Forward direction (default, sufficiency):
    Transplant the top-k selected neurons' activations from an AI sample into a
    human sample.  A flip occurs when the eval probe changes from human (0) → AI
    (1).  Tests whether AI-linked signal is *sufficient* to fool the probe.

Reverse direction (necessity via patching):
    Transplant the top-k selected neurons' activations from a human sample into
    an AI sample.  A flip occurs when the eval probe changes from AI (1) →
    human (0).  Tests whether replacing AI signal with human signal is enough to
    suppress detection.

For each k in ``k_values``, patches the top-k neurons (by selection ranking)
and compares the flip rate against patching k random neurons (baseline).

Pairs are formed within the same domain to control for domain-level artifacts.
Pairing is shuffled and repeated ``n_shuffles`` times to produce a distribution
of flip rates, avoiding dependence on any single arbitrary assignment.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ..data.activations import layer_neuron_to_global
from ..data.metadata import SampleMetadata
from .causal import patch_neurons
from ..selection.protocol import SelectionResult
from .probe_factory import EvalProbe
from .protocol import Evaluator


class PatchingEvaluator(Evaluator):
    """Evaluate sufficiency / necessity via same-domain activation patching.

    For each k in ``k_values``, patches the top-k neurons (by selection
    ranking) from a same-domain donor into a base sample.  Measures the
    fraction of patched samples where the eval probe flips its prediction.
    A random-k baseline (patching k random neurons) is computed for comparison.
    Pairings are shuffled ``n_shuffles`` times to produce a distribution.

    Args:
        k_values: List of neuron counts to patch.  If ``None``, defaults to
            ``[1, 2, 5, 10, 20, 50, 100, 200]``. Use ``-1`` as a sentinel to
            patch the full selected set (``k = len(selection.ranking)`` per
            fold). Duplicates and values exceeding the selected-set size are
            dropped; valid values preserve their YAML order.
        pairing: Pairing strategy.  Currently only ``"same_domain"``.
        n_shuffles: Number of random pairings to average over.
        n_random_draws: Number of random neuron draws for the baseline
            at each k.
        direction: ``"ai_into_human"`` (default) — AI donor activations
            transplanted into human base samples; flip is human→AI prediction.
            ``"human_into_ai"`` — human donor activations transplanted into AI
            base samples; flip is AI→human prediction.
    """

    def __init__(
        self,
        k_values: list[int] | None = None,
        pairing: str = "same_domain",
        n_shuffles: int = 20,
        n_random_draws: int = 20,
        direction: str = "ai_into_human",
    ) -> None:
        if pairing != "same_domain":
            raise ValueError(
                f"Unsupported pairing strategy: {pairing!r}; "
                f"only 'same_domain' is currently supported"
            )
        if direction not in ("ai_into_human", "human_into_ai"):
            raise ValueError(
                f"direction must be 'ai_into_human' or 'human_into_ai', got {direction!r}"
            )
        self.k_values = k_values or [1, 2, 5, 10, 20, 50, 100, 200]
        self.pairing = pairing
        self.n_shuffles = n_shuffles
        self.n_random_draws = n_random_draws
        self.direction = direction

    def __repr__(self) -> str:
        return (
            f"PatchingEvaluator(k_values={self.k_values}, "
            f"pairing={self.pairing!r}, n_shuffles={self.n_shuffles}, "
            f"direction={self.direction!r})"
        )

    def evaluate(
        self,
        test_activations: np.ndarray,
        test_labels: np.ndarray,
        test_metadata: SampleMetadata,
        selection: SelectionResult,
        eval_probe: EvalProbe,
    ) -> dict[str, Any]:
        ranked_global = [
            layer_neuron_to_global(layer, neuron)
            for layer, neuron in selection.ranking
        ]
        n_features = test_activations.shape[1]

        # Determine base and donor roles based on direction.
        # ai_into_human: human (0) is base, AI (1) is donor.  Flip: 0→1.
        # human_into_ai: AI (1) is base, human (0) is donor.  Flip: 1→0.
        base_label = 0 if self.direction == "ai_into_human" else 1
        donor_label = 1 - base_label

        base_indices = np.where(test_labels == base_label)[0]
        donor_indices = np.where(test_labels == donor_label)[0]

        donor_by_domain: dict[int, np.ndarray] = {}
        for d_id in np.unique(test_metadata.domain_ids[donor_indices]):
            mask = test_metadata.domain_ids[donor_indices] == d_id
            donor_by_domain[int(d_id)] = donor_indices[mask]

        valid_bases: list[int] = []
        base_domains: list[int] = []
        for b_idx in base_indices:
            d = int(test_metadata.domain_ids[b_idx])
            if d in donor_by_domain and len(donor_by_domain[d]) > 0:
                valid_bases.append(b_idx)
                base_domains.append(d)

        if not valid_bases:
            return self._empty_result(selection, len(test_labels))

        valid_bases_arr = np.array(valid_bases)
        base_domains_arr = np.array(base_domains)
        base_acts = test_activations[valid_bases_arr]
        base_preds = eval_probe.predict(base_acts)

        resolved_ks: list[int] = []
        seen: set[int] = set()
        for raw_k in self.k_values:
            k = len(ranked_global) if raw_k == -1 else raw_k
            if k <= 0 or k > len(ranked_global) or k in seen:
                continue
            seen.add(k)
            resolved_ks.append(k)

        patching_sweep: list[dict[str, Any]] = []

        for k in resolved_ks:

            top_k_indices = ranked_global[:k]

            # Selected top-k: shuffle pairings n_shuffles times
            selected_flip_rates = self._sweep_shuffles(
                test_activations, base_acts, base_preds,
                base_domains_arr, donor_by_domain,
                top_k_indices, eval_probe, base_label,
            )

            # Random-k baseline: for each random draw, sweep shuffles
            random_flip_rates: list[float] = []
            rng_neurons = np.random.RandomState(42)
            for _ in range(self.n_random_draws):
                rand_indices = rng_neurons.choice(
                    n_features, size=k, replace=False,
                ).tolist()
                rand_rates = self._sweep_shuffles(
                    test_activations, base_acts, base_preds,
                    base_domains_arr, donor_by_domain,
                    rand_indices, eval_probe, base_label,
                )
                random_flip_rates.append(float(np.mean(rand_rates)))

            patching_sweep.append({
                "k": k,
                "selected_flip_rate_mean": float(np.mean(selected_flip_rates)),
                "selected_flip_rate_std": float(np.std(selected_flip_rates)),
                "random_flip_rate_mean": float(np.mean(random_flip_rates)),
                "random_flip_rate_std": float(np.std(random_flip_rates)),
            })

        # Per-domain breakdown at the largest k actually used in the sweep
        max_k_used = max(p["k"] for p in patching_sweep) if patching_sweep else len(ranked_global)
        domain_neuron_indices = ranked_global[:max_k_used]
        domain_results = self._domain_breakdown(
            test_activations, base_acts, base_preds,
            base_domains, base_domains_arr, donor_by_domain,
            domain_neuron_indices, test_metadata, eval_probe, base_label,
        )

        return {
            "patching_sweep": patching_sweep,
            "flip_rate_by_domain": domain_results,
            "n_pairs_per_shuffle": len(valid_bases),
            "n_shuffles": self.n_shuffles,
            "n_random_draws": self.n_random_draws,
            "n_selected": selection.n_selected,
            "n_test": len(test_labels),
            "direction": self.direction,
        }

    def _generate_donor_indices(
        self,
        rng: np.random.RandomState,
        n_bases: int,
        base_domains_arr: np.ndarray,
        donor_by_domain: dict[int, np.ndarray],
    ) -> np.ndarray:
        """Assign a same-domain donor index to each base sample."""
        donor_idxs = np.empty(n_bases, dtype=int)
        for d_id, donor_pool in donor_by_domain.items():
            mask = base_domains_arr == d_id
            n_need = int(mask.sum())
            if n_need == 0:
                continue
            shuffled = rng.permutation(donor_pool)
            tiled = np.tile(shuffled, (n_need // len(shuffled)) + 1)[:n_need]
            donor_idxs[mask] = tiled
        return donor_idxs

    def _sweep_shuffles(
        self,
        test_activations: np.ndarray,
        base_acts: np.ndarray,
        base_preds: np.ndarray,
        base_domains_arr: np.ndarray,
        donor_by_domain: dict[int, np.ndarray],
        neuron_indices: list[int],
        eval_probe: EvalProbe,
        base_label: int,
    ) -> list[float]:
        """Run n_shuffles random pairings and return per-shuffle flip rates."""
        rng = np.random.RandomState(42)
        n_bases = len(base_acts)
        flip_rates: list[float] = []

        for _ in range(self.n_shuffles):
            donor_idxs = self._generate_donor_indices(
                rng, n_bases, base_domains_arr, donor_by_domain,
            )
            donor_acts = test_activations[donor_idxs]
            patched = patch_neurons(base_acts, donor_acts, neuron_indices)
            patched_preds = eval_probe.predict(patched)

            flips = (base_preds == base_label) & (patched_preds == 1 - base_label)
            flip_rates.append(float(flips.sum()) / n_bases)

        return flip_rates

    def _domain_breakdown(
        self,
        test_activations: np.ndarray,
        base_acts: np.ndarray,
        base_preds: np.ndarray,
        base_domains: list[int],
        base_domains_arr: np.ndarray,
        donor_by_domain: dict[int, np.ndarray],
        neuron_indices: list[int],
        test_metadata: SampleMetadata,
        eval_probe: EvalProbe,
        base_label: int,
    ) -> dict[str, dict[str, Any]]:
        """Per-domain flip rate at full selected set, averaged over shuffles."""
        rng = np.random.RandomState(42)
        n_bases = len(base_acts)
        per_shuffle_domain_flips: list[dict[int, list[bool]]] = []

        for _ in range(self.n_shuffles):
            donor_idxs = self._generate_donor_indices(
                rng, n_bases, base_domains_arr, donor_by_domain,
            )
            donor_acts = test_activations[donor_idxs]
            patched = patch_neurons(base_acts, donor_acts, neuron_indices)
            patched_preds = eval_probe.predict(patched)
            flips = (base_preds == base_label) & (patched_preds == 1 - base_label)

            domain_flips: dict[int, list[bool]] = defaultdict(list)
            for i, d in enumerate(base_domains):
                domain_flips[d].append(bool(flips[i]))
            per_shuffle_domain_flips.append(dict(domain_flips))

        all_domains = sorted(set(base_domains))
        result: dict[str, dict[str, Any]] = {}
        for d_id in all_domains:
            d_name = test_metadata.domain_names[d_id]
            rates: list[float] = []
            n_pairs = 0
            for shuffle_domains in per_shuffle_domain_flips:
                flip_list = shuffle_domains.get(d_id, [])
                if flip_list:
                    rates.append(sum(flip_list) / len(flip_list))
                    n_pairs = len(flip_list)
            result[d_name] = {
                "flip_rate_mean": float(np.mean(rates)) if rates else 0.0,
                "flip_rate_std": float(np.std(rates)) if rates else 0.0,
                "n_pairs_per_shuffle": n_pairs,
            }
        return result

    @staticmethod
    def _empty_result(
        selection: SelectionResult, n_test: int,
    ) -> dict[str, Any]:
        return {
            "patching_sweep": [],
            "flip_rate_by_domain": {},
            "n_pairs_per_shuffle": 0,
            "n_shuffles": 0,
            "n_random_draws": 0,
            "n_selected": selection.n_selected,
            "n_test": n_test,
        }
