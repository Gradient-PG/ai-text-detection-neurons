#!/usr/bin/env python
"""
One qualitative worked example of activation patching.

Finds one human-written test sample that flips to "AI" under a patch of
neurons from a same-domain AI donor, using only:
  - already-cached activations (results/activations_raid_{generator}/)
  - already-saved patching/sparse_probe artifacts (probe weights + neuron
    sets) from the main experiment run
  - a deterministic replay of the RAID loading/shuffling steps to recover
    the source text for the flipped sample and its donor

By default (--neuron-set stable) this patches the aggregated **stable set
S*** — the same 45-62-neuron-per-generator quantity headlined in the
abstract / Table 1 (e.g. 62 for cohere-chat). Use --neuron-set fold to
instead patch the per-fold selected set for one specific seed/fold (a
different, larger quantity the paper separately reports as ranging 53-92
per fold) — do NOT conflate the two sizes in the write-up.

No new model forward pass is performed anywhere in this script (no
re-encoding): text recovery replays only data loading/shuffling (pandas +
HF `datasets` operations), which is deterministic under the fixed seed=42
used throughout the pipeline. This was verified separately: replaying the
loader for gpt4 reproduces the cached per-sample domain assignment with
100% agreement (9996/9996) against `sample_metadata.npz`.

Usage:
    uv run extra_analyses/experiments/exp_02_worked_example.py
    uv run extra_analyses/experiments/exp_02_worked_example.py --generator gpt4 --seed 42 --fold 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset, concatenate_datasets  # noqa: E402

from raid_analysis.data.activations import (  # noqa: E402
    concat_all_layers,
    layer_neuron_to_global,
    load_activations,
)
from raid_analysis.data.metadata import load_metadata  # noqa: E402
from raid_analysis.data.splits import load_splits  # noqa: E402
from raid_analysis.evaluation.causal import patch_neurons  # noqa: E402
from raid_analysis.evaluation.probe_factory import load_eval_probe  # noqa: E402
from raid_analysis.experiments.config import load_config  # noqa: E402
from raid_pipeline.raid_loader import RAIDConfig, load_raid, slug  # noqa: E402

RUN_ROOT = PROJECT_ROOT / "results" / "experiments" / "20260510_rerun_main"
PATCHING_RUN_DIR = RUN_ROOT / "patching"
SPARSE_PROBE_RUN_DIR = RUN_ROOT / "sparse_probe"
PATCHING_CONFIG_PATH = PROJECT_ROOT / "config" / "experiments" / "patching.yaml"
ACTIVATIONS_ROOT = PROJECT_ROOT / "results"


def load_neuron_set(neuron_set: str, generator: str, fold_dir: Path) -> tuple[list[tuple[int, int]], str]:
    """Return the (layer, neuron) list to patch, and a label describing it.

    - "stable": the aggregated stable set S* (>= stability_threshold across
      all folds x seeds of the sparse_probe experiment) — the quantity
      reported in the abstract / Table 1 (45-62 neurons per generator).
    - "fold": the per-fold selected set for this specific seed/fold — a
      different, correctly-defined quantity that the paper separately
      reports as ranging 53-92 per fold. NOT the same as S*; do not label
      its size as "the stable set" in the write-up.
    """
    if neuron_set == "stable":
        agg_path = SPARSE_PROBE_RUN_DIR / generator / "aggregate.json"
        aggregate = json.loads(agg_path.read_text())
        neurons = [tuple(p) for p in aggregate["stable_neurons"]]
        return neurons, "stable_set_S*"
    elif neuron_set == "fold":
        eval_metrics = json.loads((fold_dir / "eval_metrics.json").read_text())
        neurons = [tuple(p) for p in eval_metrics["neuron_indices"]]
        return neurons, "per_fold_selected_set"
    else:
        raise ValueError(f"Unknown neuron_set: {neuron_set!r}")


def reconstruct_source_dataset(generator: str, seed: int, max_samples: int) -> Dataset:
    """Replay the RAID loader + balanced-subsample shuffle used during
    activation extraction, recovering row-aligned raw text.

    Mirrors `raid_pipeline.activation_extractor.ActivationExtractor
    .extract_activations`'s dataset-reordering steps (filter by label ->
    shuffle(seed) -> select -> concatenate -> shuffle(seed)), applied
    directly to the text-bearing loader output. Tokenization (which strips
    `text`) does not reorder rows, so this reproduces the exact row order
    of the cached activation arrays with `text`/`domain` intact.
    """
    cfg = RAIDConfig(model=generator, domains=None, max_samples=max_samples, seed=seed)
    ds = load_raid(cfg)

    ai_samples = ds.filter(lambda x: x["label"] == 1)
    human_samples = ds.filter(lambda x: x["label"] == 0)
    per_class = max_samples // 2

    ai_subset = ai_samples.shuffle(seed=seed).select(range(min(per_class, len(ai_samples))))
    human_subset = human_samples.shuffle(seed=seed).select(range(min(per_class, len(human_samples))))
    return concatenate_datasets([ai_subset, human_subset]).shuffle(seed=seed)


def balanced_subsample_indices(labels: np.ndarray, max_n: int, seed: int = 42) -> np.ndarray:
    """Mirror `raid_analysis.data.loading._balanced_subsample`'s index choice
    exactly, so we can recover which of the full cached indices survive into
    the max_samples-subsampled array that `load_experiment_data` produces.
    """
    rng = np.random.RandomState(seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    per_class = max_n // 2
    chosen_pos = rng.choice(pos_idx, size=min(per_class, len(pos_idx)), replace=False)
    chosen_neg = rng.choice(neg_idx, size=min(per_class, len(neg_idx)), replace=False)
    return np.sort(np.concatenate([chosen_pos, chosen_neg]))


def find_one_flip(
    test_acts: np.ndarray,
    test_labels: np.ndarray,
    test_domain_ids: np.ndarray,
    ranked_global: list[int],
    probe,
    pairing_seed: int,
):
    """Same-domain AI-donor -> human-base patch of `ranked_global`, one
    deterministic pairing draw. Returns local (within test fold) indices of
    the first flipped base sample and its donor, or (None, None).
    """
    base_idx = np.where(test_labels == 0)[0]
    donor_idx = np.where(test_labels == 1)[0]

    donor_by_domain: dict[int, np.ndarray] = {}
    for d in np.unique(test_domain_ids[donor_idx]):
        donor_by_domain[int(d)] = donor_idx[test_domain_ids[donor_idx] == d]

    rng = np.random.RandomState(pairing_seed)
    donor_for_base = np.empty(len(base_idx), dtype=int)
    for d, pool in donor_by_domain.items():
        mask = test_domain_ids[base_idx] == d
        n_need = int(mask.sum())
        if n_need == 0:
            continue
        shuffled = rng.permutation(pool)
        tiled = np.tile(shuffled, (n_need // len(shuffled)) + 1)[:n_need]
        donor_for_base[mask] = tiled

    base_acts = test_acts[base_idx]
    donor_acts = test_acts[donor_for_base]

    base_preds = probe.predict(base_acts)
    base_proba = probe.predict_proba(base_acts)[:, 1]
    patched = patch_neurons(base_acts, donor_acts, ranked_global)
    patched_preds = probe.predict(patched)
    patched_proba = probe.predict_proba(patched)[:, 1]

    flips = (base_preds == 0) & (patched_preds == 1)
    flip_positions = np.where(flips)[0]
    if len(flip_positions) == 0:
        return None

    pos = int(flip_positions[0])
    return {
        "base_local_idx": int(base_idx[pos]),
        "donor_local_idx": int(donor_for_base[pos]),
        "pre_proba": float(base_proba[pos]),
        "post_proba": float(patched_proba[pos]),
        "n_flips_this_draw": int(flips.sum()),
        "n_bases_this_draw": int(len(base_idx)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", default="cohere-chat")
    parser.add_argument("--seed", type=int, default=42, help="CV seed (matches patching run)")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--pairing-seed", type=int, default=42,
        help="RNG seed for the illustrative same-domain donor pairing draw",
    )
    parser.add_argument(
        "--neuron-set", choices=["stable", "fold"], default="stable",
        help=(
            "'stable' (default): patch the aggregated stable set S* (the "
            "45-62-neuron quantity reported in the abstract/Table 1). "
            "'fold': patch the per-fold selected set for this seed/fold "
            "instead (a different, larger quantity - 53-92 per fold - not "
            "to be confused with S*)."
        ),
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: extra_analyses/results/exp_02_worked_example_<generator>_<neuron_set>.json)",
    )
    args = parser.parse_args()

    gen = args.generator
    gen_slug = slug(gen)

    config = load_config(PATCHING_CONFIG_PATH)

    fold_dir = PATCHING_RUN_DIR / gen / f"seed_{args.seed}" / f"fold_{args.fold}"
    neuron_indices_ln, neuron_set_label = load_neuron_set(args.neuron_set, gen, fold_dir)
    ranked_global = [layer_neuron_to_global(l, n) for l, n in neuron_indices_ln]

    activations_dir = ACTIVATIONS_ROOT / f"activations_raid_{gen_slug}"
    acts_dict, labels_full, _ = load_activations(activations_dir)
    activations_full = concat_all_layers(acts_dict)
    metadata_full = load_metadata(activations_dir)

    keep = balanced_subsample_indices(labels_full, config.max_samples, seed=42)
    activations = activations_full[keep]
    labels = labels_full[keep]
    metadata = metadata_full[keep]

    splits_path = PATCHING_RUN_DIR / gen / "splits.json"
    splits_by_seed = load_splits(splits_path)
    fold = next(f for f in splits_by_seed[args.seed] if f.fold_idx == args.fold)
    test_idx = fold.test_idx

    test_acts = activations[test_idx]
    test_labels = labels[test_idx]
    test_domain_ids = metadata.domain_ids[test_idx]
    test_idx_in_full = keep[test_idx]  # maps fold-local idx -> full 9996-cache idx

    probe = load_eval_probe(fold_dir)

    result = None
    for draw_seed in range(args.pairing_seed, args.pairing_seed + 50):
        result = find_one_flip(
            test_acts, test_labels, test_domain_ids, ranked_global, probe, draw_seed,
        )
        if result is not None:
            result["pairing_seed_used"] = draw_seed
            break

    if result is None:
        print(f"No flip found for {gen} seed={args.seed} fold={args.fold} "
              f"after 50 pairing draws. Try another --generator/--fold.")
        return

    target_full_idx = int(test_idx_in_full[result["base_local_idx"]])
    donor_full_idx = int(test_idx_in_full[result["donor_local_idx"]])

    ds = reconstruct_source_dataset(gen, seed=42, max_samples=10_000)
    target_row = ds[target_full_idx]
    donor_row = ds[donor_full_idx]

    assert target_row["label"] == 0, "target row should be human"
    assert donor_row["label"] == 1, "donor row should be AI"
    assert target_row["domain"] == donor_row["domain"], "pairing must be same-domain"

    layer_counts: dict[int, int] = {}
    for layer, _ in neuron_indices_ln:
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    top_layer = max(layer_counts, key=layer_counts.get)

    report = {
        "generator": gen,
        "seed": args.seed,
        "fold": args.fold,
        "pairing_seed_used": result["pairing_seed_used"],
        "neuron_set": neuron_set_label,
        "n_neurons_patched": len(neuron_indices_ln),
        "neuron_indices_layer_neuron": neuron_indices_ln,
        "layer_counts": layer_counts,
        "most_represented_layer": top_layer,
        "target_sample": {
            "domain": target_row["domain"],
            "label": "human",
            "source_model": target_row["source_model"],
            "text": target_row["text"],
            "text_char_len": len(target_row["text"]),
            "pre_patch_pred": "human",
            "pre_patch_proba_ai": result["pre_proba"],
            "post_patch_pred": "AI",
            "post_patch_proba_ai": result["post_proba"],
        },
        "donor_sample": {
            "domain": donor_row["domain"],
            "label": "AI",
            "source_model": donor_row["source_model"],
            "text": donor_row["text"],
            "text_char_len": len(donor_row["text"]),
        },
        "draw_diagnostics": {
            "n_flips_this_draw": result["n_flips_this_draw"],
            "n_bases_this_draw": result["n_bases_this_draw"],
        },
    }

    output_path = Path(args.output) if args.output else (
        PROJECT_ROOT / "extra_analyses" / "results" / f"exp_02_worked_example_{gen_slug}_{args.neuron_set}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))

    print(f"\nGenerator: {gen}  (seed={args.seed}, fold={args.fold}, pairing_seed={result['pairing_seed_used']})")
    print(f"Neuron set: {neuron_set_label} (n={len(neuron_indices_ln)})")
    print(f"Domain: {target_row['domain']}")
    print(f"Target (human) text length: {len(target_row['text'])} chars, first 200 chars:")
    print(f"  {target_row['text'][:200]!r}")
    print(f"Donor ({donor_row['source_model']}) text length: {len(donor_row['text'])} chars, first 200 chars:")
    print(f"  {donor_row['text'][:200]!r}")
    print(f"\nPatched {len(neuron_indices_ln)} neurons ({neuron_set_label}), concentrated in layer {top_layer} "
          f"({layer_counts[top_layer]}/{len(neuron_indices_ln)} of the patched set).")
    print(f"Probe P(AI): pre-patch={result['pre_proba']:.4f}  ->  post-patch={result['post_proba']:.4f}")
    print(f"\nFull report written to {output_path}")


if __name__ == "__main__":
    main()
