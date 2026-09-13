#!/usr/bin/env python
"""
Leave-One-Generator-Out (LOGO) experiment.

Two modes (see --mode):

**generator** — For each generator G: train on all other generators, test on G.
  Risk: base/chat siblings (e.g. mistral-chat in train when holding out mistral)
  inflate transfer scores.

**family** — Partition RAID models into coarse families; for each family F:
  train on all generators not in F, test on every generator in F (pooled).
  Avoids training on mistral-chat while evaluating mistral, etc.
  Families: cohere (cohere, cohere-chat), gpt (chatgpt, gpt2, gpt3, gpt4),
  llama (llama-chat), mistral (mistral, mistral-chat), mpt (mpt, mpt-chat).

Each fold:
  1. Load activations for all generators in scope, capped at max_samples each.
  2. Build train/test masks from the held-out unit (generator or family).
  3. Per seed: L1 selection on train; full L2 + selected L2 baselines; score on test.
  4. Aggregate seeds; optional Jaccard vs sparse_probe reference (--source-dir).

Usage:
    uv run scripts/experiments/run_logo.py --mode family
    uv run scripts/experiments/run_logo.py --mode generator
    uv run scripts/experiments/run_logo.py --mode family --generators ...

Output:
    results/experiments/<run_id>/logo/
        summary.json
        <held_out_slug>/aggregate.json   # slug = generator name or family_<name>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

import yaml
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from raid_analysis.data.activations import neuron_set_to_global_indices
from raid_analysis.data.loading import load_experiment_data
from raid_analysis.evaluation.probe_factory import EvalProbe, train_eval_probe
from raid_analysis.experiments.config import ExperimentConfig, load_config
from raid_analysis.experiments.factory import build_selector
from raid_pipeline.raid_loader import ALL_RAID_MODELS

# Coarse generator families for cross-family LOGO (no shared base weights in train
# when evaluating a held-out base/chat pair).
RAID_GENERATOR_FAMILIES: dict[str, tuple[str, ...]] = {
    "cohere": ("cohere", "cohere-chat"),
    "gpt": ("chatgpt", "gpt2", "gpt3", "gpt4"),
    "llama": ("llama-chat",),
    "mistral": ("mistral", "mistral-chat"),
    "mpt": ("mpt", "mpt-chat"),
}


def _generator_to_family() -> dict[str, str]:
    m: dict[str, str] = {}
    for fam, members in RAID_GENERATOR_FAMILIES.items():
        for g in members:
            m[g] = fam
    return m


def _complete_families(generators: list[str]) -> list[str]:
    """Families where every member appears in *generators* (order: RAID table order)."""
    gen_set = set(generators)
    ordered: list[str] = []
    for fam in RAID_GENERATOR_FAMILIES:
        members = RAID_GENERATOR_FAMILIES[fam]
        if gen_set.issuperset(set(members)):
            ordered.append(fam)
    return ordered


def _load_logo_mode(config_path: Path, cli_mode: str | None) -> str:
    """Resolve logo mode: CLI wins; else logo_mode from YAML; else generator."""
    if cli_mode is not None:
        return cli_mode
    if not config_path.exists():
        return "generator"
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    mode = raw.get("logo_mode", "generator")
    if mode not in ("generator", "family"):
        raise ValueError(f"logo_mode must be 'generator' or 'family', got {mode!r}")
    return mode


# ---------------------------------------------------------------------------
# Selected-features probe helper
# ---------------------------------------------------------------------------

def _train_selected_probe(
    train_acts_selected: np.ndarray,
    train_labels: np.ndarray,
    *,
    C: float = 1.0,
    max_iter: int = 5000,
    random_state: int = 42,
) -> EvalProbe:
    """Train an L2 probe on a pre-sliced selected-feature matrix."""
    return train_eval_probe(
        train_acts_selected,
        train_labels,
        C=C,
        max_iter=max_iter,
        random_state=random_state,
    )


def _score_probe(
    probe: EvalProbe,
    acts: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    accuracy = probe.score(acts, labels)
    probas = probe.predict_proba(acts)[:, 1]
    try:
        auc = float(roc_auc_score(labels, probas))
    except ValueError:
        auc = float("nan")
    preds = probe.predict(acts)
    f1 = float(f1_score(labels, preds, zero_division=0.0))
    return {"accuracy": accuracy, "auc_roc": auc, "f1": f1}


# ---------------------------------------------------------------------------
# Reference stable set loading (optional Jaccard)
# ---------------------------------------------------------------------------

def _load_reference_stable_sets(source_dir: Path) -> dict[str, set[tuple[int, int]]]:
    """Load per-generator stable neuron sets from a sparse_probe output directory.

    Expects the layout:  <source_dir>/<generator>/aggregate.json
    Returns ``{generator: stable_set}``.
    """
    stable_sets: dict[str, set[tuple[int, int]]] = {}
    for gen_dir in sorted(source_dir.iterdir()):
        agg = gen_dir / "aggregate.json"
        if not agg.exists():
            continue
        with open(agg) as f:
            data = json.load(f)
        stable_sets[gen_dir.name] = {
            tuple(pair) for pair in data.get("stable_neurons", [])
        }
    return stable_sets


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# Single LOGO fold (generator- or family-held-out; masks define train/test)
# ---------------------------------------------------------------------------

def _run_logo_fold(
    held_out_slug: str,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    all_acts: np.ndarray,
    all_labels: np.ndarray,
    config: ExperimentConfig,
    output_dir: Path,
    reference_stable: set[tuple[int, int]] | None,
    *,
    held_out_kind: str,
    held_out_members: list[str],
) -> dict[str, Any]:
    """Run one LOGO fold; train/test partitions come from boolean masks."""

    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())
    if n_train == 0 or n_test == 0:
        raise SystemExit(
            f"Empty train or test split for {held_out_slug!r} "
            f"(train={n_train}, test={n_test}). "
            "Family mode needs at least two disjoint families in the generator list; "
            "generator mode needs at least two generators."
        )

    train_acts = all_acts[train_mask]
    train_labels = all_labels[train_mask]
    test_acts = all_acts[test_mask]
    test_labels = all_labels[test_mask]

    members_str = ", ".join(held_out_members)
    print(f"\n{'='*60}")
    print(
        f"Held-out ({held_out_kind}): {held_out_slug}  [{members_str}]"
        f"  |  train={n_train}  test={n_test}"
    )
    print(f"{'='*60}")

    selector = build_selector(config)

    seed_results: list[dict[str, Any]] = []
    all_selected_sets: list[set[tuple[int, int]]] = []

    for seed in config.seeds:
        t0 = time.time()

        # Step 1: neuron selection on train pool
        selection = selector.select(train_acts, train_labels, random_state=seed)
        selected_cols = neuron_set_to_global_indices(selection.neuron_indices)

        # Step 2a: Full L2 probe (all features) — upper-bound baseline
        full_probe = train_eval_probe(
            train_acts, train_labels,
            C=config.eval_probe_C,
            max_iter=config.eval_probe_max_iter,
            random_state=seed,
        )
        full_metrics = _score_probe(full_probe, test_acts, test_labels)

        # Step 2b: Selected L2 probe (selected features only) — main result
        if len(selected_cols) > 0:
            selected_probe = _train_selected_probe(
                train_acts[:, selected_cols], train_labels,
                C=config.eval_probe_C,
                max_iter=config.eval_probe_max_iter,
                random_state=seed,
            )
            selected_metrics = _score_probe(
                selected_probe, test_acts[:, selected_cols], test_labels,
            )
        else:
            selected_metrics = {"accuracy": float("nan"), "auc_roc": float("nan"), "f1": float("nan")}

        elapsed = time.time() - t0
        print(
            f"  seed={seed}  n_selected={selection.n_selected}"
            f"  full_acc={full_metrics['accuracy']:.4f}"
            f"  sel_acc={selected_metrics['accuracy']:.4f}"
            f"  ({elapsed:.1f}s)"
        )

        seed_results.append({
            "seed": seed,
            "n_selected": selection.n_selected,
            "full_l2": full_metrics,
            "selected_l2": selected_metrics,
        })
        all_selected_sets.append(selection.neuron_indices)

    # Aggregate across seeds
    def _mean_std(vals: list[float]) -> tuple[float, float]:
        arr = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        if not arr:
            return float("nan"), float("nan")
        return float(np.mean(arr)), float(np.std(arr))

    def _agg_metric(key: str, sub: str) -> dict[str, float]:
        vals = [r[key][sub] for r in seed_results]
        m, s = _mean_std(vals)
        return {f"{key}_{sub}_mean": m, f"{key}_{sub}_std": s}

    aggregate: dict[str, Any] = {
        "held_out_slug": held_out_slug,
        "held_out_kind": held_out_kind,
        "held_out_members": held_out_members,
    }
    for key in ("full_l2", "selected_l2"):
        for sub in ("accuracy", "auc_roc", "f1"):
            aggregate.update(_agg_metric(key, sub))

    n_sel_vals = [r["n_selected"] for r in seed_results]
    aggregate["n_selected_mean"], aggregate["n_selected_std"] = _mean_std(n_sel_vals)

    # Stable neurons (appear in ≥ stability_threshold of seeds)
    counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    for ns in all_selected_sets:
        for n in ns:
            counts[n] += 1
    total_seeds = len(config.seeds)
    stable = {n for n, c in counts.items() if c / total_seeds >= config.stability_threshold}
    aggregate["n_stable_neurons"] = len(stable)
    aggregate["stable_neurons"] = sorted([list(t) for t in stable])

    # Optional Jaccard vs reference
    if reference_stable is not None:
        aggregate["jaccard_vs_reference"] = _jaccard(stable, reference_stable)

    aggregate["n_train"] = n_train
    aggregate["n_test"] = n_test
    aggregate["n_seeds"] = total_seeds
    aggregate["seed_results"] = seed_results

    # Save
    gen_dir = output_dir / held_out_slug
    gen_dir.mkdir(parents=True, exist_ok=True)
    with open(gen_dir / "aggregate.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    return aggregate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    parser = argparse.ArgumentParser(
        description="Leave-One-Generator-Out (LOGO) experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str,
        default=str(project_root / "config" / "experiments" / "logo.yaml"),
        help="Path to YAML config (default: config/experiments/logo.yaml).",
    )
    parser.add_argument(
        "--mode",
        choices=["generator", "family"],
        default=None,
        help=(
            "generator: leave-one-generator-out (default unless logo_mode in YAML). "
            "family: leave-one-family-out — hold out entire base/chat or GPT groups "
            "(no sibling in train; recommended for paper)."
        ),
    )
    parser.add_argument(
        "--generators", nargs="+", default=None,
        help="Generators to include (default: all in config, or ALL_RAID_MODELS).",
    )
    parser.add_argument(
        "--source-dir", type=str, default=None,
        help=(
            "Path to sparse_probe output directory for Jaccard comparison. "
            "Expected layout: <source_dir>/<generator>/aggregate.json"
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory override.",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Run identifier (auto-generated timestamp if not provided).",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
    else:
        print(f"Config not found at {config_path}, using defaults.")
        config = ExperimentConfig(experiment="logo")
    config.experiment = "logo"

    logo_mode = _load_logo_mode(config_path, args.mode)

    generators = args.generators or config.generators or ALL_RAID_MODELS
    g2f = _generator_to_family()
    unknown = [g for g in generators if g not in g2f]
    if unknown:
        raise SystemExit(
            f"Generators not in RAID_GENERATOR_FAMILIES: {unknown}. "
            "Extend RAID_GENERATOR_FAMILIES in run_logo.py if RAID gained models."
        )
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(config.output_dir) / run_id / "logo"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("LOGO experiment")
    print(f"Mode: {logo_mode}")
    print(f"Generators ({len(generators)}): {', '.join(generators)}")
    print(f"Seeds: {config.seeds}")
    print(f"Max samples per generator: {config.max_samples}")
    print(f"Stability threshold: {config.stability_threshold}")
    print(f"Run ID: {run_id}")
    print(f"Output: {output_dir}")

    # Load reference stable sets (optional Jaccard)
    reference_stable_sets: dict[str, set[tuple[int, int]]] = {}
    if args.source_dir:
        source_dir = Path(args.source_dir)
        reference_stable_sets = _load_reference_stable_sets(source_dir)
        union_stable: set[tuple[int, int]] = set()
        for s in reference_stable_sets.values():
            union_stable |= s
        print(f"Reference stable sets loaded: {len(reference_stable_sets)} generators")
        print(f"Union reference stable set: {len(union_stable)} neurons")
    else:
        union_stable = set()

    # Load all generators' activations
    print("\nLoading activations...")
    all_acts_list: list[np.ndarray] = []
    all_labels_list: list[np.ndarray] = []
    generator_ids_list: list[np.ndarray] = []
    gen_id_map: dict[str, int] = {}

    for gen_id, gen in enumerate(generators):
        print(f"  Loading {gen}...", end=" ", flush=True)
        acts, labels, _ = load_experiment_data(config, gen, verbose=False)
        print(f"{len(labels)} samples")
        all_acts_list.append(acts)
        all_labels_list.append(labels)
        generator_ids_list.append(np.full(len(labels), gen_id, dtype=np.int32))
        gen_id_map[gen] = gen_id

    all_acts = np.concatenate(all_acts_list, axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)
    generator_ids = np.concatenate(generator_ids_list, axis=0)

    print(f"\nPooled dataset: {len(all_labels)} samples, {all_acts.shape[1]} features")

    # Run LOGO folds
    t_total = time.time()
    all_aggregates: list[dict[str, Any]] = []

    if logo_mode == "generator":
        for gen in generators:
            hid = gen_id_map[gen]
            train_mask = generator_ids != hid
            test_mask = generator_ids == hid
            reference = union_stable if union_stable else None
            agg = _run_logo_fold(
                held_out_slug=gen,
                train_mask=train_mask,
                test_mask=test_mask,
                all_acts=all_acts,
                all_labels=all_labels,
                config=config,
                output_dir=output_dir,
                reference_stable=reference,
                held_out_kind="generator",
                held_out_members=[gen],
            )
            all_aggregates.append(agg)
    else:
        complete_families = _complete_families(generators)
        if not complete_families:
            raise SystemExit(
                "Family mode: no complete families in the generator list. "
                "Every member of a RAID_GENERATOR_FAMILIES group must be included "
                "(e.g. both mistral and mistral-chat)."
            )
        print(f"Complete families in scope ({len(complete_families)}): {complete_families}")
        for fam in complete_families:
            members = list(RAID_GENERATOR_FAMILIES[fam])
            test_mask = np.zeros(len(all_labels), dtype=bool)
            for g in members:
                test_mask |= generator_ids == gen_id_map[g]
            train_mask = ~test_mask
            ref_fold: set[tuple[int, int]] | None = None
            if reference_stable_sets:
                ref_fold = set()
                for g in members:
                    ref_fold |= reference_stable_sets.get(g, set())
                if not ref_fold:
                    ref_fold = None
            agg = _run_logo_fold(
                held_out_slug=f"family_{fam}",
                train_mask=train_mask,
                test_mask=test_mask,
                all_acts=all_acts,
                all_labels=all_labels,
                config=config,
                output_dir=output_dir,
                reference_stable=ref_fold,
                held_out_kind="family",
                held_out_members=members,
            )
            all_aggregates.append(agg)

    # Summary across folds
    summary_rows = []
    for agg in all_aggregates:
        row: dict[str, Any] = {
            "held_out": agg["held_out_slug"],
            "held_out_kind": agg["held_out_kind"],
            "held_out_members": agg["held_out_members"],
            "n_test": agg["n_test"],
            "full_l2_accuracy_mean": agg.get("full_l2_accuracy_mean"),
            "selected_l2_accuracy_mean": agg.get("selected_l2_accuracy_mean"),
            "full_l2_auc_roc_mean": agg.get("full_l2_auc_roc_mean"),
            "selected_l2_auc_roc_mean": agg.get("selected_l2_auc_roc_mean"),
            "n_stable_neurons": agg.get("n_stable_neurons"),
            "n_selected_mean": agg.get("n_selected_mean"),
        }
        if "jaccard_vs_reference" in agg:
            row["jaccard_vs_reference"] = agg["jaccard_vs_reference"]
        summary_rows.append(row)

    elapsed_total = time.time() - t_total
    summary = {
        "run_id": run_id,
        "logo_mode": logo_mode,
        "generators": generators,
        "n_generators": len(generators),
        "elapsed_seconds": elapsed_total,
        "results": summary_rows,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print table
    print(f"\n{'='*70}")
    col_w = 26
    print(f"{'Held-out':<{col_w}} {'Test N':>7} {'Full L2':>9} {'Sel L2':>9} {'Stable':>8}")
    print(f"{'-'*(col_w + 7 + 9 + 9 + 8 + 1)}")
    for row in summary_rows:
        full_acc = row.get("full_l2_accuracy_mean")
        sel_acc = row.get("selected_l2_accuracy_mean")
        stable = row.get("n_stable_neurons")
        full_str = f"{full_acc:>9.4f}" if full_acc is not None else f"{'N/A':>9}"
        sel_str = f"{sel_acc:>9.4f}" if sel_acc is not None else f"{'N/A':>9}"
        label = row["held_out"]
        print(f"{label:<{col_w}} {row['n_test']:>7}{full_str}{sel_str} {stable:>8}")
    print(f"{'='*70}")
    print(f"\nTotal time: {elapsed_total/60:.1f} min")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
