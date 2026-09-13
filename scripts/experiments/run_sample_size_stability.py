#!/usr/bin/env python
"""
Sample-size *stability* via subsampling, swept over a (C, N) grid for one or
more generators.

For each generator G, each L1 regularisation strength C in --c-list, and each
sample size N in --n-list:
  Draw K (default 15) independent stratified subsamples of size N from the
  full activation pool (stratified by domain x label, mirroring the rest of
  the pipeline).
  Fit one L1 logistic regression at the given C on each subsample.
  Record the selected feature set per draw.

Then aggregate across the K draws per (G, C, N):
  - n_selected: mean / std / min / max
  - train_accuracy: mean / std
  - pairwise Jaccard between selected sets: mean / std / min / max
  - core stable set: features selected in >= core_threshold of K draws.

The headline metric is **pairwise Jaccard vs N** (per fixed C): it isolates the
effect of sample size on selection stability, decoupled from CV-fold variance
and knee-detector behaviour. The C sweep on top lets us pick the regularisation
strength on the same stability criterion.

Usage:
    uv run scripts/experiments/run_sample_size_stability.py
    uv run scripts/experiments/run_sample_size_stability.py \\
        --generators gpt4 llama-chat \\
        --c-list 0.005 0.01 0.02 0.05 \\
        --n-list 500 1000 2000 3500 5000 7500 10000 \\
        --k-draws 15

Output structure (C outer, N inner):
    results/experiments/{run_id}/
        config.json                                  # top-level grid config
        grid_summary.json                            # flat (G, C, N) records
        sample_size_stability/{generator}/
            config.json                              # per-generator metadata
            summary.json                             # by_c -> by_n -> metrics
            draws/c{C}/n{N}/draw_{k:02d}.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from raid_analysis.data.activations import (  # noqa: E402
    concat_all_layers,
    global_to_layer_neuron,
    load_activations,
)
from raid_analysis.data.metadata import load_metadata  # noqa: E402
from raid_pipeline.raid_loader import slug  # noqa: E402


# ---------------------------------------------------------------------------
# Stratified subsampling
# ---------------------------------------------------------------------------

def stratified_subsample(
    domain_ids: np.ndarray,
    labels: np.ndarray,
    n_target: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return ``n_target`` indices, balanced across (domain, label) buckets.

    Quota per bucket is ``n_target // (n_domains * 2)``; if a bucket has fewer
    samples than the quota, it contributes everything it has and the actual
    draw size is reported back to the caller (will be slightly below
    ``n_target``).
    """
    unique_domains = np.unique(domain_ids)
    n_buckets = len(unique_domains) * 2
    quota = max(1, n_target // n_buckets)

    chosen: list[np.ndarray] = []
    for d in unique_domains:
        for label in (0, 1):
            mask = (domain_ids == d) & (labels == label)
            bucket = np.where(mask)[0]
            n_take = min(quota, len(bucket))
            if n_take == 0:
                continue
            picked = rng.choice(bucket, size=n_take, replace=False)
            chosen.append(picked)

    if not chosen:
        return np.array([], dtype=np.int64)
    return np.sort(np.concatenate(chosen))


# ---------------------------------------------------------------------------
# Fit & metrics
# ---------------------------------------------------------------------------

def fit_one_l1(
    activations: np.ndarray,
    labels: np.ndarray,
    *,
    C: float,
    solver: str,
    max_iter: int,
    seed: int,
) -> tuple[set[tuple[int, int]], float, float]:
    """Fit a single L1 logistic regression and return the selected set.

    Returns:
        ``(selected_neurons, train_accuracy, fit_time_s)`` where
        ``selected_neurons`` is the set of ``(layer, neuron_idx)`` tuples with
        non-zero coefficients.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = scaler.fit_transform(activations)

    t0 = time.time()
    lr = LogisticRegression(
        C=C,
        l1_ratio=1.0,
        solver=solver,
        max_iter=max_iter,
        random_state=seed,
    )
    lr.fit(X, labels)
    fit_time = time.time() - t0

    coef = lr.coef_[0]
    nonzero_global = np.where(np.abs(coef) > 0)[0]
    selected = {global_to_layer_neuron(int(g)) for g in nonzero_global}

    train_acc = float(lr.score(X, labels))
    return selected, train_acc, fit_time


def pairwise_jaccard_stats(
    sets: list[set[tuple[int, int]]],
) -> dict:
    """Compute pairwise Jaccard between K selected sets.

    Returns mean / std / min / max over the K(K-1)/2 pairs.
    """
    n = len(sets)
    if n < 2:
        return {
            "mean": None, "std": None, "min": None, "max": None,
            "n_pairs": 0,
        }

    values: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sets[i], sets[j]
            union = len(a | b)
            if union == 0:
                values.append(1.0)
            else:
                values.append(len(a & b) / union)

    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_pairs": len(values),
    }


def core_stable_set(
    sets: list[set[tuple[int, int]]],
    threshold_pct: float = 80.0,
) -> tuple[int, list[tuple[int, int]]]:
    """Return the count and list of features selected in ``>= threshold_pct``
    fraction of the K draws.
    """
    if not sets:
        return 0, []

    counts: dict[tuple[int, int], int] = {}
    for s in sets:
        for neuron in s:
            counts[neuron] = counts.get(neuron, 0) + 1

    k = len(sets)
    cutoff = (threshold_pct / 100.0) * k
    core = sorted(
        neuron for neuron, c in counts.items() if c >= cutoff
    )
    return len(core), core


# ---------------------------------------------------------------------------
# Per-(generator, C, N) cell
# ---------------------------------------------------------------------------

def _format_c(c: float) -> str:
    """Stable string label for a C value (used in directory names)."""
    return f"{c:g}"


def run_cell(
    *,
    activations: np.ndarray,
    labels: np.ndarray,
    domain_ids: np.ndarray,
    n_target: int,
    n_total: int,
    k_draws: int,
    C: float,
    solver: str,
    max_iter: int,
    master_seed: int,
    core_threshold_pct: float,
    cell_dir: Path,
) -> dict:
    """Run K draws for one (C, N) cell of the grid; write per-draw JSONs and
    return the aggregated metrics dict (the same shape as the legacy
    per-N record).
    """
    cell_dir.mkdir(parents=True, exist_ok=True)

    if n_target >= n_total:
        effective_k = 1
        print(
            f"    [N={n_target}] N >= pool size ({n_total}); reducing K to 1 "
            f"(all draws would be identical)."
        )
    else:
        effective_k = k_draws

    sets_per_draw: list[set[tuple[int, int]]] = []
    n_selected_per_draw: list[int] = []
    train_acc_per_draw: list[float] = []
    n_actual_per_draw: list[int] = []
    fit_times: list[float] = []

    for k in range(effective_k):
        seed_k = master_seed + k
        rng = np.random.default_rng(seed_k)
        indices = stratified_subsample(domain_ids, labels, n_target, rng)
        X = activations[indices]
        y = labels[indices]

        selected, train_acc, fit_time = fit_one_l1(
            X, y,
            C=C, solver=solver, max_iter=max_iter, seed=seed_k,
        )

        sets_per_draw.append(selected)
        n_selected_per_draw.append(len(selected))
        train_acc_per_draw.append(train_acc)
        n_actual_per_draw.append(int(len(indices)))
        fit_times.append(fit_time)

        draw_payload = {
            "n_target": int(n_target),
            "n_actual": int(len(indices)),
            "C": float(C),
            "k_index": int(k),
            "seed": int(seed_k),
            "n_selected": int(len(selected)),
            "selected_neurons": [list(p) for p in sorted(selected)],
            "train_accuracy": float(train_acc),
            "fit_time_s": float(fit_time),
        }
        with open(cell_dir / f"draw_{k:02d}.json", "w") as f:
            json.dump(draw_payload, f)

    ns = np.asarray(n_selected_per_draw, dtype=np.float64)
    tas = np.asarray(train_acc_per_draw, dtype=np.float64)
    actuals = np.asarray(n_actual_per_draw, dtype=np.int64)

    jaccard = pairwise_jaccard_stats(sets_per_draw)
    core_n, core_neurons = core_stable_set(sets_per_draw, core_threshold_pct)

    return {
        "n_target": int(n_target),
        "C": float(C),
        "n_actual_mean": float(actuals.mean()) if actuals.size else None,
        "n_actual_std": float(actuals.std(ddof=0)) if actuals.size else None,
        "k_draws": int(effective_k),
        "n_selected": {
            "mean": float(ns.mean()) if ns.size else None,
            "std": float(ns.std(ddof=0)) if ns.size else None,
            "min": float(ns.min()) if ns.size else None,
            "max": float(ns.max()) if ns.size else None,
        },
        "train_accuracy": {
            "mean": float(tas.mean()) if tas.size else None,
            "std": float(tas.std(ddof=0)) if tas.size else None,
        },
        "pairwise_jaccard": jaccard,
        "core_stable": {
            "threshold_pct": core_threshold_pct,
            "n_neurons": int(core_n),
            "neurons": [list(p) for p in core_neurons],
        },
        "fit_time_total_s": float(np.sum(fit_times)),
    }


# ---------------------------------------------------------------------------
# Per-generator driver
# ---------------------------------------------------------------------------

def run_generator(
    *,
    generator: str,
    n_list: list[int],
    c_list: list[float],
    k_draws: int,
    solver: str,
    max_iter: int,
    master_seed: int,
    core_threshold_pct: float,
    activations_root: Path,
    output_root: Path,
    run_id: str,
) -> tuple[dict, list[dict]]:
    """Run the (C, N) grid for one generator.

    Returns:
        ``(per_generator_summary, flat_records)`` where
        ``per_generator_summary`` is the nested ``by_c -> by_n -> cell`` dict
        written to ``summary.json`` and ``flat_records`` are the rows
        contributed to the top-level ``grid_summary.json``.
    """
    output_dir = output_root / "sample_size_stability" / generator
    draws_dir = output_dir / "draws"
    output_dir.mkdir(parents=True, exist_ok=True)
    draws_dir.mkdir(parents=True, exist_ok=True)

    activations_path = activations_root / f"activations_raid_{slug(generator)}"
    print(f"\n[{generator}] loading activations from {activations_path} ...")
    acts_dict, labels, _meta_json = load_activations(activations_path)
    activations = concat_all_layers(acts_dict)
    metadata = load_metadata(activations_path)
    n_total = len(labels)
    n_features = activations.shape[1]
    print(
        f"  pool: {n_total} samples x {n_features} features "
        f"({len(metadata.domain_names)} domains)"
    )

    config_payload = {
        "generator": generator,
        "run_id": run_id,
        "sample_sizes": list(n_list),
        "c_list": list(c_list),
        "k_draws": k_draws,
        "solver": solver,
        "max_iter": max_iter,
        "seed": master_seed,
        "core_threshold_pct": core_threshold_pct,
        "n_total_pool": int(n_total),
        "n_features": int(n_features),
        "n_domains": len(metadata.domain_names),
        "domain_names": list(metadata.domain_names),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_payload, f, indent=2)

    summary_by_c: dict[str, dict[str, dict]] = {}
    flat_records: list[dict] = []
    t_start = time.time()

    for C in c_list:
        c_label = _format_c(C)
        print(f"\n  {'=' * 56}\n  [{generator}]  C = {c_label}\n  {'=' * 56}")

        by_n: dict[str, dict] = {}
        for n_target in n_list:
            cell_dir = draws_dir / f"c{c_label}" / f"n{n_target}"
            cell = run_cell(
                activations=activations,
                labels=labels,
                domain_ids=metadata.domain_ids,
                n_target=n_target,
                n_total=n_total,
                k_draws=k_draws,
                C=C,
                solver=solver,
                max_iter=max_iter,
                master_seed=master_seed,
                core_threshold_pct=core_threshold_pct,
                cell_dir=cell_dir,
            )
            by_n[str(n_target)] = cell

            ns = cell["n_selected"]
            jac = cell["pairwise_jaccard"]
            ta = cell["train_accuracy"]
            ns_mean = ns["mean"] if ns["mean"] is not None else float("nan")
            jac_mean = jac["mean"] if jac["mean"] is not None else float("nan")
            ta_mean = ta["mean"] if ta["mean"] is not None else float("nan")
            print(
                f"    N={n_target:>5d}  K={cell['k_draws']:>2d}  "
                f"n_sel={ns_mean:6.1f}  jaccard={jac_mean:6.3f}  "
                f"core={cell['core_stable']['n_neurons']:>4d}  "
                f"train_acc={ta_mean:.4f}"
            )

            flat_records.append({
                "generator": generator,
                "C": float(C),
                "N": int(n_target),
                "k_draws": cell["k_draws"],
                "n_selected_mean": ns["mean"],
                "n_selected_std": ns["std"],
                "jaccard_mean": jac["mean"],
                "jaccard_std": jac["std"],
                "core_n": cell["core_stable"]["n_neurons"],
                "train_acc_mean": ta["mean"],
                "fit_time_total_s": cell["fit_time_total_s"],
            })

        summary_by_c[c_label] = by_n

    elapsed = time.time() - t_start
    summary = {
        **config_payload,
        "elapsed_s": float(elapsed),
        "by_c": summary_by_c,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  [{generator}] done in {elapsed:.1f}s -> {output_dir}/summary.json")
    return summary, flat_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    parser = argparse.ArgumentParser(
        description=(
            "Sample-size & C stability sweep: pairwise Jaccard of L1 selected "
            "sets across K independent stratified subsamples per (C, N), for "
            "one or more generators in a single run."
        )
    )
    parser.add_argument(
        "--generators", nargs="+", type=str, default=["gpt4"],
        help="Generator slugs (e.g. gpt4 llama-chat). Default: gpt4.",
    )
    parser.add_argument(
        "--n-list", nargs="+", type=int,
        default=[500, 1000, 2000, 3500, 5000, 7500, 10000],
        help="Sample sizes to evaluate.",
    )
    parser.add_argument(
        "--c-list", nargs="+", type=float,
        default=[0.005, 0.01, 0.02, 0.05],
        help=(
            "L1 regularisation strengths to sweep (default: "
            "0.005 0.01 0.02 0.05)."
        ),
    )
    parser.add_argument(
        "--k-draws", type=int, default=15,
        help="Number of independent subsample draws per (C, N) cell.",
    )
    parser.add_argument(
        "--solver", type=str, default="liblinear",
        choices=["liblinear", "saga"],
        help="L1 solver (default: liblinear).",
    )
    parser.add_argument(
        "--max-iter", type=int, default=1000,
        help="Max iterations for the solver (default: 1000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master seed; per-draw seeds are master_seed + draw_index.",
    )
    parser.add_argument(
        "--core-threshold-pct", type=float, default=80.0,
        help="Frequency threshold for the core-stable set (default: 80%%).",
    )
    parser.add_argument(
        "--activations-root", type=str, default="results",
        help="Directory containing activations_raid_<slug> subfolders.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/experiments",
        help="Output base directory.",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Run identifier (auto-generated if omitted).",
    )

    args = parser.parse_args()

    run_id = args.run_id or (
        "stability_grid_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_root = Path(args.output_dir) / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    activations_root = Path(args.activations_root)

    n_cells = len(args.generators) * len(args.c_list) * len(args.n_list)
    print(f"Run ID:          {run_id}")
    print(f"Generators:      {', '.join(args.generators)}")
    print(f"C values:        {args.c_list}")
    print(f"Sample sizes:    {args.n_list}")
    print(f"K draws / cell:  {args.k_draws}")
    print(f"Solver:          {args.solver} (max_iter={args.max_iter})")
    print(f"Master seed:     {args.seed}")
    print(f"Total cells:     {n_cells} "
          f"({len(args.generators)} gen x {len(args.c_list)} C x "
          f"{len(args.n_list)} N)")
    print(f"Output root:     {output_root}")

    grid_config = {
        "run_id": run_id,
        "generators": list(args.generators),
        "c_list": list(args.c_list),
        "n_list": list(args.n_list),
        "k_draws": args.k_draws,
        "solver": args.solver,
        "max_iter": args.max_iter,
        "seed": args.seed,
        "core_threshold_pct": args.core_threshold_pct,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(output_root / "config.json", "w") as f:
        json.dump(grid_config, f, indent=2)

    t_start = time.time()
    all_records: list[dict] = []

    for generator in args.generators:
        _, recs = run_generator(
            generator=generator,
            n_list=args.n_list,
            c_list=args.c_list,
            k_draws=args.k_draws,
            solver=args.solver,
            max_iter=args.max_iter,
            master_seed=args.seed,
            core_threshold_pct=args.core_threshold_pct,
            activations_root=activations_root,
            output_root=output_root,
            run_id=run_id,
        )
        all_records.extend(recs)

    elapsed = time.time() - t_start

    grid_summary = {
        **grid_config,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": float(elapsed),
        "records": all_records,
    }
    grid_summary_path = output_root / "grid_summary.json"
    with open(grid_summary_path, "w") as f:
        json.dump(grid_summary, f, indent=2)

    print(f"\n{'=' * 80}\n  Cross-cell summary  ({elapsed:.1f}s total, "
          f"{len(all_records)} cells)\n{'=' * 80}")
    print(
        f"  {'generator':<14}  {'C':>7}  {'N':>6}  {'K':>3}  "
        f"{'n_sel':>7}  {'jaccard':>8}  {'core':>5}  {'train_acc':>9}"
    )
    for r in all_records:
        ns = r["n_selected_mean"]
        jac = r["jaccard_mean"]
        ta = r["train_acc_mean"]
        ns_s = f"{ns:.1f}" if ns is not None else "-"
        jac_s = f"{jac:.3f}" if jac is not None else "-"
        ta_s = f"{ta:.4f}" if ta is not None else "-"
        print(
            f"  {r['generator']:<14}  {r['C']:>7g}  {r['N']:>6d}  "
            f"{r['k_draws']:>3d}  {ns_s:>7}  {jac_s:>8}  "
            f"{r['core_n']:>5d}  {ta_s:>9}"
        )

    print(f"\n  Wrote {grid_summary_path}")


if __name__ == "__main__":
    main()
