#!/usr/bin/env python
"""
Run a single experiment by name with a YAML config.

Usage:
    uv run scripts/experiments/run_experiment.py sparse_probe
    uv run scripts/experiments/run_experiment.py ablation --config config/experiments/ablation.yaml
    uv run scripts/experiments/run_experiment.py --list
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from raid_analysis.data.loading import load_experiment_data
from raid_analysis.data.splits import (
    generate_multi_seed_splits,
    load_splits,
    save_splits,
)
from raid_analysis.experiments.config import ExperimentConfig, load_config

EXPERIMENT_NAMES = [
    "sparse_probe",
    "ablation",
    "patching",
    "confound",
    "characterize",
    "auc_comparison",
    "restricted_probe",
]


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    parser = argparse.ArgumentParser(
        description="Run a single experiment from the experiment catalog."
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        choices=EXPERIMENT_NAMES,
        help="Experiment to run.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. Uses default config if not provided.",
    )
    parser.add_argument(
        "--generator",
        type=str,
        default="gpt4",
        help=(
            "Generator to run on (default: gpt4). "
            "Used by all experiments except characterize."
        ),
    )
    parser.add_argument(
        "--generators",
        nargs="+",
        default=None,
        help=(
            "List of generators. Used only by the characterize experiment "
            "to override the YAML 'generators' field."
        ),
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Source experiment directory (for dependent experiments).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory override.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier (e.g. timestamp). Auto-generated if not provided.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        for name in EXPERIMENT_NAMES:
            print(f"  {name}")
        return

    if args.experiment is None:
        parser.error("experiment is required (or use --list)")

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        default_config = project_root / "config" / "experiments" / f"{args.experiment}.yaml"
        if default_config.exists():
            config = load_config(default_config)
        else:
            config = ExperimentConfig(experiment=args.experiment)

    config.experiment = args.experiment

    if args.generators is not None and args.experiment == "characterize":
        config.generators = args.generators

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    args.run_id = run_id

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.experiment == "characterize":
        output_dir = Path(config.output_dir) / run_id / args.experiment
    else:
        output_dir = (
            Path(config.output_dir) / run_id / args.experiment / args.generator
        )

    print(f"Experiment: {args.experiment}")
    if args.experiment == "characterize":
        print(f"Generators: {', '.join(config.generators)}")
    else:
        print(f"Generator:  {args.generator}")
    print(f"Run ID:     {run_id}")
    print(f"Output:     {output_dir}")
    print()

    t0 = time.time()

    if args.experiment == "characterize":
        _run_characterize(config, args, output_dir)
    elif args.experiment == "auc_comparison":
        _run_auc_comparison(config, args, output_dir)
    elif args.experiment == "restricted_probe":
        _run_restricted_probe(config, args, output_dir)
    else:
        _run_standard(config, args, output_dir)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


def _get_splits(config, labels, metadata, output_dir):
    """Generate or load CV splits, regenerating if sample count changed."""
    splits_path = output_dir / "splits.json"
    n_samples = len(labels)

    if splits_path.exists():
        loaded = load_splits(splits_path)
        first_folds = next(iter(loaded.values()))
        max_idx = max(
            idx
            for fold in first_folds
            for idx in (*fold.train_idx, *fold.test_idx)
        )
        if max_idx < n_samples:
            print(f"Loading existing splits from {splits_path}")
            return loaded
        print(
            f"Stale splits (max_idx={max_idx} >= n_samples={n_samples}), "
            f"regenerating"
        )

    splits_by_seed = generate_multi_seed_splits(
        labels,
        n_folds=config.n_folds,
        seeds=config.seeds,
        domain_ids=metadata.domain_ids,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_splits(splits_by_seed, splits_path)
    print(f"Saved splits to {splits_path}")
    return splits_by_seed


def _run_standard(config: ExperimentConfig, args, output_dir: Path):
    """Run a standard fold-based experiment via the generic pipeline.

    Works for both primary experiments (sparse_probe) and dependent ones
    (ablation, patching, confound) that reuse selections from a source.

    For the patching experiment both directions are run automatically:
    - ai_into_human (AI donor → human base) at ``output_dir``  [unchanged path]
    - human_into_ai (human donor → AI base) at ``<run_root>/patching_human_into_ai/<gen>``
    """
    from raid_analysis.experiments.factory import build_evaluator, build_selector
    from raid_analysis.experiments.runner import run_experiment
    from raid_analysis.experiments.source_loader import load_source_selections

    activations, labels, metadata = load_experiment_data(config, args.generator)
    splits_by_seed = _get_splits(config, labels, metadata, output_dir)

    evaluator = build_evaluator(config)

    if config.source_experiment:
        source_dir = _resolve_source(args, config)
        precomputed = load_source_selections(source_dir, splits_by_seed)
        selector = None
        print(f"Using precomputed selections from {source_dir}")
    else:
        selector = build_selector(config)
        precomputed = None

    run_experiment(
        activations=activations,
        labels=labels,
        metadata=metadata,
        splits_by_seed=splits_by_seed,
        evaluator=evaluator,
        config=config,
        selector=selector,
        precomputed_selections=precomputed,
        output_dir=output_dir,
    )

    # For patching, automatically run human_into_ai so both directions are
    # produced in a single pipeline pass.  ai_into_human stays at its existing
    # path (output_dir) for backward compatibility; human_into_ai goes to a
    # sibling directory: <run_root>/patching_human_into_ai/<generator>.
    if args.experiment == "patching":
        from raid_analysis.evaluation.patching import PatchingEvaluator

        # output_dir layout: <output_dir_root>/<run_id>/patching/<generator>
        run_root = output_dir.parent.parent
        reverse_dir = run_root / "patching_human_into_ai" / output_dir.name

        print(f"\n--- Patching: human donor → AI base (human_into_ai) ---")
        print(f"Output: {reverse_dir}\n")

        reverse_evaluator = PatchingEvaluator(
            k_values=evaluator.k_values,
            pairing=evaluator.pairing,
            n_shuffles=evaluator.n_shuffles,
            n_random_draws=evaluator.n_random_draws,
            direction="human_into_ai",
        )
        run_experiment(
            activations=activations,
            labels=labels,
            metadata=metadata,
            splits_by_seed=splits_by_seed,
            evaluator=reverse_evaluator,
            config=config,
            selector=selector,
            precomputed_selections=precomputed,
            output_dir=reverse_dir,
        )


def _run_auc_comparison(config: ExperimentConfig, args, output_dir: Path):
    """Run the AUC comparison experiment."""
    from raid_analysis.experiments.exp_auc_comparison import run_auc_comparison

    activations, labels, metadata = load_experiment_data(config, args.generator)
    splits_by_seed = _get_splits(config, labels, metadata, output_dir)

    source_dir = _resolve_source(args, config)
    _check_source_dir(source_dir, config, args)
    run_auc_comparison(
        activations, labels, metadata, splits_by_seed,
        config, source_dir, output_dir=output_dir,
    )


def _run_restricted_probe(config: ExperimentConfig, args, output_dir: Path):
    """Run the smaller-model / ablate-complement experiment (§5.1 (B))."""
    from raid_analysis.experiments.exp_restricted_probe import (
        run_restricted_probe,
    )

    activations, labels, metadata = load_experiment_data(config, args.generator)
    splits_by_seed = _get_splits(config, labels, metadata, output_dir)

    source_dir = _resolve_source(args, config)
    _check_source_dir(source_dir, config, args)
    run_restricted_probe(
        activations, labels, metadata, splits_by_seed,
        config, source_dir, output_dir=output_dir,
    )


def _run_characterize(config: ExperimentConfig, args, output_dir: Path):
    """Run the characterize experiment across multiple generators."""
    import json
    from raid_analysis.experiments.exp_characterize import run_characterize_experiment

    generators = config.generators
    source_dir = _resolve_source(args, config)

    stable_sets: dict[str, set] = {}
    activations_by_gen: dict = {}
    labels_by_gen: dict = {}

    for gen in generators:
        acts, labels, metadata = load_experiment_data(config, gen)
        activations_by_gen[gen] = acts
        labels_by_gen[gen] = labels

        gen_source = source_dir.parent / gen
        if gen_source.exists():
            aggregate_path = gen_source / "aggregate.json"
            if aggregate_path.exists():
                with open(aggregate_path) as f:
                    aggregate = json.load(f)
                stable_sets[gen] = {
                    tuple(n) for n in aggregate.get("stable_neurons", [])
                }
            else:
                print(f"  Warning: no aggregate.json for {gen}, skipping")
        else:
            print(f"  Warning: no source directory for {gen} at {gen_source}")

    if not stable_sets:
        print("No stable sets found. Run sparse_probe experiment first.")
        return

    run_characterize_experiment(
        stable_sets, activations_by_gen, labels_by_gen,
        config, output_dir=output_dir,
    )


def _resolve_source(args, config: ExperimentConfig) -> Path:
    """Resolve the source experiment directory."""
    if args.source_dir:
        return Path(args.source_dir)
    base = Path(config.output_dir) / args.run_id
    if config.source_experiment:
        return base / config.source_experiment / args.generator
    return base / "sparse_probe" / args.generator


def _check_source_dir(source_dir: Path, config: ExperimentConfig, args) -> None:
    """Raise a clear error if the source experiment directory is missing."""
    if not source_dir.exists():
        source_exp = config.source_experiment or "sparse_probe"
        raise SystemExit(
            f"\nError: source experiment directory not found:\n"
            f"  {source_dir}\n\n"
            f"'{args.experiment}' depends on '{source_exp}'. "
            f"Either:\n"
            f"  (a) Run '{source_exp}' first under the same --run-id:\n"
            f"      uv run scripts/experiments/run_experiment.py {source_exp} "
            f"--generator {args.generator} --run-id {args.run_id}\n"
            f"  (b) Point directly at an existing run with --source-dir:\n"
            f"      uv run scripts/experiments/run_experiment.py {args.experiment} "
            f"--generator {args.generator} --source-dir <path/to/{source_exp}/{args.generator}>\n"
            f"  (c) Use run_all.py, which handles dependencies automatically:\n"
            f"      uv run scripts/experiments/run_all.py "
            f"--generators {args.generator}"
        )


if __name__ == "__main__":
    main()
