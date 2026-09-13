#!/usr/bin/env python
"""
Run the full experiment sequence respecting dependencies.

Usage:
    uv run scripts/experiments/run_all.py --generators gpt4
    uv run scripts/experiments/run_all.py --generators gpt4 mistral
    uv run scripts/experiments/run_all.py --generators gpt4 mistral --skip characterize
    uv run scripts/experiments/run_all.py --generators gpt4 --only sparse_probe ablation

Experiment order (dependencies read from YAML configs via source_experiment):
    Per-generator (run independently for each generator, shared run_id):
        1. sparse_probe           (no dependencies)
        2. ablation               (depends on sparse_probe)
        3. patching               (depends on sparse_probe)
        4. confound               (depends on sparse_probe)
        5. auc_comparison         (depends on sparse_probe)
    Cross-generator (run once over all generators that succeeded):
        6. characterize           (depends on sparse_probe)

If a per-generator experiment fails for a given generator, only its dependents
for that same generator are skipped. Other generators proceed independently.
Characterize runs over the subset of generators that have a successful
sparse_probe.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

PER_GENERATOR_EXPERIMENTS = [
    "sparse_probe",
    "ablation",
    "patching",
    "confound",
    "auc_comparison",
    "restricted_probe",
]
CROSS_GENERATOR_EXPERIMENTS = [
    "characterize",
]
EXPERIMENT_ORDER = PER_GENERATOR_EXPERIMENTS + CROSS_GENERATOR_EXPERIMENTS


def _read_dependency(config_dir: Path, experiment: str) -> str | None:
    """Read source_experiment from a YAML config (if it exists)."""
    config_file = config_dir / f"{experiment}.yaml"
    if not config_file.exists():
        return None
    with open(config_file) as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("source_experiment")


def _collect_transitive_deps(
    experiment: str,
    dep_map: dict[str, str | None],
) -> set[str]:
    """Return all transitive dependencies of *experiment*."""
    deps: set[str] = set()
    current = dep_map.get(experiment)
    while current is not None:
        deps.add(current)
        current = dep_map.get(current)
    return deps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all experiments in dependency order."
    )
    parser.add_argument(
        "--generators",
        nargs="+",
        default=None,
        help=(
            "Generators to run on (one or more). "
            "Example: --generators gpt4 mistral llama-chat."
        ),
    )
    parser.add_argument(
        "--generator",
        type=str,
        default=None,
        help=(
            "Deprecated single-generator alias of --generators. "
            "Accepts exactly one name."
        ),
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config/experiments",
        help="Directory containing per-experiment YAML configs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/experiments",
        help="Base output directory.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier (e.g. timestamp). Auto-generated if not provided.",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="Experiments to skip.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only these experiments (in dependency order).",
    )
    args = parser.parse_args()

    if args.generators and args.generator:
        parser.error("Use either --generators or --generator, not both.")
    if args.generators is None and args.generator is None:
        parser.error("Either --generators or --generator is required.")
    generators: list[str] = (
        args.generators if args.generators is not None else [args.generator]
    )

    config_dir = Path(args.config_dir)

    experiments = list(EXPERIMENT_ORDER)
    if args.only:
        experiments = [e for e in experiments if e in args.only]
    experiments = [e for e in experiments if e not in args.skip]

    if not experiments:
        print("No experiments to run.")
        return

    dep_map = {exp: _read_dependency(config_dir, exp) for exp in EXPERIMENT_ORDER}

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Generators:  {', '.join(generators)}")
    print(f"Run ID:      {run_id}")
    print(f"Experiments: {', '.join(experiments)}")
    print()

    script = Path(__file__).parent / "run_experiment.py"

    per_gen_experiments = [e for e in experiments if e in PER_GENERATOR_EXPERIMENTS]
    cross_gen_experiments = [
        e for e in experiments if e in CROSS_GENERATOR_EXPERIMENTS
    ]

    failed_per_gen: set[tuple[str, str]] = set()
    skipped_per_gen: set[tuple[str, str]] = set()
    failed_cross: set[str] = set()
    skipped_cross: set[str] = set()

    t0 = time.time()

    for gen in generators:
        print(f"\n{'#'*60}")
        print(f"#  Generator: {gen}")
        print(f"{'#'*60}")

        for exp in per_gen_experiments:
            deps = _collect_transitive_deps(exp, dep_map)
            blocked_by = {d for d in deps if (gen, d) in failed_per_gen}
            if blocked_by:
                print(f"\n{'='*60}")
                print(
                    f"  SKIP {gen}/{exp} "
                    f"(dependency failed: {', '.join(sorted(blocked_by))})"
                )
                print(f"{'='*60}")
                skipped_per_gen.add((gen, exp))
                continue

            print(f"\n{'='*60}")
            print(f"  {gen}/{exp}")
            print(f"{'='*60}\n")

            cmd = [
                sys.executable, str(script),
                exp,
                "--generator", gen,
                "--run-id", run_id,
            ]

            config_file = config_dir / f"{exp}.yaml"
            if config_file.exists():
                cmd.extend(["--config", str(config_file)])

            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(
                    f"\n  FAILED: {gen}/{exp} "
                    f"(exit code {result.returncode})"
                )
                failed_per_gen.add((gen, exp))

    for exp in cross_gen_experiments:
        deps = _collect_transitive_deps(exp, dep_map)
        eligible_gens = [
            g for g in generators
            if not any((g, d) in failed_per_gen for d in deps)
        ]
        if not eligible_gens:
            print(f"\n{'='*60}")
            print(
                f"  SKIP {exp} "
                f"(no generators have successful dependencies: "
                f"{', '.join(sorted(deps))})"
            )
            print(f"{'='*60}")
            skipped_cross.add(exp)
            continue

        print(f"\n{'='*60}")
        print(f"  {exp} (generators: {', '.join(eligible_gens)})")
        print(f"{'='*60}\n")

        cmd = [
            sys.executable, str(script),
            exp,
            "--run-id", run_id,
            "--generators", *eligible_gens,
        ]

        config_file = config_dir / f"{exp}.yaml"
        if config_file.exists():
            cmd.extend(["--config", str(config_file)])

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n  FAILED: {exp} (exit code {result.returncode})")
            failed_cross.add(exp)

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")

    all_attempted: set[tuple[str, str]] = set()
    for gen in generators:
        for exp in per_gen_experiments:
            all_attempted.add((gen, exp))
    succeeded_per_gen = all_attempted - failed_per_gen - skipped_per_gen
    if succeeded_per_gen:
        by_gen: dict[str, list[str]] = {}
        for gen, exp in sorted(succeeded_per_gen):
            by_gen.setdefault(gen, []).append(exp)
        for gen, exps in sorted(by_gen.items()):
            print(f"  Succeeded [{gen}]: {', '.join(exps)}")
    if failed_per_gen:
        for gen, exp in sorted(failed_per_gen):
            print(f"  Failed    [{gen}]: {exp}")
    if skipped_per_gen:
        for gen, exp in sorted(skipped_per_gen):
            print(f"  Skipped   [{gen}]: {exp}")

    succeeded_cross = set(cross_gen_experiments) - failed_cross - skipped_cross
    if succeeded_cross:
        print(f"  Succeeded [cross-gen]: {', '.join(sorted(succeeded_cross))}")
    if failed_cross:
        print(f"  Failed    [cross-gen]: {', '.join(sorted(failed_cross))}")
    if skipped_cross:
        print(f"  Skipped   [cross-gen]: {', '.join(sorted(skipped_cross))}")

    print(f"  Elapsed:   {elapsed:.1f}s")

    if failed_per_gen or failed_cross:
        sys.exit(1)


if __name__ == "__main__":
    main()
