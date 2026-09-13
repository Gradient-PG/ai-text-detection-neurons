"""
RAID dataset loader — reads from local pre-split CSVs.

Expects the directory layout produced by ``scripts/download_raid.py``::

    data/raw/raid/
        abstracts_human.csv
        abstracts_gpt4.csv
        news_chatgpt.csv
        ...

Each CSV has columns: generation, title, domain, model

``load_raid()`` picks the relevant files for a given AI model vs human,
stratifies by domain, and returns a single balanced ``Dataset``.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
from datasets import Dataset

ALL_RAID_MODELS = [
    "chatgpt", "cohere", "cohere-chat", "gpt2", "gpt3", "gpt4",
    "llama-chat", "mistral", "mistral-chat", "mpt", "mpt-chat",
]

ALL_DOMAINS = ["abstracts", "books", "news", "reddit", "reviews", "wiki"]

DEFAULT_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "raid"


def slug(name: str) -> str:
    """Normalise a name for file/folder paths: ``mistral-chat`` -> ``mistral_chat``."""
    return name.replace("-", "_")


@dataclass
class RAIDConfig:
    """Configuration for loading a subset of RAID from local CSVs."""

    model: str = "gpt4"
    domains: Optional[List[str]] = None
    max_samples: int = 10_000
    seed: int = 42
    raw_dir: str | Path = DEFAULT_RAW_DIR


def load_raid(config: RAIDConfig | None = None) -> Dataset:
    """
    Load a balanced, domain-stratified dataset for a single AI model vs human
    from pre-split local CSVs.

    For each domain, reads ``{domain}_{model}.csv`` and ``{domain}_human.csv``,
    then takes an equal quota from every (domain, class) bucket.

    Returns a single ``Dataset`` with columns:
        text, label, title, domain, source_model
    """
    if config is None:
        config = RAIDConfig()

    raw_dir = Path(config.raw_dir)
    domains = config.domains or ALL_DOMAINS
    model_slug = slug(config.model)

    n_buckets = len(domains) * 2  # each domain has AI + human
    quota = max(config.max_samples // n_buckets, 1)

    print(f"Loading RAID from local CSVs ...")
    print(f"  Raw dir:       {raw_dir}")
    print(f"  AI model:      {config.model}")
    print(f"  Domains:       {domains}")
    print(f"  Max samples:   {config.max_samples:,}")
    print(f"  Buckets:       {n_buckets}  (domain x [human, {config.model}])")
    print(f"  Quota/bucket:  {quota}")

    rng = random.Random(config.seed)
    collected: list[dict] = []
    missing: list[str] = []

    for domain in sorted(domains):
        d_slug = slug(domain)
        for model_name, label in [(model_slug, 1), ("human", 0)]:
            csv_path = raw_dir / f"{d_slug}_{model_name}.csv"
            if not csv_path.exists():
                missing.append(str(csv_path))
                continue

            df = pd.read_csv(csv_path)
            rows = df.to_dict("records")
            rng.shuffle(rows)
            bucket = rows[:quota]

            for row in bucket:
                collected.append({
                    "text": row["generation"],
                    "label": label,
                    "title": row.get("title", ""),
                    "domain": row["domain"],
                    "source_model": row["model"],
                })

    if missing:
        print(f"\n  WARNING — missing CSV files:")
        for m in missing:
            print(f"    {m}")
        print(f"  Run: uv run scripts/download_raid.py")

    if not collected:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. "
            f"Run: uv run scripts/download_raid.py"
        )

    rng.shuffle(collected)

    model_counts: dict[str, int] = defaultdict(int)
    domain_counts: dict[str, int] = defaultdict(int)
    for rec in collected:
        model_counts[rec["source_model"]] += 1
        domain_counts[rec["domain"]] += 1

    print(f"\n  Total collected:  {len(collected):,}")
    print("  Per model:")
    for m in sorted(model_counts):
        tag = "human" if m == "human" else "AI"
        print(f"    {m:20s}  {model_counts[m]:>6,}  ({tag})")
    print("  Per domain:")
    for d in sorted(domain_counts):
        print(f"    {d:20s}  {domain_counts[d]:>6,}")

    return Dataset.from_list(collected)
