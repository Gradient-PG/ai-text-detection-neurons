#!/usr/bin/env python
"""
One-time download of the RAID train split, split into per-(domain, model) CSVs.

Streams the full RAID train CSV (~12 GB) from HuggingFace and writes clean
(no adversarial attacks) rows into separate files:

    data/raw/raid/{domain}_{model}.csv

Each file contains columns: generation, title, domain, model

Run once, then use load_raid() which reads from these local files instantly.

Usage:
    uv run scripts/download_raid.py
    uv run scripts/download_raid.py --include-attacks
    uv run scripts/download_raid.py --output-dir data/raw/raid_full
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from raid_pipeline.raid_loader import slug

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

CSV_FIELDS = ["generation", "title", "domain", "model"]


def _is_clean(attack_value) -> bool:
    return attack_value is None or attack_value in ("", "none")


def main():
    parser = argparse.ArgumentParser(
        description="Download RAID train split and split into per-(domain, model) CSVs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/raid",
        help="Directory to write CSVs to (default: data/raw/raid)",
    )
    parser.add_argument(
        "--include-attacks",
        action="store_true",
        help="Keep adversarial-attack rows (excluded by default)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  RAID DOWNLOAD & SPLIT")
    print("=" * 60)
    print(f"  Output: {output_dir}")
    print(f"  Include attacks: {args.include_attacks}")
    print()
    print("  Streaming RAID train split from HuggingFace ...")
    print("  This will take a while on the first run (~12 GB CSV).")
    print()

    stream = load_dataset(
        "liamdugan/raid", "raid", split="train", streaming=True, token=HF_TOKEN,
    )

    writers: dict[str, csv.DictWriter] = {}
    file_handles: dict[str, object] = {}
    counts: dict[str, int] = defaultdict(int)
    skipped_attacks = 0
    total = 0

    try:
        for row in tqdm(stream, desc="Streaming RAID", unit=" rows"):
            total += 1

            if not args.include_attacks and not _is_clean(row.get("attack")):
                skipped_attacks += 1
                continue

            domain = row["domain"]
            model = row["model"]
            key = f"{slug(domain)}_{slug(model)}"

            if key not in writers:
                fpath = output_dir / f"{key}.csv"
                fh = open(fpath, "w", newline="", encoding="utf-8")
                writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writers[key] = writer
                file_handles[key] = fh

            writers[key].writerow({
                "generation": row["generation"],
                "title": row["title"] or "",
                "domain": domain,
                "model": model,
            })
            counts[key] += 1

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving what we have so far ...")
    finally:
        for fh in file_handles.values():
            fh.close()

    print()
    print("=" * 60)
    print("  DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"  Total rows streamed:  {total:,}")
    print(f"  Attack rows skipped:  {skipped_attacks:,}")
    print(f"  Clean rows written:   {sum(counts.values()):,}")
    print(f"  Files created:        {len(counts)}")
    print()

    domains_seen: dict[str, int] = defaultdict(int)
    models_seen: dict[str, int] = defaultdict(int)
    for key in sorted(counts):
        fpath = output_dir / f"{key}.csv"
        size_mb = fpath.stat().st_size / (1024 * 1024)
        print(f"    {key:30s}  {counts[key]:>8,} rows  ({size_mb:.1f} MB)")
        parts = key.rsplit("_", 1)
        domain_part = key.split("_")[0]
        for d in ["abstracts", "books", "news", "reddit", "reviews", "wiki"]:
            slug_d = slug(d)
            if key.startswith(slug_d + "_"):
                domains_seen[d] += counts[key]
                model_part = key[len(slug_d) + 1:]
                models_seen[model_part] += counts[key]
                break

    print()
    print("  Per domain:")
    for d in sorted(domains_seen):
        print(f"    {d:20s}  {domains_seen[d]:>8,}")
    print("  Per model:")
    for m in sorted(models_seen):
        print(f"    {m:20s}  {models_seen[m]:>8,}")

    print(f"\n  Files saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
