"""
Tokenize a subset of the RAID benchmark and save as HuggingFace Dataset.

Loads a stratified (domain x model) subset of RAID from local pre-split CSVs
(produced by download_raid.py), tokenizes with BERT, and saves the result so
the rest of the pipeline (extract_activations, analyze_activations, notebooks)
feeds the same extraction and analysis scripts as other tokenized RAID subsets.

Usage:
    uv run scripts/tokenize_raid.py --model gpt4
    uv run scripts/tokenize_raid.py --model chatgpt --max-samples 20000
    uv run scripts/tokenize_raid.py --model gpt4 --domains news wiki

Prerequisite:
    uv run scripts/download_raid.py   (one-time, creates data/raw/raid/*.csv)

Output:
    data/processed/raid_{model}/ (tokenized HuggingFace DatasetDict)
"""

import argparse
from pathlib import Path

from raid_pipeline.dataset_tokenizer import DatasetTokenizer
from raid_pipeline.raid_loader import ALL_RAID_MODELS, RAIDConfig, load_raid, slug


def main():
    parser = argparse.ArgumentParser(
        description="Load a RAID subset (model X vs human), tokenize with BERT, and save to disk"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt4",
        help=f"AI model to compare against human (default: gpt4). "
        f"Choices: {', '.join(ALL_RAID_MODELS)}",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10_000,
        help="Max total samples to keep, balanced across (domain, model) buckets (default: 10 000)",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="RAID domains to include (default: all — abstracts books news reddit reviews wiki)",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw/raid",
        help="Directory with pre-split RAID CSVs (default: data/raw/raid)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Parent output directory (default: data/processed)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Subfolder name for the saved dataset (default: raid_{model})",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace tokenizer name",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max token sequence length (default: 512)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Tokenization batch size (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    dataset_name = args.dataset_name or f"raid_{slug(args.model)}"

    raid_cfg = RAIDConfig(
        model=args.model,
        domains=args.domains,
        max_samples=args.max_samples,
        seed=args.seed,
        raw_dir=args.raw_dir,
    )

    print("=" * 60)
    print("RAID DATASET TOKENIZATION")
    print("=" * 60)

    ds = load_raid(config=raid_cfg)

    tokenizer = DatasetTokenizer(
        tokenizer_name=args.tokenizer,
        output_dir=args.output_dir,
        max_length=args.max_length,
        random_state=args.seed,
        text_column="text",
        extra_columns=["domain", "source_model"],
    )

    output_path = tokenizer.tokenize_and_save(
        ds,
        batch_size=args.batch_size,
        dataset_name=dataset_name,
    )

    print("\n" + "=" * 60)
    print("TOKENIZATION COMPLETE")
    print("=" * 60)
    print(f"\nSaved to: {output_path}")
    print(f"\nNext step:")
    print(f"  uv run scripts/extract_activations.py \\")
    print(f"    --tokenized-path {output_path} \\")
    print(f"    --output results/activations_raid_{slug(args.model)}")


if __name__ == "__main__":
    main()
