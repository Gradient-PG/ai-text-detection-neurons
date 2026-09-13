#!/usr/bin/env python
"""
Extract BERT CLS token activations for interpretability analysis.

Expects a tokenized HuggingFace dataset on disk, as produced by
``tokenize_raid.py``.

Usage:
    uv run scripts/extract_activations.py \
        --tokenized-path data/processed/raid_gpt4 \
        --output results/activations_raid_gpt4

    uv run scripts/extract_activations.py --layers 3 6 9 12 --samples 1000
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModel
from raid_pipeline.activation_extractor import ActivationExtractor
from raid_pipeline.model_loader import BERT_MODEL_NAME, BERT_MODEL_REVISION


def main():
    parser = argparse.ArgumentParser(
        description="Extract BERT layer activations for analysis"
    )
    parser.add_argument(
        "--tokenized-path",
        type=str,
        default="data/processed/raid_tokenized",
        help="Path to tokenized dataset (e.g. data/processed/raid_gpt4)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=BERT_MODEL_NAME,
        help="Encoder model name",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        help="Layer indices to extract (1-12 for BERT-base)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples to analyze (balanced by class)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for BERT forward pass (default: 64). "
        "Lower this if you run out of memory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/activations_raid",
        help="Output directory for activations",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to extract from (default: train)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BERT ACTIVATION EXTRACTION")
    print("=" * 60)
    print(f"  Encoder: {args.encoder}")
    print(f"  Layers: {args.layers}")
    print(f"  Samples: {args.samples}")
    print(f"  Output: {args.output}\n")
    
    # Load encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading encoder model on {device}...")
    encoder = AutoModel.from_pretrained(args.encoder, revision=BERT_MODEL_REVISION)
    
    # Create extractor
    extractor = ActivationExtractor(
        encoder=encoder,
        layers=args.layers,
        device=device
    )
    
    extractor.extract_and_save(
        tokenized_dataset_path=args.tokenized_path,
        output_dir=args.output,
        batch_size=args.batch_size,
        max_samples=args.samples,
        split=args.split,
    )
    
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"\nNext step: uv run scripts/analyze_activations.py")


if __name__ == "__main__":
    main()


