"""
Activation extractor for BERT layers.
Extracts hidden states from specified transformer layers for interpretability analysis.
"""

import json

import numpy as np
import torch
from datasets import Dataset
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict


class ActivationExtractor:
    """
    Extracts activations from specified BERT layers.

    Args:
        encoder: The encoder model (BERT)
        layers: List of layer indices to extract (e.g., [3, 6, 9, 12])
        device: Device to run on ('cuda' or 'cpu')
    """

    def __init__(
        self,
        encoder,
        layers: List[int],
        device: str = "cuda"
    ):
        self.encoder = encoder
        self.layers = layers
        self.device = device
        self.encoder.to(device)
        self.encoder.eval()

        print(f"Activation extractor initialized")
        print(f"  Layers to extract: {layers}")
        print(f"  Device: {device}")

    def extract_activations(
        self,
        dataset: Dataset,
        batch_size: int = 64,
        max_samples: int = None,
        desc: str = "Extracting activations"
    ) -> tuple[Dict[int, np.ndarray], np.ndarray, Dataset]:
        """
        Extract activations from specified layers with balanced class sampling.

        Args:
            dataset: Tokenized dataset with 'input_ids', 'attention_mask', 'label'
            batch_size: Batch size for processing
            max_samples: Maximum number of samples to process (None = all), will be balanced 50/50
            desc: Progress bar description

        Returns:
            Tuple of:
              - Dictionary mapping layer_idx -> activations array (N_samples, 768)
              - Labels array (N_samples,)
              - The final (possibly shuffled/subsampled) Dataset, aligned with the arrays
        """
        from datasets import concatenate_datasets

        if max_samples:
            ai_samples = dataset.filter(lambda x: x['label'] == 1)
            human_samples = dataset.filter(lambda x: x['label'] == 0)

            samples_per_class = max_samples // 2

            ai_subset = ai_samples.shuffle(seed=42).select(range(min(samples_per_class, len(ai_samples))))
            human_subset = human_samples.shuffle(seed=42).select(range(min(samples_per_class, len(human_samples))))

            dataset = concatenate_datasets([ai_subset, human_subset]).shuffle(seed=42)

            print(f"Balanced sampling: {len(ai_subset)} AI + {len(human_subset)} Human = {len(dataset)} total")

        # Keep a reference to the aligned dataset before converting to DataLoader
        # (DataLoader strips non-tensor columns; we need it for metadata later)
        aligned_dataset = dataset

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        layer_activations = {layer: [] for layer in self.layers}
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"]

                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

                # hidden_states: (embeddings, layer1, …, layer12) — layers are 1-indexed.
                # [:, 0, :] selects the CLS token but returns a non-contiguous VIEW
                # into the full (batch, seq_len, hidden) tensor.  .numpy() on that
                # view would keep the entire tensor alive in memory.  .contiguous()
                # copies just the (batch, hidden) slice into its own storage.
                for layer_idx in self.layers:
                    cls = outputs.hidden_states[layer_idx][:, 0, :].contiguous()
                    layer_activations[layer_idx].append(cls.cpu().numpy())

                del outputs
                all_labels.append(labels.numpy())

        for layer_idx in self.layers:
            layer_activations[layer_idx] = np.vstack(layer_activations[layer_idx])

        labels_array = np.concatenate(all_labels)

        print(f"\nExtracted activations from {len(self.layers)} layers")
        print(f"  Shape per layer: {layer_activations[self.layers[0]].shape}")
        print(f"  Total samples: {len(labels_array)}")

        return layer_activations, labels_array, aligned_dataset

    def extract_and_save(
        self,
        tokenized_dataset_path: str,
        output_dir: str,
        batch_size: int = 64,
        max_samples: int = None,
        split: str = "train"
    ):
        """
        Extract activations and save to disk.

        Args:
            tokenized_dataset_path: Path to tokenized dataset
            output_dir: Directory to save activations
            batch_size: Batch size for extraction
            max_samples: Maximum samples to process
            split: Dataset split to use ('train' or 'test')
        """
        from datasets import load_from_disk

        print(f"Loading tokenized dataset from {tokenized_dataset_path}...")
        dataset_dict = load_from_disk(tokenized_dataset_path)
        dataset = dataset_dict[split]

        print(f"\nExtracting activations from {split} split...")
        activations, labels, aligned_dataset = self.extract_activations(
            dataset=dataset,
            batch_size=batch_size,
            max_samples=max_samples,
            desc=f"Extracting {split}"
        )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for layer_idx, layer_acts in activations.items():
            save_path = output_path / f"layer_{layer_idx}_activations.npy"
            np.save(save_path, layer_acts)
            print(f"  Saved layer {layer_idx}: {save_path}")

        labels_path = output_path / "labels.npy"
        np.save(labels_path, labels)
        print(f"  Saved labels: {labels_path}")

        metadata = {
            "layers": self.layers,
            "n_samples": len(labels),
            "n_ai": int((labels == 1).sum()),
            "n_human": int((labels == 0).sum()),
            "split": split,
            "activation_shape": list(activations[self.layers[0]].shape),
        }

        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved metadata: {metadata_path}")

        # Save per-sample metadata (domain, text length) aligned with the arrays
        try:
            from raid_analysis.data.metadata import (
                compute_metadata_from_dataset,
                save_metadata,
            )
            sample_meta = compute_metadata_from_dataset(aligned_dataset)
            sample_meta_path = save_metadata(sample_meta, output_path)
            print(f"  Saved sample metadata: {sample_meta_path}")
        except Exception as exc:
            print(f"  WARNING: could not save sample_metadata.npz — {exc}")
            print("  Re-run this script to regenerate it; downstream "
                  "experiments (confound analysis, per-domain "
                  "breakdowns) require it.")

        print(f"\nAll activations saved to {output_path}")
