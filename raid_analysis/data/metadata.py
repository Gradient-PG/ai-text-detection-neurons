"""Per-sample metadata aligned with activation arrays.

Metadata (text length, domain, generator) is computed once during activation
extraction and stored as ``sample_metadata.npz`` alongside the ``.npy`` files.
This ensures index-alignment with activation arrays without carrying the raw
text through the experiment pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class SampleMetadata:
    """Per-sample metadata aligned by index with activation arrays.

    All arrays have shape ``(N_samples,)``.
    """

    text_lengths: np.ndarray
    domain_ids: np.ndarray
    domain_names: list[str]

    def __len__(self) -> int:
        return len(self.text_lengths)

    def __getitem__(self, idx: np.ndarray | slice) -> "SampleMetadata":
        """Slice metadata by sample indices (for fold splitting)."""
        return SampleMetadata(
            text_lengths=self.text_lengths[idx],
            domain_ids=self.domain_ids[idx],
            domain_names=self.domain_names,
        )

    def __repr__(self) -> str:
        n = len(self)
        n_domains = len(self.domain_names)
        return f"SampleMetadata(n_samples={n}, domains={n_domains})"


def compute_metadata_from_dataset(dataset) -> SampleMetadata:
    """Extract metadata from an HF ``Dataset``.

    Expects a ``domain`` column and one of:
      - ``text`` or ``generation`` — character-level text length is computed.
      - ``input_ids`` — token count (non-padding) is used as a length proxy
        when the raw text column has been removed by tokenization.

    Args:
        dataset: A HuggingFace ``Dataset``.

    Returns:
        :class:`SampleMetadata` aligned with the dataset row order.
    """
    cols = dataset.column_names

    if "text" in cols:
        texts = dataset["text"]
        text_lengths = np.array([len(t) for t in texts], dtype=np.int64)
    elif "generation" in cols:
        texts = dataset["generation"]
        text_lengths = np.array([len(t) for t in texts], dtype=np.int64)
    elif "input_ids" in cols:
        input_ids = dataset["input_ids"]
        text_lengths = np.array(
            [int(sum(1 for tok in ids if tok != 0)) for ids in input_ids],
            dtype=np.int64,
        )
    else:
        raise ValueError(
            f"Cannot compute text lengths: dataset has columns {cols}. "
            f"Expected 'text', 'generation', or 'input_ids'."
        )

    raw_domains: Sequence[str] = dataset["domain"]
    domain_names = sorted(set(raw_domains))
    domain_to_id = {name: idx for idx, name in enumerate(domain_names)}
    domain_ids = np.array(
        [domain_to_id[d] for d in raw_domains], dtype=np.int64
    )

    return SampleMetadata(
        text_lengths=text_lengths,
        domain_ids=domain_ids,
        domain_names=domain_names,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

_METADATA_FILENAME = "sample_metadata.npz"


def save_metadata(metadata: SampleMetadata, directory: str | Path) -> Path:
    """Save metadata as ``.npz`` in *directory*.

    Returns the path to the written file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _METADATA_FILENAME
    np.savez(
        path,
        text_lengths=metadata.text_lengths,
        domain_ids=metadata.domain_ids,
        domain_names=np.array(metadata.domain_names, dtype=str),
    )
    return path


def load_metadata(directory: str | Path) -> SampleMetadata:
    """Load metadata from ``.npz`` produced by :func:`save_metadata`."""
    path = Path(directory) / _METADATA_FILENAME
    data = np.load(path, allow_pickle=False)
    return SampleMetadata(
        text_lengths=data["text_lengths"],
        domain_ids=data["domain_ids"],
        domain_names=data["domain_names"].tolist(),
    )


def metadata_exists(directory: str | Path) -> bool:
    """Check whether sample metadata has been saved in *directory*."""
    return (Path(directory) / _METADATA_FILENAME).exists()
