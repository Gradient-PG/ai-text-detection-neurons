"""
RAID data pipeline: dataset loading, tokenization, BERT activation extraction.
"""

from pathlib import Path

from .activation_extractor import ActivationExtractor
from .dataset_tokenizer import DatasetTokenizer
from .model_loader import BERT_MODEL_NAME, BERT_MODEL_REVISION, load_bert_model
from .raid_loader import ALL_DOMAINS, ALL_RAID_MODELS, RAIDConfig, load_raid, slug

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_root() -> Path:
    """Return the repository root (one level above ``raid_pipeline/``)."""
    return _PROJECT_ROOT


__all__ = [
    "ALL_DOMAINS",
    "ALL_RAID_MODELS",
    "ActivationExtractor",
    "BERT_MODEL_NAME",
    "BERT_MODEL_REVISION",
    "DatasetTokenizer",
    "RAIDConfig",
    "load_bert_model",
    "load_raid",
    "project_root",
    "slug",
]
