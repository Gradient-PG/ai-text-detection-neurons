"""
Tokenize HuggingFace ``Dataset`` / ``DatasetDict`` and save to disk for extraction.

Used by the RAID pipeline (``tokenize_raid.py``); expects ``text`` and ``label`` columns.
"""

from pathlib import Path
from typing import List, Optional

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from .model_loader import BERT_MODEL_NAME, BERT_MODEL_REVISION

_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


class DatasetTokenizer:
    """
    Tokenizes text and saves as HuggingFace Dataset on disk.

    Args:
        tokenizer_name: HuggingFace tokenizer name
        revision: Exact model revision (commit SHA) for reproducibility.
        output_dir: Parent directory for saved dataset folders.
            Defaults to ``<project_root>/data/processed``.
        max_length: Max sequence length
        random_state: Seed (reserved for future use)
        text_column: Text column name (default ``text``)
        extra_columns: Additional columns to keep (e.g. RAID ``domain``)
    """

    def __init__(
        self,
        tokenizer_name: str = BERT_MODEL_NAME,
        revision: str = BERT_MODEL_REVISION,
        output_dir: str | Path | None = None,
        max_length: int = 512,
        random_state: int = 42,
        text_column: str = "text",
        extra_columns: Optional[List[str]] = None,
    ):
        self.tokenizer_name = tokenizer_name
        self.revision = revision
        self.output_dir = Path(output_dir) if output_dir is not None else _DEFAULT_OUTPUT_DIR
        self.max_length = max_length
        self.random_state = random_state
        self.text_column = text_column
        self.extra_columns = extra_columns or []

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=revision)

    def _preprocess_dataset(self, ds):
        text_col = self.text_column

        def preprocessing_function(row):
            row[text_col] = row[text_col].lower().strip()
            return row

        cleaned = ds.map(lambda x: preprocessing_function(x), batch_size=1000)
        return cleaned

    def _tokenize_dataset(
        self,
        dataset: Dataset,
        batch_size: int = 1000,
        desc: str = "Tokenizing",
    ) -> Dataset:
        """Tokenize texts and create HuggingFace Dataset."""
        text_col = self.text_column

        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_col],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

        remove_cols = [text_col]
        for col in dataset.column_names:
            if col not in ("label", *self.extra_columns) and col != text_col:
                remove_cols.append(col)

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=batch_size,
            remove_columns=remove_cols,
            desc=desc,
        )

        format_cols = ["input_ids", "attention_mask", "label"]
        for col in self.extra_columns:
            if col in tokenized_dataset.column_names:
                format_cols.append(col)

        tokenized_dataset.set_format(type="torch", columns=format_cols)

        return tokenized_dataset

    def tokenize_and_save(
        self,
        hf_base_dataset,
        batch_size: int = 1000,
        dataset_name: str = "raid_tokenized",
    ):
        """
        Tokenize a ``Dataset`` or ``DatasetDict`` and save to disk.

        A single ``Dataset`` is stored under the ``train`` split.
        """
        if isinstance(hf_base_dataset, Dataset):
            print(f"Loaded {len(hf_base_dataset)} samples (single split)")

            preprocessed = self._preprocess_dataset(hf_base_dataset)
            tokenized = self._tokenize_dataset(
                preprocessed, batch_size=batch_size, desc="Tokenizing",
            )
            dataset_dict = DatasetDict({"train": tokenized})
        else:
            total = sum(len(hf_base_dataset[s]) for s in hf_base_dataset)
            print(f"Loaded {total} samples across {list(hf_base_dataset.keys())} splits")

            splits = {}
            for split_name in hf_base_dataset:
                preprocessed = self._preprocess_dataset(hf_base_dataset[split_name])
                splits[split_name] = self._tokenize_dataset(
                    preprocessed, batch_size=batch_size,
                    desc=f"Tokenizing {split_name}",
                )
            dataset_dict = DatasetDict(splits)

        output_path = self.output_dir / dataset_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_dict.save_to_disk(str(output_path))

        print(f"Saved to {output_path}")
        return output_path
