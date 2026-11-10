"""GIFT-Eval-style access to the fixed CastWorld A/B task release.

Numerical series use Hugging Face save_to_disk and GluonTS DataEntry fields.
The existing CastWorld task packs remain the source of agent context and tools.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


class Dataset:
    def __init__(
        self, name, board="A", storage_path=None, storage_env_var="CASTWORLD_DATA", private_path=None
    ):
        import datasets

        if board not in {"A", "B"}:
            raise ValueError("board must be A or B")
        configured = storage_path or os.getenv(storage_env_var)
        if not configured:
            raise ValueError(f"Set {storage_env_var} or pass storage_path")
        self.root = Path(configured)
        self.name, self.board = name, board
        self.private_root = Path(private_path) if private_path else None
        self.metadata = json.loads(
            (self.root / "datasets" / name / "metadata.json").read_text(encoding="utf-8")
        )
        self.splits = json.loads((self.root / "datasets" / name / "splits.json").read_text(encoding="utf-8"))
        self.hf_dataset = self._load(datasets, self.root / "datasets" / name)
        self.input_dataset = self._load(datasets, self.root / "test_inputs" / board / name)
        series_features = self.root / "datasets" / name / "features.json"
        input_features = self.root / "test_inputs" / board / name / "features.json"
        if series_features.exists() and input_features.exists():
            canonical = json.loads(series_features.read_text(encoding="utf-8"))
            actual = json.loads(input_features.read_text(encoding="utf-8"))
            for key in ("past_feat_dynamic_real", "feat_dynamic_real"):
                if canonical.get(key, []) != actual.get(key, []):
                    raise ValueError("Covariate channel order must match between series and test inputs")

    @staticmethod
    def _load(datasets, path):
        data = datasets.load_from_disk(str(path))
        columns = [
            key
            for key in ("target", "past_feat_dynamic_real", "feat_dynamic_real")
            if key in data.column_names
        ]
        return data.with_format("numpy", columns=columns, output_all_columns=True, dtype=np.float64)

    @property
    def freq(self):
        return self.hf_dataset[0]["freq"]

    @property
    def prediction_length(self):
        return self.metadata["horizon"]

    @property
    def context_length(self):
        return self.metadata["history"]

    @property
    def target_dim(self):
        return 1

    @property
    def windows(self):
        return len(self.input_dataset)

    @property
    def dataset_config(self):
        return f"{self.name}/{self.freq}/{self.board}"

    def _entry(self, row, end=None):
        value = dict(row)
        value["start"] = pd.Period(pd.Timestamp(value["start"]), freq=self.freq)
        if end is not None:
            for key in ("target", "past_feat_dynamic_real", "feat_dynamic_real"):
                if key in value:
                    value[key] = value[key][..., :end]
        return value

    @property
    def training_dataset(self):
        return [self._entry(self.hf_dataset[0], self.splits["train_end"])]

    @property
    def validation_dataset(self):
        end = self.splits["validation_end"]
        return [self._entry(self.hf_dataset[0], end)]

    @property
    def test_inputs(self):
        return [self._entry(row) for row in self.input_dataset]

    @property
    def test_data(self):
        import datasets
        from gluonts.dataset.split import split

        root = self.root if self.board == "A" else self.private_root
        if root is None:
            raise FileNotFoundError("B labels are organizer-held; use test_inputs for public inference")
        label_path = root / "test_labels" / self.board / self.name
        labels = self._load(datasets, label_path)
        inputs = self.test_inputs
        if [row["item_id"] for row in labels] != [row["item_id"] for row in inputs]:
            raise ValueError("Test inputs and labels are not aligned")
        complete = [
            dict(item, target=np.concatenate([item["target"], label["target"]]))
            for item, label in zip(inputs, labels, strict=True)
        ]
        _, template = split(complete, offset=self.context_length)
        return template.generate_instances(
            prediction_length=self.prediction_length, windows=1, max_history=self.context_length
        )

    def task_bundles(self):
        from castbench.domain_packs import load_domain_episodes

        identifiers = [str(row["item_id"]) for row in self.input_dataset]
        return load_domain_episodes(self.root / "packs" / self.board / self.name, identifiers)
