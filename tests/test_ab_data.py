"""Small round-trip checks for the optional GIFT-Eval-style data adapter."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


@unittest.skipUnless(
    importlib.util.find_spec("datasets") and importlib.util.find_spec("gluonts"),
    "Install the data extra to test the optional dataset adapter",
)
class ABDataTests(unittest.TestCase):
    def setUp(self):
        import datasets

        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.name = "example"
        self.value = 0.33333333333333337
        schema = datasets.Features(
            dict(
                item_id=datasets.Value("string"),
                start=datasets.Value("timestamp[us]"),
                freq=datasets.Value("string"),
                target=datasets.Sequence(datasets.Value("float64")),
            )
        )
        from datetime import datetime

        self.schema = schema

        def save(path, identifier, start, values):
            datasets.Dataset.from_list(
                [dict(item_id=identifier, start=start, freq="D", target=values)], features=schema
            ).save_to_disk(str(path))

        folder = self.root / "datasets" / self.name
        save(folder, self.name, datetime(2020, 1, 1), [self.value] * 10)
        (folder / "metadata.json").write_text(json.dumps(dict(history=2, horizon=1)))
        (folder / "splits.json").write_text(json.dumps(dict(train_end=4, validation_end=6, test_end=10)))
        for board in ("A", "B"):
            save(
                self.root / "test_inputs" / board / self.name,
                board + "-episode",
                datetime(2020, 1, 7),
                [self.value] * 2,
            )
        save(self.root / "test_labels/A" / self.name, "A-episode", datetime(2020, 1, 9), [self.value])

    def tearDown(self):
        self.temp.cleanup()

    def load(self, board="A"):
        from castbench.ab_data import Dataset

        return Dataset(self.name, board, storage_path=self.root)

    def test_prefix_splits(self):
        data = self.load()
        self.assertEqual(len(data.training_dataset[0]["target"]), 4)
        self.assertEqual(len(data.validation_dataset[0]["target"]), 6)
        self.assertEqual(data.prediction_length, 1)
        self.assertEqual(data.windows, 1)

    def test_float64_is_not_silently_downcast(self):
        values = self.load().test_inputs[0]["target"]
        self.assertEqual(values.dtype, np.float64)
        self.assertEqual(values[0], self.value)
        self.assertNotEqual(values[0], float(np.float32(self.value)))

    def test_standard_gluonts_test_data(self):
        from gluonts.dataset.split import TestData

        data = self.load()
        test = data.test_data
        self.assertIsInstance(test, TestData)
        pair = list(test)[0]
        self.assertEqual(pair[0]["item_id"], pair[1]["item_id"])
        self.assertEqual(len(pair[0]["target"]), 2)
        self.assertEqual(len(pair[1]["target"]), 1)
        self.assertEqual(str(pair[1]["start"]), "2020-01-09")
        self.assertEqual(len(data.test_inputs[0]["target"]), 2)

    def test_public_B_is_input_only(self):
        data = self.load("B")
        self.assertEqual(len(data.test_inputs), 1)
        with self.assertRaises(FileNotFoundError):
            data.test_data

    def test_invalid_board(self):
        from castbench.ab_data import Dataset

        with self.assertRaises(ValueError):
            Dataset(self.name, "C", storage_path=self.root)

    def test_inconsistent_covariate_order_is_rejected(self):
        canonical = dict(past_feat_dynamic_real=[], feat_dynamic_real=["temperature", "wind"])
        wrong = dict(past_feat_dynamic_real=[], feat_dynamic_real=["wind", "temperature"])
        (self.root / "datasets" / self.name / "features.json").write_text(json.dumps(canonical))
        (self.root / "test_inputs/A" / self.name / "features.json").write_text(json.dumps(wrong))
        with self.assertRaisesRegex(ValueError, "channel order"):
            self.load()


if __name__ == "__main__":
    unittest.main()
