"""Check the published target-bearing resources against the public B task timelines."""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def utc(values):
    return pd.DatetimeIndex(pd.to_datetime(values, format="mixed", utc=True)).as_unit("ns")


def disjoint(values, hidden, location):
    if set(utc(values).asi8) & hidden:
        raise ValueError("B target timestamp exposed in " + location)


def csv_axis(path, time_column):
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("gb18030")
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    columns = next(reader)
    index = columns.index(time_column)
    return [row[index] for row in reader]


def verify(root=ROOT):
    from castbench.ab_data import Dataset
    from castbench.domain_packs import load_domain_episodes

    root = Path(root)
    data = root / "benchmark_data"
    for name in ("evaluation/B", "test_labels/B"):
        if (data / name).exists():
            raise ValueError("Private B labels must not be inside the public package")
    cohorts = {board: read(data / "cohorts" / (board + ".json")) for board in ("A", "B")}
    if set(cohorts["A"]) != set(cohorts["B"]) or len(cohorts["A"]) != 25:
        raise ValueError("Public dataset inventory differs between boards")
    if any(sum(map(len, cohorts[b].values())) != count for b, count in {"A": 300, "B": 100}.items()):
        raise ValueError("Public boards must contain 300 A tasks and 100 B tasks")
    cases = histories = labels = 0
    for name in cohorts["A"]:
        bundles = {
            board: load_domain_episodes(data / "packs" / board / name, cohorts[board][name])
            for board in ("A", "B")
        }
        hidden = set(
            utc(
                [
                    t
                    for bundle, _ in bundles["B"].values()
                    for t in bundle.episode.forecast_timeline.scored_timestamps
                ]
            ).asi8
        )
        meta = read(data / "datasets" / name / "metadata.json")
        features = read(data / "datasets" / name / "features.json")
        disjoint(
            csv_axis(root / "raw_data/csv" / name / "data.csv", meta["time_column"]),
            hidden,
            name + "/source CSV",
        )
        table = read(root / "raw_data/json" / name / "data.json")
        time_index = table["columns"].index(meta["time_column"])
        disjoint([row[time_index] for row in table["data"]], hidden, name + "/source JSON")
        for part in ("train", "validation", "test"):
            disjoint(
                csv_axis(data / "datasets" / name / (part + ".csv"), meta["time_column"]),
                hidden,
                name + "/" + part,
            )
        for board in ("A", "B"):
            numerical = Dataset(name, board=board, storage_path=data)
            series = numerical.hf_dataset[0]
            start = pd.Timestamp(series["start"])
            start = start.tz_localize("UTC") if start.tz is None else start.tz_convert("UTC")
            axis = pd.date_range(start, periods=len(series["target"]), freq=meta["frequency"])
            disjoint(axis[np.isfinite(series["target"])], hidden, name + "/numerical series")
            for item in numerical.input_dataset:
                start = pd.Timestamp(item["start"])
                if start.tz is None:
                    start = start.tz_localize("UTC")
                axis = pd.date_range(start, periods=len(item["target"]), freq=meta["frequency"])
                disjoint(axis, hidden, name + "/numerical input")
                bundle = bundles[board][str(item["item_id"])][0]
                variables = {v.name: v for v in bundle.variables}
                np.testing.assert_array_equal(item["target"], bundle.episode.history.target_values)
                input_features = read(data / "test_inputs" / board / name / "features.json")
                for key in ("past_feat_dynamic_real", "feat_dynamic_real"):
                    names = features[key]
                    if input_features[key] != names:
                        raise ValueError("Numerical channel order differs between series and task input")
                    if not names:
                        continue
                    length = (
                        len(item["target"])
                        if key == "past_feat_dynamic_real"
                        else len(bundle.variables[0].values)
                    )
                    expected = np.asarray(
                        [
                            [np.nan if v is None else float(v) for v in variables[field].values[:length]]
                            for field in names
                        ]
                    )
                    np.testing.assert_allclose(
                        item[key],
                        expected,
                        rtol=0,
                        atol=0,
                        equal_nan=True,
                        err_msg="Numerical covariate masks differ from task inputs",
                    )
            for identifier, (bundle, _) in bundles[board].items():
                episode = bundle.episode
                disjoint(episode.history.timestamps, hidden, name + "/task history")
                histories += 1
                for case in bundle.samples:
                    disjoint(
                        case.history_timestamps + case.future_timestamps, hidden, name + "/training case"
                    )
                    cases += 1
                timeline = (
                    episode.history.timestamps
                    + episode.forecast_timeline.gap_timestamps
                    + episode.forecast_timeline.scored_timestamps
                )
                step = pd.tseries.frequencies.to_offset(episode.task.frequency)
                for variable in bundle.variables:
                    if variable.historical_target_lag:
                        lag = variable.historical_target_lag.lag_steps
                        for stamp, value in zip(utc(timeline), variable.values, strict=True):
                            if value is not None and (stamp - lag * step).value in hidden:
                                raise ValueError(
                                    "B target dependency exposed by a historical-target covariate"
                                )
                if board == "A":
                    import hashlib

                    label = read(
                        data
                        / "evaluation/A"
                        / name
                        / (hashlib.sha256(identifier.encode()).hexdigest() + ".json")
                    )
                    disjoint(label["timestamps"], hidden, name + "/A labels")
                    labels += 1
                    if label["timestamps"] != episode.forecast_timeline.scored_timestamps:
                        raise ValueError("A label timeline differs from task")
            if board == "A":
                import hashlib

                import datasets

                numerical_labels = datasets.load_from_disk(str(data / "test_labels/A" / name))
                for item in numerical_labels:
                    key = hashlib.sha256(item["item_id"].encode()).hexdigest() + ".json"
                    truth = read(data / "evaluation/A" / name / key)
                    np.testing.assert_array_equal(item["target"], truth["values"])
    return {
        "datasets": 25,
        "tasks": histories,
        "training_case_uses": cases,
        "public_A_labels": labels,
        "B_labels_distributed": False,
        "api_calls": 0,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
