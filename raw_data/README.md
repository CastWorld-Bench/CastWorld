# Raw Data in CSV and JSON

[CSV tables](csv) | [JSON tables](json) | [File manifest](manifest.json)

This directory provides equivalent views of the 25 released source series. These are source tables, not prepared forecasting episodes or experiment outputs. The `release/assets/raw_<dataset>.zip` archives are the authoritative public source packages. The released scope matches the A/B data package; complete organizer source series and B ground truth are not distributed here.

```text
raw_data/
  csv/<dataset>/data.csv
  json/<dataset>/data.json
  manifest.json
```

Both format directories are included in the repository. No preparation command, model, API key, or extra dependency is needed to read them. Each CSV is a byte-identical copy of the source archive's primary `data.csv`; supplementary field mappings and source reports remain in the original archives.

## Lossless JSON Table

Each JSON document contains `columns`, `data`, `row_count`, `column_count`, and source/hash metadata. `data` is an array of row arrays; positions correspond to `columns`. The original column order, row order, empty/duplicate column names, whitespace, and embedded text are preserved.

All cell values are **JSON strings representing the original CSV cell text**. This avoids rounding decimal values or large integers, changing timestamps/timezones, dropping leading zeros, or interpreting literal strings such as `NA` or `NaN` as missing values. An empty CSV cell remains `""`, not `null`. Existing raw empty columns are retained; they do not become additional benchmark features. Source-column suffixes such as `**`, `***`, and `#` are unchanged.

JSON files use UTF-8. CSV bytes retain their original encoding, recorded under `source.encoding` in the JSON document and manifest. Most sources use UTF-8; IndustryHeat-ActiveUnits uses GB18030. No time conversion, numerical coercion, imputation, renaming, row filtering, or feature engineering is applied.

Prepared A/B episodes remain separately organized in the task packages. Their typed values, permissions, timelines, splits, and evaluation configuration are unchanged. These raw-format views are not a replacement for prepared task inputs.

## Read a JSON Table

```python
import json
from pathlib import Path

table = json.loads(Path("raw_data/json/NYC-311/data.json").read_text(encoding="utf-8"))
columns = table["columns"]
rows = table["data"]
assert len(rows) == table["row_count"]
assert all(len(row) == len(columns) for row in rows)
```

For analysis with pandas, `pd.DataFrame(table["data"], columns=table["columns"])` preserves this positional representation. Convert selected named numerical columns explicitly according to the [field guide](../docs/DATASETS.md); the raw export deliberately performs no automatic type inference.

## Verify or Rebuild

```bash
python tools/export_raw_data.py check
python tools/export_raw_data.py build --out /new/empty/raw_data_copy
```

The verifier checks all 25 tables against the frozen source archives: source hashes, CSV byte identity, JSON metadata, dataset inventory/order, row/column counts, and every decoded cell. The manifest records SHA-256 identities for both formats and the canonical positional table content.

The builder is deterministic and refuses to overwrite modified exports. The directory is a convenience layer generated from the released source archives, not from organizer-held source files. It does not modify native caches, task packs, the harness, or scoring code.
