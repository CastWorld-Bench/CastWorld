# Public Release Contents

| Package | Contents |
| --- | --- |
| `assets/raw_<dataset>.zip` | Released source CSV and field report for each of 25 datasets |
| `../raw_data/csv/<dataset>/data.csv` | Byte-identical copies of the released CSV tables |
| `../raw_data/json/<dataset>/data.json` | Lossless JSON views with positional columns and original cell text |
| `assets/dataset_<dataset>.zip` | A/B tasks, training cases, domain resources, numerical datasets and public A labels |
| `assets/benchmark_protocol.zip` | A/B cohorts, policies, resource profiles and native model profiles |
| `assets/native_cache.zip` | Authorized Chronos, TimesFM, Sundial and Moirai candidates for the A/B tasks |
| `../src/castbench` | Harness, provider interfaces, tools, state/version management and scoring |

Native candidate coverage is currently 500/1,600. Data and task inputs are complete; full cache-only generation requires the remaining candidates. Live local-model inference is supported. The runner preflights selected caches before any LLM call.

Archives materialize into `benchmark_data` and `datasets`. B labels and complete organizer source series are not included. The homepage and README display A/B, each with Model and System levels. Experimental outputs and historical validation reports are not distributed.

```bash
python tools/reproduce.py prepare
python tools/verify_release_scope.py
python tools/export_raw_data.py check
python tools/verify_public_data.py
python tools/reproduce.py audit --cohort A --no-cache
python tools/reproduce.py audit --cohort B --no-cache
```

ZIPs are regular Git files, not Git LFS pointers. The [manifest](manifest.json) records every archive and member hash. The [usage guide](../docs/GETTING_STARTED.md) and [reproduction guide](../docs/REPRODUCTION.md) cover the current paths and evaluation commands.
