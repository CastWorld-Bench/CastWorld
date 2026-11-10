# Reproduction Details

## Public Release

This repository distributes the 25-dataset A/B release, the common harness and scorer, public source series, public A labels, and native forecasting-tool candidates. A contains 300 episodes and B contains 100; both support Model and System evaluation. Research-run outputs, reasoning traces, expert-review records, and historical validation logs are not included.

## Prepare and Verify

```bash
python tools/reproduce.py prepare
python tools/verify_release_scope.py
python tools/verify_dataset_documentation.py
python tools/export_raw_data.py check
python tools/verify_public_data.py
python tools/reproduce.py audit --cohort A --no-cache
python tools/reproduce.py audit --cohort B --no-cache
python tools/test_release.py
```

Preparation verifies every archive and member before extraction and refuses to overwrite modified files. Use a fresh clone for this release instead of mixing it with extracted resources from an older release. Preparation and auditing perform no model/API calls. Source CSV/JSON views are included directly under `raw_data`.

| Materialized path | Contents |
| --- | --- |
| `datasets/<dataset>` | Released source CSV and field report |
| `benchmark_data/datasets/<dataset>` | GIFT-Eval-style numerical series, chronological splits and field permissions |
| `benchmark_data/packs/A/<dataset>` | A task inputs, domain knowledge, tools, and training cases |
| `benchmark_data/packs/B/<dataset>` | B task inputs, without B labels |
| `benchmark_data/test_inputs/{A,B}` | Numerical model inputs |
| `benchmark_data/test_labels/A` | Public numerical A labels |
| `benchmark_data/evaluation/A` | Public A labels and normalization for the four-dimensional scorer |
| `benchmark_data/cohorts` | Fixed A/B episode identifiers |
| `benchmark_data/policies/{A,B}` | Resource policies |
| `benchmark_data/efficiency_profiles` | Frozen reference resource anchors |
| `benchmark_data/runtime` | Four native profiles and their cached candidates |

The public-boundary verifier checks B target timestamps against released source series, public histories, training cases and A labels. It does not require B true values. B scoring requires an external organizer label directory; the generation command has no label-path argument. See [data access](AB_DATA.md) for the numerical adapter.

## Native Tools

The new 400-task release currently includes 500 exact cached candidates out of 1,600 required. The input-only audits above can pass without complete candidate coverage; they must not be reported as full cache audits. Complete cache-only evaluation is not ready until the remaining 1,100 candidates are generated. A cache-only run checks selected candidates before starting LLM requests. Alternatively, use the existing live mode with your four deployed models.

Resource anchors retain their existing measured validation references. The copied profiles declare the new training-source bindings and A/B task policies before any new evaluation; no test outcomes are used to recalibrate anchors.

Cache-only mode is the default. Profile paths are immutable cache identifiers and do not need to exist locally. A missing cache entry fails rather than silently replacing the candidate.

For live Chronos, TimesFM, Sundial and Moirai inference, copy `benchmark_data/runtime/native_models.json` to a new local file and configure the actual interpreters, checkpoints, revisions, dependency versions, devices and checkpoint digests. Obtain weights separately, following the upstream sources and `release/third_party_notices`. Add `--native-mode live --model-profiles /absolute/path/models.json` to the run command. All four deployments are preflighted before agent generation; generated candidates are cached in the new run's local store.

Adapters are implemented once in `src/castbench/model_worker.py`. No model weights or installed Python/CUDA environment is bundled. A different deployment is a new condition, not a replacement for a frozen cache identity.

## New Runs and Scoring

Use Python 3.11 or 3.12 and install with `python -m pip install -c requirements-reproduce.txt -e ".[test,data]"`. The data extra supports the numerical adapter and complete public-boundary checks. Linux is recommended for native GPU workers; the portable runner supports Windows.

The [usage guide](GETTING_STARTED.md) covers provider profiles, explicit API opt-in, startup probes, complete A/B runs, retries and outputs. Repeat the unchanged command to resume. Changed releases, providers, credentials, model profiles or deployment settings require a new output directory.

`score` requires complete selected dataset cohorts and measured resource receipts. A scores from public labels. Organizers score B using `--labels-root /private/evaluation/B`; these labels are not loaded by the online agent. New predictions, traces and scores stay in local run directories and are not published automatically.
