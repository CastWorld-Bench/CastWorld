# A/B Data Access

CastWorld-A contains 300 forecasting episodes and CastWorld-B contains 100, both spanning 25 datasets. Each board supports systems-level predictive evaluation and model-level evaluation in the common harness. The numerical release follows the organization used by [GIFT-Eval](https://github.com/SalesforceAIResearch/gift-eval): one Hugging Face `save_to_disk` directory per dataset, a `Dataset` loader exposing training, validation, and test data, and per-model `all_results.csv` / `config.json` result files.

## Files

```text
benchmark_data/
  datasets/<dataset>/
    data-00000-of-00001.arrow
    dataset_info.json
    state.json
    metadata.json
    features.json
    splits.json
    train.csv, validation.csv, test.csv
    field_permissions.json
  test_inputs/A/<dataset>/
  test_inputs/B/<dataset>/
  test_labels/A/<dataset>/
  packs/A/<dataset>/
  packs/B/<dataset>/
  cohorts/A.json
  cohorts/B.json
  policies/A/<dataset>.json
  policies/B/<dataset>.json
  efficiency_profiles/<dataset>.json
  runtime/native_cache.jsonl
  runtime/native_models.json
```

Dataset directories contain a regular numerical series with the standard `item_id`, `start`, `freq`, and `target` fields. `past_feat_dynamic_real` contains historical numerical covariates; `feat_dynamic_real` contains declared known-future numerical covariates. `features.json` lists their names in array order. Existing CSV files and field dictionaries are retained. Missing observation slots in the full source series are represented by NaN, without interpolation; no observed value is replaced. The 400 fixed evaluation windows have regular timelines. Month-start observations use the GluonTS monthly Period frequency `M`; their actual timestamps remain unchanged in the task packs.

`test_inputs` contains one record per fixed forecasting episode. Its target contains history only. Future numerical covariates follow the original visibility masks. Non-numerical context, events, knowledge, historical cases, and tool definitions are accessed through the existing `packs` directories rather than converted into new numerical features. The A labels are public. B labels are stored separately in `CastWorld_AB_private/test_labels/B/` and are not part of the public upload.

The package includes native forecasting candidates for these fixed episodes, along with the common reference resource profiles and board-specific policy bindings. Native model weights are not included. Changing the input windows requires new candidates for the changed inputs.

## Loading

Run `python tools/reproduce.py prepare` to materialize all 25 dataset directories. Install this repository's optional data dependencies for the numerical adapter:

```bash
python -m pip install -e '.[data]'
export CASTWORLD_DATA=/path/to/CastWorld/benchmark_data
```

```python
from castbench.ab_data import Dataset

dataset = Dataset("iFLYTEK-Wind", board="A")
train = dataset.training_dataset
validation = dataset.validation_dataset
test = dataset.test_data
for model_input, future_label in test:
    pass
```

The loader is maintained once in `src/castbench/ab_data.py`. `training_dataset` is the training prefix; `validation_dataset` is the prefix through the validation endpoint, as in GIFT-Eval's cumulative context interface. The existing chronological split endpoints are retained. The numerical adapter does not resample the evaluation cohort or change its history/horizon lengths. Values remain float64 during loading.

Covariate channels use the same named order in the source series and A/B inputs. The loader rejects conflicting channel-order metadata; the public-data verifier checks channel values and masks against the Agent task bundles.

For B inference, use `Dataset(name, board="B").test_inputs`. Organizer-side evaluation can supply `private_path="/path/to/CastWorld_AB_private"` to obtain B `test_data`. An Agent run uses `dataset.task_bundles()`, which returns the original validated CastWorld bundles and registries with the same episode IDs.

## Results

The portable scorer writes a local `scores.json` containing per-episode scores and the dataset-macro summary. Organizer leaderboard ingestion can map approved aggregates to the following GIFT-Eval-style layout; these files are not pre-existing experiment exports in this repository.

```text
results/<model>/A/all_results.csv
results/<model>/A/config.json
results/<model>/B/all_results.csv
results/<model>/B/config.json
```

Each CSV has one row per dataset configuration, named `<dataset>/<frequency>/<A-or-B>`, with `model`, `domain`, `num_variates`, and `eval_metrics/...` columns. The metrics remain CastWorld's MASE, RMSSE, four dimension scores, and TotalScore; this packaging convention does not replace them with GIFT-Eval's probabilistic objectives. Per-episode forecasts, complete traces, and measurement receipts are retained by the existing execution/scoring workflow. Only completed and verified cohort results are published to a leaderboard.

The reference layout was checked against GIFT-Eval commit `82f89c3a2de1e7cd98e35dbcb6917cb4ebb6fd18`, its `src/gift_eval/data.py`, and its documented result submission format. This is CastWorld data using those standard interfaces, not a submission of CastWorld scores to the GIFT-Eval leaderboard.
