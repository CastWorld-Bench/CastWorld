# Running New Evaluations

The public package provides task data, the interaction environment, tools, and scoring. A/B leaderboards are maintained in the [README](../README.md#leaderboards) and project homepage. Researcher experiment records and human-review materials are not included.

## Standard Evaluation

1. Install the environment and run `python tools/reproduce.py prepare`.
2. Run `python tools/reproduce.py audit --cohort A --no-cache` or `--cohort B --no-cache` for input checks. Omit `--no-cache` only when verifying complete candidate coverage.
3. Configure a Chat Completions provider using `examples/portable_deepseek.json` and its named API-key environment variable.
4. Run the [documented evaluation command](GETTING_STARTED.md) with a new output directory.
5. Score the completed cohort with `python tools/reproduce.py score --out runs/your_run`.

A scores locally using public labels. B runs without labels; its organizer-side scoring requires `--labels-root /private/evaluation/B`.

Use `--dataset DATASET_ID` for complete dataset subsets. A startup probe with `--limit 1` needs a separate output directory and is not a complete benchmark result. Do not silently discard failed samples when aggregating.

## New Conditions

The registry, public bundle, resource policy, and scoring profile define an evaluation condition. Freeze changes before running. Information/tool ablations change the model-visible environment and require new generation; removing steps from an existing trace does not reproduce such an intervention.

For new window lengths, construct compatible history, forecast timestamps, context permissions, evaluator labels, normalization, and tool candidates. Existing native candidates apply only to their exact cached inputs. Use live native inference for new inputs.

## Outputs and Resume

New predictions, traces, raw provider responses, scores, and resource records are written to the selected local run directory. They remain the runner's responsibility and are excluded from Git by default. The scoring command reads these new local outputs; no author experiment outputs are needed.

Repeat the same command to resume. `--retry-failed` continues infrastructure-interrupted episodes with the same run ID, committed state, budget and receipts, retaining every recovery snapshot. Finished failed deliveries remain final and are included in scoring; unknown local errors require review. Changes to the provider, protocol, model profiles, deployment, or release manifest require a new run directory. Keep the original checkout when continuing a run created by an older release.

## Native Forecasting Tools

The default mode uses the public Chronos, TimesFM, Sundial, and Moirai candidate cache, currently covering 500 of 1,600 candidates. Complete cache-only evaluation is not ready; a selected missing candidate stops generation before any LLM request. `--native-mode live --model-profiles /absolute/path/models.json` uses your own four local deployments. This changes the deployment condition and must be recorded with the new run. See [native-tool setup](REPRODUCTION.md#native-tools).
