# Getting Started

This guide covers the portable CastWorld runtime and public A/B evaluations. A contains 300 episodes and B contains 100, both spanning 25 datasets. Both support Model and System evaluation. A includes public labels; B supplies model-visible inputs only.

Download or clone this repository and open a terminal in its root directory, containing `pyproject.toml`, `tools/`, and `release/`. Use a complete checkout with the release assets. The commands below describe the supported workflow; they do not assert that a particular provider or local environment has already passed it.

## 1. Create the runtime environment

Use Python 3.11 or 3.12 for the documented pinned environment.

```bash
python -m venv .venv
```

Activate it in a POSIX shell:

```bash
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then install the package and test dependencies:

```bash
python -m pip install -c requirements-reproduce.txt -e ".[test,data]"
```

The constraints file pins the core generation and scoring libraries. Use it with the editable installation above, rather than installing only the constraints file. Native forecasting workers use separate environments and are unnecessary for the cache-only portable workflow.

## 2. Prepare and audit the frozen inputs

Original tables can be read directly from `raw_data/csv/<dataset>/data.csv` or `raw_data/json/<dataset>/data.json`, without extraction. These equivalent raw views are separate from the prepared task packages used below. JSON preserves original cell text in positional row arrays; the [format guide](../raw_data/README.md) covers columns, missing cells, source encodings, and `python tools/export_raw_data.py check`.

```bash
python tools/reproduce.py prepare
python tools/reproduce.py audit --cohort A --no-cache
python tools/reproduce.py audit --cohort B --no-cache
```

`prepare` verifies and extracts the ZIP assets already stored under `release/assets`. It checks archive and member hashes, preserves admitted packaging exceptions, and refuses to overwrite an existing modified input. It does not download datasets or call an LLM. Allow space for the extracted inputs, the Python environment, and later run logs.

Each audit checks 25 datasets, episode identities, public bundles, policy/profile bindings, and four cached native candidates per episode. A additionally checks its public label timelines; B auditing never opens private labels. Reports are written under `.reproduction`. Run `python tools/verify_public_data.py` to check that public source series, histories, cases, and A labels do not expose B target timestamps. These checks make no LLM calls or fresh native-model inferences.

The `--no-cache` flag above omits candidate-cache verification. The expanded release currently contains 500 of 1,600 required native candidates, so complete cache-only evaluation is not ready. Do not describe these input-only checks as complete input-and-candidate audits. The run examples below use live native inference: first configure four local models and replace `/absolute/path/models.json` with your deployment profile. With complete matching caches, omit the two live-mode arguments. Cache-only runs stop before any LLM request if a selected candidate is missing. Keep the prepared relative directory structure intact; copied episode JSON files alone are insufficient for all dependencies.

## 3. Run an offline smoke test

```bash
python -m castbench.cli demo --out runs/offline_demo
python tools/test_release.py
```

The demo uses synthetic data and a scripted agent. It creates public/private sample packages, a state store, `trace.json`, and `score.json` under its output directory. It needs neither API credentials nor native model weights. Its score uses the legacy single-run smoke path and must not be reported as a robust-v1.3 A/B evaluation.

The test wrapper blocks external connections while allowing local test servers. Prepare the release inputs before running the complete suite. Extra arguments are passed to pytest, for example:

```bash
python tools/test_release.py -k portable_reproduction
```

## 4. Configure an API provider

The following examples use `examples/portable_deepseek.json`. Read that file before running it. It specifies a particular Chat Completions endpoint and model, automatic native tool calls, streaming and reasoning options, token capacities, timeouts, and transport recovery settings. For another deployment, make a local copy, set its supported options explicitly, and replace the `--provider` path in every command.

The provider must support the native `environment_action` function-call interface and report input/output token usage. The endpoint is a complete Chat Completions URL, not a Responses API URL. The output allowance plus context reserve must leave room for input. A JSON `null` temperature omits that parameter; it does not set temperature to zero.

Set the variable named by the profile's `api_key_env`. The supplied example uses `CASTWORLD_API_KEY`:

```bash
export CASTWORLD_API_KEY='YOUR_KEY'
```

In PowerShell:

```powershell
$env:CASTWORLD_API_KEY='YOUR_KEY'
```

Keep credentials out of tracked files. Portable generation requires the explicit `--allow-api` opt-in; provider charges are external, so configure an appropriate provider-side quota.

Replace `DESCRIBE_YOUR_DEPLOYMENT` below with the actual host/service and execution setup. That description is retained in the run manifest.

## 5. Check one episode in an independent probe

```bash
python tools/reproduce.py run --cohort A --dataset NYC-311 --limit 1 --provider examples/portable_deepseek.json --out runs/nyc311_probe --deployment "DESCRIBE_YOUR_DEPLOYMENT" --workers 1 --active-requests 1 --rpm 12 --native-mode live --model-profiles /absolute/path/models.json --allow-api
```

Inspect the printed `planned`, `processed`, and `valid` counts, plus `progress.json` and the episode attempt record. A completed request alone does not establish valid forecast delivery.

This probe checks startup and provider interaction. It is not a complete cohort and cannot be scored as one. `--limit` truncates the job list globally after dataset selection, while the manifest retains the full selected dataset IDs. It also becomes part of the immutable run configuration. Do not remove `--limit` and try to continue this output directory as a complete run; use the new directory in the next step.

## 6. Evaluate and score one complete dataset

The following command runs the complete NYC-311 A cohort:

```bash
python tools/reproduce.py run --cohort A --dataset NYC-311 --provider examples/portable_deepseek.json --out runs/nyc311_A --deployment "DESCRIBE_YOUR_DEPLOYMENT" --workers 1 --active-requests 1 --rpm 12 --native-mode live --model-profiles /absolute/path/models.json --allow-api
python tools/reproduce.py score --out runs/nyc311_A
```

Repeat `--dataset` to select several complete dataset cohorts. Names must match the dataset IDs exactly. Generation receives only public bundles and authorized resources. A scoring reads the distributed public A labels; B scoring requires organizer-held external labels.

`score` derives cohort and dataset selection from `run_manifest.json`. Every planned episode must have a finished trace and complete accounting. Finished failed deliveries are retained and scored by the existing invalid-delivery rules; they are not rerun until successful. Missing episodes or token receipts are not replaced with zero or estimates. Resolve infrastructure interruptions before scoring the selected cohort.

Use this portable `score` command for robust-v1.3 evaluation. The lower-level `castbench score` command defaults to a legacy single-run profile and is not an interchangeable substitute.

## 7. Run A or B

For all 300 A episodes, use a separate output directory:

```bash
python tools/reproduce.py run --cohort A --provider examples/portable_deepseek.json --out runs/deepseek_A --deployment "DESCRIBE_YOUR_DEPLOYMENT" --workers 20 --active-requests 20 --rpm 12 --native-mode live --model-profiles /absolute/path/models.json --allow-api
python tools/reproduce.py score --out runs/deepseek_A
```

For all 100 B episodes, run inference without labels. Only organizers run the subsequent scoring command with their private label directory:

```bash
python tools/reproduce.py run --cohort B --provider examples/portable_deepseek.json --out runs/deepseek_B --deployment "DESCRIBE_YOUR_DEPLOYMENT" --workers 20 --active-requests 20 --rpm 12 --native-mode live --model-profiles /absolute/path/models.json --allow-api
python tools/reproduce.py score --out runs/deepseek_B --labels-root /private/evaluation/B
```

`--workers` controls concurrent episodes, `--active-requests` bounds active LLM requests, and `--rpm` paces request starts. Rounds within an episode remain sequential. The request gate is shared within one portable process, not across separate processes. Choose these settings for the deployment before starting a run.

By default, native tool predictions come from the verified public cache. A missing candidate fails rather than silently launching another model. The paths in the frozen cache profile are identity strings; do not rewrite them for cache-only execution. To generate candidates with your own four local models, add `--native-mode live --model-profiles /absolute/path/models.json`; see [native-tool setup](REPRODUCTION.md#native-tools).

## 8. Resume or retry without changing the configuration

Repeat the original command to resume. All finished episodes are skipped, including finished failed deliveries. Add `--retry-failed` only to continue an infrastructure-interrupted run:

```bash
python tools/reproduce.py run --cohort A --dataset NYC-311 --provider examples/portable_deepseek.json --out runs/nyc311_A --deployment "DESCRIBE_YOUR_DEPLOYMENT" --workers 1 --active-requests 1 --rpm 12 --native-mode live --model-profiles /absolute/path/models.json --allow-api --retry-failed
```

Recovery retains the same run ID, committed state, interaction budget and resource receipts. Each invocation writes a separate attempt snapshot and keeps the earlier snapshots, including partial traces. It does not restart a completed failed run or select a more successful attempt. Unknown local failures are marked `needs_review` rather than retried blindly. Incomplete usage receipts remain a scoring error; recovery does not invent missing token counts.

The run identity includes the cohort, selected IDs, provider configuration, credential fingerprint, deployment description, worker/request/RPM settings, limit, runtime hashes, and release-manifest hash. Changing any of these requires a new output directory. In particular, credential rotation and altered concurrency can trigger configuration-drift rejection even if the model name is unchanged. Adding `--retry-failed` does not change the identity.

Keep live SQLite stores on local disk and do not share an active output directory between processes or hosts. An output lock prevents simultaneous portable runs in the same directory.

## 9. Find and interpret the outputs

```text
RUN_DIRECTORY/
  run_manifest.json
  progress.json
  store/
  episodes/
    DATASET/
      SHA256_OF_SAMPLE_ID/
        attempt_0001.json
        trace_a1.json
        complete.json
  scores.json
```

`run_manifest.json` records the fixed run identity. `progress.json` summarizes processed and valid episodes. Each attempt row records its outcome; a failed attempt may lack a completed trace. `trace_a1.json` contains the exported interaction record when execution and export complete. `complete.json` points to a legal delivery. Later attempts receive new numbers.

Separate scoring writes per-episode scores and the dataset-macro summary to `scores.json`. Efficiency uses complete dataset-cohort resource means and the frozen validation anchors. Preserve traces, original attempts, provider configuration, and actual resource receipts with any reported results. Review raw traces and deployment descriptions before sharing them in an anonymous package.

## 10. New conditions and live native inference

The portable script runs the released A/B tasks. Task, tool, budget, and window changes require a separately defined condition and a new output directory; they are not modifications of a completed result.

The [evaluation guide](EXPERIMENTS.md) describes creating new conditions. Information-access interventions require new generation rather than removal of steps from an existing trace. All new outputs remain in your own local run directory.

Fresh Chronos, TimesFM, Sundial, and Moirai computation requires matching upstream revisions, weights, checkpoint hashes, and worker environments. Supply a new local model-profile file with `--native-mode live --model-profiles /absolute/path/models.json`. Model weights and CUDA are not bundled. See [Reproduction Details](REPRODUCTION.md#native-tools). Do not change cache identities to make a different checkpoint appear equivalent.

## Troubleshooting

| Symptom | Check and next step |
|---|---|
| Missing archive, cohort directory, or candidate cache | Use a complete repository checkout containing `release/assets`, then run `prepare`. A sparse checkout or copied task JSON is insufficient. |
| `Refusing to overwrite a modified file` | Compare the changed input with the admitted release. Restore the intended version or use a separate checkout; preparation intentionally avoids silent replacement. |
| `Published runtime was modified` | Check changes to integrity-protected files. Use the admitted runtime or explicitly version the changed implementation instead of disabling verification. |
| Missing API credential | Set the environment variable named by the chosen provider profile in the same shell that launches the runner. |
| Tool/schema, reasoning, or streaming errors | Confirm native function-call support and profile options for that endpoint/model pair. Do not assume another provider accepts the example's options. |
| `Configuration drift on resume` | Compare provider settings, credential, cohort, limit, deployment, and concurrency parameters. Keep the original invocation or use a new output directory. |
| Failed episodes are not retried | Add `--retry-failed` to the otherwise unchanged `run` command. Inspect each recorded attempt. |
| Scoring fails after a probe or partial run | Use a new directory for a complete selected dataset cohort and obtain all required deliveries and receipts. Do not score a `--limit` probe. |
| Missing token usage or incomplete timing | Check provider receipts and interrupted trace segments. Complete-cohort efficiency cannot be reconstructed by substituting zero usage. |
| Missing frozen native prediction | Confirm package preparation and the unchanged frozen model profile. Cache-only mode intentionally forbids live substitution. |

Frozen inputs, cache outputs, and scoring definitions can be verified exactly. New LLM generations and deployment timings can vary, so retain the actual configuration and records rather than treating a model name as a guarantee of identical new results.
