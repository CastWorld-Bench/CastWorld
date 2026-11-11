# Benchmark specification

[Overview](../README.md) · [Usage guide](GETTING_STARTED.md) · [Leaderboards](../README.md#leaderboards)

## Evaluation object and design

CastWorld evaluates LLM-driven agents performing context-aware time series forecasting. The LLM organizes information acquisition, analysis, forecast construction, revision, and final selection through a shared interface. Holding the harness and task resources fixed allows LLMs to be evaluated as forecasting agents under consistent conditions.

The **context-rich dataset suite** supplies scenarios, accessible context, tools, and evaluation references. The **state-adaptive forecasting harness** instantiates episode contracts, executes actions, maintains shared state and immutable forecast versions, and records trajectories. Offline evaluation uses both the submitted forecast and that record. Model-level evaluation compares LLMs under the common harness and is the primary focus. Systems-level evaluation compares predictive accuracy across forecasting systems; controlled interventions examine budget, resource, and acquisition choices.

## Task inventory

The paper reports a 1,310-episode suite and a shared 250-episode comparison of 15 LLMs. The maintained A/B leaderboards use separate evaluation cohorts.

The public release contains 25 datasets across eight application domains, with 300 forecasting episodes on A and 100 on B. Each cohort supports Model and System evaluation. A labels are public; B labels remain organizer-held. An episode is a forecasting window, distinct from a dataset or an interaction round.

<!-- BEGIN DATASET_TABLE -->
| Application domain | Dataset | Frequency | History | Horizon | Released rows |
| --- | --- | --- | --- | --- | --- |
| Energy | iFLYTEK-PV | 15min | 96 | 96 | 34368 |
| Energy | iFLYTEK-Wind | 15min | 96 | 96 | 34272 |
| Energy | CN-GD-Load | h | 168 | 24 | 8160 |
| Energy | OPSD-DELU-Price | h | 96 | 24 | 14017 |
| Natural science | WB-T850_1 | 6h | 72 | 24 | 60988 |
| Natural science | WB-T850_2 | 6h | 72 | 24 | 60380 |
| Natural science | WB-T850_3 | 6h | 72 | 24 | 60924 |
| Natural science | SILSO-Sunspot | D | 30 | 30 | 4798 |
| Economics and finance | US-RPI | MS | 48 | 12 | 672 |
| Economics and finance | US-DFF | D | 30 | 30 | 7489 |
| Economics and finance | CN-Port-Export | D | 30 | 30 | 2511 |
| Public services | CapitalBikeshare | h | 168 | 24 | 16678 |
| Public services | NYC-311 | D | 28 | 7 | 5860 |
| Public services | US-ICSA | W-SAT | 52 | 13 | 940 |
| Public services | NYC-EMS | D | 28 | 7 | 4867 |
| Public services | CN-Flu-PosRate | W-MON | 52 | 13 | 583 |
| Retail | M5-TX3-FOODS | D | 35 | 28 | 1680 |
| Retail | RetailII-NetQty | h | 168 | 24 | 16720 |
| Industrial operations | IndustryHeat-ActiveUnits | D | 30 | 30 | 2344 |
| Industrial operations | MotorOil-GR-TI824 | 15min | 96 | 96 | 34268 |
| Web services | Abilene-SEA-NYC | 5min | 30 | 30 | 29688 |
| Web services | Wiki-EN-Views | D | 30 | 30 | 635 |
| Environment | CAMELS-SPAT_1 | D | 30 | 30 | 14969 |
| Environment | CAMELS-SPAT_2 | D | 30 | 30 | 14914 |
| Environment | AQS-ThousandOaks-O3 | h | 168 | 24 | 60648 |
<!-- END DATASET_TABLE -->

History and Horizon are measured in time steps. Frequency strings follow the packaged time axis: `MS` denotes month starts, while `W-MON` and `W-SAT` denote weekly timestamps anchored on Monday and Saturday. The [machine-readable inventory](data/dataset_inventory.csv) records the same counts and specifications. Standard policies allow up to 30 rounds and four read-only tool calls per round. Episode identifiers, schemas, permission masks, and scoring profiles are frozen in the [release manifest](../release/manifest.json) and prepared packages. `GEFCom2012-Z6` is inactive and is not counted.

The [dataset and field guide](DATASETS.md) describes each dataset and its contextual resources. Its field dictionary is available as [CSV](data/field_dictionary.csv) and [JSON](data/field_dictionary.json), with [unit interpretations](data/unit_interpretations.csv). The dictionary separates the recorded experimental metadata from explanations of field meaning and physical units.

## Information permissions

Each episode defines a target, units, observed history, forecast timestamps, resources, permissions, limits, and submission requirements. Context can include external series, events, domain knowledge, and historical analogue cases. A resource's presence in the suite does not make every field or timestamp visible.

Field- and time-specific permissions distinguish history-only variables from authorized future covariates. The suite uses controlled retrospective settings: authorized future exogenous realizations may serve as perfect forecasts under task assumptions. For NYC-311, weather uses realized or reanalysis values and scheduled events assume perfect plans without historical publication snapshots; calendar features are deterministic. Permissions specify experimental visibility, separately from historical online availability. Historical cases may expose completed continuations only before the training cutoff. The current episode's future targets remain private.

The frozen manifests and public bundles define the model-visible fields and timestamps. Source reports describe data collection and field conventions; their markers, including `*`, `**`, and `***`, do not independently determine the released permissions. Consult the [field dictionary](data/field_dictionary.csv) for each field's frozen visibility, and the public bundle for its actual values and masks. Unit explanations in the guide leave the recorded units and model-visible metadata unchanged.

Released source series and public A labels are included. B scoring reads an external organizer label directory; generation receives only the authorized public bundle, not arbitrary filesystem or label access. This is an application-level boundary; deployments executing untrusted code need corresponding filesystem or container isolation.

## Resources and operations

The paper groups forecasting capabilities into **TimeSense** for temporal diagnostics, **TimeFeat** for temporal feature extraction, **TimeZoo** for forecasting-model access, and **TimeKnowledge** for domain knowledge and historical cases. In the released registry, these capabilities use `feature.*`, `model.*`, `knowledge.*`, and `sample.*` interfaces. The four names describe functional families, not renamed API namespaces. The registry contains 49 core tool entries; domain-specific diagnostic extensions are registered separately. Chronos, TimesFM, Sundial, and Moirai are the four native forecasters.

| Resource or operation | Role |
| --- | --- |
| Contextual information | External variables, events, knowledge, and training-period cases |
| General-purpose tools | History/context analysis, retrieval, forecast construction, and public checks |
| Native forecasters | Chronos, TimesFM, Sundial, and Moirai candidates from target history and task specifications |
| Domain-specific tools | Read-only application diagnostics and domain-dependent data semantics |
| Forecast operations | Adopt, blend, generate, or revise numerical forecasts; register immutable agent versions |
| Public checks | Verify exposed constraints without using held-out targets |
| Final selection | Submit an existing valid agent-created forecast with the required horizon and units |

The portable path supports verified cached native candidates and fresh local-model inference. The expanded release currently includes 500 of 1,600 required candidates; a complete cache-only run is not yet ready. Raw candidates must become agent forecast versions before submission; context acquisition and incorporation remain agent decisions. Recomputing candidates or defining new windows requires upstream weights and separate worker environments. See [native-tool reproduction](REPRODUCTION.md#native-tools).

## Interaction and trace

The state-adaptive loop maintains acquisition state with information needs, observations, and evidence; forecast state with immutable versions, the active forecast, and public checks; and execution state with actions, errors, and remaining budget. The agent uses its authorized state view to choose the next operation.

1. The task suite instantiates an episode contract and authorized resources.
2. The LLM receives the task and current authorized state view, then selects an action.
3. The harness checks schema, state, permissions, and budget, executes an admitted action, and returns an observation or error.
4. Shared state records information needs, observations, contextual evidence, public checks, remaining budget, and forecast versions.
5. The LLM can gather more information, return to an older version, revise, check, or select a final forecast. The episode ends with a valid delivery or a recorded failure.
6. Offline evaluation uses the submitted forecast, trace, private targets, and frozen profiles.

Evidence refers to observations actually obtained, with source ranges and intended forecast intervals. Forecast versions retain numerical values, operation sources, creation rounds, and any evidence references or declared effects. This connects information requests, observations, and forecast changes. Declared-executed consistency does not itself establish causal validity.

Agents choose their operation order and can iterate between acquisition and forecasting. Revision is optional; an existing valid forecast may be submitted. Public checks never reveal current-episode forecast error. Best-created versions are retrospective analysis references and are unavailable as online feedback.

Information acquisition adds observations without automatically creating an agent version; construction can reuse earlier observations. Version counts include all accepted constructions. Revision checks require numerical differences exceeding both frozen absolute and scale-relative tolerances. The comparison baseline is the previous active agent version, or a defined operation source for an initial construction. An initial blend can therefore activate revision checks. Final-selection quality separately requires at least two agent versions.

## Evaluation dimensions

| Dimension | Measures |
| --- | --- |
| Forecast quality | MASE/RMSSE-based quality and applicable public domain metrics |
| Process execution | Valid actions, need-query alignment, revision support/control, checks, and final selection |
| Behavioral trustworthiness | Output validity, permissions, constraints, reference integrity, declared-executed consistency, and recovery |
| Resource efficiency | Quality-conditioned time/token use, redundancy, and repeated-error behavior |

The dimensions use deterministic numerical and rule-based checks, without an LLM judge. Need-query alignment combines category, variable, and time-scope compatibility with transaction admissibility; scored rejected acquisitions retain incompatibility records. Revision support checks cited application intervals against numerical changes, including scoped rationales when present. Source coverage and application intervals provide structural traceability. The default total is `0.60 Forecast + 0.20 Process + 0.15 Trust + 0.05 Efficiency`. Scores are averaged within datasets and then equally across datasets. Report raw errors, dimension profiles, applicability coverage, and resources alongside the aggregate where available.

MASE/RMSSE use positive training-derived scales frozen with the profile before testing. Scale construction, fallback rules, domain components, and efficiency anchors belong to that profile. Historical experiments may use different profiles; retain their scopes rather than pooling or retrospectively renormalizing their errors.

Process and Trust omit inapplicable checks within each group and report applicable weight as coverage. Different traces can activate different checks. Process incorporates retrospective selection quality, while Efficiency is conditioned on forecast adequacy. Invalid final submissions receive zero Efficiency and Total; missing or incomparable accounting leaves resource-based scores unavailable. The scoring command requires a fully completed selected cohort rather than an arbitrary partial run. Full definitions are implemented in `src/castbench/evaluator.py`, `domain_metrics.py`, and `efficiency.py`.

When forecast adequacy reaches its reference level, Efficiency is `70 + 0.3 C`, where `C` summarizes resource utility. Higher Efficiency therefore reflects the defined balance of adequacy and resource use; it is not simply lower token usage. The repeated-error denominator includes valid and invalid recorded attempts but excludes input capacity events generated by the environment.

## Frozen files and new runs

Preserve package structure and recorded hashes. The [usage guide](GETTING_STARTED.md) describes standard A/B evaluation. New outputs belong under `runs/` and are not part of the public input release. Final aggregate displays are maintained in the README.

## Source code map

| Component | Released implementation |
| --- | --- |
| Task contracts and data access | [`contracts.py`](../src/castbench/contracts.py), [`domain_packs.py`](../src/castbench/domain_packs.py) |
| State-adaptive interaction and provider interface | [`runner.py`](../src/castbench/runner.py), [`providers.py`](../src/castbench/providers.py), [`context_memory.py`](../src/castbench/context_memory.py) |
| Tool registration and execution | [`catalog.py`](../src/castbench/catalog.py), [`tools.py`](../src/castbench/tools.py), [`models.py`](../src/castbench/models.py) |
| Numerical forecast operations and version storage | [`forecast.py`](../src/castbench/forecast.py), [`storage.py`](../src/castbench/storage.py) |
| Offline evaluation and aggregation | [`evaluator.py`](../src/castbench/evaluator.py), [`domain_metrics.py`](../src/castbench/domain_metrics.py), [`efficiency.py`](../src/castbench/efficiency.py) |
| A/B preparation, audit, execution, and scoring | [`reproduce.py`](../tools/reproduce.py) |
