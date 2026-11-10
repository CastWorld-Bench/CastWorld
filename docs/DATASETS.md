# Datasets and Field Guide

[Benchmark specification](BENCHMARK.md) | [Evaluation workflow](GETTING_STARTED.md) | [Leaderboards](../README.md#leaderboards)

The public release supplies 25 datasets across eight domains, with 300 tasks on A and 100 on B. Each board supports Model and System evaluation. The distributed source series match the public task package. Complete organizer source series and B labels are not included.

## Raw CSV and JSON Views

Released tables are directly available under `raw_data/csv/<dataset>/data.csv` and `raw_data/json/<dataset>/data.json`. CSV copies match the released source archives byte-for-byte. JSON preserves column and row order, blank or duplicate headings, and original cell text without numerical or timestamp conversion. See the [format guide](../raw_data/README.md).

## Dataset Inventory

History and horizon lengths are time steps. Context counts exclude target, time and source-audit columns; event sources may also be numerical driver fields.

| Domain | Dataset | Frequency | History / horizon | Released rows | Context fields | History only / future authorized |
| --- | --- | --- | --- | --- | --- | --- |
| Energy | [iFLYTEK-PV](../release/assets/raw_iFLYTEK-PV.zip) | 15min | 96 / 96 | 34368 | 53 | 0 / 53 |
| Energy | [iFLYTEK-Wind](../release/assets/raw_iFLYTEK-Wind.zip) | 15min | 96 / 96 | 34272 | 54 | 0 / 54 |
| Energy | [CN-GD-Load](../release/assets/raw_CN-GD-Load.zip) | h | 168 / 24 | 8160 | 41 | 2 / 39 |
| Energy | [OPSD-DELU-Price](../release/assets/raw_OPSD-DELU-Price.zip) | h | 96 / 24 | 14017 | 91 | 10 / 81 |
| Natural science | [WB-T850_1](../release/assets/raw_WB-T850_1.zip) | 6h | 72 / 24 | 60988 | 25 | 0 / 25 |
| Natural science | [WB-T850_2](../release/assets/raw_WB-T850_2.zip) | 6h | 72 / 24 | 60380 | 25 | 0 / 25 |
| Natural science | [WB-T850_3](../release/assets/raw_WB-T850_3.zip) | 6h | 72 / 24 | 60924 | 31 | 0 / 31 |
| Natural science | [SILSO-Sunspot](../release/assets/raw_SILSO-Sunspot.zip) | D | 30 / 30 | 4798 | 23 | 19 / 4 |
| Economics and finance | [US-RPI](../release/assets/raw_US-RPI.zip) | MS | 48 / 12 | 672 | 142 | 127 / 15 |
| Economics and finance | [US-DFF](../release/assets/raw_US-DFF.zip) | D | 30 / 30 | 7489 | 45 | 27 / 18 |
| Economics and finance | [CN-Port-Export](../release/assets/raw_CN-Port-Export.zip) | D | 30 / 30 | 2511 | 30 | 15 / 15 |
| Public services | [CapitalBikeshare](../release/assets/raw_CapitalBikeshare.zip) | h | 168 / 24 | 16678 | 25 | 0 / 25 |
| Public services | [NYC-311](../release/assets/raw_NYC-311.zip) | D | 28 / 7 | 5860 | 43 | 7 / 36 |
| Public services | [US-ICSA](../release/assets/raw_US-ICSA.zip) | W-SAT | 52 / 13 | 940 | 100 | 70 / 30 |
| Public services | [NYC-EMS](../release/assets/raw_NYC-EMS.zip) | D | 28 / 7 | 4867 | 97 | 35 / 62 |
| Public services | [CN-Flu-PosRate](../release/assets/raw_CN-Flu-PosRate.zip) | W-MON | 52 / 13 | 583 | 32 | 14 / 18 |
| Retail | [M5-TX3-FOODS](../release/assets/raw_M5-TX3-FOODS.zip) | D | 35 / 28 | 1680 | 31 | 9 / 22 |
| Retail | [RetailII-NetQty](../release/assets/raw_RetailII-NetQty.zip) | h | 168 / 24 | 16720 | 33 | 15 / 18 |
| Industrial operations | [IndustryHeat-ActiveUnits](../release/assets/raw_IndustryHeat-ActiveUnits.zip) | D | 30 / 30 | 2344 | 65 | 21 / 44 |
| Industrial operations | [MotorOil-GR-TI824](../release/assets/raw_MotorOil-GR-TI824.zip) | 15min | 96 / 96 | 34268 | 24 | 24 / 0 |
| Web services | [Abilene-SEA-NYC](../release/assets/raw_Abilene-SEA-NYC.zip) | 5min | 30 / 30 | 29688 | 85 | 64 / 21 |
| Web services | [Wiki-EN-Views](../release/assets/raw_Wiki-EN-Views.zip) | D | 30 / 30 | 635 | 43 | 26 / 17 |
| Environment | [CAMELS-SPAT_1](../release/assets/raw_CAMELS-SPAT_1.zip) | D | 30 / 30 | 14969 | 40 | 6 / 34 |
| Environment | [CAMELS-SPAT_2](../release/assets/raw_CAMELS-SPAT_2.zip) | D | 30 / 30 | 14914 | 28 | 6 / 22 |
| Environment | [AQS-ThousandOaks-O3](../release/assets/raw_AQS-ThousandOaks-O3.zip) | h | 168 / 24 | 60648 | 14 | 0 / 14 |

Dataset links open the released source archive with `data.csv` and a field report. The [machine-readable inventory](data/dataset_inventory.csv) lists exact archive members. Preparation restores them at the listed paths.

## Field definitions and access

Use the complete field dictionary in [CSV](data/field_dictionary.csv) or [JSON](data/field_dictionary.json) to look up the exact field name, source description, recorded unit, access permission, and resource groups. Source descriptions retain their original wording. English column labels identify the access rules and quantity interpretations.

| Frozen profile value | Public bundle value | Available interval |
| --- | --- | --- |
| `history_only` | `history_only` | Observed history; gap and future entries are masked |
| `history_and_future` | `benchmark_known_future` | Authorized history, gap, and forecast timestamps under the task's information assumptions |

The field dictionary follows the released profile names and permissions. Source-report markers and grouping numbers describe source conventions; they are not a substitute for the final permissions. Source-audit columns are not automatically included in the agent's variable catalog. Raw data remain available for preparation and offline analysis, while generation uses the public bundle. The current task's future target values are reserved for scoring.

## Events and temporal context

Event timelines derive from named source variables. Each event records its source dependencies and time interval; future events must have authorized future-visible sources. This retains the same information boundary for numerical context and event-based queries.

For NYC-311, scheduled activities are available under the task's perfect-plan assumption, whereas other activities and issuance records marked history-only remain confined to the observed interval. Weather realizations or reanalysis can serve as perfect forecasts in the controlled retrospective setting; calendar features are deterministic. RetailII's `policy_event_flag` and `policy_event_name` follow the released future-authorized policy-context setting. These permissions specify the benchmark's information conditions, rather than historical publication snapshots. Historical case continuations end before the frozen training cutoff.

## Units and stored representations

`recorded_unit` reproduces the metadata provided in the evaluated tasks. The [unit interpretation table](data/unit_interpretations.csv) explains the physical quantity or stored representation where a separate reading is needed. It is a documentation layer and is not loaded by the runner. The representation guide does not transform the stored task arrays or source cells.

Binary event indicators and normalized temperatures are dimensionless; a threshold mentioned in a field's definition does not give the indicator a physical unit. CapitalBikeshare stores normalized temperatures: `T_C = 47 * temp - 8` and `apparent_T_C = 66 * atemp - 16` explain their physical interpretation without transforming the stored inputs. The specified IndustryHeat concentration fields are described in micrograms per cubic metre, and the Wiki high-temperature observation field is a count.

Entries marked `confirmed` provide the stated representation or physical interpretation. Entries marked `recorded` reproduce the original catalog metadata. The table documents the released inputs and supports reading the stored quantities without changing their numerical values.

## Verification

Run `python tools/verify_dataset_documentation.py` to check field coverage, profile mappings, source-report references, and documentation hashes directly against the distributed archives. This offline check makes no model calls and does not alter task inputs. The [reproduction guide](REPRODUCTION.md) describes complete-cohort task, permission, and native-candidate checks.
