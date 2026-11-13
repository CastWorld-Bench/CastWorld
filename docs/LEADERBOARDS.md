# CastWorld Leaderboards

[Project homepage](https://castworld-bench.github.io/) | [Repository](../README.md) | [Hugging Face setup](HUGGINGFACE.md)

## A/B and Model/System

| Board | Object of comparison | Episodes | Ground truth | Primary results |
| --- | --- | --- | --- | --- |
| A / Model | LLMs in the common CastWorld harness | 300 | Public | Four dimension scores and TotalScore |
| A / System | Forecasting systems | 300 | Public | MASE, RMSSE, average rank |
| B / Model | LLMs in the common CastWorld harness | 100 | Organizer-held | Four dimension scores and TotalScore |
| B / System | Forecasting systems | 100 | Organizer-held | MASE, RMSSE, average rank |

CastWorld-A contains 300 high-quality, representative forecasting episodes and CastWorld-B contains 100, both spanning 25 datasets. Model and System use the same tasks within a board, but compare different objects and metrics. The rosters contain 15 LLMs and 24 forecasting methods. The homepage selects A/B first, then Model on the left or System on the right. Model is the default view.

## Model-level scoring

Model-level evaluation assesses forecast quality (`ForecastScore`), process execution (`ProcessScore`), behavioral trustworthiness (`TrustScore`), and resource efficiency (`EfficiencyScore`) on a 0-100 scale. `TotalScore` uses the following weights:

```text
TotalScore = 0.60 ForecastScore + 0.20 ProcessScore + 0.15 TrustScore + 0.05 EfficiencyScore
```

Within each board, results are averaged within datasets and then equally across datasets. The task packages supply field/time permissions, prediction objectives, error scales, and any declared domain metrics. Full scoring uses the complete execution record, including version history and actual resource receipts. A file of prediction arrays alone supports predictive metrics, not the other three dimensions.

A and B are different task cohorts. Keep separate board identifiers, task manifests, and results. The System comparison evaluates predictive accuracy across methods and is not a four-dimensional evaluation of all external systems.

## Result sources

- A/B: approved cohort-level aggregate results in the homepage's embedded `public-board-summary` payload, cross-checked against the README tables. Current rows are pending and contain null scores.
- No independent experiment-result files are distributed. CSV downloads are constructed in the browser from the same displayed aggregate values.

Reported scores are not recomputed from rounded display components. The homepage converts sMAPE ratios to percentages for display; CSV downloads retain the source ratio.

## Maintain the project views

```bash
python tools/build_project_site.py
python tools/build_project_site.py --check
python -m unittest discover -s tests -p test_project_site.py
```

The builder checks roster consistency, numeric ranges, completion status and agreement between the homepage and README. Pending rows must have empty values. Approved completed rows can be added without changing the builder: set their status to `completed`, supply all metrics, update the corresponding README row, and set the board status to `in_progress` or `completed` as appropriate. The builder does not generate scores, access private B labels, or call an API. Keep original predictions, traces, scoring records, and provenance in the organizer's private result store. The English/Chinese switch changes presentation only; model and dataset identifiers, scores, downloads and task inputs are unchanged.

`site/` is a self-contained static site. Preview it with any static HTTP server, for example `python -m http.server 8000 --directory site`. This repository validates the site sources. The organization homepage is published from [CastWorld-Bench/castworld-bench.github.io](https://github.com/CastWorld-Bench/castworld-bench.github.io). After the checks above pass, copy the contents of `site/` to that website repository's root and push its `main` branch. Its Pages workflow publishes only the website files; raw data archives and organizer assets remain outside the deployment.
