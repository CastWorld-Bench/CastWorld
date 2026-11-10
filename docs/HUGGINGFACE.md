# Hosting CastWorld A/B on Hugging Face

[Repository](../README.md) | [Board definitions](LEADERBOARDS.md)

This guide describes the A/B leaderboard deployment. Both the project homepage and Hugging Face frontend select A/B first, with Model on the left and System on the right. Model is the default view.

## Recommended arrangement

| Component | Visibility | Role |
| --- | --- | --- |
| A/B public dataset repository | Public when released | Authorized source splits, task inputs, public A labels, metadata, and task package instructions |
| A/B leaderboard Space | Public after checking | Results display and links to the submission protocol |
| Verified result store | Organizer-private | Original scores, traces, and provenance; only approved aggregate displays are copied to the frontend |
| Submission queue | Organizer-controlled | Model revision or endpoint references, run status, and private credential references |
| Evaluation workers | Existing organizer servers | Run the same CastWorld harness and scorer |
| B label store | Organizer-only | B targets, accessed only by the offline scorer |

Start with a **Static HTML Space** for display and organizer-operated evaluations. It requires no GPU allocation in the Space. Automated submission intake can be added later with a server-backed app or a separate service. The prepared static frontend does not currently accept uploads or execute evaluations.

Hugging Face's leaderboard guide explicitly permits running the evaluation backend on your own cluster. Its example `lm_eval`/`lighteval` backend is not a substitute for CastWorld's interaction environment and trajectory evaluator.

## 1. Choose account and repository names

Create or use an organization with the intended maintainers. Suggested names are `<namespace>/CastWorld` for the dataset and `<namespace>/CastWorld-Leaderboard` for the Space. Keep original evaluation records in an organizer-private store, not a public results repository. These are suggested identifiers, not repositories already created by this project.

Use a fine-grained token scoped to the repositories being maintained. Authenticate locally with `hf auth login`, or set a token in the authorized environment. Do not put tokens in code, a public form, Git history, or browser JavaScript. Model endpoint keys belong to the private evaluation service, not the public result files.

## 2. Publish the A/B dataset assets

Upload the prepared **A/B public package**, not a complete copy of the historical paper source archive. Include a dataset card describing task schema, sampling frequency, units, window lengths, field/time permissions, usage examples, upstream data sources and their applicable terms. A contains 300 high-quality, representative forecasting episodes and B contains 100, both spanning 25 datasets. The display selects A/B first and System/Model second.

The dataset card should identify A labels as public and B labels as organizer-held. B inputs are public. Retain exact task and protocol revisions so a submission can be evaluated on the intended package.

Use the distributed public `benchmark_data` package and matching CSV/JSON source views. Do not substitute complete organizer source series or original paper task archives: they can contain B targets. The public release excludes B labels and source/history records at B target timestamps. Only approved aggregate result displays are published.

## 3. Prepare the Space source

From the CastWorld repository, export the A/B-only frontend to a new empty directory outside the repository:

```bash
python tools/build_project_site.py --hf-out /absolute/path/to/castworld-hf-space
```

The export includes `index.html`, styling, browser code, approved images, dataset metadata, and a Space README. Approved A/B display summaries are embedded in the HTML; no independent experiment-result package is included. The Space uses this configuration:

```yaml
---
title: CastWorld Leaderboard
sdk: static
app_file: index.html
pinned: false
tags:
- leaderboard
- time-series
- agents
---
```

It exports the four A/B views, without B labels or private experiment records. Use a new directory for each export to avoid mixing in unrelated old files.

The Space uses a leaderboard-first layout rather than the project homepage: the A/B tabs and results appear directly below a compact CastWorld header. Model search, family filtering, sortable columns, metric-column selection, reset, and CSV export work locally in the browser. Scores and error/resource views share the same result source. On smaller screens the model column stays visible while the metrics scroll. Fonts and icons are bundled, so the frontend does not depend on a third-party asset CDN.

## 4. Create and check the Space

On Hugging Face, select **New Space**, choose the account/organization and name, choose **Static**, and initially keep it private. Upload the exported folder with the web interface, Git, or `huggingface_hub.HfApi.upload_folder`. Confirm the A/B then System/Model hierarchy, the 15-model and 24-system rosters, blank pending scores, language switching, mobile layout and CSV links. Then make the Space public in Settings when ready.

As checked on 2026-11-01, Hugging Face documents Static Spaces as free and not requiring a paid plan. Compute-backed Gradio/Docker creation has different plan requirements; check the linked current documentation before choosing those options. An existing paid server is not needed merely to display a static board.

## 5. Define and operate submissions

For the model-level leaderboard, accept a model repository with an immutable revision, or an OpenAI-compatible endpoint/profile with credentials delivered through an organizer-private channel. Record the model identifier, inference settings, task package revision, harness revision, and run status. Run the model through the common CastWorld harness rather than accepting claimed Process/Trust/Efficiency values from a spreadsheet.

Use the existing workers and model interfaces. Prepare native forecasts for the new input windows and freeze the shared resource calibration before the formal run. Credentials and provider behavior can change; record the actual deployment used. Retry failed tasks without replacing completed deliveries silently or discarding failed attempts.

An alternative forecast-array submission can be scored with the prepared predictive adapter, but it is a predictive-only track. It cannot establish a four-dimensional model-level result.

The B scorer reads the private labels on the organizer server. The public Space displays only approved aggregate summaries embedded in its page. A static Space has no secure private scoring backend: never embed B labels, a private dataset token, or the evaluator label archive in static assets.

## 6. Publish results consistently

Publish results only after the complete board has been scored. Store board ID, model/revision, task/harness/scorer revisions, episode count, dataset count, four scores, TotalScore, predictive errors, and actual resource summaries. Keep missing results null and unranked. Use the same dataset-macro aggregation and frozen 60/20/15/5 top-level weights across the displayed models.

Maintain approved A/B aggregate values in the homepage's embedded summary and matching README tables. The builder checks their agreement and exports the same summary to the Space. Do not create independent result files or publish private scoring records.

## 7. Add automated intake if needed

Once manual scoring and publication work, add authentication, submission storage, and a queue. A compute-backed Space can provide the form while the existing servers consume the queue; the Space need not host the models. The official template separates frontend, request records, result records, and an optional evaluation backend. Choose the minimum needed components instead of deploying a second forecasting engine.

For a continuous model benchmark, the generic competition template's public/private leaderboard phases are not equivalent to CastWorld-A and CastWorld-B: those templates may reveal a private leaderboard at competition end. Check the intended lifecycle before adopting that template.

## 8. What can be automated

With access to the intended account/organization and scoped repository permissions, an authorized coding agent can create the dataset/result repositories and Space, upload reviewed assets, configure nonsecret metadata, test the deployment, and connect result synchronization. Without that account authorization, this repository supplies the tested frontend and deployment instructions but does not create remote Hugging Face resources.

## Official references

- [Leaderboard frontend, request/result repositories, and optional backend](https://huggingface.co/docs/leaderboards/main/leaderboards/building_page)
- [Static HTML Spaces](https://huggingface.co/docs/hub/spaces-sdks-static)
- [Spaces plans, visibility, and execution](https://huggingface.co/docs/hub/spaces-overview)
- [Dataset cards](https://huggingface.co/docs/hub/datasets-cards)
- [Fine-grained user access tokens](https://huggingface.co/docs/hub/security-tokens)
- [Hub-native evaluation results](https://huggingface.co/docs/hub/en/eval-results)
- [Competition lifecycle and private competition repositories](https://huggingface.co/docs/competitions/main/en/competition_repo)

The Hub-native benchmark/evaluation-results feature is documented as work in progress. It can complement the project later, but the custom Space is the direct fit for a multi-score agent benchmark containing both API-hosted models and local models.
