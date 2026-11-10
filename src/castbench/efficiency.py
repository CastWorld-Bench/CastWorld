"""Frozen resource protocol and offline, complete-cohort efficiency scoring."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .common import digest
from .contracts import Contract, Number, PositiveInt, time_axis

RESOURCE_VERSION = "castbench.resources/1.0"
SCORE_VERSION = "robust-v1.3"
Hash = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class FrozenAnchors(Contract):
    profile_hash: Hash
    time_anchor_seconds: Number = Field(gt=0)
    token_anchor: Number = Field(gt=0)


class ResourcePolicy(Contract):
    version: Literal["castbench.resources/1.0"] = RESOURCE_VERSION
    dataset_id: str = Field(min_length=1)
    split: Literal["validation", "test"]
    planned_episode_ids: list[str] = Field(min_length=1)
    source_train_hash: Hash
    test_source_train_hash: Hash | None = None
    seasonal_period: PositiveInt | None = None
    execution_conditions: dict[str, str] = Field(min_length=1)
    auxiliary_llm: Literal["none", "metered"] = "none"
    anchors: FrozenAnchors | None = None

    @model_validator(mode="after")
    def fixed_manifest(self):
        if (self.split == "test") != (self.anchors is not None):
            raise ValueError("Freeze and disclose validation anchors before a test run; not during calibration")
        if self.split == "test" and self.test_source_train_hash is not None and (
            self.source_train_hash != self.test_source_train_hash
        ):
            raise ValueError("Test training source differs from its predeclared calibration binding")
        if any(not x for x in self.planned_episode_ids):
            raise ValueError("Episode IDs must be nonempty")
        if len(set(self.planned_episode_ids)) != len(self.planned_episode_ids):
            raise ValueError("Duplicate planned Episode")
        required = {"hardware", "sample_concurrency", "tool_concurrency", "cache_policy", "preparation"}
        if not required.issubset(self.execution_conditions) or any(
            not isinstance(v, str) or not v.strip() for v in self.execution_conditions.values()
        ):
            raise ValueError("Freeze hardware, concurrency, cache policy and preparation accounting")
        return self

    def comparison_hash(self):
        values = self.model_dump(exclude={"split", "planned_episode_ids", "anchors"})
        # Validation can use an earlier training cutoff; the later test source
        # must be declared before calibration, not substituted after outcomes.
        values["source_train_hash"] = self.test_source_train_hash or self.source_train_hash
        return digest(values)


class EfficiencyProfile(Contract):
    schema_version: Literal["castbench.efficiency-profile/1.0"] = "castbench.efficiency-profile/1.0"
    policy: ResourcePolicy
    reference_agent_hash: Hash
    calibration_record_hash: Hash
    calibration_scored_end: str
    time_anchor_seconds: Number = Field(gt=0)
    token_anchor: Number = Field(gt=0)

    @model_validator(mode="after")
    def validation_only(self):
        if self.policy.split != "validation":
            raise ValueError("Resource anchors must come from frozen validation runs")
        time_axis([self.calibration_scored_end])
        return self


def public_notice(policy):
    return {
        "version": SCORE_VERSION,
        "resource_protocol": policy.version,
        "quality_reference": {
            "method": "seasonal_naive" if policy.seasonal_period else "last_observed_value",
            "seasonal_period": policy.seasonal_period,
            "fallback": "last observed history value if a required seasonal value is missing",
            "assessment": "same forecast objective, including declared domain metrics; offline only",
        },
        "resource_weights": {"time": 0.4, "tokens": 0.4, "behavior": 0.2},
        "frozen_resource_anchors": policy.anchors.model_dump() if policy.anchors is not None else None,
        "formula": {
            "time": "100/(1+dataset_mean_adjusted_seconds/T0)",
            "tokens": "100/(1+dataset_mean_tokens/K0)",
            "behavior": "100*(1-(redundant_call_ratio+repeated_error_ratio)/2)",
            "quality": "a=clip(2-F0/F,0,1); a=0 when F=0",
            "efficiency": "C=0.4T+0.4K+0.2B; E=70a+(0.6-0.3a)C; E=0 without a legal final forecast",
        },
        "instruction": (
            "Quality is compared with a fixed history-only forecast, not a SOTA threshold. "
            "Time and tokens use dataset arithmetic means over all planned Episodes. "
            "Input (including cached input) and output tokens are counted, without prices; "
            "reasoning already included in output is not counted twice. Adequate quality "
            "receives a 70-point efficiency floor and a weaker resource penalty. "
            "Rounds remain a budget, not a direct efficiency penalty. Identical calls without "
            "new information and repeated explicitly rejected actions reduce behavior efficiency."
        ),
    }


def bind_test_policy(profile, episode_ids):
    profile = EfficiencyProfile.model_validate(profile)
    policy = profile.policy.model_dump()
    policy.update(split="test", planned_episode_ids=episode_ids,
                  source_train_hash=profile.policy.test_source_train_hash or profile.policy.source_train_hash,
                  anchors={"profile_hash": digest(profile.model_dump()),
                           "time_anchor_seconds": profile.time_anchor_seconds,
                           "token_anchor": profile.token_anchor})
    return ResourcePolicy.model_validate(policy).model_dump()


def token_usage(raw):
    """Return observed input+output, never estimates or a missing-usage zero."""
    usage = raw.get("usage") if isinstance(raw, dict) else None
    if not isinstance(usage, dict):
        raise ValueError("Missing LLM token usage; new efficiency score is unavailable")
    names = ("prompt_tokens", "completion_tokens") if "prompt_tokens" in usage else (
        "input_tokens", "output_tokens"
    )
    values = [usage.get(k) for k in names]
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
        raise ValueError("Token usage needs nonnegative integer input and output counts")
    total = sum(values)
    if "total_tokens" in usage and (
        isinstance(usage["total_tokens"], bool)
        or not isinstance(usage["total_tokens"], int)
        or usage["total_tokens"] != total
    ):
        cached = [usage.get(k) for k in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens")]
        if (
            any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in cached)
            or isinstance(usage["total_tokens"], bool)
            or not isinstance(usage["total_tokens"], int)
            or sum(cached) < values[0]
            or sum(cached) + values[1] != usage["total_tokens"]
        ):
            raise ValueError("Inconsistent total token usage")
        # Some relays omit framing overhead from prompt_tokens but include it
        # in their native cache breakdown and total; require both identities.
        values[0] = sum(cached)
        total = sum(values)
    return {"input_tokens": values[0], "output_tokens": values[1], "total_tokens": total}


def _receipt_usage(receipt, evidence):
    raw = receipt["response"]
    if evidence is None:
        return token_usage(raw)
    usage = raw.get("usage")
    native = evidence["transport_usage"]
    if (
        evidence["response_hash"] != receipt["response_hash"]
        or digest(native) != evidence["transport_usage_hash"]
        or not isinstance(native, dict)
        or not isinstance(usage, dict)
        or any(native.get(k) != usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
        or not isinstance(evidence.get("transport_trace_hash"), str)
        or not evidence["transport_trace_hash"].startswith("sha256:")
        or len(evidence["transport_trace_hash"]) != 71
    ):
        raise ValueError("Transport token evidence differs from its immutable response")
    return token_usage({"usage": native})


def interval_union(intervals, duration):
    ordered = sorted(intervals)
    end, total = 0.0, 0.0
    for start, stop in ordered:
        if not all(math.isfinite(x) for x in (start, stop)) or not 0 <= start <= stop <= duration:
            raise ValueError("Invalid excluded time interval")
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def _policy(export):
    if "resource_policy" not in export["config"]:
        raise ValueError("Run predates the frozen resource protocol; do not invent historical usage")
    policy = ResourcePolicy.model_validate(export["config"]["resource_policy"])
    if (
        export["config"]["dataset_id"] != policy.dataset_id
        or export["config"]["episode_id"] not in policy.planned_episode_ids
        or export["config"]["revision_scale"]["source_train_hash"] != policy.source_train_hash
    ):
        raise ValueError("Resource policy does not match the Episode/training source")
    return policy


def measured_resources(export):
    """Audit receipts must cover every actual provider attempt and online segment."""
    policy = _policy(export)
    if export["state"]["status"] != "finished" or not export["trace"]:
        raise ValueError("Resource scoring requires a finished Episode, including failed deliveries")
    if export["trace"][-1]["state"] != export["state"]:
        raise ValueError("Final resource state does not match its journal")
    for row in export["trace"]:
        if digest(row["state"]) != row["event"]["state_hash"] or (
            digest(row["response"]) != row["event"]["response_hash"]
        ):
            raise ValueError("Resource journal integrity failure")
    log = export.get("operational_log", [])
    starts, ends, receipts, prepared, received, usage_evidence = {}, {}, {}, set(), set(), {}
    for entry in log:
        kind = entry["kind"]
        if kind in {"resource_segment_started", "resource_segment_finished"}:
            target = starts if kind.endswith("started") else ends
            sid = entry["segment_id"]
            if sid in target:
                raise ValueError("Duplicate resource segment")
            target[sid] = entry
        elif kind == "resource_llm_receipt":
            key = (entry["source"], entry["call_id"])
            if key in receipts:
                raise ValueError("Duplicate LLM resource receipt")
            if entry["source"] not in {"main", "auxiliary"}:
                raise ValueError("Unknown LLM receipt source")
            if entry["source"] == "auxiliary" and policy.auxiliary_llm != "metered":
                raise ValueError("Auxiliary LLM was not declared before this run")
            if digest(entry["response"]) != entry["response_hash"]:
                raise ValueError("LLM resource receipt integrity failure")
            receipts[key] = entry
        elif kind == "provider_prepared":
            prepared.add(entry["attempt_id"])
        elif kind == "provider_received":
            received.add(entry["attempt_id"])
        elif kind == "resource_usage_transport_evidence":
            key = (entry["source"], entry["call_id"])
            if key in usage_evidence:
                raise ValueError("Duplicate transport usage evidence")
            usage_evidence[key] = entry
    if not starts or set(starts) != set(ends):
        raise ValueError("Incomplete time measurement; resume cannot erase an interrupted segment")
    if prepared != received or prepared != {k[1] for k in receipts if k[0] == "main"}:
        raise ValueError("LLM usage receipts do not cover every actual API attempt")
    for receipt in receipts.values():
        if receipt["segment_id"] not in starts or receipt["turn"] != starts[receipt["segment_id"]]["turn"]:
            raise ValueError("LLM receipt is outside its measured online segment")
    duration = excluded = 0.0
    for sid, entry in ends.items():
        elapsed = entry["elapsed_seconds"]
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed):
            raise ValueError("Time measurement must be finite")
        if elapsed < 0 or entry["turn"] != starts[sid]["turn"]:
            raise ValueError("Invalid measured online segment")
        intervals = entry["exclusions"]
        if any(i["reason"] not in {"provider_failure", "transport_backoff"} for i in intervals):
            raise ValueError("Only explicitly observed infrastructure exclusions are supported")
        removed = interval_union([(i["start"], i["end"]) for i in intervals], elapsed)
        duration += elapsed
        excluded += removed
    covered_turns = {entry["turn"] for entry in ends.values()}
    if covered_turns != {row["turn"] for row in export["trace"]}:
        raise ValueError("Time receipts do not cover the complete committed Episode")
    if not set(usage_evidence).issubset(receipts):
        raise ValueError("Transport usage evidence has no matching LLM receipt")
    tokens = [_receipt_usage(r, usage_evidence.get(k)) for k, r in receipts.items()]
    # Receipts remain available even when a caller intentionally omits the raw request archive.
    if "provider_archive" in export:
        archive = export["provider_archive"]
        if {r["id"] for r in archive} != prepared or len(archive) != len(prepared):
            raise ValueError("API archive and resource receipt coverage differ")
        for row in archive:
            raw = json.loads(row["response"])
            if digest(raw) != receipts[("main", row["id"])]["response_hash"]:
                raise ValueError("Resource receipt differs from archived API response")
    report = {
        "elapsed_seconds": duration,
        "excluded_seconds": excluded,
        "adjusted_seconds": max(0.0, duration - excluded),
        **{k: sum(t[k] for t in tokens) for k in ("input_tokens", "output_tokens", "total_tokens")},
        "main_llm_calls": len(prepared),
        "auxiliary_llm_calls": sum(k[0] == "auxiliary" for k in receipts),
        "transport_usage_evidence_count": len(usage_evidence),
        "receipt_hash": digest({"starts": starts, "ends": ends, "llm": list(receipts.values()),
                                "transport_usage_evidence": list(usage_evidence.values())}),
    }
    if not math.isfinite(report["adjusted_seconds"]):
        raise ValueError("Resource time exceeds the numeric range")
    return report


def _agent_hash(export):
    return digest({k: export["config"].get(k) for k in (
        "provider", "catalog_hash", "model_profiles", "system_prompt_hash", "budget",
        "protocol_repair", "capacity_recovery", "completion_rescue", "resource_implementation_hash"
    )})


def _check_group(exports, split):
    policies = [_policy(e) for e in exports]
    policy = policies[0]
    if policy.split != split or any(p.model_dump() != policy.model_dump() for p in policies):
        raise ValueError("Cohort must share one frozen dataset policy and split")
    ids = [e["config"]["episode_id"] for e in exports]
    if len(ids) != len(set(ids)) or set(ids) != set(policy.planned_episode_ids):
        raise ValueError("Cohort must include exactly all planned Episodes, including failed deliveries")
    if len({_agent_hash(e) for e in exports}) != 1:
        raise ValueError("Cannot change Agent, tools, budgets or metering within a dataset cohort")
    if len({digest(e["config"].get("forecast_evaluation")) for e in exports}) != 1:
        raise ValueError("Cannot change forecast objectives within a dataset cohort")
    return policy


def calibrate_profile(exports):
    from .evaluator import stable_mean

    if not exports:
        raise ValueError("No validation resource records")
    policy = _check_group(exports, "validation")
    resources = [measured_resources(e) for e in exports]
    return EfficiencyProfile(
        policy=policy,
        reference_agent_hash=_agent_hash(exports[0]),
        calibration_record_hash=digest(sorted(
            [(e["config"]["episode_id"], r["receipt_hash"]) for e, r in zip(exports, resources)]
        )),
        calibration_scored_end=max(
            time_axis(e["config"]["episode"]["forecast_timeline"]["scored_timestamps"])[-1]
            for e in exports
        ).isoformat(),
        time_anchor_seconds=stable_mean([r["adjusted_seconds"] for r in resources]),
        token_anchor=stable_mean([r["total_tokens"] for r in resources]),
    ).model_dump()


def reference_values(episode, seasonal_period):
    history = episode["history"]["target_values"]
    valid = [x for x in history if x is not None and math.isfinite(x)]
    if not valid:
        raise ValueError("Quality reference requires public observed history")
    last = valid[-1]
    task = episode["task"]
    if seasonal_period is None or len(history) < seasonal_period:
        return [last] * task["horizon_length"]
    season = history[-seasonal_period:]
    return [
        season[(task["forecast_gap"] + i) % seasonal_period]
        if season[(task["forecast_gap"] + i) % seasonal_period] is not None else last
        for i in range(task["horizon_length"])
    ]


def behavior_score(export):
    seen_calls, seen_errors = set(), set()
    substantive = redundant = repeated_errors = attempts = 0
    for row in export["trace"]:
        event = row["event"]
        if event.get("actor") != "environment_capacity":
            attempts += 1
        for record in event["tool_records"]:
            if record["status"] not in {"ok", "partial"} or record["tool_id"].startswith("registry."):
                continue
            substantive += 1
            identity = record.get("redundancy_identity")
            if identity is not None:
                if identity in seen_calls:
                    redundant += 1
                seen_calls.add(identity)
        if not event["valid"] and event.get("actor") != "environment_capacity":
            action = event.get("action")
            if action:
                action = {k: v for k, v in action.items() if k not in {"state_version", "phase"}}
                if isinstance(action.get("payload", {}).get("calls"), list):
                    action["payload"] = {**action["payload"], "calls": [
                        {k: v for k, v in c.items() if k != "call_id"} if isinstance(c, dict) else c
                        for c in action["payload"]["calls"]
                    ]}
                identity = digest({"action": action, "code": row["response"].get("code")})
                if identity in seen_errors:
                    repeated_errors += 1
                seen_errors.add(identity)
    r1 = redundant / substantive if substantive else 0.0
    r2 = repeated_errors / attempts if attempts else 0.0
    return {
        "score": 100 * (1 - (r1 + r2) / 2),
        "redundant_call_ratio": r1,
        "repeated_error_ratio": r2,
        "successful_substantive_calls": substantive,
        "provably_redundant_calls": redundant,
        "action_attempts": attempts,
        "repeated_error_actions": repeated_errors,
    }


def efficiency_score(forecast_score, reference_score, time_score, token_score, behavior, completed=True):
    values = (forecast_score, reference_score, time_score, token_score, behavior)
    if any(isinstance(v, bool) or not math.isfinite(v) or not 0 <= v <= 100 for v in values):
        raise ValueError("Efficiency components must be finite scores in [0,100]")
    adequacy = max(0.0, min(1.0, 2 - reference_score / forecast_score)) if forecast_score > 0 else 0.0
    cost = 0.4 * time_score + 0.4 * token_score + 0.2 * behavior
    score = 70 * adequacy + (0.6 - 0.3 * adequacy) * cost if completed else 0.0
    return score, {"relative_quality_adequacy": adequacy, "resource_behavior_score": cost}


def resource_score(mean_value, anchor):
    if not math.isfinite(mean_value) or mean_value < 0 or not math.isfinite(anchor) or anchor <= 0:
        raise ValueError("Resource means must be nonnegative and anchors positive")
    # This form avoids overflowing a finite mean/anchor ratio.
    return 100 * (anchor / max(anchor, mean_value)) / (
        anchor / max(anchor, mean_value) + mean_value / max(anchor, mean_value)
    )


def implementation_hash():
    return digest({name: Path(__file__).with_name(name).read_text(encoding="utf-8")
                   for name in ("efficiency.py", "runner.py")})


def score_cohort(exports, privates, profiles):
    """All planned runs for one evaluated Agent, with one validation anchor per dataset."""
    from .contracts import ForecastEvaluation
    from .domain_metrics import augment_metrics
    from .evaluator import metrics, score_run, stable_mean

    if not exports or len(exports) != len(privates):
        raise ValueError("Provide one matched private record per planned export")
    grouped = defaultdict(list)
    for e, private in zip(exports, privates):
        grouped[e["config"]["dataset_id"]].append((e, private))
    if set(grouped) != set(profiles):
        raise ValueError("Cohort dataset scope differs from its frozen profile manifest")
    validated = {d: EfficiencyProfile.model_validate(p) for d, p in profiles.items()}
    manifest = {d: _check_group([e for e, _ in rows], "test").model_dump()
                for d, rows in sorted(grouped.items())}
    cohort_hash = digest({"manifest": manifest, "profiles": profiles})
    results = []
    for dataset, rows in sorted(grouped.items()):
        policy = ResourcePolicy.model_validate(manifest[dataset])
        profile = validated[dataset]
        if profile.policy.comparison_hash() != policy.comparison_hash():
            raise ValueError("Test and reference calibration resource/quality conditions differ")
        if policy.anchors.model_dump() != {
            "profile_hash": digest(profile.model_dump()), "time_anchor_seconds": profile.time_anchor_seconds,
            "token_anchor": profile.token_anchor,
        }:
            raise ValueError("Scoring anchors differ from those disclosed before Agent decisions")
        if set(policy.planned_episode_ids).intersection(profile.policy.planned_episode_ids):
            raise ValueError("Validation calibration and test Episode IDs must be disjoint")
        test_start = min(time_axis(e["config"]["episode"]["forecast_timeline"]["scored_timestamps"])[0]
                         for e, _ in rows)
        if time_axis([profile.calibration_scored_end])[0] >= test_start:
            raise ValueError("Validation calibration must precede the test scored windows")
        resources = [measured_resources(e) for e, _ in rows]
        mean_time = stable_mean([r["adjusted_seconds"] for r in resources])
        mean_tokens = stable_mean([r["total_tokens"] for r in resources])
        time_score = resource_score(mean_time, profile.time_anchor_seconds)
        tokens_score = resource_score(mean_tokens, profile.token_anchor)
        for (export, private), resource in zip(rows, resources):
            result = score_run(export, private, "robust-v1.2")
            config = export["config"]
            forecast_profile = (
                ForecastEvaluation.model_validate(config["forecast_evaluation"])
                if config.get("forecast_evaluation") is not None else None
            )
            reference = reference_values(config["episode"], policy.seasonal_period)
            reference_metrics = augment_metrics(
                metrics(private["values"], reference, private["normalization"], "robust-v1.2"),
                private["values"], reference, private["normalization"], forecast_profile,
                config["episode"]["history"]["target_values"][-1],
            )
            behavior = behavior_score(export)
            efficiency, parts = efficiency_score(
                result["ForecastScore"], reference_metrics["ForecastScore"], time_score, tokens_score,
                behavior["score"], result["completed"],
            )
            result.update(
                schema_version="castbench.score/1.3", evaluator_profile="robust-v1.3-resource-quality-efficiency",
                EfficiencyScore=efficiency,
                TotalScore=(0.6 * result["ForecastScore"] + 0.2 * result["ProcessScore"]
                            + 0.15 * result["TrustScore"] + 0.05 * efficiency) if result["completed"] else 0.0,
                efficiency_items={**parts, "time_score": time_score, "token_score": tokens_score,
                                  "behavior": behavior, "quality_reference_metrics": reference_metrics},
                resources=resource, MeanSampleSeconds=mean_time, MeanSampleTokens=mean_tokens,
                TimeEfficiencyScore=time_score, TokenEfficiencyScore=tokens_score,
                BehaviorEfficiencyScore=behavior["score"],
                efficiency_profile=profile.model_dump(), efficiency_profile_hash=digest(profile.model_dump()),
                efficiency_cohort_hash=cohort_hash,
                efficiency_cohort_manifest={d: p["planned_episode_ids"] for d, p in manifest.items()},
                efficiency_dataset_episode_ids=policy.planned_episode_ids,
                efficiency_resource_protocol_hash=policy.comparison_hash(),
            )
            result["evaluator_implementation_hash"] = digest({
                "base": result["evaluator_implementation_hash"], "efficiency": implementation_hash()
            })
            result["score_semantics"]["efficiency"] = (
                "fixed-validation-anchor dataset mean time/tokens plus observable redundancy, "
                "conditioned on the same-objective history-only quality reference; rounds are diagnostic only"
            )
            results.append(result)
    return results


def validate_scored_cohort(scores):
    cohorts = {s.get("efficiency_cohort_hash") for s in scores}
    if len(cohorts) != 1 or None in cohorts:
        raise ValueError("Cannot mix or omit efficiency cohort identities")
    manifests = [s["efficiency_cohort_manifest"] for s in scores]
    if any(m != manifests[0] for m in manifests):
        raise ValueError("Efficiency manifests differ")
    expected = {(d, e) for d, ids in manifests[0].items() for e in ids}
    actual = {(s["dataset_id"], s["episode_id"]) for s in scores}
    if actual != expected or len(scores) != len(expected):
        raise ValueError("Do not aggregate an incomplete/subset efficiency cohort")
    grouped = defaultdict(list)
    for score in scores:
        grouped[score["dataset_id"]].append(score)
    from .evaluator import stable_mean

    for rows in grouped.values():
        if len({s["efficiency_profile_hash"] for s in rows}) != 1:
            raise ValueError("Resource anchors differ within a dataset")
        profile = EfficiencyProfile.model_validate(rows[0]["efficiency_profile"])
        if digest(profile.model_dump()) != rows[0]["efficiency_profile_hash"]:
            raise ValueError("Resource anchor integrity failure")
        means = (stable_mean([s["resources"]["adjusted_seconds"] for s in rows]),
                 stable_mean([s["resources"]["total_tokens"] for s in rows]))
        for s in rows:
            if not math.isclose(s["MeanSampleSeconds"], means[0], rel_tol=1e-12) or not math.isclose(
                s["MeanSampleTokens"], means[1], rel_tol=1e-12
            ):
                raise ValueError("Efficiency was not computed from the complete cohort mean")
            if digest(s["efficiency_profile"]) != s["efficiency_profile_hash"]:
                raise ValueError("Resource anchor integrity failure")
            t = resource_score(means[0], profile.time_anchor_seconds)
            k = resource_score(means[1], profile.token_anchor)
            expected, _ = efficiency_score(s["ForecastScore"],
                s["efficiency_items"]["quality_reference_metrics"]["ForecastScore"], t, k,
                s["BehaviorEfficiencyScore"], s["completed"])
            if not math.isclose(s["EfficiencyScore"], expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Efficiency score does not match its frozen resources and quality")
            expected_total = (
                0.6 * s["ForecastScore"] + 0.2 * s["ProcessScore"] + 0.15 * s["TrustScore"]
                + 0.05 * expected
            ) if s["completed"] else 0.0
            if not math.isclose(s["TotalScore"], expected_total, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Total score differs from the frozen 60/20/15/5 weights")


def resampled_totals(rows):
    """Bootstrap resources must be transformed again after the sampled dataset mean."""
    from .evaluator import stable_mean

    profile = rows[0]["efficiency_profile"]
    t = resource_score(stable_mean([s["resources"]["adjusted_seconds"] for s in rows]),
                       profile["time_anchor_seconds"])
    k = resource_score(stable_mean([s["resources"]["total_tokens"] for s in rows]), profile["token_anchor"])
    return [
        (0.6 * s["ForecastScore"] + 0.2 * s["ProcessScore"] + 0.15 * s["TrustScore"]
         + 0.05 * efficiency_score(s["ForecastScore"],
             s["efficiency_items"]["quality_reference_metrics"]["ForecastScore"], t, k,
             s["BehaviorEfficiencyScore"], s["completed"])[0]) if s["completed"] else 0.0
        for s in rows
    ]
