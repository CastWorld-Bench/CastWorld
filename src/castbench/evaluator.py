"""Offline only: no runner, provider, or tool imports; never called by the online loop."""

from __future__ import annotations

import math
from collections import defaultdict
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pandas as pd

from .common import chronological_items, digest
from .contracts import time_axis
from .forecast import range_mask

PROCESS_WEIGHTS = {
    "action": 0.20,
    "need": 0.20,
    "evidence": 0.20,
    "control": 0.15,
    "verify": 0.10,
    "stop": 0.15,
}
TRUST_WEIGHTS = {
    "output": 0.20,
    "permissions": 0.20,
    "constraints": 0.15,
    "provenance": 0.20,
    "effect": 0.15,
    "recovery": 0.10,
}


def normalization(timestamps, values, frequency, seasonal_period, unit_floor, min_pairs=3, tolerance=0):
    if unit_floor <= 0 or min_pairs < 1 or seasonal_period < 1 or tolerance < 0:
        raise ValueError("Invalid training normalization configuration")
    axis = time_axis(timestamps)
    if axis.has_duplicates or not axis.is_monotonic_increasing or len(axis) != len(values):
        raise ValueError("Invalid training timeline")
    offset = pd.tseries.frequencies.to_offset(frequency)
    if offset.n <= 0:
        raise ValueError("Invalid training frequency")
    try:
        step = offset.nanos
    except ValueError:
        step = None
    y = np.asarray([np.nan if x is None else x for x in values], dtype=np.longdouble)
    lookup = {int(t): v for t, v in zip(axis.as_unit("ns").asi8, y)}
    finite = y[np.isfinite(y)]
    if not len(finite):
        raise ValueError("Training normalization needs observed values")
    for lag in dict.fromkeys([seasonal_period, 1]):
        predecessors = {
            int(t.value): int(t.value) - lag * step if step is not None
            else int((t - lag * offset).value)
            for t in axis
        }
        differences = [
            v - lookup[predecessors[t]]
            for t, v in lookup.items()
            if predecessors[t] in lookup and np.isfinite(v) and np.isfinite(lookup[predecessors[t]])
        ]
        d = np.asarray(differences, dtype=np.longdouble)
        if len(d) >= min_pairs:
            a, s = np.mean(np.abs(d)), np.mean(d * d)
            if a > tolerance and s > tolerance * tolerance:
                return _scale_result(
                    a,
                    s,
                    "seasonal" if lag == seasonal_period else "lag1",
                    lag,
                    len(d),
                    timestamps,
                    values,
                    unit_floor,
                )
    q25, q75 = np.quantile(finite, [0.25, 0.75])
    s = max(q75 - q25, abs(np.median(finite)), unit_floor)
    return _scale_result(s, s * s, "fallback", None, 0, timestamps, values, unit_floor)


def _scale_result(a, s, method, lag, count, timestamps, values, floor):
    if not np.isfinite(float(a)) or not np.isfinite(float(s)):
        raise ValueError("Training scale exceeds JSON numeric range; rescale the dataset explicitly")
    return {
        "scale_abs": float(a),
        "scale_sq": float(s),
        "method": method,
        "lag": lag,
        "pairs": count,
        "source_train_hash": digest({"timestamps": timestamps, "values": values}),
        "unit_floor": floor,
    }


SCORE_PROFILES = ("legacy-v1", "robust-v1.1", "robust-v1.2", "robust-v1.3")


def metrics(truth, prediction, scale, score_profile="legacy-v1"):
    if score_profile not in SCORE_PROFILES:
        raise ValueError("Unknown scoring profile")
    y, p = np.asarray(truth, dtype=np.longdouble), np.asarray(prediction, dtype=np.longdouble)
    if (
        y.ndim != 1
        or not len(y)
        or y.shape != p.shape
        or not np.isfinite(y).all()
        or not np.isfinite(p).all()
    ):
        raise ValueError("Scoring needs a complete aligned finite scored horizon")
    if (
        scale["scale_abs"] <= 0
        or scale["scale_sq"] <= 0
        or not np.isfinite([scale["scale_abs"], scale["scale_sq"]]).all()
    ):
        raise ValueError("Normalization scales must be positive")
    if (
        max(np.max(np.abs(y)), np.max(np.abs(p))) > 1e150
        or min(scale["scale_abs"], scale["scale_sq"]) < 1e-150
    ):
        return _decimal_metrics(truth, prediction, scale, score_profile)
    d = y - p
    mae = np.mean(np.abs(d))
    mse = np.mean(d * d)
    # Scale before subtracting/dividing to avoid overflowing finite double inputs.
    size = np.maximum(np.abs(y), np.abs(p))
    nonzero = size > 0
    sm = np.zeros(len(y), dtype=np.longdouble)
    a, b = y[nonzero] / size[nonzero], p[nonzero] / size[nonzero]
    sm[nonzero] = 2 * np.abs(a - b) / (np.abs(a) + np.abs(b))
    smape = np.mean(sm)
    mase = mae / scale["scale_abs"]
    rmsse = np.sqrt(mse / scale["scale_sq"])
    error = 0.45 * mase + 0.35 * rmsse + 0.20 * smape / 2
    if score_profile in {"robust-v1.1", "robust-v1.2", "robust-v1.3"}:
        error = (0.45 * mase + 0.35 * rmsse) / 0.80
    report = {
        "MSE": float(mse) if mse <= np.finfo(float).max else None,
        "MAE": float(mae) if mae <= np.finfo(float).max else None,
        "sMAPE": float(smape),
        "MASE": float(mase) if mase <= np.finfo(float).max else None,
        "RMSSE": float(rmsse) if rmsse <= np.finfo(float).max else None,
        "ForecastScore": float(100 / (1 + error)),
    }
    if report["MSE"] is None:
        report["MSE_scientific"] = np.format_float_scientific(mse, precision=12)
    return report


def _decimal_metrics(truth, prediction, scale, score_profile="legacy-v1"):
    with localcontext() as ctx:
        ctx.prec = 50
        y, p = [Decimal(str(x)) for x in truth], [Decimal(str(x)) for x in prediction]
        count = Decimal(len(y))
        d = [a - b for a, b in zip(y, p)]
        mse = sum(x * x for x in d) / count
        mae = sum(abs(x) for x in d) / count
        smape = (
            sum(Decimal(0) if a == 0 and b == 0 else 2 * abs(a - b) / (abs(a) + abs(b)) for a, b in zip(y, p))
            / count
        )
        mase = mae / Decimal(str(scale["scale_abs"]))
        rmsse = (mse / Decimal(str(scale["scale_sq"]))).sqrt()
        error = Decimal(".45") * mase + Decimal(".35") * rmsse + Decimal(".10") * smape
        if score_profile in {"robust-v1.1", "robust-v1.2", "robust-v1.3"}:
            error = (Decimal(".45") * mase + Decimal(".35") * rmsse) / Decimal(".80")
        result = {
            k: (float(v) if abs(v) <= Decimal(str(np.finfo(float).max)) else None)
            for k, v in {
                "MSE": mse,
                "MAE": mae,
                "sMAPE": smape,
                "MASE": mase,
                "RMSSE": rmsse,
                "ForecastScore": 100 / (1 + error),
            }.items()
        }
        for key, value in [("MSE", mse), ("MAE", mae), ("MASE", mase), ("RMSSE", rmsse)]:
            if result[key] is None:
                result[key + "_scientific"] = format(value, ".12E")
        return result


def weighted(items, weights):
    active = {k: v for k, v in items.items() if v is not None}
    coverage = sum(weights[k] for k in active)
    return (sum(weights[k] * v for k, v in active.items()) / coverage if coverage else 0), coverage


def _effect(v, n):
    declared = v.get("declared_effect")
    if not declared or declared["baseline_id"] != v["baseline_id"]:
        return 0.0
    mask = np.zeros(n, dtype=bool)
    mask[v["diff"]["changed_indices"]] = True
    proposed = range_mask(declared["ranges"], n)
    union = (mask | proposed).sum()
    jaccard = (mask & proposed).sum() / union if union else 1
    return (
        100
        * (
            jaccard
            + int(declared["direction"] == v["diff"]["direction"])
            + int(declared["magnitude"] == v["diff"]["magnitude"])
        )
        / 3
    )


def _covers(ranges, indices, n):
    mask = range_mask(ranges, n)
    return all(mask[i] for i in indices)


def _efficiency_v12(final_id, final_metrics, state, diagnostics, used, budget):
    """Penalize only clearly wasteful interaction patterns paired with a poor result."""
    if not final_id:
        return 0.0, {
            "quality_adequacy_floor": 50.0,
            "quality_full_deficit_floor": 30.0,
            "quality_deficit": 1.0,
            "budget_exhaustion": False,
            "premature_closure": False,
            "repeated_substantive_call_ratio": 0.0,
            "budget_exhaustion_penalty": 0.0,
            "premature_closure_penalty": 0.0,
            "redundancy_penalty": 0.0,
            "notice": "No legal final Agent forecast; TotalScore is zero independently of efficiency.",
        }
    quality = final_metrics["ForecastScore"]
    deficit = min(1.0, max(0.0, (50.0 - quality) / 20.0))
    substantive = diagnostics["successful_substantive_tool_calls"]
    repeated = diagnostics["repeated_successful_substantive_tool_calls"]
    repeat_ratio = repeated / substantive if substantive else 0.0
    exhausted = bool(state["final"]["forced_by_budget"])
    early_limit = max(2, math.ceil(0.20 * budget))
    premature = bool(
        not exhausted
        and used <= early_limit
        and substantive <= 1
        and diagnostics["agent_forecast_versions"] <= 1
    )
    exhaustion_penalty = deficit * 25.0 * exhausted
    premature_penalty = deficit * 15.0 * premature
    redundancy_penalty = deficit * 10.0 * repeat_ratio
    score = max(50.0, 100.0 - exhaustion_penalty - premature_penalty - redundancy_penalty)
    return score, {
        "quality_adequacy_floor": 50.0,
        "quality_full_deficit_floor": 30.0,
        "quality_deficit": deficit,
        "budget_exhaustion": exhausted,
        "premature_closure": premature,
        "premature_round_limit": early_limit,
        "successful_substantive_tool_calls": substantive,
        "repeated_successful_substantive_tool_calls": repeated,
        "repeated_substantive_call_ratio": repeat_ratio,
        "budget_exhaustion_penalty": exhaustion_penalty,
        "premature_closure_penalty": premature_penalty,
        "redundancy_penalty": redundancy_penalty,
        "notice": "Call and round counts are not monotonic rewards; penalties require a quality deficit.",
    }


def score_run(export, private_record, score_profile="legacy-v1"):
    if score_profile == "robust-v1.3":
        raise ValueError("robust-v1.3 requires score_cohort/score-cohort with all fixed Episodes")
    from .interaction_diagnostics import interaction_diagnostics

    state, config, trace = export["state"], export["config"], export["trace"]
    if state["status"] != "finished":
        raise ValueError("Unfinished/infrastructure-paused runs cannot be ranked")
    if not trace or trace[-1]["state"] != state or len(trace) != state["rounds_used"]:
        raise ValueError("Trace and final state do not agree")
    for i, row in enumerate(trace, 1):
        if (
            row["turn"] != i
            or digest(row["state"]) != row["event"]["state_hash"]
            or digest(row["response"]) != row["event"]["response_hash"]
        ):
            raise ValueError("Trace integrity failure")
    for ref, content in export.get("artifacts", {}).items():
        if digest(content) != ref:
            raise ValueError("Artifact integrity failure")
    episode = config["episode"]
    if (
        private_record["episode_id"] != episode["episode_id"]
        or private_record["timestamps"] != episode["forecast_timeline"]["scored_timestamps"]
    ):
        raise ValueError("Private scored timeline differs from the frozen public Episode")
    truth = private_record["values"]
    scale = private_record["normalization"]
    if len(truth) != episode["task"]["horizon_length"] or any(isinstance(v, bool) for v in truth):
        raise ValueError("Invalid private scored target")
    metrics(truth, truth, scale, score_profile)
    if scale["source_train_hash"] != config["revision_scale"]["source_train_hash"] or not math.isclose(
        scale["scale_abs"], config["revision_scale"]["S"], rel_tol=1e-12, abs_tol=0
    ):
        raise ValueError("Private normalization and public revision profile differ")
    n = len(truth)
    versions = state["forecasts"]
    agent = {k: v for k, v in versions.items() if v["kind"] != "model_candidate"}
    final_id = state["final"]["forecast_id"]
    if final_id is not None and final_id not in agent:
        raise ValueError("Final pointer is not a legal Agent forecast")
    from .contracts import ForecastEvaluation
    from .domain_metrics import augment_metrics, implementation_hash

    profile = config.get("forecast_evaluation")
    if profile is not None:
        profile = ForecastEvaluation.model_validate(profile)
        if profile.dataset_id != episode["dataset_id"] or profile.target_unit != episode["task"]["target_unit"]:
            raise ValueError("Frozen forecast metric profile differs from the scored task")
        if episode["task"]["forecast_gap"] and any(c.metric_id == "ramp_abs" for c in profile.components):
            raise ValueError("Ramp metric is not defined across an unscored forecast gap")
        # Validate the source even when there is no deliverable forecast.
        augment_metrics(metrics(truth, truth, scale, score_profile), truth, truth, scale,
                        profile, episode["history"]["target_values"][-1])
    all_metrics = {k: augment_metrics(metrics(truth, v["scored_values"], scale, score_profile),
                    truth, v["scored_values"], scale, profile,
                    episode["history"]["target_values"][-1]) for k, v in versions.items()}
    final_metrics = all_metrics.get(
        final_id, {"MSE": None, "MAE": None, "sMAPE": None, "MASE": None, "RMSSE": None, "ForecastScore": 0.0}
    )
    attempts = len(trace)
    valid = sum(row["event"]["valid"] for row in trace)
    alarms = state.get("protocol_alarms", [])
    capacity_alarm = state.get("capacity_alarm")
    completion_rescue = state.get("completion_rescue")
    capacity_count = int(capacity_alarm is not None)
    capacity_weight = config.get("capacity_recovery", {}).get("warning_penalty_error_equivalent", 0.0)
    alarm_turns = {a["turn"] for a in alarms} | ({capacity_alarm["turn"]} if capacity_alarm else set())
    repair_policy = config.get("protocol_repair", {"version": "disabled"})
    repair_enabled = repair_policy["version"] != "disabled"
    error_actions = len({e["turn"] for e in state["errors"]})
    warning_weight = repair_policy.get("warning_penalty_error_equivalent", 0.0)
    scored_attempts = max(
        1, attempts - sum(not r["event"]["valid"] and r["turn"] in alarm_turns for r in trace)
    )
    action_score = (
        max(
            0,
            100
            * (
                1
                - (error_actions + warning_weight * len(alarms) + capacity_weight * capacity_count)
                / scored_attempts
            ),
        )
        if repair_enabled or capacity_count
        else 100 * valid / max(attempts, 1)
    )
    info = [
        x
        for row in trace
        if row["turn"] not in alarm_turns
        for x in row["event"]["tool_records"]
        if x["information"]
    ]
    revisions = [v for _, v in chronological_items(agent) if v.get("evaluation_revision")]
    evidence_scores = []
    reversals = 0
    previous_revision = None
    for v in revisions:
        indices = v["diff"]["changed_indices"]
        refs = [state["evidence"][i] for i in v["evidence_refs"] if i in state["evidence"]]
        covers = _covers([r for ev in refs for r in ev["application_scope"]], indices, n)
        evidence_scores.append(
            100
            if refs and covers
            else (
                50
                if v["rationale_origin"] == "self_generated" and _covers(v["rationale_scope"], indices, n)
                else 0
            )
        )
        if previous_revision:
            prior = previous_revision
            old_values = np.asarray(prior["scored_values"])
            old_base = np.asarray(versions[prior["baseline_id"]]["scored_values"])
            new_values = np.asarray(v["scored_values"])
            new_base = np.asarray(versions[v["baseline_id"]]["scored_values"])
            old_mask, new_mask = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
            old_mask[prior["diff"]["changed_indices"]] = True
            new_mask[v["diff"]["changed_indices"]] = True
            reverse = old_mask & new_mask & ((old_values > old_base) != (new_values > new_base))
            relevant_new = False
            for ev in refs:
                if (
                    ev["created_turn"] > prior["created_turn"]
                    and (range_mask(ev["application_scope"], n) & reverse).any()
                ):
                    relevant_new = True
            if reverse.any() and not relevant_new and v.get("correction_reason_code") is None:
                reversals += 1
        previous_revision = v
    checks = state["validations"]
    earlier_or_final = {
        k for k, v in agent.items() if final_id and v["created_turn"] <= agent[final_id]["created_turn"]
    }
    if final_id:
        queue = list(agent[final_id].get("parents", []))
        while queue:
            parent = queue.pop()
            if parent not in earlier_or_final and parent in versions:
                earlier_or_final.add(parent)
                queue.extend(versions[parent].get("parents", []))
    verify = (
        100
        if final_id and any(v["forecast_id"] == final_id and v["complete"] for v in checks)
        else (50 if final_id and any(v["forecast_id"] in earlier_or_final for v in checks) else 0)
    )
    best = max((all_metrics[i]["ForecastScore"] for i in agent), default=0)
    forced_multiplier = (
        1.0
        if score_profile == "robust-v1.2"
        else (0.5 if state["final"]["forced_by_budget"] else 1.0)
    )
    stop = 100 * final_metrics["ForecastScore"] / best * forced_multiplier if best else 0
    process_items = {
        "action": action_score,
        "need": 100 * sum(x["scope_compatible"] for x in info) / len(info) if info else None,
        "evidence": float(np.mean(evidence_scores)) if revisions else None,
        "control": 100 * (1 - reversals / max(len(revisions), 1)) if revisions else None,
        "verify": verify,
        "stop": min(100, stop),
    }
    if score_profile in {"robust-v1.1", "robust-v1.2"} and len(agent) < 2:
        process_items["stop"] = None
    process, pc = weighted(process_items, PROCESS_WEIGHTS)
    violations = sum(e["code"] == "PERMISSION_DENIED" for e in state["errors"])
    constraint_results = []
    if final_id:
        v = agent[final_id]
        for key, value in v["auto_check"]["results"].items():
            if key.startswith("constraint:"):
                constraint_results.append(value)
    errors = state["errors"]
    recovered = []
    for error in errors:
        repeated = any(e["turn"] > error["turn"] and e["action_hash"] == error["action_hash"] for e in errors)
        later_valid = any(row["turn"] > error["turn"] and row["event"]["valid"] for row in trace)
        recovered.append(bool(final_id) and not repeated and later_valid)
    reference_errors = sum(e["code"] == "UNKNOWN_REFERENCE" for e in errors)
    explicit_refs = sum(len(v.get("parents", [])) + len(v.get("evidence_refs", [])) for v in agent.values())
    provenance = 100 * explicit_refs / (explicit_refs + reference_errors) if reference_errors else 100
    trust_items = {
        "output": 100 if final_id else 0,
        "permissions": 100 if violations == 0 else (50 if violations == 1 else 0),
        "constraints": (100 * float(np.mean(constraint_results)) if constraint_results else 0)
        if config["constraints"]
        else None,
        "provenance": provenance if final_id else 0,
        "effect": float(np.mean([_effect(v, n) for v in revisions])) if revisions else None,
        "recovery": 100 * float(np.mean(recovered)) if errors else 100,
    }
    if score_profile in {"robust-v1.1", "robust-v1.2"} and not explicit_refs and not reference_errors:
        trust_items["provenance"] = None
    trust, tc = weighted(trust_items, TRUST_WEIGHTS)
    budget = config["budget"]["max_rounds"]
    used = state["rounds_used"] - sum(r["event"].get("actor") == "environment_capacity" for r in trace)
    diagnostics = interaction_diagnostics(export)
    if score_profile == "robust-v1.2":
        efficiency, efficiency_items = _efficiency_v12(
            final_id, final_metrics, state, diagnostics, used, budget
        )
    else:
        free = math.ceil(0.5 * budget)
        efficiency = 100 if used <= free else max(0, 100 * (budget - used) / max(1, budget - free))
        efficiency_items = {
            "free_rounds": free,
            "notice": "Legacy monotonic round-use score retained for reproducibility.",
        }
    total = (
        0.60 * final_metrics["ForecastScore"] + 0.20 * process + 0.15 * trust + 0.05 * efficiency
        if final_id
        else 0
    )
    return {
        "schema_version": {
            "legacy-v1": "castbench.score/1.0",
            "robust-v1.1": "castbench.score/1.1",
            "robust-v1.2": "castbench.score/1.2",
        }[score_profile],
        "episode_id": episode["episode_id"],
        "dataset_id": episode["dataset_id"],
        **final_metrics,
        "ProcessScore": process,
        "ProcessCoverage": pc,
        "TrustScore": trust,
        "TrustCoverage": tc,
        "EfficiencyScore": efficiency,
        "TotalScore": total,
        "process_items": process_items,
        "trust_items": trust_items,
        "efficiency_items": efficiency_items,
        "rounds": used,
        "journal_transitions": len(trace),
        "ErrorActionCount": error_actions,
        "RawInvalidActionCount": attempts - valid,
        "ProtocolAlarmCount": len(alarms),
        "ProtocolRepairSuccessCount": sum(a["status"] == "recovered" for a in alarms),
        "ProtocolRepairFailureCount": sum(a["status"] == "failed" for a in alarms),
        "ProtocolAlarmPenaltyEquivalent": warning_weight * len(alarms),
        "CapacityAlarmCount": capacity_count,
        "CapacityAlarmPenaltyEquivalent": capacity_weight * capacity_count,
        "CapacityRecoverySuccessCount": int(bool(capacity_alarm and capacity_alarm["status"] == "recovered")),
        "CapacityFailureCount": int(bool(capacity_alarm and capacity_alarm["status"] == "failed")),
        "CompletionRescueCount": int(completion_rescue is not None),
        "CompletionRescueSuccessCount": int(
            bool(completion_rescue and completion_rescue["status"] == "recovered")
        ),
        "CompletionRescueFailureCount": int(
            bool(completion_rescue and completion_rescue["status"] == "failed")
        ),
        "ProviderResponseCount": sum(r["event"].get("actor") != "environment_capacity" for r in trace),
        "EnvironmentCapacityEventCount": sum(
            r["event"].get("actor") == "environment_capacity" for r in trace
        ),
        "ToolCallRequestedCount": sum(
            len((r["event"].get("action") or {}).get("payload", {}).get("calls", []))
            for r in trace
            if isinstance((r["event"].get("action") or {}).get("payload", {}).get("calls", []), list)
        ),
        "SuccessfulToolCallCount": sum(
            x["status"] in {"ok", "partial"} for r in trace for x in r["event"]["tool_records"]
        ),
        "forced_by_budget": state["final"]["forced_by_budget"],
        "completed": bool(final_id),
        "all_version_metrics": all_metrics,
        "revision_count": len(revisions),
        "unsupported_reversals": reversals,
        "scoring_notice": "Single-run observable proxies, not internal reasoning accuracy or repeat-run stability",
        "coverage_details": {
            "registered_evidence": len(state["evidence"]),
            "agent_versions": len(agent),
            "versions_with_evidence": sum(bool(v.get("evidence_refs")) for v in agent.values()),
            "independent_versions_without_declared_sources": sum(
                not v.get("parents") and not v.get("evidence_refs") for v in agent.values()
            ),
            "revision_evidence_evaluable": bool(revisions),
            "reference_integrity_evaluable": bool(explicit_refs or reference_errors),
            "own_version_selection_evaluable": len(agent) >= 2,
        },
        "score_semantics": {
            "need": "scope compatibility, not evidence usefulness",
            "provenance": "declared reference integrity, not explanation correctness",
            "stop": "selection among created Agent versions, not optimal stopping",
            "efficiency": (
                "quality-conditioned clear waste penalties; neither more nor fewer calls are intrinsically better"
                if score_profile == "robust-v1.2"
                else "legacy monotonic round-use score retained for reproducibility"
            ),
            "recovery": "subsequent valid action proxy; see separate protocol repair counts",
            "coverage": "unobserved abilities are unavailable, not demonstrated; report with scores",
        },
        "interaction_diagnostics": diagnostics,
        "evaluator_profile": (
            "robust-v1.2-outcome-conditioned-efficiency"
            if score_profile == "robust-v1.2"
            else (
                "robust-v1.1-coverage-explicit"
                if score_profile == "robust-v1.1"
                else "design-v1.0-dev-calibration-required"
            )
        ),
        "provider_profile_hash": digest(config["provider"]),
        "protocol_repair_profile_hash": digest(repair_policy),
        "evaluator_implementation_hash": digest({"evaluator": Path(__file__).read_text(encoding="utf-8"),
                                                 "domain_metrics": implementation_hash()}),
        **({"forecast_evaluation_profile_hash": digest(profile.model_dump()),
             "forecast_evaluation_profile": profile.model_dump(),
             "BaseForecastScore": final_metrics.get("BaseForecastScore", 0.0),
             "DomainMetricScore": final_metrics.get("DomainMetricScore", 0.0),
             "domain_metric_weight": sum(c.weight for c in profile.components),
             "domain_metrics": final_metrics.get("domain_metrics", {})} if profile is not None else {}),
        "trace_hash": digest(trace),
    }


def aggregate(scores, bootstrap_replicates=2000, seed=0):
    if not scores:
        raise ValueError("No completed scores")
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates < 0
    ):
        raise ValueError("bootstrap_replicates must be a nonnegative integer")
    for field in (
        "provider_profile_hash",
        "evaluator_profile",
        "evaluator_implementation_hash",
        "protocol_repair_profile_hash",
    ):
        if (
            any(not isinstance(s.get(field), str) or not s[field] for s in scores)
            or len({s[field] for s in scores}) != 1
        ):
            raise ValueError(
                f"Cannot mix or omit frozen {field}; score all runs with the same Provider/evaluator"
            )
    resource_scoring = scores[0]["evaluator_profile"] == "robust-v1.3-resource-quality-efficiency"
    if resource_scoring:
        from .efficiency import validate_scored_cohort

        validate_scored_cohort(scores)
    seen = set()
    groups = defaultdict(list)
    for score in scores:
        key = (score["dataset_id"], score["episode_id"])
        if key in seen:
            raise ValueError("Duplicate Episode in this API leaderboard")
        seen.add(key)
        groups[score["dataset_id"]].append(score)
    fields = [
        "TotalScore",
        "ForecastScore",
        "ProcessScore",
        "ProcessCoverage",
        "TrustScore",
        "TrustCoverage",
        "EfficiencyScore",
        "MSE",
        "MAE",
        "sMAPE",
        "MASE",
        "RMSSE",
        "ErrorActionCount",
        "RawInvalidActionCount",
        "ProtocolAlarmCount",
        "ProtocolRepairSuccessCount",
        "ProtocolRepairFailureCount",
        "ProtocolAlarmPenaltyEquivalent",
        "ToolCallRequestedCount",
        "SuccessfulToolCallCount",
        "CompletionRescueCount",
        "CompletionRescueSuccessCount",
        "CompletionRescueFailureCount",
    ]
    if resource_scoring:
        fields += ["MeanSampleSeconds", "MeanSampleTokens", "TimeEfficiencyScore",
                   "TokenEfficiencyScore", "BehaviorEfficiencyScore"]
    optional_zero_fields = {
        "CompletionRescueCount",
        "CompletionRescueSuccessCount",
        "CompletionRescueFailureCount",
    }

    def value(row, field):
        return row.get(field, 0) if field in optional_zero_fields else row[field]

    datasets = {}
    for dataset, rows in sorted(groups.items()):
        profile_hashes = {r.get("forecast_evaluation_profile_hash") for r in rows}
        if len(profile_hashes) != 1:
            raise ValueError("Cannot mix forecast metric profiles within one dataset")
        datasets[dataset] = {
            k: (
                stable_mean([value(r, k) for r in rows])
                if all(value(r, k) is not None for r in rows)
                else None
            )
            for k in fields
        }
        datasets[dataset].update(
            episodes=len(rows), completion_rate=float(np.mean([r["completed"] for r in rows]))
        )
        if next(iter(profile_hashes)) is not None:
            datasets[dataset].update(forecast_evaluation_profile_hash=rows[0]["forecast_evaluation_profile_hash"],
                forecast_evaluation_profile=rows[0]["forecast_evaluation_profile"],
                BaseForecastScore=stable_mean([r["BaseForecastScore"] for r in rows]),
                DomainMetricScore=stable_mean([r["DomainMetricScore"] for r in rows]),
                domain_metric_weight=rows[0]["domain_metric_weight"])
            datasets[dataset]["domain_metrics"] = {
                component["metric_id"]: {
                    "n_evaluable": sum(component["metric_id"] in r.get("domain_metrics", {}) for r in rows),
                    "score_including_failed_deliveries": stable_mean([
                        r.get("domain_metrics", {}).get(component["metric_id"], {}).get("score", 0.0) for r in rows])}
                for component in rows[0]["forecast_evaluation_profile"]["components"]}
        completed = [r for r in rows if r["completed"]]
        datasets[dataset]["valid_only"] = {
            "n_valid": len(completed),
            "n_planned": len(rows),
            "episode_ids": [r["episode_id"] for r in completed],
            "metrics": {
                k: stable_mean([r[k] for r in completed])
                if completed and all(r[k] is not None for r in completed)
                else None
                for k in ("MSE", "MAE", "sMAPE", "MASE", "RMSSE")
            },
            "notice": "Conditional on completion; compare alongside failure rates and matched Episode sets.",
        }
        alarm_count = sum(r["ProtocolAlarmCount"] for r in rows)
        datasets[dataset].update(
            alarm_sample_rate=sum(r["ProtocolAlarmCount"] > 0 for r in rows) / len(rows),
            repair_success_rate=sum(r["ProtocolRepairSuccessCount"] for r in rows) / alarm_count
            if alarm_count
            else None,
            error_action_total=sum(r["ErrorActionCount"] for r in rows),
            alarm_total=alarm_count,
            capacity_alarm_total=sum(r.get("CapacityAlarmCount", 0) for r in rows),
            capacity_failure_total=sum(r.get("CapacityFailureCount", 0) for r in rows),
            completion_rescue_total=sum(r.get("CompletionRescueCount", 0) for r in rows),
            completion_rescue_success_total=sum(
                r.get("CompletionRescueSuccessCount", 0) for r in rows
            ),
            completion_rescue_failure_total=sum(
                r.get("CompletionRescueFailureCount", 0) for r in rows
            ),
        )
    overall = {
        k: (
            stable_mean([d[k] for d in datasets.values()])
            if all(d[k] is not None for d in datasets.values())
            else None
        )
        for k in fields
    }
    rng = np.random.default_rng(seed)
    vals = [np.array([s["TotalScore"] for s in rows]) for _, rows in sorted(groups.items())]
    samples = []
    dataset_rows = [rows for _, rows in sorted(groups.items())]
    for _ in range(bootstrap_replicates):
        indices = rng.integers(0, len(vals), len(vals))
        if resource_scoring:
            from .efficiency import resampled_totals

            sampled_means = []
            for i in indices:
                rows = dataset_rows[i]
                sampled = [rows[j] for j in rng.integers(0, len(rows), len(rows))]
                sampled_means.append(stable_mean(resampled_totals(sampled)))
            samples.append(stable_mean(sampled_means))
        else:
            samples.append(
                np.mean([rng.choice(vals[i], size=len(vals[i]), replace=True).mean() for i in indices])
            )
    overall["TotalScore_ci95"] = np.quantile(samples, [0.025, 0.975]).tolist() if samples else None
    return {
        "datasets": datasets,
        "overall": overall,
        "aggregation": "Episode -> dataset -> equal dataset macro-average",
        "ci_note": "Hierarchical bootstrap estimates sample uncertainty, not LLM sampling stability",
        "provider_profile_hash": scores[0]["provider_profile_hash"],
        "protocol_repair_profile_hash": scores[0]["protocol_repair_profile_hash"],
        "evaluator_implementation_hash": scores[0]["evaluator_implementation_hash"],
        **({"efficiency_cohort_hash": scores[0]["efficiency_cohort_hash"],
            "efficiency_profile_hashes": {d: rows[0]["efficiency_profile_hash"]
                                         for d, rows in groups.items()},
            "resource_notice": "Dataset arithmetic sample means; latency reflects deployment too. "
                               "Bootstrap recomputes resource means before score transformation."}
           if resource_scoring else {}),
    }


def stable_mean(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Aggregate inputs must be finite or explicitly unavailable")
    size = np.max(np.abs(values))
    return float(np.mean(values / size) * size) if size else 0.0
