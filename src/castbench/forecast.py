from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from .common import AgentError, finite_array
from .contracts import ForecastArgs, IndexRange, PublicBundle, RevisionScale


def bounds(r: IndexRange, n: int) -> tuple[int, int]:
    end = n if r.end is None else r.end
    if not 0 <= r.start < end <= n:
        raise AgentError("Range is outside the selected timeline")
    return r.start, end


def range_mask(ranges, n):
    mask = np.zeros(n, dtype=bool)
    for raw in ranges:
        r = IndexRange.model_validate(raw)
        if r.timeline == "scored":
            start, end = bounds(r, n)
            mask[start:end] = True
    return mask


def validate_ranges(ranges, scored_n, gap_n):
    for raw in ranges:
        r = IndexRange.model_validate(raw)
        bounds(r, scored_n if r.timeline == "scored" else gap_n)


def diff_report(before, after, scale: RevisionScale):
    if max(abs(x) for x in [*before, *after]) > 1e150 or scale.S < 1e-150:
        # Windows longdouble is only float64; use Decimal at extreme scales.
        with localcontext() as ctx:
            ctx.prec = 50
            epsilon = max(
                Decimal(str(scale.absolute_tolerance)),
                Decimal(str(scale.relative_tolerance)) * Decimal(str(scale.S)),
            )
            differences = [Decimal(str(a)) - Decimal(str(b)) for b, a in zip(before, after)]
            indices = [i for i, d in enumerate(differences) if abs(d) > epsilon]
            if not indices:
                return {
                    "changed_indices": [],
                    "direction": "unchanged",
                    "magnitude": "none",
                    "mean_abs_normalized": 0.0,
                }
            active = [differences[i] for i in indices]
            positive = sum(d > 0 for d in active) / len(active)
            value = sum(abs(d) for d in active) / len(active) / Decimal(str(scale.S))
            edges = [Decimal(str(e)) for e in scale.magnitude_edges]
            return {
                "changed_indices": indices,
                "direction": "increase"
                if positive >= scale.direction_fraction
                else ("decrease" if 1 - positive >= scale.direction_fraction else "mixed"),
                "magnitude": "small" if value < edges[0] else ("moderate" if value < edges[1] else "large"),
                "mean_abs_normalized": float(min(value, Decimal(str(np.finfo(float).max)))),
                **(
                    {"mean_abs_normalized_scientific": format(value, ".12E")}
                    if value > Decimal(str(np.finfo(float).max))
                    else {}
                ),
            }
    d = np.asarray(after, dtype=np.longdouble) - np.asarray(before, dtype=np.longdouble)
    epsilon = max(scale.absolute_tolerance, scale.relative_tolerance * scale.S)
    mask = np.abs(d) > epsilon
    if not mask.any():
        return {
            "changed_indices": [],
            "direction": "unchanged",
            "magnitude": "none",
            "mean_abs_normalized": 0.0,
        }
    active = d[mask]
    positive = np.mean(active > 0)
    direction = (
        "increase"
        if positive >= scale.direction_fraction
        else ("decrease" if 1 - positive >= scale.direction_fraction else "mixed")
    )
    magnitude_value = np.mean(np.abs(active)) / scale.S
    edges = scale.magnitude_edges
    magnitude = (
        "small" if magnitude_value < edges[0] else ("moderate" if magnitude_value < edges[1] else "large")
    )
    return {
        "changed_indices": np.flatnonzero(mask).tolist(),
        "direction": direction,
        "magnitude": magnitude,
        "mean_abs_normalized": min(float(magnitude_value), 1.7976931348623157e308),
    }


@np.errstate(over="ignore")
def check_values(values, bundle: PublicBundle, checks=None):
    checks = bundle.default_check_profile if checks is None else checks
    if not checks or not set(checks) <= set(bundle.default_check_profile):
        raise AgentError("Check profile must be nonempty and use available checks")
    v = np.asarray(values, dtype=np.longdouble)
    results = {}
    warnings = []
    if "timeline" in checks:
        results["timeline"] = len(v) == bundle.episode.task.horizon_length
    if "finite" in checks:
        results["finite"] = bool(np.isfinite(v).all())
    if "shape" in checks:
        h = np.asarray([x for x in bundle.episode.history.target_values if x is not None])
        scale = bundle.revision_scale.S
        shape_ok = not (len(v) > 1 and np.max(np.abs(np.diff(v))) > 10 * scale)
        results["shape"] = bool(shape_ok)
        if not shape_ok:
            warnings.append("Large step relative to frozen training scale")
        if len(h) and (np.min(v) < np.min(h) - 5 * scale or np.max(v) > np.max(h) + 5 * scale):
            warnings.append("Outside recent historical support; this is not a rejection")
    if "domain" in checks:
        for c in bundle.constraints:
            if c.kind == "minimum":
                ok = bool(np.all(v >= c.value))
            elif c.kind == "maximum":
                ok = bool(np.all(v <= c.value))
            else:
                ok = bool(np.all(np.abs(np.diff(v)) <= c.value))
            results["constraint:" + c.constraint_id] = ok
            if not ok:
                warnings.append(f"Constraint {c.constraint_id} failed ({'hard' if c.hard else 'soft'})")
    return {
        "checks": list(checks),
        "results": results,
        "warnings": warnings,
        "complete": set(checks) == set(bundle.default_check_profile),
    }


def _profile(value, n, name):
    if value is None:
        raise AgentError(f"{name} is required")
    return finite_array(value, n) if isinstance(value, list) else np.full(n, value, dtype=float)


def create_forecast(args: ForecastArgs, state, bundle: PublicBundle):
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _create_forecast(args, state, bundle)
    except (FloatingPointError, OverflowError) as exc:
        raise AgentError(
            "Requested operation exceeds finite numeric support; no version was created"
        ) from exc


def _create_forecast(args: ForecastArgs, state, bundle: PublicBundle):
    versions = state["forecasts"]
    n = bundle.episode.task.horizon_length
    gap_n = bundle.episode.task.forecast_gap
    op = args.operation
    parents = []

    def get(ref):
        if not ref or ref not in versions:
            raise AgentError("Unknown forecast in this Episode", "UNKNOWN_REFERENCE")
        if ref not in parents:
            parents.append(ref)
        return versions[ref]

    def array(version, timeline):
        value = version[timeline + "_values"]
        if value is None:
            raise AgentError("Source has no gap forecast")
        return np.asarray(value, dtype=float).copy()

    if op == "constant_forecast":
        if args.unit != bundle.episode.task.target_unit:
            raise AgentError("Explicit target unit required")
        if args.constant_value is None:
            raise AgentError("constant_value is required; the Agent must choose the value")
        if gap_n == 0 and args.gap_constant_value is not None:
            raise AgentError("No gap exists in this Episode")
        scored = np.full(n, args.constant_value, dtype=float)
        gap = (
            None if args.gap_constant_value is None else np.full(gap_n, args.gap_constant_value, dtype=float)
        )
        kind = "agent_generated"
    elif op == "submit_full_forecast":
        if args.unit != bundle.episode.task.target_unit:
            raise AgentError("Explicit target unit required")
        if args.scored_timestamps != bundle.episode.forecast_timeline.scored_timestamps:
            raise AgentError("Scored timestamps do not exactly match the public timeline")
        scored = finite_array(args.scored_values, n)
        gap = None
        if args.gap_values is not None or args.gap_timestamps is not None:
            if args.gap_timestamps != bundle.episode.forecast_timeline.gap_timestamps:
                raise AgentError("Gap timestamps do not match")
            gap = finite_array(args.gap_values, gap_n)
        kind = "agent_generated"
    elif op == "adopt_candidate":
        source = get(args.candidate_id)
        if source["kind"] != "model_candidate":
            raise AgentError("adopt_candidate requires a model_candidate")
        scored, gap = array(source, "scored"), source["gap_values"]
        gap = None if gap is None else np.asarray(gap).copy()
        kind = "agent_adopted"
    else:
        if args.declared_effect is None:
            raise AgentError("A structured declared_effect is required for numerical revisions")
        kind = "agent_blended" if op == "blend_forecasts" else "agent_revised"
        timeline = args.range.timeline
        length = n if timeline == "scored" else gap_n
        start, end = bounds(args.range, length)
        if op == "blend_forecasts":
            if len(args.source_ids) < 2 or len(set(args.source_ids)) != len(args.source_ids):
                raise AgentError("Blend requires at least two distinct sources")
            weights = finite_array(args.weights, len(args.source_ids))
            if (weights < 0).any() or abs(np.sum(weights) - 1) > 1e-8:
                raise AgentError("Weights must be nonnegative and sum to one within 1e-8")
            sources = [get(i) for i in args.source_ids]
            partial = start != 0 or end != length
            if partial and args.base_id is None:
                raise AgentError("Partial blend requires base_id")
            base = get(args.base_id) if args.base_id else sources[0]
            scored = array(base, "scored")
            gap = None if base["gap_values"] is None else array(base, "gap")
            target = scored if timeline == "scored" else gap
            if target is None:
                raise AgentError("Base has no gap forecast")
            target[start:end] = np.sum(
                [w * array(s, timeline)[start:end] for w, s in zip(weights, sources)], axis=0
            )
        else:
            parent = get(args.parent_id)
            scored = array(parent, "scored")
            gap = None if parent["gap_values"] is None else array(parent, "gap")
            target = scored if timeline == "scored" else gap
            if target is None:
                raise AgentError("Parent has no gap forecast")
            if op == "add_offset":
                target[start:end] += _profile(args.offset, end - start, "offset")
            elif op == "scale_values":
                target[start:end] *= _profile(args.factor, end - start, "factor")
            elif op == "adjust_segments":
                if not args.segments:
                    raise AgentError("segments cannot be empty")
                touched = set()
                for segment in args.segments:
                    seg = segment.model_dump(exclude_none=True)
                    if set(seg) - {"range", "offset", "factor"} or ("offset" in seg) == ("factor" in seg):
                        raise AgentError("Each segment has a range and exactly one offset or factor")
                    r = IndexRange.model_validate(seg["range"])
                    if r.timeline != timeline:
                        raise AgentError("All segments must share the operation timeline")
                    a, b = bounds(r, length)
                    if a < start or b > end or touched.intersection(range(a, b)):
                        raise AgentError("Overlapping or out-of-range segments")
                    touched.update(range(a, b))
                    key = "offset" if "offset" in seg else "factor"
                    val = seg[key]
                    if isinstance(val, bool):
                        raise AgentError("Boolean is not a numeric profile")
                    p = _profile(val, b - a, key)
                    target[a:b] = target[a:b] + p if key == "offset" else target[a:b] * p
            elif op == "shift_time":
                if args.steps is None or args.fill_policy is None:
                    raise AgentError("steps and explicit fill_policy are required")
                old = target[start:end].copy()
                idx = np.arange(len(old)) - args.steps
                valid = (idx >= 0) & (idx < len(old))
                new = old.copy()
                new[valid] = old[idx[valid]]
                if args.fill_policy == "edge":
                    new[~valid] = old[np.clip(idx[~valid], 0, len(old) - 1)]
                elif args.fill_policy == "explicit_values":
                    new[~valid] = finite_array(args.fill_values, int((~valid).sum()))
                target[start:end] = new
            elif op == "smooth_revision_delta":
                proposed = array(get(args.proposed_id), timeline)
                if args.window % 2 == 0 or args.window > end - start:
                    raise AgentError("Smoothing window must be odd and fit the selected range")
                delta = proposed[start:end] - target[start:end]
                smooth = (
                    uniform_filter1d(delta, size=args.window, mode="nearest")
                    if args.method == "moving_average"
                    else median_filter(delta, size=args.window, mode="nearest")
                )
                target[start:end] += smooth
            elif op == "replace_segment":
                if (args.values is None) == (args.source_id is None):
                    raise AgentError("Provide exactly one replacement values or source_id")
                target[start:end] = (
                    finite_array(args.values, end - start)
                    if args.values is not None
                    else array(get(args.source_id), timeline)[start:end]
                )
        if not np.isfinite(scored).all() or (gap is not None and not np.isfinite(gap).all()):
            raise AgentError("Operation produced non-finite values; no version was created")

    report = check_values(scored, bundle)
    for c in bundle.constraints:
        if c.hard and not report["results"].get("constraint:" + c.constraint_id, True):
            raise AgentError(f"Hard constraint {c.constraint_id} failed", "HARD_CONSTRAINT")
    for ref in args.evidence_refs:
        if ref not in state["evidence"]:
            raise AgentError("Evidence reference does not exist", "UNKNOWN_REFERENCE")
    previous = state["active_forecast_id"]
    operation_baseline = (
        (args.base_id or parents[0])
        if kind == "agent_blended"
        else (args.parent_id if kind == "agent_revised" else None)
    )
    baseline = previous or operation_baseline
    if args.declared_effect:
        if args.declared_effect.baseline_id != baseline:
            raise AgentError(
                "declared_effect must compare to the previous Agent version (or first operation source)"
            )
        validate_ranges([r.model_dump() for r in args.declared_effect.ranges], n, gap_n)
    validate_ranges([r.model_dump() for r in args.rationale_scope], n, gap_n)
    diff = (
        diff_report(versions[baseline]["scored_values"], scored, bundle.revision_scale) if baseline else None
    )
    return {
        "kind": kind,
        "parents": parents,
        "previous_agent_forecast_id": previous,
        "scored_values": scored.tolist(),
        "gap_values": None if gap is None else gap.tolist(),
        "unit": bundle.episode.task.target_unit,
        "operation": op,
        "baseline_id": baseline,
        "diff": diff,
        "evaluation_revision": bool(diff and diff["changed_indices"]),
        "declared_effect": args.declared_effect.model_dump() if args.declared_effect else None,
        "evidence_refs": args.evidence_refs,
        "rationale_origin": args.rationale_origin,
        "rationale_scope": [r.model_dump() for r in args.rationale_scope],
        "correction_reason_code": args.correction_reason_code,
        "auto_check": report,
    }
