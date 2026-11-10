"""Frozen, offline domain losses. Packages supply data, never executable scorers."""

from __future__ import annotations

from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

from .common import digest
from .contracts import ForecastEvaluation


def implementation_hash():
    return digest({name: Path(__file__).with_name(name).read_text(encoding="utf-8")
                   for name in ("domain_metrics.py", "contracts.py")})


def public_notice(profile):
    if profile is None:
        return None
    data = profile.model_dump()
    definitions = {
        "peak_abs": "abs(max(prediction)-max(truth)); peak magnitude only, not peak timing.",
        "window_total_abs": "abs(sum(prediction)-sum(truth)); opposite-signed point errors can cancel.",
        "ramp_abs": "mean(abs((prediction[t]-prediction[t-1])-(truth[t]-truth[t-1]))); both t=0 anchors equal the last historical target.",
        "asymmetric_abs": "mean(overprediction_cost*max(prediction-truth,0)+underprediction_cost*max(truth-prediction,0)); requires a disclosed cost basis.",
    }
    data.update(profile_hash=digest(profile.model_dump()),
        base_forecast_weight=1.0 - sum(c.weight for c in profile.components),
        loss_definitions={c.metric_id: definitions[c.metric_id] for c in profile.components},
        normalization="Each raw loss is divided by the frozen training scale_abs. Component score=100/(1+scaled_loss). window_total_abs additionally divides by horizon length.",
        formula="ForecastScore=base_forecast_weight*BaseForecastScore+sum(weight*component_score). Raw MAE/MSE/sMAPE/MASE/RMSSE are unchanged.",
        decision_notice="The extra objective has a small weight. Do not raise a peak, smooth a curve, or change its total merely to look favorable: these losses compare with unknown truth and do not reward larger or smoother predictions by themselves. Shape checks are not accuracy checks.",
        information_boundary="Future targets are private; do not infer metric values online. All weights and cost parameters are fixed before this Episode.")
    return data


def domain_scores(truth, prediction, scale, profile, last_observed):
    profile = ForecastEvaluation.model_validate(profile)
    if profile.source_train_hash != scale["source_train_hash"]:
        raise ValueError("Domain metric normalization differs from its frozen training source")
    y, p = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Domain metrics require matching, finite, complete vectors")
    if any(isinstance(v, (bool, np.bool_)) for v in (*truth, *prediction)):
        raise ValueError("Boolean values are not numeric forecasts")
    a = scale["scale_abs"]
    if isinstance(a, bool) or not np.isfinite(a) or a <= 0:
        raise ValueError("Domain metrics require a positive finite training scale")
    output = {}
    # Decimal also handles valid extreme input where a float subtraction/sum overflows.
    with localcontext() as ctx:
        ctx.prec = 50
        yd, pd = [Decimal(str(v)) for v in y], [Decimal(str(v)) for v in p]
        errors = [b - a for a, b in zip(yd, pd)]
        n, denominator = Decimal(len(y)), Decimal(str(a))
        for component in profile.components:
            metric = component.metric_id
            divisor = denominator
            if metric == "peak_abs":
                loss = abs(max(pd) - max(yd))
            elif metric == "window_total_abs":
                loss = abs(sum(errors))
                divisor *= n
            elif metric == "ramp_abs":
                if isinstance(last_observed, bool) or last_observed is None or not np.isfinite(last_observed):
                    raise ValueError("Ramp metric requires a finite historical anchor")
                # The shared observed anchor makes the first error e_1, not an omitted edge.
                prior_errors = [Decimal(0)] + errors[:-1]
                loss = sum(abs(e - previous) for e, previous in zip(errors, prior_errors)) / n
            elif metric == "asymmetric_abs":
                over, under = Decimal(str(component.overprediction_cost)), Decimal(str(component.underprediction_cost))
                loss = sum(over * e if e >= 0 else under * -e for e in errors) / n
            else:
                raise ValueError("Unsupported domain metric: " + metric)
            scaled = loss / divisor
            score = float(Decimal(100) / (1 + scaled))
            def display(value):
                return float(value) if value <= Decimal(str(np.finfo(float).max)) else None
            output[metric] = {"raw_loss": display(loss), "raw_loss_scientific": format(loss, ".12E"),
                "scaled_loss": display(scaled), "score": score, "weight": component.weight,
                "raw_unit": profile.target_unit, "description": component.description}
    return output


def augment_metrics(base, truth, prediction, scale, profile, last_observed):
    if profile is None:
        return base
    profile = ForecastEvaluation.model_validate(profile)
    parts = domain_scores(truth, prediction, scale, profile, last_observed)
    weight = sum(component.weight for component in profile.components)
    domain_score = sum(part["weight"] * part["score"] for part in parts.values()) / weight
    return {**base, "BaseForecastScore": base["ForecastScore"], "DomainMetricScore": domain_score,
        "ForecastScore": (1 - weight) * base["ForecastScore"] + weight * domain_score,
        "domain_metrics": parts, "forecast_evaluation_profile": profile.model_dump(),
        "forecast_evaluation_profile_hash": digest(profile.model_dump()),
        "domain_metric_weight": weight}
