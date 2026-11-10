from __future__ import annotations

import base64
import io
import threading

import numpy as np
from scipy import signal, stats

from .common import AgentError, clean_numbers

_plot_lock = threading.Lock()


def summary(x):
    x = np.asarray(x, dtype=float)
    v = x[np.isfinite(x)]
    if not len(v):
        return {"count": len(x), "valid_count": 0, "mean": None, "median": None}
    return clean_numbers(
        {
            "count": len(x),
            "valid_count": len(v),
            "mean": np.mean(v),
            "median": np.median(v),
            "std": np.std(v),
            "variance": np.var(v),
            "min": np.min(v),
            "max": np.max(v),
            "quantiles": dict(zip(["q05", "q25", "q75", "q95"], np.quantile(v, [0.05, 0.25, 0.75, 0.95]))),
            "coefficient_of_variation": np.std(v) / abs(np.mean(v)) if np.mean(v) != 0 else None,
        }
    )


def contiguous(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [[int(a), int(b)] for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def _prepare(x, policy):
    valid = np.isfinite(x)
    if valid.all():
        return x.copy(), []
    if policy != "linear" or valid.sum() < 2:
        return None, ["Method requires complete data; explicit linear interpolation is available"]
    # No boundary extrapolation: unsupported leading/trailing missing points remain inapplicable.
    if not valid[0] or not valid[-1]:
        return None, ["Linear policy does not extrapolate missing boundaries"]
    return np.interp(np.arange(len(x)), np.flatnonzero(valid), x[valid]), [
        "Explicit linear interpolation of internal missing points"
    ]


def analyze(name, x, timestamps, params):
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    v = x[valid]
    result = {
        "status": "partial" if not valid.all() else "ok",
        "warnings": [],
        "coverage": {"requested_points": len(x), "valid_points": int(valid.sum())},
        "structured_result": {},
    }
    if not len(v):
        result.update(status="not_applicable", warnings=["No finite observations"])
        return result
    out = {"parameters": params, "statistics": summary(x)}
    if name == "window_summary":
        out.update(
            start=timestamps[0], end=timestamps[-1], recent=summary(x[-min(len(x), max(1, len(x) // 4)) :])
        )
    elif name == "basic_statistics":
        pass
    elif name == "data_quality_profile":
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        out.update(
            missing_indices=np.flatnonzero(~valid).tolist(),
            missing_runs=contiguous(~valid),
            outlier_candidate_indices=np.flatnonzero(valid & (np.abs(x - med) > 6 * 1.4826 * mad)).tolist()
            if mad > 0
            else [],
            flat_pair_runs=contiguous(np.isfinite(x[:-1]) & np.isfinite(x[1:]) & (x[:-1] == x[1:])),
            outlier_rule="6 scaled MAD, diagnostic only; undefined when MAD=0",
        )
    elif name == "robust_distribution":
        q25, q75 = np.quantile(v, [0.25, 0.75])
        varying = np.ptp(v) > 0
        out.update(
            mad=np.median(np.abs(v - np.median(v))),
            iqr=q75 - q25,
            skew=stats.skew(v, bias=False) if len(v) >= 3 and varying else None,
            excess_kurtosis=stats.kurtosis(v, bias=False) if len(v) >= 4 and varying else None,
            zero_fraction=np.mean(v == 0),
            tukey_tail_fraction=np.mean((v < q25 - 1.5 * (q75 - q25)) | (v > q75 + 1.5 * (q75 - q25))),
        )
    elif name == "multiscale_trend":
        reports = []
        for scale in params["scales"]:
            if scale > len(x) or scale < 3:
                reports.append({"scale": scale, "status": "not_applicable"})
                continue
            part = x[-scale:]
            mask = np.isfinite(part)
            if mask.sum() < 3:
                reports.append({"scale": scale, "status": "not_applicable"})
                continue
            slope, intercept, low, high = stats.theilslopes(part[mask], np.flatnonzero(mask))
            reports.append(
                {
                    "scale": scale,
                    "slope_per_step": slope,
                    "intercept": intercept,
                    "ci95": [low, high],
                    "valid_count": int(mask.sum()),
                    "direction": "up" if low > 0 else ("down" if high < 0 else "uncertain"),
                }
            )
        out["scales"] = reports
    elif name in {
        "seasonality_spectrum",
        "autocorrelation_profile",
        "robust_decomposition",
        "change_point_regime",
    }:
        filled, notes = _prepare(x, params.get("missing_policy", "reject"))
        result["warnings"] += notes
        if filled is None or len(x) < 8:
            result.update(
                status="not_applicable", warnings=result["warnings"] + ["Insufficient complete support"]
            )
            return result
        out["imputed_indices"] = np.flatnonzero(~valid).tolist()
        if name == "seasonality_spectrum":
            frequencies, power = signal.periodogram(filled, detrend="linear")
            mask = (frequencies > 0) & (frequencies >= 2.5 / len(x))
            candidates = np.flatnonzero(mask)
            order = sorted(candidates, key=lambda i: (-power[i], i))[:5] if power.sum() > 0 else []
            out["periods"] = [
                {"period_steps": 1 / frequencies[i], "power_fraction": power[i] / power.sum()} for i in order
            ]
            result["warnings"].append(
                "Periodogram peaks are candidates; short windows and harmonics can mislead"
            )
        elif name == "autocorrelation_profile":
            from statsmodels.tsa.stattools import acf, pacf

            lag = min(params["max_lag"], len(x) // 2 - 1)
            if np.ptp(filled) == 0:
                out.update(acf=None, pacf=None, max_lag=lag)
                result.update(status="not_applicable", warnings=["ACF/PACF undefined for a constant series"])
            else:
                out.update(
                    acf=acf(filled, nlags=lag, fft=False),
                    pacf=pacf(filled, nlags=lag, method="ywm"),
                    max_lag=lag,
                )
        elif name == "robust_decomposition":
            from statsmodels.tsa.seasonal import STL

            period = params["period"]
            if len(x) < 2 * period:
                result.update(status="not_applicable", warnings=["At least two periods are required"])
                return result
            fit = STL(filled, period=period, robust=True).fit()
            denom = np.var(fit.resid + fit.seasonal)
            out.update(
                trend=fit.trend,
                seasonal=fit.seasonal,
                residual=fit.resid,
                seasonal_strength=max(0, 1 - np.var(fit.resid) / denom) if denom > 0 else 0,
                reconstruction_max_error=np.max(np.abs(filled - fit.trend - fit.seasonal - fit.resid)),
            )
        else:
            minimum = params["min_segment"]
            if minimum < 3:
                raise AgentError("Regime segments require at least three points")
            cuts = [0, len(x)]
            scale = np.std(filled)
            floor = max(scale**2 * 1e-8, np.finfo(float).tiny)

            def segment_cost(a, b):
                part = filled[a:b]
                t = np.arange(b - a, dtype=float)
                slope = np.polyfit(t, part, 1)[0] if len(part) > 2 else 0
                residual = part - (part.mean() + slope * (t - t.mean()))
                return len(part) * np.log(max(float(np.mean(residual**2)), floor))

            for _ in range(params["max_changes"]):
                best = None
                for a, b in zip(cuts[:-1], cuts[1:]):
                    for c in range(a + minimum, b - minimum + 1):
                        gain = segment_cost(a, b) - segment_cost(a, c) - segment_cost(c, b)
                        if best is None or gain > best[0]:
                            best = (gain, c)
                if scale == 0 or best is None or best[0] <= 3 * params["penalty"] * np.log(max(len(x), 2)):
                    break
                cuts = sorted(cuts + [best[1]])
            segments = [
                {
                    "start": a,
                    "end": b,
                    "slope_per_step": float(np.polyfit(np.arange(b - a), filled[a:b], 1)[0]),
                    **summary(filled[a:b]),
                }
                for a, b in zip(cuts[:-1], cuts[1:])
            ]
            out.update(
                change_indices=cuts[1:-1],
                segments=segments,
                current_regime_length=cuts[-1] - cuts[-2],
                method="penalized binary segmentation of linear Gaussian regimes; three parameters per segment",
            )
            result["warnings"].append(
                "Descriptive level/trend/variance segmentation, not a causal regime identification"
            )
    elif name == "volatility_profile":
        window = params["window"]
        if window > len(x):
            result.update(status="not_applicable", warnings=["Window exceeds available series"])
            return result
        rolling = [summary(x[max(0, i - window + 1) : i + 1]).get("std") for i in range(len(x))]
        out.update(rolling_std=rolling, window=window, finite_count=int(valid.sum()))
    elif name == "ramp_turning_points":
        pairs = valid[:-1] & valid[1:]
        ramps = np.diff(x)
        ids = sorted(np.flatnonzero(pairs), key=lambda i: (-abs(ramps[i]), int(i)))[:10]
        turns = np.flatnonzero(valid[:-2] & valid[1:-1] & valid[2:] & (ramps[:-1] * ramps[1:] < 0)) + 1
        out.update(
            top_ramps=[{"start": int(i), "end": int(i + 1), "delta_per_step": ramps[i]} for i in ids],
            turning_indices=turns.tolist(),
        )
    elif name in {"time_series_to_text", "time_series_to_image"}:
        s = summary(x)
        out["text"] = (
            f"{len(v)}/{len(x)} finite points; mean={s['mean']:.6g}, median={s['median']:.6g}, min={s['min']:.6g}, max={s['max']:.6g}. No causal inference."
        )
        if name == "time_series_to_image":
            with _plot_lock:
                from matplotlib.backends.backend_agg import FigureCanvasAgg
                from matplotlib.figure import Figure

                fig = Figure(figsize=(10, 3), layout="constrained")
                FigureCanvasAgg(fig)
                ax = fig.subplots()
                ax.plot(np.arange(len(x)), x, color="#137c8b", linewidth=1.4)
                ax.set(xlabel="Time step", ylabel="Value", title="Public series")
                ax.grid(alpha=0.2)
                buffer = io.BytesIO()
                fig.savefig(buffer, format="png", dpi=140, metadata={"Software": "CastBench"})
            out.update(
                image_png_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
                plot_data={"timestamps": timestamps, "values": clean_numbers(x)},
                text_core_note="Image is supplementary; exact numeric and textual views are provided",
            )
    else:
        raise AgentError("Unknown feature algorithm")
    result["structured_result"] = clean_numbers(out)
    return result


def associations(target, variables, max_lag=0):
    y = np.asarray(target, dtype=float)
    output = {}
    for name, values in variables.items():
        x = np.asarray(values, dtype=float)
        rows = []
        for lag in range(-max_lag, max_lag + 1):
            # Positive lag compares x[t-lag] with y[t]. All points are historical.
            a, b = (x[:-lag], y[lag:]) if lag > 0 else ((x[-lag:], y[:lag]) if lag < 0 else (x, y))
            mask = np.isfinite(a) & np.isfinite(b)
            a, b = a[mask], b[mask]
            useful = len(a) >= 3 and np.ptp(a) > 0 and np.ptp(b) > 0
            half = len(a) // 2
            blocks = []
            for aa, bb in [(a[:half], b[:half]), (a[half:], b[half:])]:
                blocks.append(
                    float(stats.pearsonr(aa, bb).statistic)
                    if len(aa) >= 3 and np.ptp(aa) > 0 and np.ptp(bb) > 0
                    else None
                )
            strata = []
            if useful and lag == 0:
                edges = np.unique(np.quantile(a, [0, 0.25, 0.5, 0.75, 1]))
                for j, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
                    selected = (a >= lower) & ((a <= upper) if j == len(edges) - 2 else (a < upper))
                    strata.append(
                        {
                            "covariate_range": [float(lower), float(upper)],
                            "count": int(selected.sum()),
                            "target_mean": float(np.mean(b[selected])) if selected.any() else None,
                        }
                    )
            rows.append(
                {
                    "lag": lag,
                    "valid_pairs": len(a),
                    "pearson": stats.pearsonr(a, b).statistic if useful else None,
                    "spearman": stats.spearmanr(a, b).statistic if useful else None,
                    "half_window_pearson": blocks,
                    "covariate_strata": strata,
                }
            )
        output[name] = rows
    return clean_numbers(
        {
            "associations": output,
            "lag_convention": "positive = covariate leads target",
            "warning": "Historical association is not causality; no future target is used",
        }
    )
