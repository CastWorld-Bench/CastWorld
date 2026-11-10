"""Reviewed, read-only domain adapters. No filesystem, private labels, or LLM calls."""

from __future__ import annotations

import hashlib
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field

from .catalog import Registry
from .common import AgentError
from .contracts import Contract, Number, Scope


class DomainScope(Scope):
    timeline: Literal["history", "scored"] = "scored"


class DomainArgs(Contract):
    scope: DomainScope = Field(default_factory=DomainScope)


class DegreeArgs(DomainArgs):
    balance_temperature_f: Number = Field(default=65.0, ge=-50, le=120)


class HistoryArgs(DomainArgs):
    scope: DomainScope = Field(default_factory=lambda: DomainScope(timeline="history"))


INPUTS = {
    "weather_units": ["temp", "atemp", "hum", "windspeed"],
    "weather_exposure": ["rain_flag", "snow_flag", "storm_flag", "precip_1h_mm", "snow_depth_mm", "temp"],
    "observation_quality": [
        "precip_observed_flag",
        "precip_allocated_flag",
        "sky_obscured_flag",
        "cloud_valid_hours",
    ],
    "calendar_transition": ["workingday", "weekday"],
    "history_demand_shape": [],
    "degree_hours": ["temperature_2m_f"],
    "station_dispersion": [f"station_{i}" for i in range(1, 12)] + ["temperature_2m_f"],
    "moisture_context": [
        "temperature_2m_f",
        "dew_point_2m_f",
        "relative_humidity_2m_pct",
        "vapour_pressure_deficit_kpa",
    ],
    "event_exposure": [
        "is_day_off",
        "dst_transition_window",
        "heatwave_severity_f",
        "coldwave_severity_f",
        "noaa_severe_event_any_va",
        "federal_holiday",
    ],
    "history_load_shape": [],
    "radiation_budget": ["shortwave_radiation_fused", "diffuse_radiation_fused", "direct_radiation_fused"],
    "spatial_radiation": [
        "station_direct_radiation_fused",
        "station_direct_radiation_spread",
        "direct_radiation_fused",
    ],
    "snow_dust_exposure": [
        "high_dust_proxy",
        "persistent_snow_proxy",
        "station_precipitation_any",
        "snowfall_24h_sum",
        "icing_risk_score",
    ],
    "thermal_context": ["temperature_2m_fused", "wind_speed_10m_fused", "station_temperature_2m_fused"],
    "history_low_light_power": ["shortwave_radiation_fused"],
    "wind_resource": ["hf_wind_speed_80m_fused", "wind_gusts_10m_fused"],
    "vertical_shear": ["hf_wind_speed_80m_fused", "hf_wind_speed_120m_fused"],
    "direction_coherence": ["station_wind_direction_80m_fused", "station_wind_direction_80m_resultant"],
    "icing_exposure": [
        "icing_risk_score",
        "hf_temperature_80m_fused",
        "relative_humidity_2m_fused",
        "snowfall_24h_sum",
    ],
    "history_wind_power": ["hf_wind_speed_80m_fused"],
}

TOOLS = {
    "bike": {
        "weather_units": (
            "context",
            "Decode normalized temperature, apparent temperature, humidity and wind; no fitted response",
        ),
        "weather_exposure": (
            "context",
            "Rain/snow/storm duration and wet/cold co-occurrence; preserve precipitation provenance",
        ),
        "observation_quality": (
            "context",
            "Separate observed precipitation, allocated precipitation and obscured sky",
        ),
        "calendar_transition": (
            "context",
            "Working-day/weekend context on the explicitly requested window",
        ),
        "history_demand_shape": (
            "feature",
            "Observed hourly demand shares and ramps; no future demand access",
        ),
    },
    "gef": {
        "degree_hours": (
            "context",
            "Heating/cooling degree-hours from regional temperature; explicit reference, not a fitted load law",
        ),
        "station_dispersion": (
            "feature",
            "11-station temperature center, spread, extremes and departures from ERA5",
        ),
        "moisture_context": (
            "context",
            "Temperature/dewpoint/RH/VPD together, without counting redundant evidence as independent",
        ),
        "event_exposure": (
            "context",
            "Holiday, day-off, DST, heat/cold wave and regional severe-weather exposure",
        ),
        "history_load_shape": ("feature", "Historical hourly load, peak-to-mean and adjacent ramps"),
    },
    "sunny": {
        "radiation_budget": (
            "feature",
            "GHI/DHI/horizontal direct irradiation and diffuse fraction; never add DNI to GHI",
        ),
        "spatial_radiation": (
            "context",
            "Station direct irradiance center/spread and grid discrepancy, not a sensor-error verdict",
        ),
        "snow_dust_exposure": (
            "context",
            "Snow, dust, precipitation and icing proxies; not measured panel coverage/loss",
        ),
        "thermal_context": (
            "context",
            "Ambient temperature and cooling-wind context, not an invented cell-temperature model",
        ),
        "history_low_light_power": (
            "feature",
            "Observed signed net power under low external irradiance; no zero clipping",
        ),
    },
    "windy": {
        "wind_resource": (
            "feature",
            "Wind-speed and cubic resource proxies in m/s; not a turbine power curve",
        ),
        "vertical_shear": ("feature", "80-to-120m power-law exponent only at valid positive speeds"),
        "direction_coherence": (
            "context",
            "Circular temporal wind direction plus supplied spatial directional coherence",
        ),
        "icing_exposure": (
            "context",
            "Supplied icing score with temperature, humidity and snow; not shutdown probability",
        ),
        "history_wind_power": (
            "feature",
            "Historical wind-power association and negative net-power share; no causal fit",
        ),
    },
}


def _stats(x):
    a = np.asarray(x, float)
    good = a[np.isfinite(a)]
    return {
        "n_valid": int(len(good)),
        "n_total": int(len(a)),
        "mean": float(good.mean()) if len(good) else None,
        "min": float(good.min()) if len(good) else None,
        "max": float(good.max()) if len(good) else None,
    }


def _corr(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    return (
        float(np.corrcoef(a[valid], b[valid])[0, 1])
        if valid.sum() >= 3 and np.std(a[valid]) > 0 and np.std(b[valid]) > 0
        else None
    )


def _handler(domain, name, bundle, args):
    e, variables = bundle["episode"], {v["name"]: v for v in bundle["variables"]}
    if e["domain_id"] != domain:
        raise AgentError("Domain tool does not belong to this Episode", "PERMISSION_DENIED")
    h = e["task"]["lookback_length"]
    all_times = (
        e["history"]["timestamps"]
        + e["forecast_timeline"]["gap_timestamps"]
        + e["forecast_timeline"]["scored_timestamps"]
    )
    scope = DomainScope.model_validate(args["scope"])
    offset = 0 if scope.timeline == "history" else h + e["task"]["forecast_gap"]
    length = h if scope.timeline == "history" else e["task"]["horizon_length"]
    end = length if scope.end is None else scope.end
    if not 0 <= scope.start < end <= length:
        raise AgentError("Domain scope is empty or outside the Episode")
    a, z = offset + scope.start, offset + end
    dt = pd.tseries.frequencies.to_offset(e["task"]["frequency"]).nanos / 3.6e12
    used, units, missing = [], {}, []
    warnings = ["Descriptive evidence only; no fixed correction magnitude or causal effect is implied."]
    history_tool = name.startswith("history_")
    if history_tool and scope.timeline != "history":
        return {
            "status": "not_applicable",
            "structured_result": {},
            "warnings": ["This tool only reads the historical target. Set scope.timeline=history."],
            "observed_scope": {**scope.model_dump(), "variables": []},
        }

    def read(col, unit=None):
        if col not in variables:
            raise AgentError(f"Domain input missing: {col}", "PERMISSION_DENIED")
        v = variables[col]
        if unit is not None and v["unit"] != unit:
            raise AgentError(f"Expected unit {unit} for {col}; found {v['unit']}")
        used.append(col)
        units[col] = v["unit"]
        values = v["values"][a:z]
        if any(x is None for x in values):
            missing.append(col)
        return np.asarray([np.nan if x is None else x for x in values], float)

    def exposure(cols):
        return {
            col: {
                "positive_hours": float(np.sum((v := read(col)) > 0) * dt),
                "mean": _stats(v)["mean"],
                "maximum": _stats(v)["max"],
            }
            for col in cols
        }

    def historical():
        used.append(e["task"]["target_name"])
        units[e["task"]["target_name"]] = e["task"]["target_unit"]
        y = np.asarray(e["history"]["target_values"][a:z], float)
        if not np.isfinite(y).all():
            missing.append(e["task"]["target_name"])
        return y

    result = {}
    if name in {"history_demand_shape", "history_load_shape"}:
        y = historical()
        local = pd.to_datetime(all_times[a:z], utc=True).tz_convert(e["task"]["timezone"])
        mean = _stats(y)["mean"]
        result = {
            "target": _stats(y),
            "mean_absolute_adjacent_ramp": _stats(np.abs(np.diff(y)))["mean"],
            "peak_to_mean": float(np.nanmax(y) / mean) if mean and np.isfinite(y).any() else None,
            "hourly_observations": [
                {"hour": int(t.hour), "value": float(v) if np.isfinite(v) else None} for t, v in zip(local, y)
            ],
        }
    elif name == "weather_units":
        result = {
            "temperature_c": _stats(47 * read("temp", "normalized") - 8),
            "apparent_temperature_c": _stats(66 * read("atemp", "normalized") - 16),
            "relative_humidity_pct": _stats(100 * read("hum", "fraction")),
            "wind_kmh": _stats(67 * read("windspeed", "normalized")),
            "formulas": [
                "temp_C=47*temp-8",
                "atemp_C=66*atemp-16",
                "RH_pct=100*hum",
                "wind_kmh=67*windspeed",
            ],
        }
    elif name == "weather_exposure":
        result = exposure(["rain_flag", "snow_flag", "storm_flag"])
        rain, snow, temp = (
            read("precip_1h_mm", "mm"),
            read("snow_depth_mm", "mm"),
            47 * read("temp", "normalized") - 8,
        )
        result.update(
            precipitation_mm=float(np.sum(rain)),
            snow_depth_mm=_stats(snow),
            cold_wet_hours=float(np.sum((temp <= 0) & ((rain > 0) | (snow > 0))) * dt),
        )
        warnings.append(
            "Hourly rainfall includes daily-total allocation. A positive amount alone does not prove observed hourly rain."
        )
    elif name == "observation_quality":
        result = exposure(["precip_observed_flag", "precip_allocated_flag", "sky_obscured_flag"])
        result["cloud_valid_hours"] = _stats(read("cloud_valid_hours", "hour"))
        warnings.append(
            "Daily counts repeated on each row must not be summed. Obscured sky is not 9/8 cloud cover."
        )
    elif name == "calendar_transition":
        dayoff = read("workingday")
        weekend = np.isin(read("weekday"), [0, 6])
        result = {
            "working_hours": float(np.sum(dayoff == 1) * dt),
            "weekend_hours": float(weekend.sum() * dt),
            "weekday_codes": sorted(set(read("weekday").astype(int).tolist())),
        }
    elif name == "degree_hours":
        x = read("temperature_2m_f", "degF")
        base = args["balance_temperature_f"]
        result = {
            "reference_temperature_f": base,
            "heating_degree_hours_f": float(np.maximum(base - x, 0).sum() * dt),
            "cooling_degree_hours_f": float(np.maximum(x - base, 0).sum() * dt),
            "temperature_f": _stats(x),
            "unit": "degF*hour",
        }
        warnings.append(
            "65F is a conventional reference, not the empirically identified balance temperature of Zone 6."
        )
    elif name == "station_dispersion":
        x = np.vstack([read(f"station_{i}", "degF") for i in range(1, 12)])
        mean = np.mean(x, axis=0)
        result = {
            "regional_mean_f": _stats(mean),
            "station_std_f": _stats(np.std(x, axis=0)),
            "max_minus_min_f": _stats(np.max(x, axis=0) - np.min(x, axis=0)),
            "regional_minus_era5_f": _stats(mean - read("temperature_2m_f", "degF")),
        }
    elif name == "moisture_context":
        temp, dew = read("temperature_2m_f", "degF"), read("dew_point_2m_f", "degF")
        result = {
            "dewpoint_depression_f": _stats(temp - dew),
            "humidity_pct": _stats(read("relative_humidity_2m_pct", "percent")),
            "vpd_kpa": _stats(read("vapour_pressure_deficit_kpa", "kPa")),
        }
        warnings.append(
            "Humidity, dewpoint and VPD are coupled descriptions, not three independent causal factors."
        )
    elif name == "event_exposure":
        result = exposure(
            [
                "is_day_off",
                "dst_transition_window",
                "heatwave_severity_f",
                "coldwave_severity_f",
                "noaa_severe_event_any_va",
            ]
        )
        col = "federal_holiday"
        used.append(col)
        vals = variables[col]["values"][a:z]
        result["holiday_labels"] = sorted({v for v in vals if v})
        if variables[col]["visibility"] == "history_only" and scope.timeline == "scored":
            missing.append(col)
        warnings.append(
            "NOAA is a statewide proxy, not confirmed local event exposure. Wave indicators already encode temperature."
        )
    elif name == "radiation_budget":
        ghi, dhi, direct = (
            read(c, "W/m2")
            for c in ["shortwave_radiation_fused", "diffuse_radiation_fused", "direct_radiation_fused"]
        )
        lit = ghi > 1
        result = {
            "ghi_irradiation_wh_m2": float(ghi.sum() * dt),
            "dhi_irradiation_wh_m2": float(dhi.sum() * dt),
            "direct_horizontal_irradiation_wh_m2": float(direct.sum() * dt),
            "diffuse_fraction_at_ghi_gt1": _stats(dhi[lit] / ghi[lit]),
            "ghi_minus_direct_minus_diffuse_wm2": _stats(ghi - direct - dhi),
            "low_light_hours_ghi_le1": float((ghi <= 1).sum() * dt),
        }
        warnings.append(
            "The low-light threshold is a numerical diagnostic, not astronomical sunset. Hourly held means are integrated with the 15-minute step."
        )
    elif name == "spatial_radiation":
        center, spread, grid = (
            read(c, "W/m2")
            for c in [
                "station_direct_radiation_fused",
                "station_direct_radiation_spread",
                "direct_radiation_fused",
            ]
        )
        result = {
            "station_center_wm2": _stats(center),
            "spatial_std_wm2": _stats(spread),
            "station_minus_grid_wm2": _stats(center - grid),
        }
        warnings.append(
            "Grid-to-site discrepancy can reflect spatial averaging; it does not by itself identify faulty weather."
        )
    elif name == "snow_dust_exposure":
        result = exposure(["high_dust_proxy", "persistent_snow_proxy", "station_precipitation_any"])
        result.update(
            snowfall_rolling24h_cm=_stats(read("snowfall_24h_sum", "cm")),
            icing_score=_stats(read("icing_risk_score", "fraction")),
        )
        warnings.append(
            "Never sum overlapping 24-hour snow totals; snow depth proxy does not measure snow on modules. Dust air concentration is not deposited dirt."
        )
    elif name == "thermal_context":
        result = {
            "ambient_c": _stats(read("temperature_2m_fused", "degC")),
            "wind_10m_mps": _stats(read("wind_speed_10m_fused", "km/h") / 3.6),
            "station_ambient_c": _stats(read("station_temperature_2m_fused", "degC")),
        }
        warnings.append(
            "Module temperature also depends on irradiance and mounting; ambient temperature alone cannot supply a calibrated power derating."
        )
    elif name == "history_low_light_power":
        y, light = historical(), read("shortwave_radiation_fused", "W/m2")
        low = light <= 1
        result = {
            "low_light_net_power": _stats(y[low]),
            "negative_net_power_fraction": float(np.mean(y < 0)),
            "low_light_count": int(low.sum()),
            "all_history": _stats(y),
        }
        warnings.append(
            "Negative readings are retained; auxiliary consumption is possible, not a verified explanation of every point."
        )
    elif name == "wind_resource":
        v = read("hf_wind_speed_80m_fused", "km/h") / 3.6
        gust = read("wind_gusts_10m_fused", "km/h") / 3.6
        result = {
            "wind80_mps": _stats(v),
            "gust10_mps": _stats(gust),
            "mean_v_cubed_m3_s3": float(np.mean(v**3)),
            "energy_equivalent_speed_mps": float(np.cbrt(np.mean(v**3))),
        }
        warnings.append(
            "Cubic resource is not electric power. Rated saturation, cut-out and availability need turbine-specific metadata."
        )
    elif name == "vertical_shear":
        lo, hi = read("hf_wind_speed_80m_fused", "km/h"), read("hf_wind_speed_120m_fused", "km/h")
        valid = (lo > 0) & (hi > 0)
        result = {
            "power_law_alpha": _stats(np.log(hi[valid] / lo[valid]) / np.log(120 / 80)),
            "excluded_nonpositive_or_missing_points": int((~valid).sum()),
            "formula": "log(v120/v80)/log(120/80)",
        }
        warnings.append(
            "An observed shear exponent is not a universal site parameter; no extrapolation beyond these heights is performed."
        )
    elif name == "direction_coherence":
        angles = read("station_wind_direction_80m_fused", "degree")
        rad = np.deg2rad(angles)
        s, c = float(np.sin(rad).mean()), float(np.cos(rad).mean())
        r = float(np.hypot(s, c))
        result = {
            "temporal_circular_mean_deg": float(np.degrees(np.arctan2(s, c)) % 360) if r > 1e-8 else None,
            "temporal_resultant": r,
            "spatial_resultant": _stats(read("station_wind_direction_80m_resultant", "fraction")),
        }
        warnings.append(
            "Temporal concentration and supplied cross-station coherence measure different things. Opposite directions can have undefined mean direction."
        )
    elif name == "icing_exposure":
        score = read("icing_risk_score", "fraction")
        result = {
            "icing_score": _stats(score),
            "score_histogram": {str(q): int(np.sum(score == q)) for q in [0, 0.25, 0.5, 0.75, 1]},
            "temperature80_c": _stats(read("hf_temperature_80m_fused", "degC")),
            "humidity_pct": _stats(read("relative_humidity_2m_fused", "percent")),
            "snow_rolling24h_cm": _stats(read("snowfall_24h_sum", "cm")),
        }
        warnings.append(
            "Score>0 means at least one condition holds, not observed icing. The score is not a probability or confirmed turbine stop."
        )
    elif name == "history_wind_power":
        y, v = historical(), read("hf_wind_speed_80m_fused", "km/h") / 3.6
        result = {
            "historical_pearson_v_power": _corr(v, y),
            "historical_pearson_v3_power": _corr(v**3, y),
            "negative_net_power_fraction": float(np.mean(y < 0)),
            "power": _stats(y),
        }
        warnings.append(
            "One-day association is weak evidence, not a fitted power curve or permission to reverse the physical mechanism."
        )
    else:
        raise AgentError("Unknown domain operation")
    if missing:
        result = {"unavailable_inputs": sorted(set(missing))}
        warnings.append("No derived value is returned when a required input is hidden or missing.")
    return {
        "status": "not_applicable" if missing else "ok",
        "structured_result": result,
        "warnings": warnings,
        "observed_scope": {**scope.model_dump(), "variables": sorted(set(used))},
        "units": units,
        "coverage": {"requested_points": z - a, "missing_inputs": sorted(set(missing))},
    }


def domain_registry(domain: str, registry: Registry | None = None):
    if domain not in TOOLS:
        raise ValueError(f"Unregistered domain: {domain}")
    registry = Registry() if registry is None else registry
    code_hash = "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name, (category, description) in TOOLS[domain].items():
        registry.register_extension(
            f"domain.{domain}.{name}",
            DegreeArgs
            if name == "degree_hours"
            else HistoryArgs
            if name.startswith("history_")
            else DomainArgs,
            partial(_handler, domain, name),
            description,
            code_hash,
            category=category,
        )
        registry.manifests[f"domain.{domain}.{name}"].update(
            input_variables=INPUTS[name], reads_target_history=name.startswith("history_")
        )
    return registry
