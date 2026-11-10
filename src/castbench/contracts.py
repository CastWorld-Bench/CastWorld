from __future__ import annotations

from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
Number = Annotated[FiniteFloat, Field(strict=True)]


class Task(Contract):
    target_name: str = Field(min_length=1)
    target_unit: str = Field(min_length=1)
    frequency: str
    timezone: str
    lookback_length: PositiveInt
    forecast_gap: NonnegativeInt
    horizon_length: PositiveInt


class History(Contract):
    timestamps: list[str]
    target_values: list[Number | None]
    quality_mask_ref: str


class Timeline(Contract):
    gap_timestamps: list[str]
    scored_timestamps: list[str]


class Resources(Contract):
    variable_catalog_id: str
    covariate_resource_id: str
    event_resource_id: str
    knowledge_pack_id: str
    sample_index_id: str
    revision_scale_profile_id: str
    tool_catalog_snapshot_id: str


def time_axis(strings: list[str]) -> pd.DatetimeIndex:
    if any(pd.Timestamp(s).tzinfo is None for s in strings):
        raise ValueError("Timestamps require explicit UTC offsets")
    return pd.DatetimeIndex(pd.to_datetime(strings, utc=True))


def regular_axis(axis: pd.DatetimeIndex, frequency: str, timezone: str = "UTC") -> bool:
    offset = pd.tseries.frequencies.to_offset(frequency)
    if offset.n <= 0 or axis.has_duplicates or not axis.is_monotonic_increasing:
        return False
    try:
        return bool(np.all(np.diff(axis.as_unit("ns").asi8) == offset.nanos))
    except ValueError:
        # Month starts and anchored weeks preserve calendar positions, not a
        # fabricated fixed number of seconds per month.
        local = axis.tz_convert(timezone)
        expected = pd.date_range(local[0], periods=len(local), freq=offset)
        return bool(local.equals(expected))


class Episode(Contract):
    schema_version: Literal["castbench.episode/1.0"] = "castbench.episode/1.0"
    episode_id: str
    dataset_id: str
    domain_id: str
    forecast_origin: str
    task: Task
    history: History
    forecast_timeline: Timeline
    resources: Resources
    tool_profile_id: str
    domain_pack_id: str
    budget_profile_id: str

    @model_validator(mode="after")
    def aligned(self):
        t, h, f = self.task, self.history, self.forecast_timeline
        ZoneInfo(t.timezone)
        if len(h.target_values) != t.lookback_length or len(h.timestamps) != t.lookback_length:
            raise ValueError("History length mismatch")
        if len(f.gap_timestamps) != t.forecast_gap or len(f.scored_timestamps) != t.horizon_length:
            raise ValueError("Future length mismatch")
        axis = time_axis(h.timestamps + f.gap_timestamps + f.scored_timestamps)
        if not regular_axis(axis, t.frequency, t.timezone):
            raise ValueError("Episode must have a complete regular ordered axis")
        if time_axis([self.forecast_origin])[0] != axis[t.lookback_length - 1]:
            raise ValueError("forecast_origin must be the last observed timestamp")
        if not any(v is not None for v in h.target_values):
            raise ValueError("History has no observed target")
        return self


class Scope(Contract):
    timeline: Literal["history", "gap", "scored", "all"] = "all"
    start: NonnegativeInt = 0
    end: PositiveInt | None = None
    variables: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def ordered(self):
        if self.end is not None and self.end < self.start:
            raise ValueError("Scope end precedes start")
        return self


class IndexRange(Contract):
    timeline: Literal["gap", "scored"] = "scored"
    start: NonnegativeInt = 0
    end: PositiveInt | None = None

    @model_validator(mode="after")
    def increasing(self):
        if self.end is not None and self.end <= self.start:
            raise ValueError("Ranges are half-open and must be nonempty")
        return self


class RevisionScale(Contract):
    profile_id: str
    version: str = "1.0.0"
    target_unit: str
    S: Number = Field(gt=0)
    absolute_tolerance: Number = Field(ge=0)
    relative_tolerance: Number = Field(default=1e-6, ge=0)
    magnitude_edges: tuple[Number, Number] = (0.25, 1.0)
    direction_fraction: Number = Field(default=0.9, gt=0.5, le=1)
    source_train_hash: str

    @model_validator(mode="after")
    def edges(self):
        if not 0 < self.magnitude_edges[0] < self.magnitude_edges[1]:
            raise ValueError("Invalid magnitude edges")
        return self


class DeclaredEffect(Contract):
    baseline_id: str
    ranges: list[IndexRange]
    direction: Literal["increase", "decrease", "mixed", "unchanged"]
    magnitude: Literal["small", "moderate", "large", "none"]


class Segment(Contract):
    range: IndexRange
    offset: Number | list[Number] | None = None
    factor: Number | list[Number] | None = None

    @model_validator(mode="after")
    def one_operation(self):
        if (self.offset is None) == (self.factor is None):
            raise ValueError("Segment requires exactly one offset or factor")
        return self


class ForecastArgs(Contract):
    operation: Literal[
        "adopt_candidate",
        "submit_full_forecast",
        "constant_forecast",
        "add_offset",
        "scale_values",
        "adjust_segments",
        "blend_forecasts",
        "shift_time",
        "smooth_revision_delta",
        "replace_segment",
    ]
    candidate_id: str | None = None
    parent_id: str | None = None
    proposed_id: str | None = None
    base_id: str | None = None
    source_id: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    weights: list[Number] = Field(default_factory=list)
    scored_values: list[Number] | None = None
    scored_timestamps: list[str] | None = None
    gap_values: list[Number] | None = None
    gap_timestamps: list[str] | None = None
    unit: str | None = None
    constant_value: Number | None = None
    gap_constant_value: Number | None = None
    range: IndexRange = Field(default_factory=IndexRange)
    offset: Number | list[Number] | None = None
    factor: Number | list[Number] | None = None
    segments: list[Segment] = Field(default_factory=list)
    steps: int | None = Field(default=None, strict=True)
    fill_policy: Literal["keep_parent", "edge", "explicit_values"] | None = None
    fill_values: list[Number] | None = None
    window: PositiveInt = 3
    method: Literal["moving_average", "median"] = "moving_average"
    values: list[Number] | None = None
    declared_effect: DeclaredEffect | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    rationale_origin: Literal["self_generated", "evidence_based", "unspecified"] = "unspecified"
    rationale_scope: list[IndexRange] = Field(default_factory=list)
    correction_reason_code: (
        Literal["arithmetic_fix", "alignment_fix", "constraint_response", "reconsideration"] | None
    ) = None

    @model_validator(mode="after")
    def operation_parameters(self):
        common = {
            "operation",
            "declared_effect",
            "evidence_refs",
            "rationale_origin",
            "rationale_scope",
            "correction_reason_code",
        }
        allowed = {
            "constant_forecast": {"constant_value", "gap_constant_value", "unit"},
            "submit_full_forecast": {
                "scored_values",
                "scored_timestamps",
                "gap_values",
                "gap_timestamps",
                "unit",
            },
            "adopt_candidate": {"candidate_id"},
            "add_offset": {"parent_id", "range", "offset"},
            "scale_values": {"parent_id", "range", "factor"},
            "adjust_segments": {"parent_id", "range", "segments"},
            "blend_forecasts": {"source_ids", "weights", "base_id", "range"},
            "shift_time": {"parent_id", "range", "steps", "fill_policy", "fill_values"},
            "smooth_revision_delta": {"parent_id", "proposed_id", "range", "window", "method"},
            "replace_segment": {"parent_id", "source_id", "range", "values"},
        }[self.operation] | common
        for name in self.model_fields_set - allowed:
            default = type(self).model_fields[name].get_default(call_default_factory=True)
            if getattr(self, name) != default:
                raise ValueError(f"Parameter {name} does not apply to {self.operation}")
        if (
            self.operation == "shift_time"
            and self.fill_values is not None
            and self.fill_policy != "explicit_values"
        ):
            raise ValueError("fill_values requires explicit_values fill_policy")
        return self


class ToolCall(Contract):
    call_id: str = Field(min_length=1, max_length=128)
    tool_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    need_ids: list[str] = Field(default_factory=list)
    purpose: str | None = Field(default=None, min_length=1, max_length=2000)


class ToolResult(Contract):
    status: Literal["ok", "partial", "not_applicable", "tool_error"]
    structured_result: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    coverage: dict[str, Any] = Field(default_factory=dict)
    observed_scope: dict[str, Any]
    units: dict[str, str]
    factual_summary: str
    provenance: dict[str, Any]


class NeedUpdate(Contract):
    need_id: str
    status: Literal["resolved", "abandoned"]
    resolution_note: str = Field(min_length=1)


class Action(Contract):
    action_type: Literal[
        "DECLARE_NEED",
        "CALL_TOOL",
        "REGISTER_EVIDENCE",
        "CREATE_FORECAST",
        "CHECK_FORECAST",
        "REQUEST_STATE",
        "FINALIZE",
    ]
    state_version: NonnegativeInt
    phase: Literal["GAP", "ACQUIRE", "GROUND", "REVISE", "VERIFY", "STOP"] | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    need_updates: list[NeedUpdate] = Field(default_factory=list)


class DeclareNeedArgs(Contract):
    description: str = Field(min_length=1)
    kind: Literal["feature", "context", "model", "knowledge", "sample", "registry", "domain"]
    scope: Scope = Field(default_factory=Scope)


class EvidenceArgs(Contract):
    source_observation_ids: list[str] = Field(min_length=1)
    scope: Scope
    application_scope: list[IndexRange]
    relation_type: Literal["direct_context", "historical_analogy", "general_knowledge"]
    statement: str = ""


class CheckArgs(Contract):
    forecast_id: str
    checks: list[str] | None = None


class StateRequestArgs(Contract):
    section: str
    ref: str | None = None
    offset: NonnegativeInt = 0
    limit: PositiveInt = 128
    path: list[str | NonnegativeInt] = Field(default_factory=list)


class FinalizeArgs(Contract):
    forecast_id: str
    stop_reason: str = Field(min_length=1)


class BatchArgs(Contract):
    calls: list[ToolCall] = Field(min_length=1)


class Budget(Contract):
    max_rounds: PositiveInt = 30
    max_calls_per_round: PositiveInt = 4
    recent_rounds: PositiveInt = Field(
        default=4,
        description="Legacy pack compatibility; capacity-driven context policy supersedes this cutoff",
    )
    max_inline_bytes: PositiveInt = 8192
    max_state_page: PositiveInt = 256
    state_summary_limit: PositiveInt = 12
    summary_text_characters: PositiveInt = 256


class Constraint(Contract):
    constraint_id: str
    kind: Literal["minimum", "maximum", "max_abs_step"]
    value: Number
    hard: bool = False

    @model_validator(mode="after")
    def valid_step_bound(self):
        if self.kind == "max_abs_step" and self.value < 0:
            raise ValueError("Absolute step constraint cannot be negative")
        return self


class HistoricalTargetLag(Contract):
    kind: Literal["audited_target_lag"] = "audited_target_lag"
    lag_steps: int = Field(strict=True, ge=1)
    audit_reference: str = Field(min_length=1)


class Variable(Contract):
    name: str
    dtype: Literal["numeric", "categorical", "boolean"]
    unit: str
    description: str
    target_derived: bool = Field(strict=True)
    historical_target_lag: HistoricalTargetLag | None = None
    visibility: Literal["benchmark_known_future", "history_only"]
    lineage: str = Field(min_length=1)
    values: list[Any]

    @model_validator(mode="after")
    def typed(self):
        if self.target_derived != (self.historical_target_lag is not None):
            raise ValueError("Target-derived covariates require an explicit audited historical lag")
        if self.historical_target_lag and self.dtype != "numeric":
            raise ValueError("An audited target lag must be numeric")
        for value in self.values:
            if value is None:
                continue
            if self.dtype == "numeric" and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
            ):
                raise ValueError("Numeric covariate has non-finite/non-numeric value")
            if self.dtype == "boolean" and not isinstance(value, bool):
                raise ValueError("Boolean covariate must use JSON booleans")
            if self.dtype == "categorical" and not isinstance(value, str):
                raise ValueError("Categorical covariate must use strings")
        return self


class KnowledgeEntry(Contract):
    entry_id: str
    topic: str
    text: str
    source: str
    version: str
    scope: Literal["general", "domain"]
    variables: list[str] = Field(default_factory=list)
    conflict_group: str | None = None
    language: str = "unspecified"
    search_aliases: list[str] = Field(default_factory=list)


class TrainingCase(Contract):
    sample_id: str
    partition: Literal["train"]
    history_timestamps: list[str]
    history_target: list[Number]
    future_timestamps: list[str]
    future_target: list[Number]
    context: dict[str, list[Any]] = Field(default_factory=dict)
    event_group: str | None = None


class Event(Contract):
    event_id: str
    type: str
    start: str
    end: str
    description: str
    source: str
    target_derived: Literal[False]
    depends_on: list[str] = Field(default_factory=list)
    intensity: Number | None = None

    @model_validator(mode="after")
    def ordered(self):
        axis = time_axis([self.start, self.end])
        if axis[0] > axis[1]:
            raise ValueError("Event end precedes start")
        return self


class DomainMetric(Contract):
    metric_id: Literal["peak_abs", "ramp_abs", "window_total_abs", "asymmetric_abs"]
    weight: Annotated[FiniteFloat, Field(strict=True, gt=0, le=0.1)]
    description: str = Field(min_length=1)
    overprediction_cost: Annotated[FiniteFloat, Field(strict=True, gt=0)] = 1.0
    underprediction_cost: Annotated[FiniteFloat, Field(strict=True, gt=0)] = 1.0
    cost_basis: str | None = None

    @model_validator(mode="after")
    def cost_semantics(self):
        if self.metric_id == "asymmetric_abs":
            if not self.cost_basis or not self.cost_basis.strip():
                raise ValueError("Asymmetric loss requires a disclosed business cost basis")
        elif self.overprediction_cost != 1.0 or self.underprediction_cost != 1.0 or self.cost_basis:
            raise ValueError("Cost parameters only apply to asymmetric_abs")
        return self


class ForecastEvaluation(Contract):
    schema_version: Literal["castbench.forecast_metrics/1.0"] = "castbench.forecast_metrics/1.0"
    profile_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    target_unit: str = Field(min_length=1)
    source_train_hash: str = Field(min_length=1)
    components: list[DomainMetric] = Field(min_length=1)

    @model_validator(mode="after")
    def bounded_weights(self):
        ids = [component.metric_id for component in self.components]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate domain metric")
        if sum(component.weight for component in self.components) > 0.1 + 1e-12:
            raise ValueError("Domain metric weights must sum to at most 0.10")
        return self


class PublicBundle(Contract):
    """One Episode capability view. No private truth fields are accepted."""

    episode: Episode
    budget: Budget = Field(default_factory=Budget)
    revision_scale: RevisionScale
    task_description: str
    forecast_evaluation: ForecastEvaluation | None = None
    variables: list[Variable] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    knowledge: list[KnowledgeEntry] = Field(default_factory=list)
    samples: list[TrainingCase] = Field(default_factory=list)
    sample_library_cutoff: str
    constraints: list[Constraint] = Field(default_factory=list)
    default_check_profile: list[str] = Field(
        default_factory=lambda: ["timeline", "finite", "shape", "domain"]
    )

    @model_validator(mode="after")
    def isolated(self):
        e = self.episode
        if self.revision_scale.target_unit != e.task.target_unit:
            raise ValueError("Scale and target units differ")
        if self.revision_scale.profile_id != e.resources.revision_scale_profile_id:
            raise ValueError("Scale profile ID differs")
        if self.forecast_evaluation is not None:
            profile = self.forecast_evaluation
            if (profile.dataset_id != e.dataset_id or profile.target_unit != e.task.target_unit
                    or profile.source_train_hash != self.revision_scale.source_train_hash):
                raise ValueError("Forecast evaluation profile differs from the public task/scale")
            if any(c.metric_id == "ramp_abs" for c in profile.components):
                if e.task.forecast_gap or e.history.target_values[-1] is None:
                    raise ValueError("Ramp metric requires gap=0 and a finite last historical target")
        if len({v.name for v in self.variables}) != len(self.variables):
            raise ValueError("Duplicate variables")
        n = e.task.lookback_length + e.task.forecast_gap + e.task.horizon_length
        for v in self.variables:
            if v.name == e.task.target_name or len(v.values) != n:
                raise ValueError("Covariates require complete public axis and cannot shadow target")
            if v.visibility == "history_only" and any(
                x is not None for x in v.values[e.task.lookback_length :]
            ):
                raise ValueError("History-only covariates must mask gap and future values at source")
            if v.historical_target_lag:
                if v.unit != e.task.target_unit:
                    raise ValueError("Target-lag covariate must keep the original target unit")
                lag = v.historical_target_lag.lag_steps
                for index, value in enumerate(v.values):
                    if value is None:
                        continue
                    dependency = index - lag
                    if dependency >= e.task.lookback_length:
                        raise ValueError("Target-lag covariate depends on an unobserved future target")
                    if dependency >= 0:
                        expected = e.history.target_values[dependency]
                        if expected is None or not np.isclose(value, expected, rtol=1e-10, atol=1e-12):
                            raise ValueError("Target-lag value disagrees with the observed history")
        hidden = {v.name for v in self.variables if v.visibility == "history_only"}
        for event in self.events:
            if hidden and time_axis([event.end])[0] > time_axis([e.forecast_origin])[0]:
                if not event.depends_on or hidden.intersection(event.depends_on):
                    raise ValueError(
                        "Future events must declare visible dependencies when covariates are hidden"
                    )
        cutoff = time_axis([self.sample_library_cutoff])[0]
        if cutoff > time_axis([e.forecast_origin])[0]:
            raise ValueError("Training cutoff cannot follow the forecast origin")
        if len({s.sample_id for s in self.samples}) != len(self.samples):
            raise ValueError("Duplicate training case IDs")
        for s in self.samples:
            if (
                len(s.history_target) != len(s.history_timestamps)
                or not s.history_target
                or len(s.future_target) != len(s.future_timestamps)
                or not s.future_target
            ):
                raise ValueError("Invalid training case lengths")
            axis = time_axis(s.history_timestamps + s.future_timestamps)
            if not axis.is_monotonic_increasing or axis.has_duplicates or axis[-1] > cutoff:
                raise ValueError("Training case crosses frozen cutoff or has invalid axis")
        if len({k.entry_id for k in self.knowledge}) != len(self.knowledge):
            raise ValueError("Duplicate knowledge IDs")
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ValueError("Duplicate event IDs")
        if not set(self.default_check_profile) <= {"timeline", "finite", "shape", "domain"}:
            raise ValueError("Unknown check profile")
        if not {"timeline", "finite"} <= set(self.default_check_profile):
            raise ValueError("Default checks must include timeline and finite values")
        if self.constraints and "domain" not in self.default_check_profile:
            raise ValueError("Domain checks are required when constraints exist")
        if len({c.constraint_id for c in self.constraints}) != len(self.constraints):
            raise ValueError("Duplicate constraint IDs")
        lower = max(
            (c.value for c in self.constraints if c.hard and c.kind == "minimum"), default=-float("inf")
        )
        upper = min(
            (c.value for c in self.constraints if c.hard and c.kind == "maximum"), default=float("inf")
        )
        if lower > upper:
            raise ValueError("Hard minimum and maximum constraints are contradictory")
        variables = {v.name: v for v in self.variables}
        for s in self.samples:
            axis = time_axis(s.history_timestamps + s.future_timestamps)
            if not regular_axis(axis, e.task.frequency, e.task.timezone):
                raise ValueError("Training cases must preserve the dataset frequency")
            total = len(s.history_target) + len(s.future_target)
            for key, values in s.context.items():
                if key not in variables or len(values) != total:
                    raise ValueError("Training context must align and use declared variables")
                Variable.model_validate({**variables[key].model_dump(exclude={"values"}), "values": values})
        return self
