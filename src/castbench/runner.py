from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import date
from itertools import permutations
from uuid import uuid4

from pydantic import ValidationError

from .capacity import POLICY as CAPACITY_POLICY
from .capacity import ContextCapacityExceeded, agent_rounds, reduce_packet
from .capacity import feedback as capacity_feedback
from .capacity import limits as capacity_limits
from .catalog import Registry
from .common import (
    AgentError,
    InfrastructureError,
    TransientInfrastructureError,
    chronological_items,
    digest,
    finite_array,
    json_text,
    runtime_versions,
)
from .completion_rescue import POLICY as COMPLETION_RESCUE_POLICY
from .completion_rescue import action_schema as completion_rescue_schema
from .completion_rescue import close_rescue, open_rescue
from .completion_rescue import packet as completion_rescue_packet
from .context_memory import build_messages
from .contracts import (
    Action,
    BatchArgs,
    CheckArgs,
    DeclareNeedArgs,
    EvidenceArgs,
    FinalizeArgs,
    ForecastArgs,
    IndexRange,
    PublicBundle,
    Scope,
    StateRequestArgs,
    ToolCall,
    ToolResult,
)
from .forecast import check_values, create_forecast, diff_report, validate_ranges
from .protocol_repair import POLICY, check_retry, close_alarm, open_alarm
from .providers import ACTION_GUIDE, SYSTEM_PROMPT, decode, protocol_examples
from .tools import ToolView, execute_readonly

ACTION_PAYLOAD_SCHEMAS = {
    "DECLARE_NEED": DeclareNeedArgs,
    "REGISTER_EVIDENCE": EvidenceArgs,
    "CREATE_FORECAST": ForecastArgs,
    "CHECK_FORECAST": CheckArgs,
    "REQUEST_STATE": StateRequestArgs,
    "FINALIZE": FinalizeArgs,
    "CALL_TOOL": BatchArgs,
}


def initial_state(episode_id):
    return {
        "episode_id": episode_id,
        "state_version": 0,
        "rounds_used": 0,
        "phase": "GAP",
        "phase_history": [],
        "cycle": 0,
        "pending_cycle": False,
        "status": "running",
        "needs": {},
        "observations": {},
        "evidence": {},
        "forecasts": {},
        "active_forecast_id": None,
        "validations": [],
        "errors": [],
        "protocol_alarms": [],
        "repair_pending": None,
        "completion_rescue": None,
        "final": None,
    }


def public_interaction_budget(bundle, state):
    maximum = bundle.budget.max_rounds
    used = agent_rounds(state)
    remaining = max(0, maximum - used)
    closing_threshold = max(2, math.ceil(0.10 * maximum))
    if remaining == 0:
        stage = "exhausted"
    elif remaining == 1:
        stage = "final_round"
    elif remaining <= closing_threshold:
        stage = "closing"
    else:
        stage = "normal"
    return {
        "max_rounds": maximum,
        "agent_rounds_used": used,
        "rounds_remaining": remaining,
        "max_calls_per_round": bundle.budget.max_calls_per_round,
        "budget_stage": stage,
        "notice": (
            "The round budget is a ceiling, not a target. Tool count and round count are not "
            "rewarded monotonically; stop when the forecast is adequately supported."
        ),
    }


def _keys(payload, required, optional=()):
    if (
        not isinstance(payload, dict)
        or set(required) - set(payload)
        or set(payload) - set(required) - set(optional)
    ):
        raise AgentError(f"Expected payload keys {required}; optional {optional}")


def _compatible(need_scope, call_scope, bundle):
    n, c = Scope.model_validate(need_scope), Scope.model_validate(call_scope)
    if n.variables and not set(c.variables) <= set(n.variables):
        return False
    view = ToolView(bundle, {}, None, None)
    a, b = view.indices(n)
    x, y = view.indices(c)
    if x < a or y > b:
        return False
    return True


def error_fingerprint(action, raw):
    if action is None:
        return digest(raw)
    value = action.model_dump(exclude={"state_version", "phase"})
    calls = value.get("payload", {}).get("calls", [])
    if isinstance(calls, list):
        for call in calls:
            if isinstance(call, dict):
                call.pop("call_id", None)
    return digest(value)


def observed_stage(action):
    """Label the executed semantics, without making the label a routing constraint."""
    if action.action_type == "CALL_TOOL":
        calls = action.payload.get("calls")
        tools = (
            [c.get("tool_id", "") for c in calls if isinstance(c, dict)] if isinstance(calls, list) else []
        )
        if tools and all(isinstance(t, str) and t.startswith("registry.") for t in tools):
            return "GAP"
        if len(tools) == 1 and isinstance(tools[0], str) and tools[0].startswith("sandbox."):
            return (
                "VERIFY"
                if tools[0] in {"sandbox.validate_forecast", "sandbox.compare_versions"}
                else "REVISE"
            )
        return "ACQUIRE"
    if action.action_type == "REQUEST_STATE":
        section = action.payload.get("section")
        return (
            "GAP"
            if isinstance(section, str) and section in {"summary", "schema", "phase_history"}
            else "ACQUIRE"
        )
    return {
        "DECLARE_NEED": "GAP",
        "REGISTER_EVIDENCE": "GROUND",
        "CREATE_FORECAST": "REVISE",
        "CHECK_FORECAST": "VERIFY",
        "FINALIZE": "STOP",
    }[action.action_type]


class Runner:
    def __init__(
        self,
        bundle: PublicBundle,
        store,
        provider,
        run_id,
        registry=None,
        model_service=None,
        fault_hook=None,
        protocol_repair=False,
        continuation_from=None,
        verified_bundle_hash=None,
        resource_policy=None,
    ):
        if verified_bundle_hash is None:
            # A caller may reuse its mutable models to configure another run.
            # This run owns a validated snapshot, not live references to those inputs.
            bundle = PublicBundle.model_validate(bundle.model_dump())
            bundle_hash = digest(bundle.model_dump())
        else:
            if not (
                isinstance(verified_bundle_hash, str)
                and verified_bundle_hash.startswith("sha256:")
                and len(verified_bundle_hash) == 71
            ):
                raise ValueError("verified_bundle_hash must be a sha256 content identity")
            # Batch domain-pack loading already validated this Episode and owns the
            # object for the duration of this run. Shared libraries are read-only;
            # Agent state remains isolated in Store under this run_id.
            bundle_hash = verified_bundle_hash
        self.bundle, self.store, self.provider, self.run_id = bundle, store, provider, run_id
        self.registry = deepcopy(registry) if registry is not None else Registry()
        # Counterbalance presentation by public calendar date, never by model performance or truth.
        models = sorted(k for k in self.registry.manifests if k.startswith("model."))
        if len(models) == 4:
            ordinal = date.fromisoformat(
                bundle.episode.forecast_timeline.scored_timestamps[0][:10]
            ).toordinal()
            self.registry.model_order = list(permutations(models))[(ordinal % 4) * 6 + (ordinal // 4) % 6]
        self.model_service = model_service
        self.fault_hook = fault_hook or (lambda point: None)
        self.protocol_repair = bool(protocol_repair)
        self._frozen_protocol_repair = self.protocol_repair
        self.resource_policy = None
        if resource_policy is not None:
            from .efficiency import ResourcePolicy

            self.resource_policy = ResourcePolicy.model_validate(resource_policy)
            if (
                self.resource_policy.dataset_id != bundle.episode.dataset_id
                or bundle.episode.episode_id not in self.resource_policy.planned_episode_ids
                or self.resource_policy.source_train_hash != bundle.revision_scale.source_train_hash
            ):
                raise ValueError("Resource policy differs from the public Episode/training source")
        self._resource_policy_hash = digest(
            self.resource_policy.model_dump() if self.resource_policy is not None else None
        )
        config = {
            "runtime": runtime_versions(),
            "bundle_hash": bundle_hash,
            "catalog_hash": self.registry.snapshot,
            "model_presentation_order": list(getattr(self.registry, "model_order", ())),
            "context_policy": "capacity_factual_memory/2.0",
            "capacity_recovery": CAPACITY_POLICY if self.capacity_enabled else {"version": "disabled"},
            "protocol_repair": POLICY if self.protocol_repair else {"version": "disabled"},
            "completion_rescue": (
                COMPLETION_RESCUE_POLICY if self.protocol_repair else {"version": "disabled"}
            ),
            "provider": provider.manifest(),
            "model_profiles": {k: v.model_dump() for k, v in model_service.profiles.items()}
            if model_service
            else {},
            "budget": bundle.budget.model_dump(),
            "revision_scale": bundle.revision_scale.model_dump(),
            "constraints": [x.model_dump() for x in bundle.constraints],
            "default_check_profile": bundle.default_check_profile,
            "dataset_id": bundle.episode.dataset_id,
            "episode_id": bundle.episode.episode_id,
            "episode": bundle.episode.model_dump(),
            "system_prompt_hash": digest(SYSTEM_PROMPT),
            "schema": "castbench.run/1.0",
        }
        if bundle.forecast_evaluation is not None:
            config["forecast_evaluation"] = bundle.forecast_evaluation.model_dump()
        if self.resource_policy is not None:
            from .efficiency import implementation_hash

            config["resource_policy"] = self.resource_policy.model_dump()
            config["resource_implementation_hash"] = implementation_hash()
        self._provider_hash = digest(config["provider"])
        self._model_profiles_hash = digest(config["model_profiles"])
        if continuation_from is not None:
            store.migrate_capacity_config(run_id, continuation_from, config)
        store.start(run_id, config, initial_state(bundle.episode.episode_id))

    @property
    def capacity_enabled(self):
        return bool(getattr(getattr(self.provider, "profile", None), "capacity_recovery", False))

    def _check_frozen_services(self):
        models = (
            {k: v.model_dump() for k, v in self.model_service.profiles.items()} if self.model_service else {}
        )
        if (
            digest(self.provider.manifest()) != self._provider_hash
            or digest(models) != self._model_profiles_hash
            or self.protocol_repair != self._frozen_protocol_repair
            or digest(self.resource_policy.model_dump() if self.resource_policy is not None else None)
            != self._resource_policy_hash
        ):
            raise InfrastructureError(
                "Provider/model configuration changed after run start; restore the frozen "
                "configuration to resume, or create a new run for the changed configuration"
            )

    def state(self):
        return self.store.get(self.run_id)

    def packet(self, state):
        # Array values live in retrievable artifacts, not repeatedly in the state summary.
        visible = deepcopy(state)
        interaction_budget = public_interaction_budget(self.bundle, state)
        visible["forecasts"] = {
            k: {a: b for a, b in v.items() if a not in {"scored_values", "gap_values"}}
            for k, v in visible["forecasts"].items()
        }
        maximum = self.bundle.budget.state_summary_limit
        text_limit = self.bundle.budget.summary_text_characters
        visible["collection_counts"] = {
            name: len(state[name]) for name in ("needs", "observations", "evidence", "forecasts")
        }
        for name in ("needs", "observations", "evidence", "forecasts"):
            entries = chronological_items(visible[name])
            chosen = entries[-maximum:]
            if name == "forecasts" and state["active_forecast_id"] is not None:
                active = state["active_forecast_id"]
                if active not in dict(chosen):
                    chosen = chosen[1:] + [(active, visible[name][active])]
            visible[name] = dict(chosen)
            for value in visible[name].values():
                for text_key in ("description", "statement", "resolution_note"):
                    if isinstance(value.get(text_key), str):
                        value[text_key] = value[text_key][:text_limit]
                if isinstance(value.get("scope"), dict):
                    timestamps = value["scope"].pop("timestamps", None)
                    if timestamps:
                        value["scope"]["timestamp_extent"] = [timestamps[0], timestamps[-1]]
                if value.get("diff"):
                    changed = value["diff"].pop("changed_indices", [])
                    value["diff"]["changed_count"] = len(changed)
        visible["summary_notice"] = (
            "Recent bounded summaries only; complete records remain available through REQUEST_STATE."
        )
        visible["validations"] = visible["validations"][-4:]
        visible["errors"] = visible["errors"][-4:]
        visible["phase_history_count"] = len(visible["phase_history"])
        visible["phase_history"] = visible["phase_history"][-maximum:]
        # Compatibility scalars remain while the explicit structure distinguishes
        # Agent rounds from state_version and environment-only capacity events.
        visible["rounds_remaining"] = interaction_budget["rounds_remaining"]
        visible["agent_rounds_used"] = interaction_budget["agent_rounds_used"]
        if state.get("environment_capacity_events"):
            visible["llm_rounds_used"] = agent_rounds(state)
        visible["protocol_retry_available"] = bool(state.get("repair_pending"))
        visible["completion_rescue_available"] = bool(
            (state.get("completion_rescue") or {}).get("status") == "pending"
        )
        # The full diagnostics are pinned once below, not repeated twice in state.
        visible["protocol_alarms"] = [
            {k: v for k, v in alarm.items() if k != "message"} for alarm in state.get("protocol_alarms", [])
        ]
        if visible.get("repair_pending"):
            visible["repair_pending"].pop("message", None)
        repair_feedback = None
        if state.get("repair_pending"):
            original = next(
                (
                    row
                    for row in self.store.trace(self.run_id)
                    if row["turn"] == state["repair_pending"]["turn"]
                ),
                None,
            )
            if original is None or "protocol_alarm" not in original["response"]:
                raise InfrastructureError(
                    "Pending protocol repair is missing its committed diagnostic feedback"
                )
            repair_feedback = original["response"]["protocol_alarm"]
        packet = {
            "episode": self.bundle.episode.model_dump(),
            "task_description": self.bundle.task_description,
            "state": visible,
            "interaction_budget": interaction_budget,
            "tool_cards": self.registry.cards(detailed=False),
            "action_guide": ACTION_GUIDE,
            "protocol_examples": protocol_examples(state),
            "window_contract": {
                "model_call_arguments": {
                    "frequency": self.bundle.episode.task.frequency,
                    "forecast_gap": self.bundle.episode.task.forecast_gap,
                    "horizon_length": self.bundle.episode.task.horizon_length,
                },
                "scored_array_length": self.bundle.episode.task.horizon_length,
                "gap_array_length": self.bundle.episode.task.forecast_gap,
                "instruction": "Model calls require these three arguments. Forecast arrays contain exactly one number per timestamp. Use a scalar offset/factor for a uniform transformation rather than repeating it in a vector. A full-horizon range ends at horizon_length (exclusive), not horizon_length-1.",
            },
            "variable_catalog": [v.model_dump(exclude={"values"}) for v in self.bundle.variables],
            "revision_scale": self.bundle.revision_scale.model_dump(),
            "constraints": [c.model_dump() for c in self.bundle.constraints],
            "protocol_repair_policy": POLICY if self.protocol_repair else {"version": "disabled"},
            "pending_protocol_repair_feedback": repair_feedback,
            "default_check_profile": self.bundle.default_check_profile,
            "catalog_snapshot": self.registry.snapshot,
            "max_calls_per_round": self.bundle.budget.max_calls_per_round,
            "model_tool_profiles": [
                profile.model_dump(exclude={"python_executable", "checkpoint"})
                for name, profile in sorted(
                    self.model_service.profiles.items(),
                    key=lambda item: (
                        list(getattr(self.registry, "model_order", ())).index(
                            "model." + item[0] + "_forecast"
                        )
                        if "model." + item[0] + "_forecast" in getattr(self.registry, "model_order", ())
                        else 99
                    ),
                )
            ]
            if self.model_service
            else [],
        }
        if self.bundle.forecast_evaluation is not None:
            from .domain_metrics import public_notice

            packet["forecast_evaluation"] = public_notice(self.bundle.forecast_evaluation)
        if self.resource_policy is not None:
            from .efficiency import public_notice as efficiency_notice

            packet["efficiency_evaluation"] = efficiency_notice(self.resource_policy)
        if (state.get("completion_rescue") or {}).get("status") == "pending":
            packet["completion_rescue"] = completion_rescue_packet(
                state["completion_rescue"], state["state_version"]
            )
        if self.capacity_enabled:
            packet["token_limits"] = capacity_limits(self.provider.profile)
            packet["capacity_recovery_policy"] = CAPACITY_POLICY
            if state.get("capacity_alarm"):
                packet = reduce_packet(packet)
                packet["capacity_warning"] = capacity_feedback(
                    self.provider.profile,
                    state["capacity_alarm"]["kind"],
                    state["state_version"],
                    state["capacity_alarm"].get("observed_input_tokens"),
                )
                packet["capacity_warning"]["status"] = state["capacity_alarm"]["status"]
        return packet

    def messages(self, state):
        messages, self._context_report = build_messages(
            self.provider, self.store, self.run_id, self.packet(state), SYSTEM_PROMPT
        )
        return messages

    def _accept(self, state):
        turn = state["state_version"] + 1
        raw = self.store.accepted(self.run_id, turn)
        if self.store.has_accepted(self.run_id, turn):
            return raw
        schema = Action.model_json_schema()
        if (state.get("completion_rescue") or {}).get("status") == "pending":
            schema = completion_rescue_schema(
                state["completion_rescue"],
                state["state_version"],
                self.bundle.episode.forecast_timeline.scored_timestamps,
            )
        request = self.provider.request(self.messages(state), schema)
        self.store.audit(self.run_id, "context_prepared", {"turn": turn, **self._context_report})
        attempts = getattr(getattr(self.provider, "profile", None), "max_transport_attempts", 1)
        for attempt in range(attempts):
            aid = self.store.prepare(self.run_id, turn, request)
            started = time.perf_counter()
            try:
                raw = self.provider.respond(request)
            except InfrastructureError as exc:
                failed = {"transport_error": str(exc)}
                self.store.receive(aid, failed, complete=False, error=str(exc))
                if self.resource_policy is not None:
                    self._resource_exclude(started, time.perf_counter(), "provider_failure")
                    self._resource_llm_receipt("main", aid, failed, turn)
                if not isinstance(exc, TransientInfrastructureError) or attempt + 1 == attempts:
                    raise
                backoff_started = time.perf_counter()
                time.sleep(min(2**attempt, 8))
                if self.resource_policy is not None:
                    self._resource_exclude(backoff_started, time.perf_counter(), "transport_backoff")
                continue
            self.store.receive(aid, raw, complete=True)
            if self.resource_policy is not None:
                self._resource_llm_receipt("main", aid, raw, turn)
            self.fault_hook("after_provider_accept")
            return self.store.accepted(self.run_id, turn)
        raise InfrastructureError("Provider did not return")

    def run(self):
        with self.store.writer(self.run_id):
            try:
                if self.store.replay(self.run_id) != self.state():
                    raise InfrastructureError("Checkpoint differs from committed trace")
                while self.state()["status"] == "running":
                    self._step()
                return self.state()
            except InfrastructureError:
                raise
            except Exception as exc:
                # Unexpected host failures must not be turned into charged Agent mistakes.
                self.store.audit(
                    self.run_id,
                    "environment_failure",
                    {"type": type(exc).__name__, "message": str(exc)[:2000]},
                )
                raise InfrastructureError(
                    f"Environment paused: {type(exc).__name__}: {str(exc)[:500]}"
                ) from exc

    def step(self):
        with self.store.writer(self.run_id):
            if self.store.replay(self.run_id) != self.state():
                raise InfrastructureError("Checkpoint differs from committed trace")
            if self.state()["status"] != "running":
                return self.state()
            return self._step()

    def _step(self):
        if self.resource_policy is None:
            return self._step_impl()
        self._check_frozen_services()
        self._resource_segment_id = uuid4().hex
        self._resource_started = time.perf_counter()
        self._resource_exclusions = []
        turn = self.state()["state_version"] + 1
        self.store.audit(self.run_id, "resource_segment_started", {
            "segment_id": self._resource_segment_id, "turn": turn,
        })
        try:
            return self._step_impl()
        finally:
            elapsed = time.perf_counter() - self._resource_started
            self.store.audit(self.run_id, "resource_segment_finished", {
                "segment_id": self._resource_segment_id, "turn": turn,
                "elapsed_seconds": elapsed, "exclusions": self._resource_exclusions,
            })
            self._resource_segment_id = None

    def _resource_exclude(self, start, end, reason):
        self._resource_exclusions.append({
            "start": start - self._resource_started, "end": end - self._resource_started,
            "reason": reason,
        })

    def _resource_llm_receipt(self, source, call_id, raw, turn):
        self.store.audit(self.run_id, "resource_llm_receipt", {
            "segment_id": self._resource_segment_id, "turn": turn, "source": source,
            "call_id": call_id, "response": raw, "response_hash": digest(raw),
        })

    def record_auxiliary_llm_response(self, call_id, raw):
        """Tool adapters must call this for every auxiliary LLM attempt, including failures."""
        if (
            self.resource_policy is None or self.resource_policy.auxiliary_llm != "metered"
            or not getattr(self, "_resource_segment_id", None)
        ):
            raise InfrastructureError("Auxiliary usage requires a declared active resource segment")
        self._resource_llm_receipt("auxiliary", call_id, raw, self.state()["state_version"] + 1)

    def _step_impl(self):
        self._check_frozen_services()
        before = self.state()
        rescue_pending = (before.get("completion_rescue") or {}).get("status") == "pending"
        if self.store.replay(self.run_id) != before:
            raise InfrastructureError("Checkpoint differs from committed trace")
        try:
            raw = self._accept(before)
        except ContextCapacityExceeded as exc:
            if not self.capacity_enabled:
                raise
            # An unfit initial task is a deployment error, not an Agent failure.
            self._check_initial_capacity()
            return self._capacity_event(before, "input", count=exc.count)
        if self.capacity_enabled and isinstance(raw, dict):
            choices = raw.get("choices")
            if (
                isinstance(choices, list)
                and choices
                and isinstance(choices[0], dict)
                and choices[0].get("finish_reason") == "length"
            ):
                return self._capacity_event(before, "output")
        state = deepcopy(before)
        turn = before["state_version"] + 1
        refs, records = [], []
        action = None
        actual_stage = None
        try:
            action = decode(raw)
            if action.state_version != before["state_version"]:
                raise AgentError("Stale state_version", "STATE_VERSION")
            if before.get("repair_pending"):
                check_retry(action, before["repair_pending"])
            if rescue_pending and action.action_type != "CREATE_FORECAST":
                raise AgentError(
                    "The completion rescue accepts CREATE_FORECAST only",
                    "COMPLETION_RESCUE_NOT_RESOLVED",
                )
            actual_stage = observed_stage(action)
            state["phase"] = actual_stage
            if state["active_forecast_id"] and actual_stage in {"GAP", "ACQUIRE"}:
                state["pending_cycle"] = True
            response = self.dispatch(action, state, turn, refs, records)
            if self.protocol_repair and any(e["turn"] == turn for e in state["errors"]):
                raise AgentError(
                    "Tool returned an explicit tool_error; the action did not complete successfully",
                    "TOOL_ERROR",
                )
            if before.get("repair_pending") and before["repair_pending"]["intent"]["kind"] in {
                "forecast_creation",
                "missing_forecast",
            }:
                if state["active_forecast_id"] == before["active_forecast_id"]:
                    raise AgentError(
                        "Retry must produce a legal Agent forecast, not merely inspect a candidate",
                        "REPAIR_NOT_RESOLVED",
                    )
            for update in action.need_updates:
                need = state["needs"].get(update.need_id)
                if need is None or need["status"] not in {"open", "observed"}:
                    raise AgentError("Only an open/observed Need may be resolved or abandoned")
                need.update(status=update.status, resolution_note=update.resolution_note)
            state["phase_history"].append(
                {
                    "turn": turn,
                    "from_stage": before["phase"],
                    "observed_stage": actual_stage,
                    "intent_stage": action.phase,
                    "intent_mismatch": bool(action.phase and action.phase != actual_stage),
                    "cause": "agent_action",
                }
            )
            valid = True
            if before.get("repair_pending"):
                close_alarm(state, turn, True)
                response["protocol_repair_result"] = "recovered; alarm retained, no error action charged"
            if rescue_pending:
                if state["active_forecast_id"] is None:
                    raise AgentError(
                        "Completion rescue did not create a legal Agent forecast",
                        "COMPLETION_RESCUE_NOT_RESOLVED",
                    )
                close_rescue(state, turn, True)
                response["completion_rescue_result"] = (
                    "recovered; the Agent supplied the numerical decision under a narrowed format"
                )
        except (AgentError, ValidationError) as exc:
            state = deepcopy(before)
            refs, records = [], []
            if action and action.action_type == "CALL_TOOL":
                attempts = action.payload.get("calls", [])
                if isinstance(attempts, list):
                    for call in attempts:
                        if (
                            isinstance(call, dict)
                            and isinstance(call.get("tool_id"), str)
                            and not call["tool_id"].startswith("sandbox.")
                        ):
                            records.append(
                                {
                                    "tool_id": call["tool_id"],
                                    "information": True,
                                    "scope_compatible": False,
                                    "status": "agent_error",
                                }
                            )
            code = exc.code if isinstance(exc, AgentError) else "INVALID_ARGUMENT"
            response = {"status": "agent_error", "code": code, "message": str(exc)[:4000]}
            response["recovery"] = {
                "notice": "Rejected action made no forecast/state changes. Correct inputs; do not repeat unchanged.",
                "hint": "Use protocol_examples in the current packet. Read-only returned metadata is not input.",
                "schema_query": {
                    "action_type": "REQUEST_STATE",
                    "state_version": turn,
                    "payload": {"section": "schema", "ref": "ACTION"},
                },
            }
            if action and action.action_type == "CALL_TOOL":
                response["recovery"]["schema_query"]["payload"]["ref"] = "CALL_TOOL"
                response["recovery"]["hint"] = (
                    "CALL_TOOL payload contains calls only. Each calls[] item has call_id, tool_id, arguments, "
                    "and purpose (or compatible need_ids). purpose/need_ids are not tool arguments. "
                    "Read tool-specific argument schemas with REQUEST_STATE schema ref=<tool_id>."
                )
            elif action:
                response["recovery"]["schema_query"]["payload"]["ref"] = action.action_type
                if action.action_type == "REQUEST_STATE":
                    response["recovery"]["hint"] = (
                        "Read a forecast with section=forecasts and ref=<forecast ID>. "
                        "section=artifact instead requires an authorized sha256 artifact_ref from this run; "
                        "a forecast ID is not a hash. path applies only to artifacts."
                    )
            if (
                action
                and action.action_type == "CREATE_FORECAST"
                and action.payload.get("operation") == "adopt_candidate"
            ):
                response["recovery"]["hint"] = (
                    'For unchanged adoption use payload={"operation":"adopt_candidate","candidate_id":"<existing candidate ID>"}. '
                    "Do not include scored_values, timestamps, unit, forecast_id, or nested example labels. "
                    "For a changed prediction choose an editing operation or submit_full_forecast instead."
                )
            if code == "HARD_CONSTRAINT":
                response["recovery"]["hint"] = (
                    "The operation parsed, but its numerical forecast violates a declared hard domain constraint. "
                    "Repeating the same adoption, or adding declared_effect without changing values, cannot fix this. "
                    "Use the public candidate values and hard_constraints below to explicitly edit/generate "
                    "a compliant prediction or choose another existing candidate. The pending retry must "
                    "still create a forecast, not merely inspect one. "
                    "The environment will not silently clip or replace your values."
                )
                response["recovery"]["hard_constraints"] = [
                    constraint.model_dump() for constraint in self.bundle.constraints if constraint.hard
                ]
            if rescue_pending:
                close_rescue(state, turn, False)
                state["errors"].append(
                    {"turn": turn, **response, "action_hash": error_fingerprint(action, raw)}
                )
                response["completion_rescue_result"] = (
                    "failed; no host-selected or host-repaired prediction was substituted"
                )
            elif self.protocol_repair and not before.get("protocol_alarms"):
                open_alarm(
                    state, action, response, exc, self.bundle, self.registry, ACTION_PAYLOAD_SCHEMAS, turn
                )
                response["status"] = "protocol_warning"
            else:
                if before.get("repair_pending"):
                    close_alarm(state, turn, False)
                    response["protocol_repair_result"] = (
                        "failed; one error action charged, retry exemption exhausted"
                    )
                state["errors"].append(
                    {"turn": turn, **response, "action_hash": error_fingerprint(action, raw)}
                )
            valid = False
        if state.get("capacity_alarm", {}).get("status") == "pending":
            state["capacity_alarm"].update(status="recovered", recovery_turn=turn)
            response["capacity_recovery_result"] = (
                "Output fits capacity; protocol validity is scored separately."
            )
        state["state_version"], state["rounds_used"] = turn, turn
        if (
            self.protocol_repair
            and agent_rounds(state) >= self.bundle.budget.max_rounds
            and state["status"] == "running"
            and state["active_forecast_id"] is None
            and not state.get("repair_pending")
            and state.get("completion_rescue") is None
        ):
            response["completion_rescue"] = open_rescue(state, self.bundle, turn)
        if agent_rounds(state) >= self.bundle.budget.max_rounds and state["status"] == "running":
            if not state.get("repair_pending") and not (
                (state.get("completion_rescue") or {}).get("status") == "pending"
            ):
                state["status"] = "finished"
                state["final"] = {
                    "forecast_id": state["active_forecast_id"],
                    "forced_by_budget": True,
                    "stop_reason": "round_budget_exhausted",
                }
                state["phase_history"].append(
                    {
                        "turn": turn,
                        "from_stage": state["phase"],
                        "observed_stage": "STOP",
                        "intent_stage": None,
                        "intent_mismatch": False,
                        "cause": "round_budget_exhausted",
                    }
                )
                state["phase"] = "STOP"
        response.update(
            state_version=turn,
            rounds_remaining=public_interaction_budget(self.bundle, state)["rounds_remaining"],
            interaction_budget=public_interaction_budget(self.bundle, state),
            final=state["final"],
            protocol_retry_available=bool(state.get("repair_pending")),
            completion_rescue_available=bool(
                (state.get("completion_rescue") or {}).get("status") == "pending"
            ),
        )
        event = {
            "action": action.model_dump() if action else None,
            "valid": valid,
            "tool_records": records,
            "state_hash": digest(state),
            "response_hash": digest(response),
            "stage_transition": {
                "from_stage": before["phase"],
                "to_stage": state["phase"],
                "observed_stage": actual_stage,
                "intent_stage": action.phase if action else None,
                "intent_mismatch": bool(
                    action and action.phase and actual_stage and action.phase != actual_stage
                ),
                "applied": valid,
            },
        }
        self.fault_hook("before_commit")
        response = self.store.commit(self.run_id, before["state_version"], state, response, event, refs)
        self.fault_hook("after_commit")
        return response

    def _check_initial_capacity(self):
        packet = self.packet(initial_state(self.bundle.episode.episode_id))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json_text(packet)},
        ]
        count = self.provider.count_input_tokens(self.provider.request(messages, Action.model_json_schema()))
        if count > capacity_limits(self.provider.profile)["input_tokens"]:
            raise InfrastructureError(
                "Initial public task cannot fit configured capacity; increase deployment capacity before evaluation"
            )

    def _capacity_event(self, before, kind, count=None):
        if kind == "input" and before.get("capacity_alarm", {}).get("status") == "pending":
            raise InfrastructureError(
                "The capacity feedback packet itself cannot fit; the Agent has not received its warning yet. "
                "Fix the public presentation/deployment instead of charging an unseen warning as an Agent failure."
            )
        turn = before["state_version"] + 1
        state = deepcopy(before)
        warning = capacity_feedback(self.provider.profile, kind, turn, count)
        if not state.get("capacity_alarm"):
            state["capacity_alarm"] = {
                "turn": turn,
                "kind": kind,
                "status": "pending",
                "observed_input_tokens": count,
            }
            response = {"status": "capacity_warning", "capacity_warning": warning}
        else:
            state["capacity_alarm"].update(status="failed", failure_turn=turn, failure_kind=kind)
            response = {
                "status": "agent_error",
                "code": warning["code"],
                "message": "Capacity exceeded again after the one warning; this Episode failed.",
            }
            state["errors"].append(
                {"turn": turn, **response, "action_hash": digest({"capacity_failure": kind})}
            )
            if state.get("repair_pending"):
                close_alarm(state, turn, False)
            state["status"], state["phase"] = "finished", "STOP"
            state["final"] = {
                "forecast_id": None,
                "forced_by_budget": False,
                "stop_reason": "capacity_exhausted",
            }
            state["phase_history"].append(
                {
                    "turn": turn,
                    "from_stage": before["phase"],
                    "observed_stage": "STOP",
                    "intent_stage": None,
                    "intent_mismatch": False,
                    "cause": "capacity_exhausted",
                }
            )
        state["state_version"], state["rounds_used"] = turn, turn
        if kind == "input":
            state["environment_capacity_events"] = state.get("environment_capacity_events", 0) + 1
        response.update(
            state_version=turn,
            final=state["final"],
            rounds_remaining=public_interaction_budget(self.bundle, state)["rounds_remaining"],
            interaction_budget=public_interaction_budget(self.bundle, state),
        )
        event = {
            "actor": "environment_capacity" if kind == "input" else "agent",
            "action": None,
            "valid": False,
            "tool_records": [],
            "state_hash": digest(state),
            "response_hash": digest(response),
            "capacity_event": kind,
            "api_called": kind == "output",
        }
        self.fault_hook("before_commit")
        result = self.store.commit(self.run_id, before["state_version"], state, response, event)
        self.fault_hook("after_commit")
        return result

    def _put_forecast(self, forecast, state, turn, refs):
        fid = "forecast_" + str(len(state["forecasts"]) + 1)
        forecast.update(forecast_id=fid, created_turn=turn)
        ref = self.store.put_blob(forecast)
        refs.append(ref)
        forecast["artifact_ref"] = ref
        state["forecasts"][fid] = forecast
        if forecast["kind"] != "model_candidate":
            if state["cycle"] == 0 or state["pending_cycle"]:
                state["cycle"] += 1
                state["pending_cycle"] = False
            state["active_forecast_id"] = fid
        return {
            "forecast_id": fid,
            "artifact_ref": ref,
            "kind": forecast["kind"],
            "read_forecast": {"section": "forecasts", "ref": fid, "offset": 0, "limit": min(256, self.bundle.budget.max_state_page)},
            "diff": forecast.get("diff"),
            "warnings": forecast.get("auto_check", {}).get("warnings", []),
        }

    def _forecast(self, p, state, turn, refs):
        args = ForecastArgs.model_validate(p)
        return self._put_forecast(create_forecast(args, state, self.bundle), state, turn, refs)

    def _check(self, p, state, turn):
        _keys(p, ["forecast_id"], ["checks"])
        fid = p["forecast_id"]
        if fid not in state["forecasts"]:
            raise AgentError("Unknown forecast", "UNKNOWN_REFERENCE")
        checks = p.get("checks")
        if checks is not None and (
            not isinstance(checks, list) or not all(isinstance(c, str) for c in checks)
        ):
            raise AgentError("checks must be a list of public check names")
        validation = {
            "forecast_id": fid,
            "turn": turn,
            **check_values(state["forecasts"][fid]["scored_values"], self.bundle, checks),
        }
        state["validations"].append(validation)
        return validation

    def dispatch(self, action, state, turn, refs, records):
        kind = action.action_type
        p = ACTION_PAYLOAD_SCHEMAS[kind].model_validate(action.payload).model_dump(exclude_none=True)
        if kind == "DECLARE_NEED":
            _keys(p, ["description", "kind"], ["scope"])
            if not isinstance(p["description"], str) or not p["description"].strip():
                raise AgentError("Need description must be nonempty text")
            if p["kind"] not in {"feature", "context", "model", "knowledge", "sample", "registry", "domain"}:
                raise AgentError("Unknown Need kind")
            nid = "need_" + str(len(state["needs"]) + 1)
            state["needs"][nid] = {
                **p,
                "scope": Scope.model_validate(p.get("scope", {})).model_dump(),
                "status": "open",
                "created_turn": turn,
            }
            return {"status": "ok", "need_id": nid}
        if kind == "CREATE_FORECAST":
            return {"status": "ok", **self._forecast(p, state, turn, refs)}
        if kind == "CHECK_FORECAST":
            return {"status": "ok", "validation": self._check(p, state, turn)}
        if kind == "FINALIZE":
            _keys(p, ["forecast_id", "stop_reason"])
            v = state["forecasts"].get(p["forecast_id"])
            if v is None or v["kind"] == "model_candidate":
                raise AgentError("Finalize a legal Agent version, not a raw model candidate")
            if not isinstance(p["stop_reason"], str) or not p["stop_reason"].strip():
                raise AgentError("Provide a stop reason")
            state["status"] = "finished"
            state["final"] = {**p, "forced_by_budget": False}
            return {"status": "ok"}
        if kind == "REGISTER_EVIDENCE":
            _keys(p, ["source_observation_ids", "scope", "application_scope", "relation_type"], ["statement"])
            sources = p["source_observation_ids"]
            if (
                not isinstance(sources, list)
                or not sources
                or any(x not in state["observations"] for x in sources)
            ):
                raise AgentError("Evidence must cite actual observations", "UNKNOWN_REFERENCE")
            scope = Scope.model_validate(p["scope"])
            application = [IndexRange.model_validate(r).model_dump() for r in p["application_scope"]]
            validate_ranges(
                application, self.bundle.episode.task.horizon_length, self.bundle.episode.task.forecast_gap
            )
            if p["relation_type"] not in {"direct_context", "historical_analogy", "general_knowledge"}:
                raise AgentError("Unknown evidence relation type")
            for source in sources:
                observed = state["observations"][source]["scope"]
                obs_scope = {k: v for k, v in observed.items() if k in Scope.model_fields}
                if not _compatible(obs_scope, scope.model_dump(), self.bundle):
                    raise AgentError(
                        "Evidence source scope exceeds its observations: "
                        f"source {source} observed_scope={json_text(obs_scope)}, "
                        f"requested_scope={json_text(scope.model_dump())}. "
                        "Every cited source must cover the declared source scope. "
                        "Register separate evidence entries for sources with different timelines or variables; "
                        "use application_scope to describe where the evidence applies to the forecast, "
                        "not to claim that historical observations were observed in the future."
                    )
            eid = "evidence_" + str(len(state["evidence"]) + 1)
            state["evidence"][eid] = {
                **p,
                "scope": scope.model_dump(),
                "application_scope": application,
                "created_turn": turn,
            }
            return {"status": "ok", "evidence_id": eid}
        if kind == "REQUEST_STATE":
            return {"status": "ok", "data": self._read_state(p, state, refs)}
        if kind == "CALL_TOOL":
            return self._tools(p, state, turn, refs, records)
        raise AgentError("Unknown action")

    def _read_state(self, p, state, refs=None):
        _keys(p, ["section"], ["ref", "offset", "limit", "path"])
        offset, limit = p.get("offset", 0), p.get("limit", 128)
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= self.bundle.budget.max_state_page
        ):
            raise AgentError("Invalid pagination")
        section = p["section"]
        if p.get("path") and section != "artifact":
            raise AgentError("path is only applicable to artifact retrieval")
        if section == "summary":
            return self.packet(state)["state"]
        if section == "phase_history":
            return state["phase_history"][offset : offset + limit]
        if section == "history":
            rows = self.store.trace(self.run_id)
            # Only committed public interactions from this run; never private targets or other episodes.
            page = []
            for row in rows[offset : offset + min(limit, 4)]:
                item = {
                    "turn": row["turn"],
                    "action": row["event"].get("action"),
                    "response": row["response"],
                    "provider_response": self.store.accepted(self.run_id, row["turn"]),
                }
                if (
                    refs is not None
                    and len(json_text(item).encode("utf-8")) > self.bundle.budget.max_inline_bytes
                ):
                    ref = self.store.put_blob(item)
                    refs.append(ref)
                    item = {
                        "turn": row["turn"],
                        "artifact_ref": ref,
                        "notice": "Large original interaction; use REQUEST_STATE artifact with path and pagination.",
                    }
                page.append(item)
            return {
                "total_turns": len(rows),
                "offset": offset,
                "turns": page,
                "next_offset": min(len(rows), offset + len(page)),
            }
        if section == "schema":
            ref = p.get("ref")
            if ref == "ACTION":
                return Action.model_json_schema()
            if ref in ACTION_PAYLOAD_SCHEMAS:
                return ACTION_PAYLOAD_SCHEMAS[ref].model_json_schema()
            if ref not in self.registry.manifests:
                raise AgentError("Unknown schema reference")
            return self.registry.manifests[ref]
        if section == "artifact":
            value = self.store.read_blob(self.run_id, p.get("ref"))
            for component in p.get("path", []):
                if isinstance(value, dict) and isinstance(component, str) and component in value:
                    value = value[component]
                elif isinstance(value, list) and isinstance(component, int) and 0 <= component < len(value):
                    value = value[component]
                else:
                    raise AgentError("Artifact path does not exist", "UNKNOWN_REFERENCE")
            if isinstance(value, list):
                selected = value[offset : offset + limit]
                page = {"values": selected, "offset": offset, "total_items": len(value)}
                if len(json_text(page).encode("utf-8")) <= self.bundle.budget.max_inline_bytes:
                    page["next_offset"] = min(len(value), offset + len(selected))
                    return page
                # Item-count pagination alone cannot bound nested training cases.
                # Keep originals accessible by index without silently truncating data.
                page = {
                    "encoding": "indexed_items",
                    "items": [],
                    "offset": offset,
                    "total_items": len(value),
                    "next_offset": offset,
                    "notice": "Large items are references, not data. Follow read and then select fields with path.",
                }
                for index, item in enumerate(selected, offset):
                    entry = {"index": index, "value": item}
                    if len(json_text(entry).encode("utf-8")) > self.bundle.budget.max_inline_bytes // 2:
                        entry = {
                            "index": index,
                            "deferred": True,
                            "value_type": type(item).__name__,
                            "read": {
                                "section": "artifact",
                                "ref": p.get("ref"),
                                "path": [*p.get("path", []), index],
                                "offset": 0,
                                "limit": min(128, self.bundle.budget.max_state_page),
                            },
                        }
                    candidate = {**page, "items": [*page["items"], entry], "next_offset": index + 1}
                    if page["items"] and len(json_text(candidate).encode("utf-8")) > self.bundle.budget.max_inline_bytes:
                        break
                    page = candidate
                return page
            if isinstance(value, dict) and not p.get("path"):
                return {
                    "keys": list(value),
                    "usage": "Use path to select fields/arrays; arrays support offset/limit.",
                }
            # Arrays page by points; other JSON pages use 32 characters per requested unit.
            text = json_text(value)
            return {
                "encoding": "json_text",
                "total_characters": len(text),
                "offset": offset,
                "text": text[offset : offset + limit * 32],
                "next_offset": min(len(text), offset + limit * 32),
                "children": [
                    {
                        "key": key,
                        "read": {
                            "section": "artifact",
                            "ref": p.get("ref"),
                            "path": [*p.get("path", []), key],
                            "offset": 0,
                            "limit": min(256, self.bundle.budget.max_state_page),
                        },
                    }
                    for key in value
                ]
                if isinstance(value, dict)
                else [],
            }
        if section not in {"forecasts", "observations", "evidence", "needs"}:
            raise AgentError("State section is not public", "PERMISSION_DENIED")
        if "ref" in p:
            if p["ref"] not in state[section]:
                raise AgentError("Unknown reference", "UNKNOWN_REFERENCE")
            value = deepcopy(state[section][p["ref"]])
            for key in ("scored_values", "gap_values"):
                if isinstance(value.get(key), list):
                    value[key] = value[key][offset : offset + limit]
            return value
        return chronological_items(state[section])[offset : offset + limit]

    def _tools(self, p, state, turn, refs, records):
        _keys(p, ["calls"])
        if (
            not isinstance(p["calls"], list)
            or not 1 <= len(p["calls"]) <= self.bundle.budget.max_calls_per_round
        ):
            raise AgentError("Invalid batch size")
        calls = [ToolCall.model_validate(c) for c in p["calls"]]
        if len({c.call_id for c in calls}) != len(calls):
            raise AgentError("Duplicate call_id within round")
        prepared = []
        for call in calls:
            args = self.registry.validate(call.tool_id, call.arguments)
            manifest = self.registry.manifests[call.tool_id]
            if len(calls) > 1 and not manifest["parallel_safe"]:
                raise AgentError("Only independent read-only tools may be batched")
            if manifest["category"] == "sandbox":
                prepared.append((call, args, [], None))
                continue
            cat = manifest["category"]
            call_scope = Scope.model_validate(args.get("scope", {})).model_dump()
            call_scope["variables"] = args.get("variables", manifest.get("input_variables", []))
            if manifest.get("reads_target_history"):
                call_scope["variables"] = [self.bundle.episode.task.target_name, *call_scope["variables"]]
            if cat == "feature" and "series_ref" in args:
                series_ref = args["series_ref"]
                if series_ref == "episode.history.target":
                    if call_scope["timeline"] == "all":
                        call_scope["timeline"] = "history"
                    call_scope["variables"] = [
                        self.bundle.episode.task.target_name,
                        *args.get("variables", []),
                    ]
                elif series_ref.startswith("covariate:"):
                    call_scope["variables"] = [series_ref.split(":", 1)[1]]
            elif cat == "knowledge" and "variable" in args:
                call_scope["variables"] = [args["variable"]]
            if cat == "model":
                call_scope = {
                    "timeline": "history",
                    "start": 0,
                    "end": self.bundle.episode.task.lookback_length,
                    "variables": [self.bundle.episode.task.target_name],
                }
            ids = list(call.need_ids)
            if call.purpose is not None:
                if not call.purpose.strip():
                    raise AgentError("purpose cannot be blank")
                nid = "need_" + str(len(state["needs"]) + 1)
                state["needs"][nid] = {
                    "description": call.purpose,
                    "kind": cat,
                    "scope": call_scope,
                    "status": "open",
                    "created_turn": turn,
                }
                ids.append(nid)
            if not ids:
                raise AgentError("Information tools require purpose or compatible need_ids")
            for nid in ids:
                need = state["needs"].get(nid)
                if (
                    need is None
                    or need["status"] not in {"open", "observed"}
                    or need["kind"] != cat
                    or not _compatible(need["scope"], call_scope, self.bundle)
                ):
                    raise AgentError("Need is terminal, unknown, or incompatible with this call")
            prepared.append((call, args, ids, call_scope))
        if prepared[0][0].tool_id.startswith("sandbox."):
            call, args, _, _ = prepared[0]
            name = call.tool_id.split(".")[1]
            if name == "validate_forecast":
                output = {"validation": self._check(args, state, turn)}
            elif name == "compare_versions":
                ids = args["forecast_ids"]
                if len(ids) < 2 or any(i not in state["forecasts"] for i in ids):
                    raise AgentError("Comparison needs at least two known versions")
                output = {
                    "comparisons": [
                        {
                            "before": ids[0],
                            "after": i,
                            **diff_report(
                                state["forecasts"][ids[0]]["scored_values"],
                                state["forecasts"][i]["scored_values"],
                                self.bundle.revision_scale,
                            ),
                        }
                        for i in ids[1:]
                    ]
                }
            else:
                output = self._forecast(args, state, turn, refs)
            records.append(
                {"tool_id": call.tool_id, "information": False, "call_id": call.call_id, "status": "ok"}
            )
            result = ToolResult(
                status="ok",
                structured_result=output,
                observed_scope={"timeline": "scored"},
                units={"values": self.bundle.episode.task.target_unit},
                factual_summary=f"{call.tool_id}: ok; immutable state updated as requested.",
                provenance={
                    "tool_id": call.tool_id,
                    "arguments": args,
                    "tool_version": "1.0.0",
                    "implementation_digest": self.registry.manifests[call.tool_id]["implementation_digest"],
                },
            ).model_dump()
            return {"status": "ok", "results": [{"call_id": call.call_id, "result": result}]}
        snapshot = deepcopy(state)

        def run_one(item):
            call, args, _, _ = item
            key = digest({"tool": call.tool_id, "args": args, "call_id": call.call_id})
            prior = self.store.staged(self.run_id, turn, key)
            if prior is not None:
                return prior
            started = time.perf_counter()
            value = execute_readonly(
                call.tool_id,
                args,
                ToolView(self.bundle, snapshot, self.store, self.run_id),
                self.registry,
                self.model_service,
            )
            self.store.stage(self.run_id, turn, key, value)
            self.store.audit(
                self.run_id,
                "tool_executed",
                {
                    "turn": turn,
                    "call_id": call.call_id,
                    "tool_id": call.tool_id,
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                },
            )
            return value

        if len(prepared) == 1:
            outputs = [run_one(prepared[0])]
        else:
            with ThreadPoolExecutor(max_workers=len(prepared)) as pool:
                futures = [pool.submit(run_one, item) for item in prepared]
                outputs = []
                errors = []
                for future in futures:
                    try:
                        outputs.append(future.result())
                    except Exception as exc:
                        errors.append(exc)
                if errors:
                    infra = next(
                        (x for x in errors if not isinstance(x, (AgentError, ValidationError))), None
                    )
                    raise infra or errors[0]
        results = []
        for (call, args, nids, call_scope), output in zip(prepared, outputs):
            if output["status"] not in {"ok", "partial", "not_applicable", "tool_error"}:
                raise InfrastructureError("Tool returned an invalid status")
            artifact = self.store.put_blob(output)
            refs.append(artifact)
            oid = "observation_" + str(len(state["observations"]) + 1)
            scope = output["observed_scope"]
            state["observations"][oid] = {
                "tool_id": call.tool_id,
                "scope": scope,
                "artifact_ref": artifact,
                "need_ids": nids,
                "created_turn": turn,
                "status": output["status"],
            }
            if output["status"] in {"ok", "partial"}:
                for nid in nids:
                    state["needs"][nid]["status"] = "observed"
            elif output["status"] == "tool_error":
                state["errors"].append(
                    {
                        "turn": turn,
                        "status": "tool_error",
                        "code": "TOOL_ERROR",
                        "message": output["factual_summary"],
                        "action_hash": digest({"tool_id": call.tool_id, "arguments": args}),
                    }
                )
            result = {
                "call_id": call.call_id,
                "observation_id": oid,
                "artifact_ref": artifact,
                "status": output["status"],
            }
            result["read_result"] = {
                "section": "artifact",
                "ref": artifact,
                "path": ["structured_result"],
                "offset": 0,
                "limit": min(256, self.bundle.budget.max_state_page),
            }
            result["result_fields"] = list(output["structured_result"])
            if call.tool_id.startswith("model.") and output["status"] == "ok":
                prediction = output["structured_result"]
                try:
                    finite_array(prediction["gap_values"], self.bundle.episode.task.forecast_gap)
                    finite_array(prediction["scored_values"], self.bundle.episode.task.horizon_length)
                except (AgentError, KeyError, TypeError) as exc:
                    raise InfrastructureError(
                        "Model adapter returned an invalid candidate; nothing was committed"
                    ) from exc
                forecast = {
                    "kind": "model_candidate",
                    "parents": [],
                    "previous_agent_forecast_id": None,
                    "gap_values": prediction["gap_values"],
                    "scored_values": prediction["scored_values"],
                    "unit": self.bundle.episode.task.target_unit,
                    "source_observation_id": oid,
                    "model_tool": call.tool_id,
                }
                registered = self._put_forecast(forecast, state, turn, refs)
                result.update({k: v for k, v in registered.items() if k != "artifact_ref"})
                result["forecast_artifact_ref"] = registered["artifact_ref"]
            if call.tool_id.startswith("model.") and output["status"] == "ok":
                # Same point-only view for every model; full distributions/provenance remain in the artifact.
                result["result"] = {
                    **output,
                    "structured_result": {
                        k: output["structured_result"][k] for k in ("gap_values", "scored_values")
                    },
                    "factual_summary": "Ordered point predictions, one value per task timestamp, not quantiles. "
                    "No private error estimates. Full model output is in artifact_ref.",
                }
                result["presentation"] = "uniform_point_forecast_view"
            elif len(json_text(output).encode("utf-8")) <= self.bundle.budget.max_inline_bytes:
                result["result"] = output
            else:
                result.update(
                    summary=output["factual_summary"],
                    large_result=True,
                    retrieve_with="REQUEST_STATE artifact",
                )
            records.append(
                {
                    "call_id": call.call_id,
                    "tool_id": call.tool_id,
                    "information": True,
                    "arguments_hash": digest(args),
                    "need_ids": nids,
                    "scope_compatible": True,
                    "status": output["status"],
                    **({"redundancy_identity": digest({
                        "tool_id": call.tool_id,
                        "arguments": args,
                        "bundle_hash": self.store.config(self.run_id)["bundle_hash"],
                        "catalog_hash": self.registry.snapshot,
                        "returned_information": output,
                    })} if self.resource_policy is not None and call.tool_id.split(".")[0]
                       in {"feature", "context", "knowledge", "sample", "domain"} else {}),
                }
            )
            results.append(result)
        return {"status": "ok", "results": results}
