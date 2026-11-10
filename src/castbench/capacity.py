"""One shared input/output capacity warning; never repair numerical predictions."""

from copy import deepcopy

from .common import InfrastructureError

POLICY = {
    "version": "one_capacity_warning/1.0",
    "max_warnings_per_episode": 1,
    "shared_input_output_allowance": True,
    "warning_penalty_error_equivalent": 0.05,
    "second_overflow": "finish_episode_as_error_without_fallback_prediction",
}


def agent_rounds(state):
    """Journal transitions include host notifications; those do not consume an LLM round."""
    return state["rounds_used"] - state.get("environment_capacity_events", 0)


class ContextCapacityExceeded(InfrastructureError):
    def __init__(self, count, limit):
        self.count, self.limit = count, limit
        super().__init__(
            f"Public packet and factual memory exceed context input capacity ({count}>{limit}); "
            "no API call made and no Agent round charged."
        )


def limits(profile):
    return {
        "context_tokens": profile.context_window_tokens,
        "input_tokens": profile.context_window_tokens
        - profile.max_output_tokens
        - profile.context_safety_tokens,
        "output_tokens_including_reasoning": profile.max_output_tokens,
        "safety_tokens": profile.context_safety_tokens,
    }


def feedback(profile, kind, turn, count=None):
    return {
        "code": "INPUT_CAPACITY" if kind == "input" else "OUTPUT_TRUNCATED",
        "observed_input_tokens": count,
        "limits": limits(profile),
        "instruction": (
            "This is the ONE shared input/output capacity warning for this Episode. "
            "Your existing progress and all originals are preserved. On the next response, "
            "finish one complete environment_action within the output limit, including reasoning. "
            "Use concise reasoning and reference existing artifacts/forecasts instead of copying them. "
            "Retrieve smaller pages or specific fields, not entire histories or repeated schemas. "
            "Complete independent forecasts remain allowed: supply EVERY required scored point. "
            "Do not change your numerical judgment just to save tokens. If a protocol repair is pending, "
            "correct that same action. Another input overflow or truncated output ends THIS Episode as an error."
        ),
        "retrieval_example": {
            "action_type": "REQUEST_STATE",
            "state_version": turn,
            "payload": {"section": "history", "offset": 0, "limit": 1},
        },
        "example_notice": "Format only; not a required next action and not an override of a pending protocol repair.",
    }


def reduce_packet(packet):
    """A recoverable index view; keep the full task, timeline, cards and safety contract."""
    from .context_memory import compact_value

    result = deepcopy(packet)
    state = result["state"]
    for name in ("needs", "observations", "evidence", "forecasts"):
        state[name] = {
            key: {
                **compact_value(value),
                "read_original": {"section": name, "ref": key},
            }
            for key, value in state[name].items()
        }
    for name in ("errors", "validations", "phase_history"):
        state[name] = compact_value(state.get(name, []))
    pending = result.get("pending_protocol_repair_feedback")
    if pending:
        # The next action must repair the same intent; it cannot spend that opportunity
        # retrieving schemas/examples discarded by lossy compression.
        result["pending_protocol_repair_feedback"] = deepcopy(pending)
        result["pending_protocol_repair_feedback"]["read_original"] = {
            "section": "history",
            "offset": pending["turn"] - 1,
            "limit": 1,
        }
    result["capacity_index_view"] = (
        "Bounded factual excerpts; not complete reasoning. All original data and interactions remain retrievable."
    )
    return result
