"""One format-constrained completion turn after the ordinary Episode budget.

The environment never chooses or changes forecast values here. It only narrows the
wire contract so that a capable Agent does not lose an otherwise formed numerical
decision to avoidable schema errors on the final ordinary round.
"""

from __future__ import annotations

from copy import deepcopy

POLICY = {
    "version": "one_completion_rescue/1.0",
    "max_attempts_per_episode": 1,
    "trigger": "ordinary_budget_exhausted_without_legal_agent_forecast",
    "allowed_decisions": ["adopt_candidate", "constant_forecast", "submit_full_forecast"],
    "numeric_policy": "agent_chosen_only; no host selection, padding, clipping, or fallback",
}


def open_rescue(state, bundle, turn):
    candidates = [
        forecast_id
        for forecast_id, value in state["forecasts"].items()
        if value.get("kind") == "model_candidate"
    ]
    rescue = {
        "status": "pending",
        "opened_turn": turn,
        "attempt_turn": None,
        "candidate_ids": candidates,
        "horizon_length": bundle.episode.task.horizon_length,
        "unit": bundle.episode.task.target_unit,
    }
    state["completion_rescue"] = rescue
    return deepcopy(rescue)


def close_rescue(state, turn, success):
    state["completion_rescue"].update(
        status="recovered" if success else "failed",
        attempt_turn=turn,
    )


def packet(rescue, state_version):
    candidates = list(rescue["candidate_ids"])
    examples = []
    if candidates:
        examples.append(
            {
                "action_type": "CREATE_FORECAST",
                "state_version": state_version,
                "payload": {"operation": "adopt_candidate", "candidate_id": candidates[0]},
            }
        )
    examples.append(
        {
            "action_type": "CREATE_FORECAST",
            "state_version": state_version,
            "payload": {
                "operation": "constant_forecast",
                "constant_value": "<your already chosen finite number>",
                "unit": rescue["unit"],
            },
        }
    )
    return {
        **deepcopy(rescue),
        "instruction": (
            "The ordinary round budget is exhausted without a legal Agent forecast. This is the "
            "single format-constrained completion turn. Submit CREATE_FORECAST now. Preserve your "
            "own numerical judgment: adopt an existing candidate, submit your chosen constant, or "
            "submit the complete scored horizon. Do not inspect more evidence, register evidence, "
            "or include optional rationale fields. The environment will not choose a candidate or "
            "repair forecast values for you."
        ),
        "legal_candidate_ids": candidates,
        "exact_examples": examples,
    }


def action_schema(rescue, state_version, scored_timestamps):
    """A narrow native function schema; it constrains representation, never values."""
    alternatives = []
    if rescue["candidate_ids"]:
        alternatives.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["operation", "candidate_id"],
                "properties": {
                    "operation": {"const": "adopt_candidate"},
                    "candidate_id": {"type": "string", "enum": list(rescue["candidate_ids"])},
                },
            }
        )
    alternatives.extend(
        [
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["operation", "constant_value", "unit"],
                "properties": {
                    "operation": {"const": "constant_forecast"},
                    "constant_value": {"type": "number"},
                    "unit": {"const": rescue["unit"]},
                },
            },
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["operation", "scored_values", "scored_timestamps", "unit"],
                "properties": {
                    "operation": {"const": "submit_full_forecast"},
                    "scored_values": {
                        "type": "array",
                        "minItems": rescue["horizon_length"],
                        "maxItems": rescue["horizon_length"],
                        "items": {"type": "number"},
                    },
                    "scored_timestamps": {"const": list(scored_timestamps)},
                    "unit": {"const": rescue["unit"]},
                },
            },
        ]
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["action_type", "state_version", "payload"],
        "properties": {
            "action_type": {"const": "CREATE_FORECAST"},
            "state_version": {"const": state_version},
            "payload": {"oneOf": alternatives},
        },
    }
