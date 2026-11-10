"""Deterministic conversation compaction; originals remain in the run journal."""

from copy import deepcopy

from .capacity import ContextCapacityExceeded
from .common import InfrastructureError, digest, json_text
from .contracts import Action
from .providers import replayable_tool_calls


def compact_value(value, depth=0):
    """Keep references and facts, not invented interpretations or rewritten reasoning."""
    if isinstance(value, str):
        return value if len(value) <= 240 else {"excerpt": value[:240], "truncated": True}
    if isinstance(value, list):
        if len(value) > 6:
            return {
                "items": len(value),
                "preview": [compact_value(v, depth + 1) for v in value[:2]],
                "truncated": True,
            }
        return [compact_value(v, depth + 1) for v in value]
    if isinstance(value, dict):
        if depth >= 5:
            return {"keys": list(value), "truncated": True}
        return {
            k: compact_value(v, depth + 1)
            for k, v in value.items()
            if k not in {"provenance", "input_schema", "output_schema"}
        }
    return value


def turn_memory(row):
    action = row["event"].get("action") or {}
    response = row["response"]
    payload = action.get("payload", {})
    refs = {
        k: v
        for k, v in payload.items()
        if k
        in {
            "operation",
            "candidate_id",
            "parent_id",
            "source_ids",
            "forecast_id",
            "section",
            "ref",
            "evidence_refs",
            "source_observation_ids",
            "stop_reason",
            "statement",
            "description",
        }
    }
    calls = payload.get("calls", [])
    if isinstance(calls, list):
        refs["tools"] = [c.get("tool_id") for c in calls if isinstance(c, dict)]
    outcomes = {
        k: response[k]
        for k in ("status", "code", "message", "forecast_id", "evidence_id", "need_id")
        if k in response
    }
    if isinstance(response.get("results"), list):
        outcomes["results"] = [
            {k: r[k] for k in ("observation_id", "forecast_id", "artifact_ref", "status") if k in r}
            for r in response["results"]
            if isinstance(r, dict)
        ]
    # This is an index, not a lossless semantic summary. Detailed arguments and reasoning stay retrievable.
    return {
        "turn": row["turn"],
        "valid": row["event"]["valid"],
        "action_type": action.get("action_type"),
        "action_excerpt": compact_value(refs),
        "response_excerpt": compact_value(outcomes),
        "read_original": {"section": "history", "offset": row["turn"] - 1, "limit": 1},
    }


def replay_group(provider, raw, row):
    native_errors = getattr(getattr(provider, "profile", None), "replay_rejected_messages", False)
    if not row["event"]["valid"] and not native_errors:
        # Originals remain archived; rejected arrays are not demonstrations to repeat.
        return [
            {
                "role": "user",
                "content": json_text(
                    {
                        "rejected_action": turn_memory(row),
                        "rejected_arguments": compact_value(
                            (row["event"].get("action") or {}).get("payload", {})
                        ),
                        "prior_action_response": row["response"],
                        "raw_response_archived": True,
                    }
                ),
            }
        ]
    try:
        assistant = provider.assistant_message(raw)
    except (TypeError, IndexError, KeyError, AttributeError):
        assistant = {}
    if not isinstance(assistant, dict):
        assistant = {}
    calls = assistant.get("tool_calls") or []
    if assistant.get("role") == "assistant" and replayable_tool_calls(calls):
        replies = [
            {"role": "tool", "tool_call_id": c["id"], "content": json_text(row["response"])} for c in calls
        ]
        return [assistant, *(replies or [{"role": "user", "content": json_text(row["response"])}])]
    return [
        {
            "role": "user",
            "content": json_text({"prior_action_response": row["response"], "raw_response_archived": True}),
        }
    ]


def build_messages(provider, store, run_id, packet, system_prompt):
    rows = store.trace(run_id)
    groups = []
    for row in rows:
        if not store.has_accepted(run_id, row["turn"]):
            if row["event"].get("actor") != "environment_capacity":
                raise InfrastructureError("Committed action is missing its provider response")
            groups.append([{"role": "user", "content": json_text(row["response"])}])
        else:
            groups.append(replay_group(provider, store.accepted(run_id, row["turn"]), row))
    profile = getattr(provider, "profile", None)
    keep_from = 0
    while True:
        current = deepcopy(packet)
        if keep_from:
            current["conversation_memory"] = {
                "kind": "deterministic_factual_excerpts_not_complete_reasoning",
                "compacted_turns": [1, keep_from],
                "originals_preserved": True,
                "turns": [turn_memory(row) for row in rows[:keep_from]],
            }
            if packet.get("capacity_index_view"):
                current["conversation_memory"]["turns"] = current["conversation_memory"]["turns"][-8:]
                current["conversation_memory"]["earlier_turns"] = {
                    "count": max(0, keep_from - 8),
                    "read_original": {"section": "history", "offset": 0, "limit": 1},
                }
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json_text(current)},
        ]
        messages.extend(m for group in groups[keep_from:] for m in group)
        if rows:
            state = packet["state"]
            messages.append(
                {
                    "role": "user",
                    "content": json_text(
                        {
                            "current_action_cursor": {
                                "state_version": state["state_version"],
                                "active_forecast_id": state.get("active_forecast_id"),
                                "rounds_remaining": state.get("rounds_remaining"),
                                "interaction_budget": packet.get("interaction_budget"),
                                "repair_pending": state.get("repair_pending"),
                            },
                            "notice": "This is the latest environment state, not another task. Use this state_version for your next action; earlier versions in conversation history are obsolete.",
                        }
                    ),
                }
            )
        if profile is None:
            return messages, {"mode": "test_provider_unbounded", "compacted_rounds": 0}
        count = provider.count_input_tokens(provider.request(messages, Action.model_json_schema()))
        hard_limit = profile.context_window_tokens - profile.max_output_tokens - profile.context_safety_tokens
        trigger = int(hard_limit * profile.context_trigger_fraction)
        # Keep native assistant/tool pairs whole. Preserve the newest pair when it fits the hard limit.
        if count <= trigger or (keep_from >= max(0, len(rows) - 1) and count <= hard_limit):
            return messages, {
                "mode": profile.token_count_mode,
                "input_token_count": count,
                "input_hard_limit": hard_limit,
                "compression_trigger": trigger,
                "compacted_rounds": keep_from,
                "verbatim_rounds": len(rows) - keep_from,
                "message_hash": digest(messages),
            }
        if keep_from < len(rows):
            keep_from += 1
            continue
        raise ContextCapacityExceeded(count, hard_limit)
