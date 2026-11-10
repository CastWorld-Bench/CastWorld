from castbench.interaction_diagnostics import interaction_diagnostics


def test_diagnostics_count_information_between_versions_without_scoring_it():
    forecasts = {
        "model": {"kind": "model_candidate", "model_tool": "model.chronos", "scored_values": [1, 2]},
        "v1": {"kind": "agent", "created_turn": 2, "scored_values": [1, 2]},
        "v2": {
            "kind": "agent",
            "created_turn": 4,
            "scored_values": [2, 3],
            "previous_agent_forecast_id": "v1",
        },
        "v3": {
            "kind": "agent",
            "created_turn": 6,
            "scored_values": [2, 3],
            "previous_agent_forecast_id": "v2",
        },
    }
    trace = [
        {
            "turn": 1,
            "event": {
                "valid": True,
                "tool_records": [{"tool_id": "model.chronos", "status": "ok", "information": True}],
            },
        },
        {
            "turn": 3,
            "event": {
                "valid": True,
                "tool_records": [
                    {"tool_id": "context.calendar_profile", "status": "ok", "information": True}
                ],
            },
        },
        {
            "turn": 5,
            "event": {
                "valid": True,
                "tool_records": [
                    {"tool_id": "knowledge.explain_variable", "status": "partial", "information": True}
                ],
            },
        },
        {
            "turn": 7,
            "event": {
                "valid": False,
                "tool_records": [
                    {"tool_id": "feature.basic_statistics", "status": "error", "information": True}
                ],
            },
        },
    ]
    export = {
        "state": {"forecasts": forecasts, "final": {"forecast_id": "v3"}, "evidence": {}},
        "trace": trace,
    }
    result = interaction_diagnostics(export)
    assert result["successful_substantive_tool_calls"] == 3
    assert result["repeated_successful_substantive_tool_calls"] == 0
    assert result["information_tool_calls"] == 3
    assert result["nonmodel_information_tool_calls"] == 2
    assert result["new_information_then_changed_prediction"] == 1
    assert result["agent_forecast_versions"] == 3
    assert result["final_identical_model_tools"] == []
    export["state"]["final"] = {"forecast_id": "v1"}
    assert interaction_diagnostics(export)["final_identical_model_tools"] == ["model.chronos"]


def test_failed_or_empty_episode_has_no_implicit_prediction():
    result = interaction_diagnostics({"state": {"forecasts": {}, "final": None, "evidence": {}}, "trace": []})
    assert result["successful_substantive_tool_calls"] == 0
    assert result["repeated_successful_substantive_tool_calls"] == 0
    assert result["agent_forecast_versions"] == 0
    assert result["final_identical_model_tools"] == []
    assert result["new_information_then_changed_prediction"] == 0
    export = {
        "state": {"forecasts": {}, "final": None, "evidence": {}},
        "trace": [{"event": {"valid": False, "actor": "environment_capacity"}}],
    }
    assert interaction_diagnostics(export)["invalid_action_rounds"] == 0


def test_repeated_information_is_not_counted_as_new_evidence():
    export = {"state": {"forecasts": {}, "final": None, "evidence": {}}, "trace": []}
    for turn, args_hash in enumerate(["same", "same", "different"], 1):
        export["trace"].append(
            {
                "turn": turn,
                "event": {
                    "valid": True,
                    "action": {"action_type": "CALL_TOOL"},
                    "tool_records": [
                        {
                            "tool_id": "feature.basic_statistics",
                            "status": "ok",
                            "information": True,
                            "arguments_hash": args_hash,
                        }
                    ],
                },
            }
        )
    result = interaction_diagnostics(export)
    assert result["successful_substantive_tool_calls"] == 3
    assert result["repeated_successful_substantive_tool_calls"] == 1
    assert result["repeated_successful_information_calls"] == 1
    assert result["information_tool_calls"] == 3
    assert result["invalid_action_rounds"] == 0
