from pathlib import Path

from castbench.common import load_json, write_json
from scripts.run_local_joint_probe import refresh_summary


def test_subset_rerun_keeps_all_completed_summary_rows(tmp_path):
    for eid in ("bike.one", "gef.one"):
        root = tmp_path / eid
        write_json(
            root / "trace.json",
            {"config": {"episode_id": eid}, "state": {"final": {"forecast_id": "forecast_1"}}},
        )
        write_json(root / "score.json", {"completed": True})
    write_json(tmp_path / "sunny.one" / "paused_trace.json", {"state": {"status": "running"}})
    write_json(tmp_path / "summary.json", [{"episode_id": "gef.one"}])
    refresh_summary(Path(tmp_path))
    assert [x["episode_id"] for x in load_json(tmp_path / "summary.json")] == ["bike.one", "gef.one"]
    progress = load_json(tmp_path / "progress.json")
    assert progress["llm_interaction_rounds"] is None
    assert progress["completion_rescues"] == 0
    assert progress["successful_completion_rescues"] == 0
    assert progress["failed_completion_rescues"] == 0
