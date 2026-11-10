from copy import deepcopy

import pytest
from conftest import call, final, run_actions, submit
from pydantic import ValidationError

from castbench.catalog import Registry
from castbench.common import digest, write_json
from castbench.contracts import Contract, PublicBundle
from castbench.data import build_packages, synthetic_fixture
from castbench.domain_packs import load_domain_episode, load_domain_episodes
from castbench.domains import TOOLS, domain_registry
from castbench.tools import ToolView, execute_readonly


def test_history_only_masked_before_serialization(tmp_path):
    frame, config = synthetic_fixture()
    config["variables"][0]["visibility"] = "history_only"
    build_packages(frame, config, tmp_path / "public", tmp_path / "private")
    from castbench.common import load_json

    b = PublicBundle.model_validate(load_json(next((tmp_path / "public/dev").glob("*.json"))))
    h = b.episode.task.lookback_length
    assert b.variables[0].values[:h] == frame.temperature.iloc[133:181].tolist()
    assert b.variables[0].values[h:] == [None] * 14
    assert all(x is not None for x in b.samples[0].context["temperature"])
    view = ToolView(b, {}, None, "test")
    r = execute_readonly(
        "context.get_variable_slice",
        Registry().validate(
            "context.get_variable_slice", {"variables": ["temperature"], "scope": {"timeline": "scored"}}
        ),
        view,
        Registry(),
    )
    assert r["structured_result"]["variables"]["temperature"]["values"] == [None] * 12


def test_hidden_future_rejected_at_contract_boundary(bundle):
    raw = bundle.model_dump()
    raw["variables"][0]["visibility"] = "history_only"
    with pytest.raises(ValidationError, match="mask gap and future"):
        PublicBundle.model_validate(raw)


def test_hidden_event_alias_rejected(bundle):
    raw = bundle.model_dump()
    h = raw["episode"]["task"]["lookback_length"]
    raw["variables"][0]["visibility"] = "history_only"
    raw["variables"][0]["values"][h:] = [None] * (len(raw["variables"][0]["values"]) - h)
    ts = raw["episode"]["forecast_timeline"]["scored_timestamps"]
    raw["events"] = [
        dict(
            event_id="e",
            type="temperature",
            start=ts[0],
            end=ts[-1],
            description="hot",
            source="computed",
            target_derived=False,
            depends_on=["temperature"],
        )
    ]
    with pytest.raises(ValidationError, match="visible dependencies"):
        PublicBundle.model_validate(raw)


@pytest.mark.parametrize("domain", list(TOOLS))
def test_domain_manifest_uses_six_families(domain):
    r = domain_registry(domain)
    assert len(r.extensions) == 5
    assert all(r.manifests[k]["category"] in {"context", "feature"} for k in r.extensions)
    assert all(k.startswith(f"domain.{domain}.") for k in r.extensions)
    with pytest.raises(ValueError):
        domain_registry("untrusted.import")
    history_id = next(k for k in r.extensions if ".history_" in k)
    assert r.validate(history_id, {})["scope"]["timeline"] == "history"
    assert r.manifests[history_id]["reads_target_history"]


def test_feature_extension_does_not_require_generic_series_ref(tmp_path, bundle):
    r = Registry()
    r.register_extension(
        "domain.summary",
        Contract,
        lambda b, p: {
            "structured_result": {"n": 1},
            "observed_scope": {"timeline": "history", "variables": ["load"]},
            "units": {"load": "unit"},
        },
        "Test feature extension",
        "sha256:test",
        category="feature",
    )
    store, runner = run_actions(
        tmp_path, bundle, [call("domain.summary"), submit(bundle), final()], registry=r
    )
    obs = runner.state()["observations"]["observation_1"]
    assert obs["scope"]["timeline"] == "history"
    assert store.read_blob("test", obs["artifact_ref"])["units"] == {"load": "unit"}


def domain_bundle(bundle, domain, values):
    b = bundle.model_dump()
    b["episode"]["domain_id"] = domain
    n = (
        len(b["episode"]["history"]["target_values"])
        + len(b["episode"]["forecast_timeline"]["gap_timestamps"])
        + len(b["episode"]["forecast_timeline"]["scored_timestamps"])
    )
    b["samples"] = []
    b["variables"] = [
        dict(
            name=k,
            dtype="numeric",
            unit=u,
            description=k,
            target_derived=False,
            visibility="benchmark_known_future",
            lineage="synthetic unit fixture",
            values=([x] * n if not isinstance(x, list) else x),
        )
        for k, (x, u) in values.items()
    ]
    return PublicBundle.model_validate(b)


def execute(b, name, args=None):
    r = domain_registry(b.episode.domain_id)
    tool_id = f"domain.{b.episode.domain_id}.{name}"
    return execute_readonly(tool_id, r.validate(tool_id, args or {}), ToolView(b, {}, None, "test"), r)


def test_gef_degree_hours_formula(bundle):
    b = domain_bundle(bundle, "gef", {"temperature_2m_f": (75.0, "degF")})
    x = execute(b, "degree_hours")["structured_result"]
    assert x["heating_degree_hours_f"] == 0
    assert x["cooling_degree_hours_f"] == 120


def test_solar_radiation_does_not_add_dni(bundle):
    b = domain_bundle(
        bundle,
        "sunny",
        {
            "shortwave_radiation_fused": (100.0, "W/m2"),
            "diffuse_radiation_fused": (20.0, "W/m2"),
            "direct_radiation_fused": (80.0, "W/m2"),
        },
    )
    x = execute(b, "radiation_budget")["structured_result"]
    assert x["ghi_irradiation_wh_m2"] == 1200
    assert x["diffuse_fraction_at_ghi_gt1"]["mean"] == pytest.approx(0.2)
    assert x["ghi_minus_direct_minus_diffuse_wm2"]["mean"] == 0


def test_wind_shear_and_circle(bundle):
    b = domain_bundle(
        bundle,
        "windy",
        {"hf_wind_speed_80m_fused": (8.0, "km/h"), "hf_wind_speed_120m_fused": (12.0, "km/h")},
    )
    assert execute(b, "vertical_shear")["structured_result"]["power_law_alpha"]["mean"] == pytest.approx(1)
    b = domain_bundle(
        bundle,
        "windy",
        {
            "station_wind_direction_80m_fused": ([359.0, 1.0] * 31, "degree"),
            "station_wind_direction_80m_resultant": (1.0, "fraction"),
        },
    )
    angle = execute(b, "direction_coherence")["structured_result"]["temporal_circular_mean_deg"]
    assert min(angle, 360 - angle) < 1e-8


def test_hidden_domain_input_produces_no_false_zero(bundle):
    b = domain_bundle(bundle, "gef", {"temperature_2m_f": (75.0, "degF")}).model_dump()
    b["variables"][0]["visibility"] = "history_only"
    b["variables"][0]["values"][48:] = [None] * 14
    result = execute(PublicBundle.model_validate(b), "degree_hours")
    assert result["status"] == "not_applicable"
    assert set(result["structured_result"]) == {"unavailable_inputs"}


def test_wrong_units_and_wrong_domain_rejected(bundle):
    b = domain_bundle(bundle, "gef", {"temperature_2m_f": (75.0, "degC")})
    with pytest.raises(Exception, match="Expected unit"):
        execute(b, "degree_hours")
    r = domain_registry("bike")
    with pytest.raises(Exception, match="does not belong"):
        execute_readonly(
            "domain.bike.weather_units",
            r.validate("domain.bike.weather_units", {}),
            ToolView(b, {}, None, "test"),
            r,
        )


def test_domain_pack_hash_and_path_verification(tmp_path, bundle):
    r = domain_registry("bike")
    raw = bundle.model_dump()
    raw["episode"]["domain_id"] = "bike"
    raw["samples"] = []
    shared = {"training_cases": [], "knowledge": []}
    write_json(tmp_path / "shared.json", shared)
    write_json(tmp_path / "test.json", raw)
    tool_hash = digest({k: v for k, v in r.manifests.items() if k in r.extensions})
    manifest = {
        "schema_version": "castbench.domain_pack/1.0",
        "status": "ready",
        "domain_id": "bike",
        "episodes": [
            {"episode_id": raw["episode"]["episode_id"], "path": "test.json", "sha256": digest(raw)}
        ],
        "shared": {"path": "shared.json", "sha256": digest(shared)},
        "tool_manifest_hash": tool_hash,
    }
    write_json(tmp_path / "manifest.json", manifest)
    b, _ = load_domain_episode(tmp_path, raw["episode"]["episode_id"])
    assert b.episode.domain_id == "bike"
    second = deepcopy(raw)
    second["episode"]["episode_id"] += ".second"
    write_json(tmp_path / "second.json", second)
    manifest["episodes"].append(
        {
            "episode_id": second["episode"]["episode_id"],
            "path": "second.json",
            "sha256": digest(second),
        }
    )
    write_json(tmp_path / "manifest.json", manifest)
    ids = [raw["episode"]["episode_id"], second["episode"]["episode_id"]]
    batched = load_domain_episodes(tmp_path, ids)
    assert list(batched) == ids
    assert batched[ids[0]][0].episode == b.episode
    assert batched[ids[0]][0].samples is not batched[ids[1]][0].samples
    assert batched[ids[0]][0].knowledge is not batched[ids[1]][0].knowledge
    with pytest.raises(ValueError, match="nonempty and unique"):
        load_domain_episodes(tmp_path, [])
    manifest["episodes"].pop()
    bad = deepcopy(manifest)
    bad["shared"]["path"] = "../elsewhere.json"
    write_json(tmp_path / "manifest.json", bad)
    with pytest.raises(ValueError, match="inside the public pack"):
        load_domain_episode(tmp_path, raw["episode"]["episode_id"])
    write_json(tmp_path / "manifest.json", manifest)
    write_json(tmp_path / "shared.json", {"training_cases": [], "knowledge": [{"untrusted": "changed"}]})
    with pytest.raises(ValueError, match="hash mismatch"):
        load_domain_episode(tmp_path, raw["episode"]["episode_id"])
