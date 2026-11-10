import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace

import pytest
from conftest import final, submit

from castbench.common import (
    AgentError,
    InfrastructureError,
    TransientInfrastructureError,
    load_json,
    write_json,
)
from castbench.models import ModelProfile, ModelService
from castbench.providers import ScriptedProvider
from castbench.runner import Runner
from castbench.scheduler import run_jobs
from castbench.storage import Store


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_concurrency(value):
    with pytest.raises(ValueError):
        run_jobs([], lambda x: x, value)
    with pytest.raises(ValueError):
        ModelService([], None, max_parallel_workers=value)


def test_bounded_parallelism_and_single_coordinator():
    lock, barrier = threading.Lock(), threading.Barrier(2)
    active = peak = 0
    callbacks = []

    def work(i):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        if i < 2:
            barrier.wait(timeout=5)
        time.sleep(0.01)
        with lock:
            active -= 1
        return i

    values = run_jobs(range(8), work, 2, lambda value: callbacks.append(threading.get_ident()))
    assert sorted(values) == list(range(8)) and peak == 2
    assert set(callbacks) == {threading.get_ident()}


def test_failure_drains_active_without_scheduling_more():
    barrier = threading.Barrier(2)
    called, completed = [], []

    def work(i):
        called.append(i)
        barrier.wait(timeout=5)
        if i == 0:
            raise InfrastructureError("injected outage")
        time.sleep(0.1)
        completed.append(i)
        return i

    with pytest.raises(InfrastructureError, match="injected outage"):
        run_jobs(range(10), work, 2)
    assert sorted(called) == [0, 1] and completed == [1]


def test_external_stop_admits_no_jobs():
    stop = threading.Event()
    stop.set()
    with pytest.raises(InfrastructureError, match="stopped"):
        run_jobs([1], lambda _: pytest.fail("must not run"), stop=stop)


def test_transient_failure_requeues_only_its_job_and_preserves_other_progress():
    attempts = {0: 0, 1: 0, 2: 0}
    completed, retries = [], []

    def work(job):
        attempts[job] += 1
        if job == 0 and attempts[job] == 1:
            raise TransientInfrastructureError("temporary disconnect")
        completed.append(job)
        return job

    values = run_jobs(
        range(3),
        work,
        2,
        retry_if=lambda exc: isinstance(exc, TransientInfrastructureError),
        max_retries=2,
        retry_delays=(0.01,),
        on_retry=lambda job, exc, count, delay: retries.append((job, count, delay)),
        job_label=lambda job: f"sample-{job}",
    )
    assert sorted(values) == [0, 1, 2]
    assert attempts == {0: 2, 1: 1, 2: 1}
    assert sorted(completed) == [0, 1, 2]
    assert retries == [(0, 1, 0.01)]


def test_exhausted_transient_job_does_not_block_remaining_jobs_and_has_compact_identity():
    called, exhausted = [], []

    def work(job):
        called.append(job)
        if job == 0:
            raise TransientInfrastructureError("still disconnected")
        return job

    with pytest.raises(InfrastructureError, match="sample-0") as error:
        run_jobs(
            range(3),
            work,
            1,
            retry_if=lambda exc: isinstance(exc, TransientInfrastructureError),
            max_retries=1,
            retry_delays=(0.0,),
            on_retry_exhausted=lambda job, exc, count: exhausted.append((job, count)),
            job_label=lambda job: f"sample-{job}",
            continue_after_retry_exhaustion=True,
        )
    assert called == [0, 0, 1, 2]
    assert exhausted == [(0, 1)]
    assert "PublicBundle" not in str(error.value)


def test_aggregate_failure_stops_admission():
    called = []

    def fail(_):
        raise OSError("cannot persist summary")

    with pytest.raises(InfrastructureError, match="cannot persist summary"):
        run_jobs(range(5), lambda i: called.append(i), 1, on_complete=fail)
    assert called == [0]


def test_coordinator_lock_rejects_duplicate_writer(tmp_path):
    store = Store(tmp_path / "lock")
    with store.writer("evaluation"):
        with pytest.raises(InfrastructureError, match="active writer"):
            with store.writer("evaluation"):
                pytest.fail("duplicate writer")
    with store.writer("evaluation"):
        pass


def test_model_gate_caps_uncached_workers_and_releases_after_error(tmp_path, bundle, monkeypatch):
    profile = ModelProfile(
        name="chronos",
        engine="chronos_original",
        python_executable="fake",
        checkpoint="fake",
        revision="pinned",
        checkpoint_digest="sha256:fake",
        max_context=24,
        max_horizon=30,
        dependency_versions={"torch": "test"},
    )
    service = ModelService([profile], Store(tmp_path), max_parallel_workers=1)
    active = peak = 0
    lock = threading.Lock()
    fail_once = [True]

    def fake_run(command, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(active, peak)
        time.sleep(0.01)
        with lock:
            active -= 1
        if fail_once.pop() if fail_once else False:
            return SimpleNamespace(returncode=1, stderr="injected")
        request = load_json(command[-2])
        write_json(command[-1], {"point_values": [1.0] * request["prediction_length"]})
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("castbench.models.subprocess.run", fake_run)
    with pytest.raises(InfrastructureError):
        service.predict("chronos", bundle.episode)
    episodes = []
    for index in range(4):
        episode = deepcopy(bundle.episode)
        episode.history.target_values[-1] += index + 1
        episodes.append(episode)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda episode: service.predict("chronos", episode), episodes))
    assert all(result["status"] == "ok" for result in results)
    assert peak == 1


def test_npu_pool_isolates_physical_device_and_releases_after_failure(
    tmp_path, bundle, monkeypatch
):
    profile = ModelProfile(
        name="chronos",
        engine="chronos_original",
        python_executable="fake",
        checkpoint="fake",
        revision="pinned",
        checkpoint_digest="sha256:fake",
        max_context=24,
        max_horizon=30,
        dependency_versions={"torch": "test"},
        device="npu:0",
    )
    service = ModelService([profile], Store(tmp_path), npu_devices=[3])
    environments = []

    def fake_run(command, **kwargs):
        environments.append(kwargs["env"])
        if len(environments) == 1:
            return SimpleNamespace(returncode=1, stderr="injected")
        request = load_json(command[-2])
        write_json(command[-1], {"point_values": [1.0] * request["prediction_length"]})
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("castbench.models.subprocess.run", fake_run)
    with pytest.raises(InfrastructureError):
        service.predict("chronos", bundle.episode)
    assert service.predict("chronos", bundle.episode)["status"] == "ok"
    assert [env["ASCEND_RT_VISIBLE_DEVICES"] for env in environments] == ["3", "3"]
    assert all(env["CUDA_VISIBLE_DEVICES"] == "" for env in environments)


def test_cuda_worker_enables_deterministic_cublas_workspace(tmp_path, bundle, monkeypatch):
    profile = ModelProfile(
        name="chronos",
        engine="chronos_original",
        python_executable="fake",
        checkpoint="fake",
        revision="pinned",
        checkpoint_digest="sha256:fake",
        max_context=24,
        max_horizon=30,
        dependency_versions={"torch": "test"},
        device="cuda:0",
    )
    service = ModelService([profile], Store(tmp_path))
    environments = []

    def fake_run(command, **kwargs):
        environments.append(kwargs["env"])
        request = load_json(command[-2])
        write_json(command[-1], {"point_values": [1.0] * request["prediction_length"]})
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("castbench.models.subprocess.run", fake_run)
    assert service.predict("chronos", bundle.episode)["status"] == "ok"
    assert environments[0]["ASCEND_RT_VISIBLE_DEVICES"] == ""
    assert environments[0]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_parallel_runners_isolate_state_forecast_alarm_and_grants(tmp_path, bundle):
    store = Store(tmp_path / "shared")

    def work(i):
        values = [float(20 + i)] * bundle.episode.task.horizon_length
        actions = [submit(bundle, values), final()]
        if i == 0:
            actions.insert(0, submit(bundle, values[:-1]))
        runner = Runner(bundle, store, ScriptedProvider(actions), f"sample{i}", protocol_repair=True)
        runner.run()
        assert store.replay(f"sample{i}") == runner.state()
        return runner.state()

    states = run_jobs(range(4), work, 2)
    assert sorted(s["forecasts"]["forecast_1"]["scored_values"][0] for s in states) == [20, 21, 22, 23]
    assert len(store.get("sample0")["protocol_alarms"]) == 1
    assert not store.get("sample1")["protocol_alarms"]
    ref = store.put_blob({"private_to_sample": 0})
    with pytest.raises(AgentError):
        store.read_blob("sample1", ref)


def test_local_capacity_rejects_shrunken_slot(monkeypatch):
    from castbench.providers import ProviderProfile
    from scripts.run_local_joint_probe import check_local_capacity

    profile = ProviderProfile(
        endpoint="http://localhost:18080/v1/chat/completions",
        model="qwen",
        api_key_env="CASTBENCH_TEST_KEY",
        token_count_mode="llama_cpp",
    )
    monkeypatch.setenv("CASTBENCH_TEST_KEY", "test")

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"total_slots": 2, "default_generation_settings": {"n_ctx": 16384}}

    monkeypatch.setattr("scripts.run_local_joint_probe.httpx.get", lambda *a, **k: Response())
    with pytest.raises(InfrastructureError, match="context per slot"):
        check_local_capacity(profile, 2)


def test_local_capacity_allows_queued_episodes(monkeypatch):
    from castbench.providers import ProviderProfile
    from scripts.run_local_joint_probe import check_local_capacity

    profile = ProviderProfile(
        endpoint="http://localhost:18080/v1/chat/completions",
        model="qwen",
        api_key_env="CASTBENCH_TEST_KEY",
        token_count_mode="llama_cpp",
    )
    monkeypatch.setenv("CASTBENCH_TEST_KEY", "test")

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"total_slots": 4, "default_generation_settings": {"n_ctx": 49152}}

    monkeypatch.setattr("scripts.run_local_joint_probe.httpx.get", lambda *a, **k: Response())
    result = check_local_capacity(profile, 50)
    assert result["slots"] == 4
    assert result["queued_episodes"] == 46
