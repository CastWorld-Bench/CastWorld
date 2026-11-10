"""Public A/B inference never opens B labels or falls back to paper labels."""

import argparse
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("public_runner", ROOT / "tools/reproduce.py")
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def test_old_paper_cohorts_are_not_aliased_to_different_tasks():
    for name in ("full", "lite"):
        with pytest.raises(ValueError, match="A and B"):
            RUN.cohort(name)


def test_B_scoring_requires_explicit_external_labels(tmp_path):
    RUN.write(tmp_path / "run_manifest.json", {"cohort": "B", "selected_ids": {}})
    with pytest.raises(PermissionError, match="organizer-held"):
        RUN.score(argparse.Namespace(out=tmp_path, labels_root=None))


def test_A_scoring_rejects_replacement_labels(tmp_path):
    RUN.write(tmp_path / "run_manifest.json", {"cohort": "A", "selected_ids": {}})
    with pytest.raises(ValueError, match="only used"):
        RUN.score(argparse.Namespace(out=tmp_path, labels_root=tmp_path))


def test_bundles_have_no_hidden_label_export():
    assert not (ROOT / "benchmark_data/evaluation/B").exists()
    assert not (ROOT / "benchmark_data/test_labels/B").exists()


@pytest.mark.parametrize('board,wrong', [('A', 125), ('B', 125)])
def test_retired_board_sizes_are_rejected(monkeypatch, board, wrong):
    names = [f'dataset_{i}' for i in range(25)]
    monkeypatch.setattr(RUN, 'cohort', lambda _: {name: list(range(wrong // 25)) for name in names})
    with pytest.raises(ValueError, match='Episode count mismatch'):
        RUN.audit(board, cache=False)


def test_boundary_check_normalizes_timestamp_resolution():
    spec = importlib.util.spec_from_file_location('public_boundary', ROOT/'tools/verify_public_data.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hidden = set(module.utc(['2020-01-02T00:00:00Z']).asi8)
    times = module.utc(['2020-01-02']).as_unit('us')
    with pytest.raises(ValueError, match='exposed'):
        module.disjoint(times, hidden, 'test')
    module.disjoint(['2020-01-01'], hidden, 'test')


def test_missing_cache_preflight_does_not_start_llm():
    from types import SimpleNamespace
    from castbench.common import InfrastructureError
    calls = []
    class Missing:
        def predict(self, name, episode):
            calls.append(name)
            raise InfrastructureError('absent')
    jobs = [('dataset', 'sample', SimpleNamespace(episode=object()), None)]
    with pytest.raises(InfrastructureError, match='No LLM requests were started'):
        RUN.verify_cached_candidates(Missing(), jobs)
    assert calls == ['chronos', 'timesfm', 'sundial', 'moirai']
