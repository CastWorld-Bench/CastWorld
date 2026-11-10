"""Public input archives must not become a channel for private experiment exports."""
import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location('public_scope', Path(__file__).resolve().parents[1] / 'tools/verify_release_scope.py')
SCOPE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCOPE)


def test_allowed_resources_exclude_experiment_records():
    assert SCOPE.allowed_member('datasets/example/data.csv')
    assert SCOPE.allowed_member(SCOPE.DATA+'evaluation/A/example/label.json')
    assert SCOPE.allowed_member(SCOPE.DATA+'runtime/native_cache.jsonl')
    for name in ('results/scores.json', 'outputs/paper/episodes/model/trace.json',
                 SCOPE.DATA+'evaluation/B/example/label.json', SCOPE.DATA+'test_labels/B/example/label.json',
                 'datasets/example/provenance/old_design.md', 'datasets/example/domain_design.md',
                 'outputs/experiment/reviewer_ratings.xlsx', 'release/implementation/raw_response.json'):
        assert not SCOPE.allowed_member(name)


def test_rejects_independent_result_exports():
    assert SCOPE.private_exports({'README.md', 'site/index.html', 'docs/data/dataset_inventory.csv'}) == []
    assert len(SCOPE.private_exports({'results/table.csv', 'site/data/lite.csv', 'site/data/leaderboards.json',
                                     'release/validation/old.xml'})) == 4


def test_source_archive_ignores_unpublished_bytecode(tmp_path):
    (tmp_path/'README.md').write_text('Public')
    (tmp_path/'__pycache__').mkdir()
    (tmp_path/'__pycache__/private.pyc').write_bytes(b'cache')
    assert SCOPE.published_paths(tmp_path) == {'README.md'}
