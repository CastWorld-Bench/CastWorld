import argparse
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

from conftest import final, submit


@pytest.fixture
def portable():
    source = Path(__file__).resolve().parents[1] / 'tools/reproduce.py'
    spec = importlib.util.spec_from_file_location('portable_reproduction_test', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def offline_execution(tmp_path, monkeypatch, portable, bundle):
    from castbench.catalog import Registry
    monkeypatch.setenv('PORTABLE_TEST_KEY', 'not-a-real-key')
    provider = tmp_path/'provider.json'
    portable.write(provider, {'endpoint': 'https://invalid.example/v1/chat/completions',
                              'model': 'scripted-offline-test', 'api_key_env': 'PORTABLE_TEST_KEY',
                              'context_window_tokens': 1000000})
    sample, dataset = bundle.episode.episode_id, bundle.episode.dataset_id
    monkeypatch.setattr(portable, 'cohort', lambda which: {dataset: [sample]})
    monkeypatch.setattr(portable, 'load_bundles', lambda *args: {sample: (bundle, Registry())})
    monkeypatch.setattr(portable, 'import_cache', lambda store: None)
    monkeypatch.setattr(portable, 'cached_service', lambda store: None)
    monkeypatch.setattr(portable, 'verify_cached_candidates', lambda service, jobs: len(jobs)*4)
    monkeypatch.setattr(portable, 'policy', lambda *args: None)
    return argparse.Namespace(allow_api=True, out=tmp_path/'run', provider=provider,
        cohort='A', dataset=None, workers=1, active_requests=1, rpm=600000,
        deployment='offline scripted test, no network', limit=None, retry_failed=False)


def tiny_release(portable, root, name, content):
    (root / 'release/assets').mkdir(parents=True)
    archive = root / 'release/assets/test.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr(name, content)
    portable.write(root / 'release/manifest.json', {'assets': [
        {'path': 'release/assets/test.zip', 'sha256': portable.sha(archive)}],
        'files': {name: {'bytes': len(content), 'sha256': portable.hashlib.sha256(content).hexdigest()}},
        'datasets': ['test']})


def test_prepare_checks_bytes_and_is_idempotent(tmp_path, monkeypatch, portable):
    monkeypatch.setattr(portable, 'ROOT', tmp_path)
    tiny_release(portable, tmp_path, 'frozen/input.json', b'{"a":1}')
    assert portable.prepare()['verified_members'] == 1
    assert portable.prepare()['api_calls'] == 0
    (tmp_path / 'frozen/input.json').write_bytes(b'changed')
    with pytest.raises(FileExistsError):
        portable.prepare()


def test_prepare_rejects_archive_traversal(tmp_path, monkeypatch, portable):
    monkeypatch.setattr(portable, 'ROOT', tmp_path)
    tiny_release(portable, tmp_path, '../escape.json', b'{}')
    with pytest.raises(ValueError, match='Unsafe archive'):
        portable.prepare()
    assert not (tmp_path.parent / 'escape.json').exists()


def test_generation_requires_opt_in(tmp_path, portable):
    with pytest.raises(PermissionError):
        portable.run(argparse.Namespace(allow_api=False, out=tmp_path))


def test_live_native_requires_explicit_local_profiles(portable):
    with pytest.raises(ValueError, match='--model-profiles'):
        portable.live_service(None, None)


def test_live_native_preflights_user_deployments(tmp_path, monkeypatch, portable):
    from castbench import models
    raw = json.loads((Path(__file__).resolve().parents[1] / 'release/manifest.json').read_text())
    assert len(raw['datasets']) == 25
    profiles = tmp_path / 'models.json'
    profiles.write_text('[{"name":"user-deployment"}]')
    class Profile:
        @staticmethod
        def model_validate(value):
            return value
    class Service:
        def __init__(self, items, store, **kwargs):
            self.items = items
            self.checked = False
        def preflight(self):
            self.checked = True
    monkeypatch.setattr(models, 'ModelProfile', Profile)
    monkeypatch.setattr(models, 'ModelService', Service)
    service = portable.live_service(None, profiles)
    assert service.checked and service.items == [{'name': 'user-deployment'}]


def test_published_source_fingerprint_rejects_edits(tmp_path, monkeypatch, portable):
    monkeypatch.setattr(portable, 'ROOT', tmp_path)
    source = tmp_path / 'src/worker.py'
    source.parent.mkdir()
    source.write_text('original', encoding='utf-8')
    portable.write(tmp_path / 'release/source_integrity.json', {
        'files': {'src/worker.py': {'sha256': portable.sha(source)}}})
    portable.verify_source_tree()
    source.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='Published runtime was modified'):
        portable.verify_source_tree()


@pytest.mark.parametrize('board', ['A', 'B'])
def test_scripted_portable_run_and_resume(tmp_path, monkeypatch, portable, bundle, board):
    from castbench.catalog import Registry
    from castbench.providers import OpenAICompatibleProvider, ScriptedProvider
    monkeypatch.setenv('PORTABLE_TEST_KEY', 'not-a-real-key')
    provider = tmp_path / 'provider.json'
    portable.write(provider, {'endpoint': 'https://invalid.example/v1/chat/completions',
                              'model': 'scripted-offline-test', 'api_key_env': 'PORTABLE_TEST_KEY',
                              'context_window_tokens': 1000000})
    scripted = ScriptedProvider([submit(bundle), final()])
    monkeypatch.setattr(OpenAICompatibleProvider, 'respond', lambda self, request: scripted.respond(request))
    sample = bundle.episode.episode_id
    dataset = bundle.episode.dataset_id
    monkeypatch.setattr(portable, 'cohort', lambda which: {dataset: [sample]})
    monkeypatch.setattr(portable, 'load_bundles', lambda *args: {sample: (bundle, Registry())})
    monkeypatch.setattr(portable, 'import_cache', lambda store: None)
    monkeypatch.setattr(portable, 'cached_service', lambda store: None)
    monkeypatch.setattr(portable, 'verify_cached_candidates', lambda service, jobs: len(jobs) * 4)
    monkeypatch.setattr(portable, 'policy', lambda *args: None)
    args = argparse.Namespace(allow_api=True, out=tmp_path / 'run', provider=provider,
        cohort=board, dataset=None, workers=1, active_requests=1, rpm=600000,
        deployment='offline scripted test, no network', limit=None, retry_failed=False)
    result = portable.run(args)
    assert result == {'planned': 1, 'processed': 1, 'valid': 1}
    assert portable.run(args) == result
    trace_path = next((args.out / 'episodes').rglob('trace_a1.json'))
    trace = json.loads(trace_path.read_text(encoding='utf-8'))
    assert trace['state']['final']
    assert 'evaluator_private' not in json.dumps(trace['config'])
    assert len(list((args.out / 'episodes').rglob('attempt_*.json'))) == 1
    args.deployment = 'changed deployment'
    with pytest.raises(ValueError, match='Configuration drift'):
        portable.run(args)


def test_finished_failure_is_retained_and_not_retried(monkeypatch, portable, bundle, offline_execution):
    from castbench.providers import OpenAICompatibleProvider, ScriptedProvider
    bundle.budget.max_rounds = 1
    scripted = ScriptedProvider([final()])
    calls = []
    def respond(self, request):
        calls.append(request)
        return scripted.respond(request)
    monkeypatch.setattr(OpenAICompatibleProvider, 'respond', respond)
    args = offline_execution
    assert portable.run(args) == {'planned': 1, 'processed': 1, 'valid': 0}
    args.retry_failed = True
    assert portable.run(args) == {'planned': 1, 'processed': 1, 'valid': 0}
    assert len(calls) == 1
    directory = next((args.out/'episodes'/bundle.episode.dataset_id).iterdir())
    done = portable.read(directory/'complete.json')
    assert done['status'] == 'finished' and not done['legal_delivery']
    assert len(list(directory.glob('attempt_*.json'))) == 1
    trace = portable.read(args.out/done['trace'])
    assert trace['state']['status'] == 'finished'


def test_infrastructure_retry_resumes_same_state_and_preserves_receipts(monkeypatch, portable, bundle, offline_execution):
    from castbench.common import InfrastructureError
    from castbench.providers import OpenAICompatibleProvider, ScriptedProvider
    scripted = ScriptedProvider([submit(bundle), final()])
    calls = []
    def respond(self, request):
        calls.append(request)
        if len(calls) == 2:
            raise InfrastructureError('synthetic transport interruption')
        return scripted.respond(request)
    monkeypatch.setattr(OpenAICompatibleProvider, 'respond', respond)
    args = offline_execution
    assert portable.run(args)['valid'] == 0
    directory = next((args.out/'episodes'/bundle.episode.dataset_id).iterdir())
    failed = portable.read(directory/'attempt_0001.json')
    assert failed['status'] == 'needs_recovery'
    partial = portable.read(args.out/failed['trace'])
    assert len(partial['trace']) == 1
    args.retry_failed = True
    assert portable.run(args)['valid'] == 1
    done = portable.read(directory/'complete.json')
    assert done['run_id'] == failed['run_id']
    assert done['recovery_of_attempt'] == 1
    trace = portable.read(args.out/done['trace'])
    assert len(trace['trace']) == 2
    assert trace['trace'][0] == partial['trace'][0]
    assert len(trace['provider_archive']) == 3
    assert portable.read(directory/'attempt_0001.json') == failed


def test_unknown_initialization_failure_is_isolated_for_review(monkeypatch, portable, bundle, offline_execution):
    from castbench import runner
    class BrokenRunner:
        def __init__(self, *args, **kwargs):
            raise ValueError('synthetic local configuration failure')
    monkeypatch.setattr(runner, 'Runner', BrokenRunner)
    args = offline_execution
    assert portable.run(args)['valid'] == 0
    directory = next((args.out/'episodes'/bundle.episode.dataset_id).iterdir())
    failure = portable.read(directory/'attempt_0001.json')
    assert failure['status'] == 'needs_review'
    assert failure['trace_export_error_type'] == 'TypeError'
    args.retry_failed = True
    assert portable.run(args)['valid'] == 0
    assert len(list(directory.glob('attempt_*.json'))) == 1
