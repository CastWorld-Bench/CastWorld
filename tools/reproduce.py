"""Portable, explicit-opt-in evaluation of the public CastWorld A/B tasks."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import threading
import time
import zipfile
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'benchmark_data'
sys.path.insert(0, str(ROOT / 'src'))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + str(threading.get_ident()) + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data := stream.read(4 * 1024**2):
            h.update(data)
    return h.hexdigest()


def verify_source_tree():
    path = ROOT / 'release/source_integrity.json'
    if path.exists():
        for name, record in read(path)['files'].items():
            if sha(ROOT / name) != record['sha256']:
                raise ValueError('Published runtime was modified: ' + name)


def prepare():
    manifest = read(ROOT / 'release/manifest.json')
    count = 0
    windows_paths = {}
    for asset in manifest['assets']:
        path = ROOT / asset['path']
        if sha(path) != asset['sha256']:
            raise ValueError('Archive checksum mismatch: ' + asset['path'])
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                original_name = member.filename
                portable_name = original_name
                if os.name == 'nt':
                    portable_name = '/'.join(quote(part, safe=' !#$&\'()+,-.;=@[]^_`{}~')
                                             if any(c in part for c in '<>:"|?*') else part
                                             for part in original_name.split('/'))
                    if portable_name != original_name:
                        windows_paths[original_name] = portable_name
                target = (ROOT / portable_name).resolve()
                if not target.is_relative_to(ROOT) or member.filename.startswith(('/', '\\')):
                    raise ValueError('Unsafe archive member')
                if member.is_dir():
                    continue
                expected = manifest['files'][member.filename]
                data = archive.read(member)
                if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
                    raise ValueError('Member checksum mismatch: ' + member.filename)
                if member.filename in manifest.get('materialization_exclusions', []):
                    count += 1
                    continue
                if target.exists():
                    if sha(target) != expected['sha256']:
                        raise FileExistsError('Refusing to overwrite a modified file: ' + member.filename)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                count += 1
        print(json.dumps({'prepared_archive': asset['path'], 'verified_members': count}), flush=True)
    if windows_paths:
        write(ROOT / '.reproduction/windows_paths.json', windows_paths)
    return {'verified_members': count, 'datasets': len(manifest['datasets']), 'api_calls': 0}


def cohort(which):
    if which not in {'A', 'B'}:
        raise ValueError('The public release supplies cohorts A and B')
    return read(DATA / 'cohorts' / (which+'.json'))


def pack(which, dataset):
    return DATA / 'packs' / which / dataset


def policy(which, dataset):
    return read(DATA / 'policies' / which / (dataset + '.json'))


def profiles(which, datasets):
    root = DATA / 'efficiency_profiles'
    return {dataset: read(root / (dataset + '.json')) for dataset in datasets}


def import_cache(store):
    source = DATA / 'runtime/native_cache.jsonl'
    with store.connection() as connection, source.open(encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            value = json.dumps(record['value'], ensure_ascii=False, allow_nan=False)
            existing = connection.execute('SELECT value FROM model_cache WHERE key=?', (record['key'],)).fetchone()
            if existing is not None and json.loads(existing[0]) != record['value']:
                raise ValueError('Different native output under the same immutable cache key')
            connection.execute('INSERT OR IGNORE INTO model_cache VALUES(?,?)', (record['key'], value))


def cached_service(store):
    # Public native-tool candidates are optional; --native-mode live regenerates them with local models.
    from castbench.common import InfrastructureError
    from castbench.models import ModelProfile, ModelService
    cached = store.cached_model

    def only_cached(key, value=None):
        result = cached(key, value)
        if result is None:
            raise InfrastructureError('Frozen native forecast is absent; live substitution is disabled')
        return result

    store.cached_model = only_cached
    # These historical paths are part of the cache identity, NOT accessed in cache-only mode.
    return ModelService([ModelProfile.model_validate(p) for p in read(DATA / 'runtime/native_models.json')], store,
                        max_parallel_workers=1)


def live_service(store, model_profiles):
    from castbench.models import ModelProfile, ModelService
    if model_profiles is None:
        raise ValueError('Live native inference requires --model-profiles with four local model deployments')
    service = ModelService([ModelProfile.model_validate(p) for p in read(model_profiles)], store,
                           max_parallel_workers=1)
    service.preflight()
    return service


def verify_cached_candidates(service, jobs):
    from castbench.common import InfrastructureError
    missing = []
    for dataset, sample, bundle, _ in jobs:
        for model in ('chronos', 'timesfm', 'sundial', 'moirai'):
            try:
                service.predict(model, bundle.episode)
            except InfrastructureError:
                missing.append((dataset, sample, model))
    if missing:
        raise InfrastructureError(
            f'Native cache preflight failed: {len(missing)} candidates unavailable. '
            'No LLM requests were started. Supply complete matching caches or use '
            '--native-mode live --model-profiles /absolute/path/models.json. '
            f'First unavailable candidate: {missing[0]}'
        )
    return len(jobs) * 4


def audit(which, cache=True):
    from castbench.domain_packs import load_domain_episodes
    from castbench.efficiency import ResourcePolicy, EfficiencyProfile
    from castbench.storage import Store
    ids = cohort(which)
    if len(ids) != 25 or any('GEFCom' in d for d in ids):
        raise ValueError('Dataset scope mismatch')
    if sum(map(len, ids.values())) != {'A': 300, 'B': 100}[which]:
        raise ValueError('Episode count mismatch')
    service = None
    if cache:
        cache_id = sha(DATA / 'runtime/native_cache.jsonl')[:16]
        store = Store(Path(tempfile.gettempdir()) / 'castworld-reproduction-audit' / cache_id)
        import_cache(store)
        service = cached_service(store)
    records = []
    for dataset, samples in ids.items():
        bundles = load_domain_episodes(pack(which, dataset), samples)
        pol = ResourcePolicy.model_validate(policy(which, dataset))
        prof = EfficiencyProfile.model_validate(profiles(which, [dataset])[dataset])
        if pol.comparison_hash() != prof.policy.comparison_hash():
            raise ValueError('Resource profile mismatch: ' + dataset)
        if set(samples) != set(pol.planned_episode_ids):
            raise ValueError('Policy and cohort mismatch')
        for sample, (bundle, _) in bundles.items():
            if which == 'A':
                label = DATA / 'evaluation/A' / dataset / (hashlib.sha256(sample.encode()).hexdigest() + '.json')
                truth = read(label)
                if truth['timestamps'] != bundle.episode.forecast_timeline.scored_timestamps:
                    raise ValueError('Public label timeline mismatch: ' + sample)
            if service:
                for model in ('chronos', 'timesfm', 'sundial', 'moirai'):
                    result = service.predict(model, bundle.episode)
                    if result['status'] != 'ok':
                        raise ValueError('Frozen candidate unavailable: ' + sample + ' / ' + model)
        first = next(iter(bundles.values()))[0].episode
        records.append({'dataset': dataset, 'episodes': len(samples), 'history': len(first.history.target_values),
                        'horizon': first.task.horizon_length, 'gap': first.task.forecast_gap})
        print(json.dumps({'audited_dataset': dataset, 'episodes': len(samples),
                          'native_predictions': len(samples) * 4 if cache else 0}), flush=True)
    result = {'cohort': which, 'datasets': records, 'episodes': sum(map(len, ids.values())),
              'native_predictions_verified': sum(map(len, ids.values())) * 4 if cache else 0, 'api_calls': 0}
    write(ROOT / '.reproduction' / ('audit_' + which + '.json'), result)
    return result


class RequestGate:
    def __init__(self, active, rpm):
        self.slots = threading.BoundedSemaphore(active)
        self.lock = threading.Lock()
        self.interval = 60 / rpm
        self.next_start = 0.0

    def enter(self):
        self.slots.acquire()
        with self.lock:
            now = time.monotonic()
            delay = max(0, self.next_start - now)
            self.next_start = max(now, self.next_start) + self.interval
        if delay:
            time.sleep(delay)


@contextmanager
def output_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.run.lock').open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            if stream.seek(0, 2) == 0:
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def run(args):
    if not args.allow_api:
        raise PermissionError('API generation requires explicit --allow-api')
    with output_lock(Path(args.out).resolve()):
        return run_locked(args)


def run_locked(args):
    from castbench.common import InfrastructureError
    from castbench.providers import OpenAICompatibleProvider, ProviderProfile
    from castbench.runner import Runner
    from castbench.storage import Store
    provider = ProviderProfile.model_validate(read(args.provider))
    if not os.environ.get(provider.api_key_env):
        raise ValueError('Set the environment variable named by provider.api_key_env')
    selected = cohort(args.cohort)
    native_mode = getattr(args, 'native_mode', 'cache')
    model_profiles = getattr(args, 'model_profiles', None)
    if model_profiles is not None and native_mode != 'live':
        raise ValueError('--model-profiles requires --native-mode live')
    if native_mode == 'live' and model_profiles is None:
        raise ValueError('Live native inference requires --model-profiles')
    if args.dataset:
        selected = {d: selected[d] for d in args.dataset}
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    identity = {'schema': 'castworld.portable-run/1.0', 'cohort': args.cohort, 'selected_ids': selected,
                'recovery_policy': 'resume_infrastructure_only/1.0',
                'provider': provider.model_dump(), 'sample_workers': args.workers,
                'max_active_requests': args.active_requests, 'requests_per_minute': args.rpm,
                'deployment': args.deployment, 'candidate_mode': 'live-native' if native_mode == 'live' else 'frozen-cache-only',
                'native_profiles_sha256': sha(model_profiles) if model_profiles else None,
                'score_protocol': 'robust-v1.3', 'limit': args.limit,
                'runtime_sha256': {p.name: sha(p) for p in sorted((ROOT / 'src/castbench').glob('*.py'))},
                'release_manifest_sha256': sha(ROOT / 'release/manifest.json')
                    if (ROOT / 'release/manifest.json').exists() else None,
                'credential_sha256': hashlib.sha256(os.environ[provider.api_key_env].encode()).hexdigest()}
    stamp = out / 'run_manifest.json'
    if stamp.exists() and read(stamp) != identity:
        raise ValueError('Configuration drift on resume; use a new output directory')
    write(stamp, identity)
    store = Store(out / 'store')
    if native_mode == 'live':
        service = live_service(store, model_profiles)
    else:
        import_cache(store)
        service = cached_service(store)
    gate = RequestGate(args.active_requests, args.rpm)

    class LimitedProvider(OpenAICompatibleProvider):
        def respond(self, request):
            gate.enter()
            try:
                return super().respond(request)
            finally:
                gate.slots.release()

    jobs = []
    for dataset, samples in selected.items():
        bundles = load_bundles(args.cohort, dataset, samples)
        jobs.extend((dataset, s, *bundles[s]) for s in samples)
    if args.limit:
        jobs = jobs[:args.limit]
    if native_mode == 'cache':
        verify_cached_candidates(service, jobs)

    def one(dataset, sample, bundle, registry):
        key = hashlib.sha256(sample.encode()).hexdigest()
        dest = out / 'episodes' / dataset / key
        complete = dest / 'complete.json'
        if complete.exists() and read(complete).get('status') == 'finished':
            return read(complete)
        prior = sorted(dest.glob('attempt_*.json'))
        records = [read(path) for path in prior]
        terminal = [row for row in records if row.get('status') == 'finished']
        if terminal:
            if len(terminal) != 1:
                raise ValueError('Multiple finished attempts require audit; automatic selection is forbidden')
            write(complete, terminal[0])
            return terminal[0]
        if records and (not args.retry_failed or records[-1].get('status') != 'needs_recovery'):
            return records[-1]
        attempt = len(prior) + 1
        # A technical recovery continues the committed state, budget and resource receipts.
        run_id = (records[-1].get('run_id') or out.name + ':' + sample + ':a' + str(records[-1]['attempt'])) \
            if records else out.name + ':' + sample + ':a1'
        connection = LimitedProvider(provider)
        try:
            # Generation receives only the PublicBundle and registry; no evaluator labels.
            state = Runner(bundle, store, connection, run_id, model_service=service, registry=registry,
                           resource_policy=policy(args.cohort, dataset)).run()
            export = store.export(run_id, include_raw=True)
            write(dest / ('trace_a' + str(attempt) + '.json'), export)
            row = {'dataset': dataset, 'sample_id': sample, 'attempt': attempt,
                   'status': state['status'], 'legal_delivery': bool((state.get('final') or {}).get('forecast_id')),
                   'trace': str((dest / ('trace_a' + str(attempt) + '.json')).relative_to(out))}
        except Exception as exc:
            row = {'dataset': dataset, 'sample_id': sample, 'attempt': attempt,
                   'status': 'needs_recovery' if isinstance(exc, InfrastructureError) else 'needs_review',
                   'legal_delivery': False, 'error_type': type(exc).__name__,
                   'error': re.sub(r'sk-[A-Za-z0-9_-]+', '[REDACTED]', str(exc))[:1600]}
            try:
                export = store.export(run_id, include_raw=True)
                write(dest / ('trace_a' + str(attempt) + '.json'), export)
                row['trace'] = str((dest / ('trace_a' + str(attempt) + '.json')).relative_to(out))
            except Exception as trace_error:
                row['trace_export_error_type'] = type(trace_error).__name__
        finally:
            connection.client.close()
        row['run_id'] = run_id
        if records:
            row['recovery_of_attempt'] = records[-1]['attempt']
        write(dest / ('attempt_' + str(attempt).zfill(4) + '.json'), row)
        if row['status'] == 'finished':
            write(complete, row)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(one, *job) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
            result = {'planned': len(jobs), 'processed': len(rows),
                      'valid': sum(r.get('legal_delivery', False) for r in rows), 'rows': rows}
            write(out / 'progress.json', result)
            print(json.dumps({k: result[k] for k in ('planned', 'processed', 'valid')}), flush=True)
    return {k: result[k] for k in ('planned', 'processed', 'valid')}


def load_bundles(which, dataset, samples):
    from castbench.domain_packs import load_domain_episodes
    return load_domain_episodes(pack(which, dataset), samples)


def score(args):
    from castbench.efficiency import score_cohort
    from castbench.evaluator import aggregate
    out = Path(args.out).resolve()
    manifest = read(out / 'run_manifest.json')
    board = manifest['cohort']
    private_labels = getattr(args, 'labels_root', None)
    if board == 'B':
        if private_labels is None:
            raise PermissionError('B scoring requires an organizer-held --labels-root; B inference needs no labels')
        label_root = Path(private_labels).resolve()
    elif board == 'A':
        if private_labels is not None:
            raise ValueError('--labels-root is only used for organizer-side B scoring')
        label_root = DATA / 'evaluation/A'
    else:
        raise ValueError('The public release supplies cohorts A and B')
    exports, labels = [], []
    for dataset, samples in manifest['selected_ids'].items():
        for sample in samples:
            key = hashlib.sha256(sample.encode()).hexdigest()
            directory = out / 'episodes' / dataset / key
            done = read(directory / 'complete.json')
            if done.get('status') != 'finished':
                raise ValueError('Scoring requires a finished run, including genuine failed deliveries')
            attempts = [read(path) for path in sorted(directory.glob('attempt_*.json'))]
            terminal = [row for row in attempts if row.get('status') == 'finished']
            if len(terminal) != 1 or terminal[0].get('trace') != done.get('trace'):
                raise ValueError('Finished attempt provenance differs; automatic retry selection is forbidden')
            exports.append(read(out / done['trace']))
            labels.append(read(label_root / dataset / (key + '.json')))
    rows = score_cohort(exports, labels, profiles(manifest['cohort'], manifest['selected_ids']))
    result = {'scores': rows, 'summary': aggregate(rows)}
    write(out / 'scores.json', result)
    return result['summary']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    check = sub.add_parser('audit')
    check.add_argument('--cohort', choices=['A', 'B'], default='A')
    check.add_argument('--no-cache', action='store_true')
    generate = sub.add_parser('run')
    generate.add_argument('--cohort', choices=['A', 'B'], default='A')
    generate.add_argument('--provider', required=True, type=Path)
    generate.add_argument('--out', required=True, type=Path)
    generate.add_argument('--deployment', required=True)
    generate.add_argument('--dataset', action='append')
    generate.add_argument('--workers', type=int, default=20)
    generate.add_argument('--active-requests', type=int, default=20)
    generate.add_argument('--rpm', type=float, default=12)
    generate.add_argument('--limit', type=int)
    generate.add_argument('--allow-api', action='store_true')
    generate.add_argument('--retry-failed', action='store_true')
    generate.add_argument('--native-mode', choices=['cache', 'live'], default='cache')
    generate.add_argument('--model-profiles', type=Path)
    scoring = sub.add_parser('score')
    scoring.add_argument('--out', required=True, type=Path)
    scoring.add_argument('--labels-root', type=Path,
                         help='Organizer-only B label directory, containing <dataset>/<episode-hash>.json')
    args = parser.parse_args()
    verify_source_tree()
    if args.command == 'prepare':
        result = prepare()
    elif args.command == 'audit':
        result = audit(args.cohort, not args.no_cache)
    elif args.command == 'score':
        result = score(args)
    else:
        if args.workers < 1 or args.active_requests < 1 or args.rpm <= 0 or (args.limit is not None and args.limit < 1):
            parser.error('Concurrency, RPM and limit must be positive')
        result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
