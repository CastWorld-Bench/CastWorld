"""Validate public inputs and toolkit resources, excluding private experiment records."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = 'benchmark_data/'


def published_paths(root):
    if (root / '.git').exists():
        result = subprocess.run(['git', 'ls-files', '-z'], cwd=root, capture_output=True, check=True)
        return {name for name in result.stdout.decode('utf-8').split('\0') if name and (root/name).is_file()}
    return {p.relative_to(root).as_posix() for p in root.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def private_exports(paths):
    return sorted(name for name in paths if name.startswith(('results/', 'release/validation/'))
                  or name == 'site/data/leaderboards.json'
                  or (name.startswith('site/data/') and name.endswith('.csv')))


def allowed_member(name):
    if '/provenance/' in name or name.endswith('/domain_design.md'):
        return False
    if name.startswith('datasets/'):
        return bool(re.fullmatch(r'datasets/[A-Za-z0-9_.-]+/(data\.csv|report\.md)', name))
    if name in {DATA+'manifest.json', DATA+'runtime/native_models.json',
                DATA+'runtime/native_cache.jsonl', 'release/third_party_sources.json'}:
        return True
    return any(name.startswith(prefix) for prefix in (
        DATA+'datasets/', DATA+'profiles/', DATA+'packs/A/', DATA+'packs/B/',
        DATA+'cohorts/', DATA+'policies/A/', DATA+'policies/B/', DATA+'efficiency_profiles/',
        DATA+'test_inputs/A/', DATA+'test_inputs/B/', DATA+'test_labels/A/', DATA+'evaluation/A/',
    ))


def main():
    manifest = json.loads((ROOT/'release/manifest.json').read_text(encoding='utf-8'))
    failures, members = [], set()
    forbidden_keys = {'reasoning_content', 'raw_response', 'provider_response', 'reviewer_id', 'agent_versions'}
    def inspect(value, name):
        if isinstance(value, dict):
            if forbidden_keys & value.keys(): failures.append('Private record fields: '+name)
            for child in value.values(): inspect(child, name)
        elif isinstance(value, list):
            for child in value: inspect(child, name)
    for asset in manifest['assets']:
        path = ROOT/asset['path']
        if path.stat().st_size != asset['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != asset['sha256']:
            failures.append('Archive identity: '+asset['path'])
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir(): continue
                name, data = info.filename, archive.read(info)
                members.add(name)
                if not allowed_member(name): failures.append('Outside public scope: '+name)
                record = manifest['files'].get(name)
                if record is None or len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
                    failures.append('Member identity: '+name)
                if name.endswith('.json'): inspect(json.loads(data), name)
    if members != set(manifest['files']): failures.append('Archive inventory differs from manifest')
    if len(manifest['datasets']) != 25: failures.append('Dataset scope')
    fingerprint = json.loads((ROOT/'release/source_integrity.json').read_text(encoding='utf-8'))
    for name, record in fingerprint['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != record['sha256']:
            failures.append('Runtime identity: '+name)
    failures.extend('Independent experiment export: '+name for name in private_exports(published_paths(ROOT)))
    print(json.dumps(dict(archives=len(manifest['assets']), members=len(members), datasets=25,
                         runtime_files=len(fingerprint['files']), errors=failures, api_calls=0), indent=2))
    if failures: raise SystemExit(1)


if __name__ == '__main__': main()
