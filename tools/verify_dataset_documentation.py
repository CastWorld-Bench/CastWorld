"""Verify dataset documentation against frozen release archives, without extraction."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_csv(path):
    return list(csv.DictReader(path.open(encoding='utf-8-sig', newline='')))


def report_fields(text):
    """Read field dictionaries, excluding prose and correlation tables."""
    rows = []
    column = None
    permission_column = None
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.lstrip().startswith('|'):
            column = permission_column = None
            continue
        cells = [v.strip() for v in re.split(r'(?<!\\)\|', line.strip().strip('|'))]
        headers = [i for i, value in enumerate(cells) if value in {'字段名', '字段名称'}]
        if headers:
            column = headers[0]
            permission_column = cells.index('冻结评测权限') if '冻结评测权限' in cells else None
            continue
        if column is None or all(re.fullmatch(r':?-+:?', cell) for cell in cells):
            continue
        if column < len(cells):
            name = cells[column].strip('`')
            if name:
                rows.append((name, cells[permission_column] if permission_column is not None else None, lineno))
    return rows


def main():
    failures = []
    checks = 0

    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(message)

    folder = ROOT / 'docs/data'
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    release = json.loads((ROOT / 'release/manifest.json').read_text(encoding='utf-8'))
    for name, record in manifest['files'].items():
        data = (ROOT / name).read_bytes()
        check(len(data) == record['bytes'] and digest(data) == record['sha256'], 'Documentation identity: ' + name)

    inventory = read_csv(folder / 'dataset_inventory.csv')
    official = {r['dataset_id']: r for r in release['task_inventory']}
    rows = read_csv(folder / 'field_dictionary.csv')
    dictionary = json.loads((folder / 'field_dictionary.json').read_text(encoding='utf-8'))
    check(dictionary['fields'] == rows, 'CSV and JSON field dictionaries differ')
    unit_rows = read_csv(folder / 'unit_interpretations.csv')
    units = {(r['dataset_id'], r['field']): r for r in unit_rows}
    by_field = {(r['dataset_id'], r['field']): r for r in rows}
    check(len(by_field) == len(rows) == manifest['fields'], 'Field count or duplicate key')
    check(len(units) == len(unit_rows) == manifest['unit_interpretations'], 'Unit interpretation count or duplicate key')
    check(set(official) == {r['dataset_id'] for r in inventory} == set(release['datasets']), 'Dataset scope differs')
    check(len(inventory) == len(dictionary['datasets']) == manifest['datasets'], 'Dataset count differs')
    archived = {}
    for composite, record in manifest['frozen_inputs'].items():
        archive_name, member_name = composite.split('!', 1)
        if archive_name not in archived:
            archived[archive_name] = zipfile.ZipFile(ROOT / archive_name)
        data = archived[archive_name].read(member_name)
        check(len(data) == record['bytes'] and digest(data) == record['sha256'], 'Source identity: ' + composite)
        check(digest(data) == release['files'][member_name]['sha256'], 'Release/source identity: ' + member_name)

    documented_keys = set()
    for record in inventory:
        dataset = record['dataset_id']
        for key, value in official[dataset].items():
            check(record[key] == value, f'Inventory mismatch: {dataset}/{key}')
        task_archive = archived[record['task_archive']]
        profile = json.loads(task_archive.read(record['profile_member']))
        raw_archive = archived[record['raw_archive']]
        raw_member = next(n for n in raw_archive.namelist() if n.endswith('/data.csv'))
        raw_data = raw_archive.read(raw_member)
        try:
            raw_text = raw_data.decode('utf-8-sig')
        except UnicodeDecodeError:
            raw_text = raw_data.decode('gb18030')
        header = next(csv.reader(io.StringIO(raw_text)))
        columns = {name for name in header if name.strip()}
        check(len(columns) == int(record['raw_named_columns']), 'Raw column count: ' + dataset)
        check(len(profile['fields']) == int(record['context_fields']), 'Context count: ' + dataset)
        check(len(profile['groups'].get('events', [])) == int(record['event_source_fields']), 'Event-source count: ' + dataset)
        future_count = sum(v['visibility'] == 'history_and_future' for v in profile['fields'].values())
        check(future_count == int(record['future_authorized_fields']), 'Future-field count: ' + dataset)
        check(len(profile['fields']) - future_count == int(record['history_only_fields']), 'History-field count: ' + dataset)
        report = raw_archive.read(record['report_member']).decode('utf-8-sig')
        source_fields = report_fields(report)
        check(len(source_fields) == len(columns) and {name for name, _, _ in source_fields} == columns,
              'Source-report dictionary differs from CSV: ' + dataset)
        for name, permission, _ in source_fields:
            if permission is not None and name in profile['fields']:
                expected = {'history_and_future': '历史与许可未来', 'history_only': '仅历史'}[profile['fields'][name]['visibility']]
                check(permission == expected, f'Report/profile permission: {dataset}/{name}')
        for name, value in profile['fields'].items():
            key = (dataset, name)
            documented_keys.add(key)
            row = by_field.get(key)
            check(row is not None, f'Missing documented field: {key}')
            if row is None:
                continue
            for dest, source in [('description', 'description'), ('recorded_unit', 'unit'), ('profile_visibility', 'visibility')]:
                check(row[dest] == value[source], f'Frozen metadata mismatch: {key}/{dest}')
            check(row['public_visibility'] == {'history_only': 'history_only', 'history_and_future': 'benchmark_known_future'}[value['visibility']], f'Public visibility: {key}')
            check(row['numeric'] == str(value['numeric']).lower(), f'Numeric type: {key}')
            check(row['event_source'] == str(name in profile['groups'].get('events', [])).lower(), f'Event source: {key}')
            check(row['groups'] == ';'.join(g for g, names in profile['groups'].items() if name in names), f'Groups: {key}')
            if key in units:
                interpretation = units[key]
                for property_name in ['recorded_unit', 'interpretation_unit', 'interpretation_status', 'quantity_type', 'representation']:
                    check(row[property_name] == interpretation[property_name], f'Interpretation mismatch: {key}/{property_name}')
                check(interpretation['source_member'] == record['raw_archive'] + '!' + record['report_member'], f'Source member: {key}')
                lines = report.splitlines()
                n = int(interpretation['source_line']) - 1
                check(0 <= n < len(lines) and lines[n] == interpretation['source_quote'] and f'`{name}`' in lines[n], f'Interpretation source quote: {key}')
                check(any(field == name and lineno == n + 1 for field, _, lineno in source_fields), f'Interpretation must cite a field definition: {key}')
                check(interpretation['interpretation_status'] != 'confirmed' or bool(interpretation['interpretation_unit']), f'Missing confirmed unit: {key}')
            else:
                check(row['interpretation_unit'] == ('' if value['unit'] == 'source_unit_unverified' else value['unit']), f'Unexpected unit conversion: {key}')
    check(documented_keys == set(by_field), 'Unexpected documented field')
    check(set(units) <= documented_keys, 'Interpretation refers to missing field')

    for archive in archived.values():
        archive.close()
    print(json.dumps({'datasets': len(inventory), 'field_definitions': len(rows), 'unit_interpretations': len(units),
                      'checks': checks, 'errors': failures, 'new_model_calls': 0}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
