"""Build and verify lossless CSV/JSON views of the frozen raw dataset tables."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'castworld.raw-table/1.0'
MANIFEST_SCHEMA = 'castworld.raw-data-exports/1.0'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')


def write_identical_or_new(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError('Refusing to overwrite a modified export: '+str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def source_tables(root):
    release = json.loads((root/'release/manifest.json').read_text(encoding='utf-8'))
    with (root/'docs/data/dataset_inventory.csv').open(encoding='utf-8-sig', newline='') as stream:
        inventory = list(csv.DictReader(stream))
    ids = [row['dataset_id'] for row in inventory]
    if len(ids) != len(set(ids)) or set(ids) != set(release['datasets']):
        raise ValueError('Raw export and frozen dataset inventories differ')
    assets = {asset['path']: asset for asset in release['assets']}
    for row in inventory:
        dataset = row['dataset_id']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', dataset):
            raise ValueError('Unsafe dataset identifier: '+dataset)
        archive_name = row['raw_archive']
        archive = (root/archive_name).resolve()
        if not archive.is_relative_to(root.resolve()) or archive_name not in assets:
            raise ValueError('Raw archive is outside the frozen release')
        archive_data = archive.read_bytes()
        if sha(archive_data) != assets[archive_name]['sha256']:
            raise ValueError('Raw archive checksum mismatch: '+archive_name)
        with zipfile.ZipFile(io.BytesIO(archive_data)) as source:
            names = [name for name in source.namelist() if name.endswith('/data.csv')]
            if len(names) != 1:
                raise ValueError('Expected exactly one primary data.csv: '+dataset)
            member = names[0]
            raw = source.read(member)
        expected = release['files'][member]
        if len(raw) != expected['bytes'] or sha(raw) != expected['sha256']:
            raise ValueError('Frozen CSV identity mismatch: '+dataset)
        yield dataset, archive_name, member, raw


def parse_table(raw):
    encoding = 'utf-8-sig' if raw.startswith(b'\xef\xbb\xbf') else 'utf-8'
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError:
        encoding = 'gb18030'
        text = raw.decode(encoding)
    reader = csv.reader(io.StringIO(text, newline=''), strict=True)
    try:
        columns = next(reader)
    except StopIteration as error:
        raise ValueError('Raw CSV has no header') from error
    if not columns:
        raise ValueError('Raw CSV has an empty header row')
    rows = []
    for index, row in enumerate(reader, 1):
        if len(row) != len(columns):
            raise ValueError(f'CSV row {index} has {len(row)} cells; expected {len(columns)}')
        rows.append(row)
    return encoding, columns, rows


def table_digest(columns, rows):
    digest = hashlib.sha256()
    for row in [columns, *rows]:
        digest.update(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def make_document(dataset, archive, member, raw):
    encoding, columns, rows = parse_table(raw)
    return {
        'schema': SCHEMA, 'dataset_id': dataset, 'cell_encoding': 'csv-text',
        'source': {'archive': archive, 'member': member, 'sha256': sha(raw), 'encoding': encoding},
        'row_count': len(rows), 'column_count': len(columns), 'columns': columns,
        'table_sha256': table_digest(columns, rows), 'data': rows,
    }


def document_bytes(document):
    # One positional row per line preserves duplicate/blank headings and keeps diffs readable.
    metadata = {key: value for key, value in document.items() if key != 'data'}
    prefix = json_bytes(metadata).decode('utf-8').rstrip()[:-1].rstrip()
    body = ',\n'.join('    '+json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
                      for row in document['data'])
    return (prefix+',\n  "data": [\n'+body+'\n  ]\n}\n').encode('utf-8')


def entry_for(dataset, raw, encoded, document):
    return {
        'dataset_id': dataset, 'rows': document['row_count'], 'columns': document['column_count'],
        'source': document['source'], 'table_sha256': document['table_sha256'],
        'csv': {'path': 'csv/'+dataset+'/data.csv', 'sha256': sha(raw), 'bytes': len(raw)},
        'json': {'path': 'json/'+dataset+'/data.json', 'sha256': sha(encoded), 'bytes': len(encoded)},
    }


def verify_pair(raw, encoded, expected_document):
    document = json.loads(encoded)
    if document != expected_document:
        raise ValueError('JSON metadata, field order, or cell values differ from the original CSV')
    encoding, columns, rows = parse_table(raw)
    if columns != document['columns'] or rows != document['data'] or encoding != document['source']['encoding']:
        raise ValueError('CSV/JSON cell-level round trip mismatch')


def build(root=ROOT, out=None):
    root = Path(root).resolve()
    out = Path(out).resolve() if out is not None else root/'raw_data'
    entries = []
    for dataset, archive, member, raw in source_tables(root):
        document = make_document(dataset, archive, member, raw)
        encoded = document_bytes(document)
        verify_pair(raw, encoded, document)
        entry = entry_for(dataset, raw, encoded, document)
        for kind, data in [('csv', raw), ('json', encoded)]:
            write_identical_or_new(out/entry[kind]['path'], data)
        entries.append(entry)
        print(json.dumps({'exported': dataset, 'rows': entry['rows'], 'columns': entry['columns']}), flush=True)
    manifest = {'schema': MANIFEST_SCHEMA, 'cell_encoding': 'csv-text', 'datasets': entries}
    write_identical_or_new(out/'manifest.json', json_bytes(manifest))
    return check(root, out)


def check(root=ROOT, out=None):
    root = Path(root).resolve()
    out = Path(out).resolve() if out is not None else root/'raw_data'
    manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    if manifest['schema'] != MANIFEST_SCHEMA or manifest['cell_encoding'] != 'csv-text':
        raise ValueError('Unexpected raw export schema')
    entries = manifest['datasets']
    cells = rows = 0
    for entry, (dataset, archive, member, raw) in zip(entries, source_tables(root), strict=True):
        if entry['dataset_id'] != dataset:
            raise ValueError('Exported dataset order or inventory differs from the frozen release')
        document = make_document(dataset, archive, member, raw)
        encoded = (out/'json'/dataset/'data.json').read_bytes()
        if entry != entry_for(dataset, raw, encoded, document):
            raise ValueError('Export manifest or file hash mismatch: '+dataset)
        if (out/'csv'/dataset/'data.csv').read_bytes() != raw:
            raise ValueError('CSV copy is not byte-identical: '+dataset)
        verify_pair(raw, encoded, document)
        rows += entry['rows']
        cells += entry['rows']*entry['columns']
    expected = {entry[kind]['path'] for entry in entries for kind in ('csv', 'json')}
    actual = {path.relative_to(out).as_posix() for kind in ('csv', 'json')
              for path in (out/kind).rglob('*') if path.is_file()}
    if actual != expected:
        raise ValueError('Unexpected or missing raw CSV/JSON files')
    return {'datasets': len(entries), 'rows': rows, 'cells_verified': cells,
            'csv_files': len(entries), 'json_files': len(entries), 'api_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build', 'check'])
    parser.add_argument('--out', type=Path, help='Export root; default: raw_data in this repository')
    args = parser.parse_args()
    result = build(out=args.out) if args.command == 'build' else check(out=args.out)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
