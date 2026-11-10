"""Lossless raw views must preserve cell text, source bytes, and positional headings."""
import csv
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('raw_export', ROOT/'tools/export_raw_data.py')
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


def fixture_release(root, raw, dataset='Example'):
    archive = root/'release/assets/raw.zip'
    archive.parent.mkdir(parents=True)
    member = 'datasets/'+dataset+'/data.csv'
    with zipfile.ZipFile(archive, 'w') as output:
        output.writestr(member, raw)
    release = {'datasets': [dataset], 'assets': [{'path': 'release/assets/raw.zip',
                'sha256': EXPORT.sha(archive.read_bytes())}],
               'files': {member: {'sha256': EXPORT.sha(raw), 'bytes': len(raw)}}}
    (root/'release/manifest.json').write_bytes(EXPORT.json_bytes(release))
    inventory = root/'docs/data/dataset_inventory.csv'
    inventory.parent.mkdir(parents=True)
    with inventory.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dataset_id', 'raw_archive'])
        writer.writerow([dataset, 'release/assets/raw.zip'])


def test_round_trip_precision_blanks_duplicate_names_and_text(tmp_path):
    columns = ['', '', 'time', 'large', 'decimal**', 'event***', 'source#', 'missing', 'literal', 'code']
    values = ['', 'index', '2020-03-08T01:30:00-08:00', '9007199254740993',
              '0.123456789012345678901234567890', 'quote: "yes", comma,\r\nnext',
              '\u4e2d\u6587', '', 'NaN', '001']
    text = io.StringIO(newline='')
    csv.writer(text, lineterminator='\r\n').writerows([columns, values])
    raw = b'\xef\xbb\xbf'+text.getvalue().encode('utf-8')
    fixture_release(tmp_path, raw)
    first = EXPORT.build(tmp_path)
    exported = tmp_path/'raw_data'
    csv_path = exported/'csv/Example/data.csv'
    json_path = exported/'json/Example/data.json'
    original_json = json_path.read_bytes()
    document = json.loads(original_json)
    assert csv_path.read_bytes() == raw
    assert document['columns'] == columns
    assert document['data'] == [values]
    assert document['source']['encoding'] == 'utf-8-sig'
    assert all(type(value) is str for value in document['data'][0])
    assert first['cells_verified'] == 10 and first['rows'] == 1
    assert EXPORT.build(tmp_path) == EXPORT.check(tmp_path) == first
    assert json_path.read_bytes() == original_json


def test_gb18030_is_decoded_without_modifying_csv(tmp_path):
    raw = 'time,label\n2020-01-01,\u5de5\u4e1a\u70ed\u6e90\n'.encode('gb18030')
    fixture_release(tmp_path, raw)
    EXPORT.build(tmp_path)
    document = json.loads((tmp_path/'raw_data/json/Example/data.json').read_bytes())
    assert document['source']['encoding'] == 'gb18030'
    assert document['data'][0][1] == '\u5de5\u4e1a\u70ed\u6e90'
    assert (tmp_path/'raw_data/csv/Example/data.csv').read_bytes() == raw


@pytest.mark.parametrize('raw', [b'', b'\n', b'a,b\n1\n', b'a,b\n1,2,3\n'])
def test_rejects_empty_or_ragged_sources(raw):
    with pytest.raises(ValueError):
        EXPORT.parse_table(raw)


@pytest.mark.parametrize('kind,filename', [('csv', 'data.csv'), ('json', 'data.json')])
def test_modified_exports_are_detected_and_not_overwritten(tmp_path, kind, filename):
    fixture_release(tmp_path, b'time,value\n2020-01-01,1.00\n')
    EXPORT.build(tmp_path)
    path = tmp_path/'raw_data'/kind/'Example'/filename
    changed = path.read_bytes()+b' '
    path.write_bytes(changed)
    with pytest.raises(ValueError):
        EXPORT.check(tmp_path)
    with pytest.raises(FileExistsError):
        EXPORT.build(tmp_path)
    assert path.read_bytes() == changed


def test_tampered_values_fail_even_with_updated_file_hash(tmp_path):
    fixture_release(tmp_path, b'a,b\n1,2\n')
    EXPORT.build(tmp_path)
    path = tmp_path/'raw_data/json/Example/data.json'
    document = json.loads(path.read_bytes())
    document['data'][0][1] = '3'
    encoded = EXPORT.document_bytes(document)
    path.write_bytes(encoded)
    manifest_path = tmp_path/'raw_data/manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    manifest['datasets'][0]['json'].update(sha256=EXPORT.sha(encoded), bytes=len(encoded))
    manifest_path.write_bytes(EXPORT.json_bytes(manifest))
    with pytest.raises(ValueError, match='cell values'):
        EXPORT.check(tmp_path)


def test_unexpected_export_file_is_rejected(tmp_path):
    fixture_release(tmp_path, b'a\n1\n')
    EXPORT.build(tmp_path)
    (tmp_path/'raw_data/json/Example/extra.json').write_bytes(b'{}')
    with pytest.raises(ValueError, match='Unexpected or missing'):
        EXPORT.check(tmp_path)


def test_source_archive_tampering_is_rejected(tmp_path):
    fixture_release(tmp_path, b'a\n1\n')
    (tmp_path/'release/assets/raw.zip').write_bytes(b'changed')
    with pytest.raises(ValueError, match='archive checksum'):
        EXPORT.build(tmp_path)


def test_unsafe_dataset_identifier_is_rejected(tmp_path):
    fixture_release(tmp_path, b'a\n1\n', '../escape')
    with pytest.raises(ValueError, match='Unsafe dataset identifier'):
        EXPORT.build(tmp_path)


def test_repository_export_inventory():
    release = json.loads((ROOT/'release/manifest.json').read_bytes())
    manifest = json.loads((ROOT/'raw_data/manifest.json').read_bytes())
    assert len(manifest['datasets']) == 25
    assert {entry['dataset_id'] for entry in manifest['datasets']} == set(release['datasets'])
    for entry in manifest['datasets']:
        for kind in ('csv', 'json'):
            assert (ROOT/'raw_data'/entry[kind]['path']).stat().st_size == entry[kind]['bytes']
