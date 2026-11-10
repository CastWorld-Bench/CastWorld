"""Maintain the project UI from displayed summaries, without private experiment files."""
from __future__ import annotations

import argparse
import csv
from html.parser import HTMLParser
import io
import json
import math
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT/'site'
MODEL_FIELDS = ['rank','model','total_score','forecast','process','trust','efficiency','MASE','RMSSE','sMAPE','time_s','tokens']
SYSTEM_FIELDS = ['model', 'MASE', 'RMSSE', 'average_rank']
BOARD_ORDER = ['a-model', 'a-system', 'b-model', 'b-system']


def validate_result(row, fields, roster_size):
    assert set(row) == set(fields) | {'status'}, 'Unexpected result fields'
    assert row['status'] in {'awaiting_evaluation', 'completed'}, 'Unknown result status'
    if row['status'] == 'awaiting_evaluation':
        assert all(row[key] is None for key in fields if key != 'model'), 'Pending results must be empty'
        return
    for key in fields:
        if key == 'model':
            continue
        value = row[key]
        assert type(value) in {int, float} and math.isfinite(value), 'Invalid numeric result: '+key
        assert value >= 0, 'Negative result: '+key
        if key in {'forecast', 'process', 'trust', 'efficiency', 'total_score'}:
            assert value <= 100, 'Score outside 0-100: '+key
        elif key == 'sMAPE':
            assert value <= 2, 'sMAPE must be a ratio, not a percentage'
        elif key in {'rank', 'average_rank'}:
            assert 1 <= value <= roster_size, 'Invalid rank'
            if key == 'rank':
                assert value == int(value), 'Rank must be an integer'


def check_display(value, cell, key):
    if value is None:
        assert cell == '-', 'Missing result must be displayed as pending'
    elif key == 'model':
        assert cell == value, 'Model identifier mismatch'
    else:
        tolerance = .000050001 if key in {'MASE', 'RMSSE'} else .005001
        assert abs(float(cell)-value) <= tolerance, 'README result differs: '+key


class SummaryParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.parts = []
    def handle_starttag(self, tag, attrs):
        if tag == 'script' and dict(attrs).get('id') == 'public-board-summary': self.active = True
    def handle_endtag(self, tag):
        if tag == 'script': self.active = False
    def handle_data(self, data):
        if self.active: self.parts.append(data)


def readme_table(readme, name):
    block = re.search('<!-- BEGIN '+name+' -->(.*?)<!-- END '+name+' -->', readme, re.S)
    assert block, name
    table = block[1].strip()
    rows = list(csv.reader(io.StringIO(table), delimiter='|'))[2:]
    return table, [[re.sub(r'<[^>]*>|\*\*', '', value).strip() for value in cells[1:-1]] for cells in rows]


def build_data():
    parser = SummaryParser()
    parser.feed((SITE/'index.html').read_text(encoding='utf-8'))
    data = json.loads(''.join(parser.parts))
    assert data['schema_version'] == 'castworld.project-board/2.0'
    assert data['cohorts'] == [dict(id='a', episodes=300, visibility='public'),
                               dict(id='b', episodes=100, visibility='hidden')]
    assert [b['id'] for b in data['boards']] == BOARD_ORDER
    assert data['weights'] == dict(forecast=.60, process=.20, trust=.15, efficiency=.05)
    readme = (ROOT/'README.md').read_text(encoding='utf-8')
    rosters = {}
    for level, column, count in [('model', 1, 15), ('system', 0, 24)]:
        _, rows = readme_table(readme, 'A_'+level.upper()+'_TABLE')
        names = [row[column] for row in rows]
        assert len(names) == len(set(names)) == count, 'Invalid '+level+' roster'
        rosters[level] = set(names)
    for board in data['boards']:
        level, cohort = board['level'], board['cohort']
        fields = SYSTEM_FIELDS if level == 'system' else MODEL_FIELDS
        assert board['id'] == cohort+'-'+level
        assert board['episodes'] == {'a': 300, 'b': 100}[cohort]
        assert board['datasets'] == 25
        names = [row['model'] for row in board['rows']]
        assert len(names) == len(set(names)) == len(rosters[level]) and set(names) == rosters[level], 'Board roster mismatch'
        for row in board['rows']:
            validate_result(row, fields, len(names))
        completed = sum(row['status'] == 'completed' for row in board['rows'])
        status = 'completed' if completed == len(names) else 'in_progress' if completed else 'awaiting_evaluation'
        assert board['status'] == status, 'Board completion status mismatch'
        marker = board['id'].upper().replace('-', '_')+'_TABLE'
        _, rows = readme_table(readme, marker)
        displayed_fields = SYSTEM_FIELDS if level == 'system' else MODEL_FIELDS[:7]
        assert len(rows) == len(board['rows']), 'README row count differs: '+marker
        for displayed, record in zip(rows, board['rows'], strict=True):
            assert len(displayed) == len(displayed_fields), 'README column count differs: '+marker
            for cell, key in zip(displayed, displayed_fields, strict=True):
                check_display(record[key], cell, key)
    datasets = []
    with (ROOT/'docs/data/dataset_inventory.csv').open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            datasets.append(dict(id=row['dataset_id'],domain=row['application_domain'],frequency=row['frequency'],
                                 history=int(row['history_steps']),horizon=int(row['horizon_steps'])))
    assert len(datasets)==25
    return data, datasets


def json_bytes(value):
    return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()


def outputs():
    _, datasets = build_data()
    result = {SITE/'data/datasets.json':json_bytes(datasets), SITE/'.nojekyll':b''}
    for name in ['overview.png','data-profiles.png']:
        result[SITE/'assets'/name]=(ROOT/'docs/assets'/name).read_bytes()
    for name in ['castworld-mark.svg','atlas-background.png']:
        result[SITE/'assets'/name]=(ROOT/'docs/assets/brand'/name).read_bytes()
    result[SITE/'assets/favicon.svg']=(ROOT/'docs/assets/brand/castworld-mark.svg').read_bytes()
    return result


def export_hf(destination):
    destination = Path(destination).resolve()
    if destination==ROOT or ROOT in destination.parents or (destination.exists() and any(destination.iterdir())):
        raise ValueError('Use a new empty directory outside the repository')
    destination.mkdir(parents=True,exist_ok=True)
    data, datasets = build_data()
    for name in ['index.html','styles.css','i18n.js','app.js','assets']:
        source, target = SITE/name, destination/name
        if source.is_dir(): shutil.copytree(source,target)
        else: shutil.copyfile(source,target)
    html=(destination/'index.html').read_text(encoding='utf-8')
    payload=json.dumps(data,ensure_ascii=False,indent=2).replace('<','\\u003c')
    html=re.sub(r'(<script type="application/json" id="public-board-summary">).*?(</script>)',lambda m:m[1]+'\n'+payload+'\n'+m[2],html,flags=re.S)
    html=html.replace('data-view="project"','data-view="leaderboard"').replace('<title>CastWorld | Agentic Time Series Forecasting</title>','<title>CastWorld Leaderboard | A / B</title>')
    (destination/'index.html').write_bytes(html.encode())
    (destination/'data').mkdir()
    (destination/'data/datasets.json').write_bytes(json_bytes(datasets))
    (destination/'README.md').write_bytes(b'---\ntitle: CastWorld Leaderboard\nsdk: static\napp_file: index.html\npinned: false\n---\n\n# CastWorld A/B Leaderboard\n\nUpload this folder to a Static HTML Space. Display summaries are embedded in the page. No experiment records or private labels are included.\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--hf-out',type=Path)
    args=parser.parse_args()
    for path, data in outputs().items():
        if args.check:
            if not path.is_file() or path.read_bytes()!=data: raise SystemExit('Stale asset: '+str(path))
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(data)
    if args.hf_out: export_hf(args.hf_out)
    print('Project display verified')


if __name__=='__main__': main()
