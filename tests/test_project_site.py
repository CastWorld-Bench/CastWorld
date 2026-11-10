"""Read-only project view checks; no model or provider dependencies."""
import importlib.util
import copy
from html.parser import HTMLParser
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("project_builder", ROOT / "tools/build_project_site.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class ProjectSiteTests(unittest.TestCase):
    def setUp(self):
        self.data, _ = BUILDER.build_data()
        self.boards = {row["id"]: row for row in self.data["boards"]}

    def test_two_cohorts_four_boards(self):
        self.assertEqual(set(self.boards), {"a-system", "a-model", "b-system", "b-model"})
        self.assertEqual([self.boards[k]["episodes"] for k in ("a-system", "a-model", "b-system", "b-model")],
                         [300, 300, 100, 100])
        self.assertTrue(all(b["datasets"] == 25 for b in self.boards.values()))
        self.assertEqual([c['episodes'] for c in self.data['cohorts']], [300, 100])
        html = (ROOT / 'site/index.html').read_text(encoding='utf-8')
        self.assertIn('id="cohort-tabs"', html)
        self.assertIn('class="board-tabs level-tabs"', html)
        self.assertEqual([b['id'] for b in self.data['boards']], ['a-model', 'a-system', 'b-model', 'b-system'])
        self.assertIn('id += "-model"', (ROOT/'site/app.js').read_text(encoding='utf-8'))

    def test_retired_result_displays_removed(self):
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        for marker in ('LLM_TABLE', 'BASELINE_TABLE'):
            self.assertNotIn('<!-- BEGIN '+marker+' -->', readme)
        order = [readme.index('<!-- BEGIN '+name+' -->') for name in
                 ('A_MODEL_TABLE', 'A_SYSTEM_TABLE', 'B_MODEL_TABLE', 'B_SYSTEM_TABLE')]
        self.assertEqual(order, sorted(order))
        html = (ROOT/'site/index.html').read_text(encoding='utf-8')
        self.assertNotIn('id="insights"', html)
        for folder in ('docs/assets', 'site/assets'):
            for name in ('motivation.png', 'workflow-study.png'):
                self.assertFalse((ROOT/folder/name).exists())

    def test_rosters_and_result_states(self):
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        for level, model_column, count, fields in [
            ('model', 1, 15, BUILDER.MODEL_FIELDS),
            ('system', 0, 24, BUILDER.SYSTEM_FIELDS),
        ]:
            _, rows = BUILDER.readme_table(readme, 'A_'+level.upper()+'_TABLE')
            roster = [row[model_column] for row in rows]
            self.assertEqual(len(set(roster)), count)
            for cohort in ('a', 'b'):
                board = self.boards[cohort+'-'+level]
                self.assertEqual([row['model'] for row in board['rows']], roster)
                for row in board['rows']:
                    BUILDER.validate_result(row, fields, count)
                    if row['status'] == 'awaiting_evaluation':
                        for key in fields:
                            if key != 'model':
                                self.assertIsNone(row[key])

    def test_no_independent_result_exports(self):
        self.assertFalse((ROOT / 'site/data/leaderboards.json').exists())
        self.assertEqual(list((ROOT / 'site/data').glob('*.csv')), [])
        js = (ROOT / 'site/app.js').read_text(encoding='utf-8')
        self.assertIn('new Blob(', js)

    def test_data_and_images_match(self):
        datasets = json.loads((ROOT / "site/data/datasets.json").read_text(encoding="utf-8"))
        self.assertEqual(len(datasets), 25)
        self.assertEqual(len({row["domain"] for row in datasets}), 8)
        self.assertNotIn("GEFCom2012-Z6", {row["id"] for row in datasets})
        for name in ("overview.png", "data-profiles.png"):
            self.assertEqual((ROOT / "docs/assets" / name).read_bytes(),
                             (ROOT / "site/assets" / name).read_bytes())

    def test_brand_is_shared(self):
        source = (ROOT / "docs/assets/brand/castworld-mark.svg").read_bytes()
        for name in ("castworld-mark.svg", "favicon.svg"):
            self.assertEqual(source, (ROOT / "site/assets" / name).read_bytes())
        self.assertIn(b"#3983B3", source)
        self.assertIn(b"#31A68E", source)
        self.assertIn(b"#E6AA50", source)
        self.assertIn("docs/assets/brand/readme-banner.png",
                      (ROOT / "README.md").read_text(encoding="utf-8"))
        self.assertEqual((ROOT / "docs/assets/brand/atlas-background.png").read_bytes(),
                         (ROOT / "site/assets/atlas-background.png").read_bytes())
        self.assertTrue((ROOT / "site/assets/fonts/Manrope.ttf").is_file())

    def test_readme_heading_icons(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for icon in ("orbit", "trophy", "bot", "chart-no-axes-combined", "workflow", "rocket", "book-open", "fingerprint"):
            self.assertIn('src="docs/assets/icons/'+icon+'.svg"', readme)
            self.assertTrue((ROOT / "docs/assets/icons" / (icon+'.svg')).is_file())

    def test_research_sections_present(self):
        html = (ROOT / "site/index.html").read_text(encoding="utf-8")
        for section in ("overview", "framework", "datasets"):
            self.assertIn('id="'+section+'"', html)
        for family in ("CastSense", "CastFeat", "CastZoo", "CastKnowledge"):
            self.assertIn(family, html)

    def test_generated_files_current(self):
        for path, expected in BUILDER.outputs().items():
            self.assertEqual(path.read_bytes(), expected, str(path))

    def test_no_label_assets_or_credentials(self):
        self.assertEqual(self.data["weights"], dict(forecast=.6, process=.2, trust=.15, efficiency=.05))
        for path in (ROOT / "site").rglob("*"):
            if path.is_file():
                self.assertNotIn("private", path.parts)
                self.assertNotIn("evaluation", path.parts)
                if path.suffix in {".json", ".js", ".html"}:
                    text = path.read_text(encoding="utf-8")
                    self.assertNotIn("Bearer sk-", text)
                    self.assertNotIn('"target_values"', text)

    def test_huggingface_export_is_AB_only(self):
        with tempfile.TemporaryDirectory(prefix="castworld-space-") as temporary:
            target = Path(temporary) / "space"
            BUILDER.export_hf(target)
            parser = BUILDER.SummaryParser()
            parser.feed((target / 'index.html').read_text(encoding='utf-8'))
            data = json.loads(''.join(parser.parts))
            self.assertEqual(data, self.data)
            self.assertEqual([board["id"] for board in data["boards"]], ['a-model', 'a-system', 'b-model', 'b-system'])
            for name in ('app.js', 'i18n.js', 'styles.css'):
                self.assertEqual((target / name).read_bytes(), (ROOT / 'site' / name).read_bytes())
            self.assertEqual({path.name for path in (target / "data").iterdir()}, {'datasets.json'})
            self.assertIn('data-view="leaderboard"', (target / "index.html").read_text(encoding="utf-8"))
            self.assertNotIn('data-view="project"', (target / "index.html").read_text(encoding="utf-8"))
            with self.assertRaises(ValueError):
                BUILDER.export_hf(target)

    def test_all_static_copy_has_chinese_translation(self):
        script = (ROOT / 'site/i18n.js').read_text(encoding='utf-8')
        translations = json.loads(re.search(r'const translations = (\{.*?\n  \});', script, re.S)[1])
        names = {'CastWorld', 'Cast', 'World', 'CastSense', 'CastFeat', 'CastZoo', 'CastKnowledge',
                 'GitHub', 'MASE', 'RMSSE', '中文'}
        names.update(row['model'] for board in self.boards.values() for row in board['rows'])
        missing = []

        class CopyParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.excluded = []

            def handle_starttag(self, tag, attrs):
                if tag in {'script', 'style', 'pre', 'code', 'noscript'}:
                    self.excluded.append(tag)
                for name, value in attrs:
                    if name in {'title', 'aria-label', 'alt', 'placeholder'} and value:
                        if value != 'Switch to Chinese':
                            self.check(value)
                    if tag == 'meta' and name == 'content' and value.startswith(('CastWorld', '25 datasets.')):
                        self.check(value)

            def handle_endtag(self, tag):
                if self.excluded and tag == self.excluded[-1]:
                    self.excluded.pop()

            def handle_data(self, data):
                if not self.excluded:
                    self.check(data)

            def check(self, value):
                value = ' '.join(value.split())
                if re.search('[A-Za-z]', value) and value not in names and value not in translations:
                    missing.append(value)

        CopyParser().feed((ROOT / 'site/index.html').read_text(encoding='utf-8'))
        self.assertEqual(missing, [])
        for source, translated in translations.items():
            if source != '{cohort} - {level}':
                self.assertRegex(translated, r'[\u4e00-\u9fff]', source)
            self.assertEqual(sorted(re.findall(r'\{\w+\}', source)), sorted(re.findall(r'\{\w+\}', translated)), source)
        self.assertIn('localStorage.setItem("castworld-language", language)', script)
        self.assertIn('localStorage.getItem("castworld-language")', script)
        app = (ROOT / 'site/app.js').read_text(encoding='utf-8')
        for key in re.findall(r'\bt\("([^"\n]+)"', app):
            self.assertIn(key, translations)

    def test_builder_rejects_roster_mismatch_and_pending_values(self):
        original_root, original_site = BUILDER.ROOT, BUILDER.SITE
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        html = (ROOT / 'site/index.html').read_text(encoding='utf-8')
        with tempfile.TemporaryDirectory(prefix='castworld-site-check-') as temporary:
            fake_root = Path(temporary)
            fake_site = fake_root / 'site'
            fake_site.mkdir()
            try:
                BUILDER.ROOT, BUILDER.SITE = fake_root, fake_site
                (fake_site / 'index.html').write_text(html, encoding='utf-8')
                (fake_root / 'README.md').write_text(readme.replace('GPT-6 Astra', 'Unknown model', 1), encoding='utf-8')
                with self.assertRaisesRegex(AssertionError, 'roster mismatch'):
                    BUILDER.build_data()
                (fake_root / 'README.md').write_text(readme, encoding='utf-8')
                (fake_site / 'index.html').write_text(html.replace('"MASE": null', '"MASE": 1.2345', 1), encoding='utf-8')
                with self.assertRaisesRegex(AssertionError, 'Pending results must be empty'):
                    BUILDER.build_data()
            finally:
                BUILDER.ROOT, BUILDER.SITE = original_root, original_site

    def test_completed_aggregate_can_be_maintained_without_changing_builder(self):
        original_root, original_site = BUILDER.ROOT, BUILDER.SITE
        data = copy.deepcopy(self.data)
        board = data['boards'][0]
        board['status'] = 'in_progress'
        row = board['rows'][0]
        row.update(status='completed', rank=1, total_score=73.25, forecast=70, process=75,
                   trust=80, efficiency=85, MASE=1.1, RMSSE=1.2, sMAPE=.1, time_s=10, tokens=100.5)
        readme = (ROOT/'README.md').read_text(encoding='utf-8')
        old = '| - | '+row['model']+' | - | - | - | - | - |'
        new = '| 1 | '+row['model']+' | 73.25 | 70 | 75 | 80 | 85 |'
        self.assertIn(old, readme)
        readme = readme.replace(old, new, 1)
        html = (ROOT/'site/index.html').read_text(encoding='utf-8')
        html = re.sub(r'(<script type="application/json" id="public-board-summary">).*?(</script>)',
                      lambda m: m[1]+json.dumps(data)+m[2], html, flags=re.S)
        with tempfile.TemporaryDirectory(prefix='castworld-maintenance-test-') as temporary:
            fake_root = Path(temporary)
            fake_site = fake_root/'site'
            fake_site.mkdir()
            (fake_root/'docs/data').mkdir(parents=True)
            shutil.copyfile(ROOT/'docs/data/dataset_inventory.csv', fake_root/'docs/data/dataset_inventory.csv')
            (fake_root/'README.md').write_text(readme, encoding='utf-8')
            (fake_site/'index.html').write_text(html, encoding='utf-8')
            try:
                BUILDER.ROOT, BUILDER.SITE = fake_root, fake_site
                accepted, _ = BUILDER.build_data()
                self.assertEqual(accepted['boards'][0]['rows'][0]['total_score'], 73.25)
                self.assertEqual(accepted['boards'][0]['rows'][0]['tokens'], 100.5)
            finally:
                BUILDER.ROOT, BUILDER.SITE = original_root, original_site

    def test_invalid_completed_scores_are_rejected(self):
        row = dict(self.boards['a-system']['rows'][0], status='completed', MASE=-1, RMSSE=1, average_rank=1)
        with self.assertRaisesRegex(AssertionError, 'Negative result'):
            BUILDER.validate_result(row, BUILDER.SYSTEM_FIELDS, 24)


if __name__ == "__main__":
    unittest.main()
