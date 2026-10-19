from pathlib import Path

from tractor.coverage_assembly import sha
from tractor.document_build import plan_item
from tractor.document_refill import refill
from tractor.jsonio import dumps, loads


def test_refill_preserves_old_plan_and_targets_only_missing_cell(tmp_path):
    import tractor.document_build as build
    spec = loads(Path('schemas/extraction/invoices.json').read_text())
    item = plan_item(spec, 'train', 0, 42, 'single_object')
    original = dumps(item) + '\n'
    (tmp_path / 'plan.jsonl').write_text(original)
    plan_hash = sha(tmp_path / 'plan.jsonl')
    code = Path(build.__file__)
    recipe = {'plan_sha256': plan_hash, 'code_sha256': sha(code),
              'facts_code_sha256': sha(code.with_name('document_facts.py')),
              'scenario_code_sha256': sha(code.with_name('document_scenarios.py')),
              'prompts': {k: Path(f'configs/document-{k}.txt').read_text() for k in ['facts', 'render', 'review']}}
    (tmp_path / 'recipe.json').write_text(dumps(recipe))
    (tmp_path / 'manifest.json').write_text(dumps({'files': {'plan.jsonl': plan_hash},
        'counts': {'train:invoices': 1}, 'config': {'seed': 42}}))
    (tmp_path / 'scale-audit.json').write_text(dumps({'passed': True, 'recipe_sha256': sha(tmp_path / 'recipe.json')}))
    (tmp_path / 'campaign-status.json').write_text(dumps({'status': 'planned_candidates_exhausted', 'cells': {
        'train:invoices:single_object:de': {'accepted': 0, 'target': 1},
        'train:invoices:single_object:en': {'accepted': 1, 'target': 1}}}))
    report = refill(tmp_path)
    assert (tmp_path / 'plan.jsonl').read_text().startswith(original)
    rows = [loads(line) for line in (tmp_path / 'plan.jsonl').read_text().splitlines()]
    assert len({r['id'] for r in rows}) == len(rows)
    assert all((r['split'], r['shape'], r['language']) == ('train', 'single_object', 'de') for r in rows[1:])
    assert rows[0] == item
    assert set(report['added_candidates']) == {'train:invoices:single_object:de'}
    assert loads((tmp_path / 'recipe.json').read_text())['plan_sha256'] == sha(tmp_path / 'plan.jsonl')
    assert next((tmp_path / 'refills').glob('*/plan.jsonl')).read_text() == original
