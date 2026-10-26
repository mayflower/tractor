"""Verify legacy inference invariants before reusing full, frozen Dev predictions."""

import ast
import hashlib
import subprocess
import tomllib
from pathlib import Path

from tractor.biqwen import LEGACY_SCHEMA, schema_for, verify_bundle
from tractor.coverage_assembly import sha
from tractor.generalist_data import read_episodes
from tractor.jsonio import dumps, loads
from tractor.model import field_ids
from tractor.pipeline_evaluation import evaluate_pipeline, typed_requests_for
from tractor.source_policy import excluded_source

root = Path('reports/biqwen-correction-20261026/legacy-baseline')
root.mkdir(exist_ok=True)
if (root / 'verification.json').exists():
    raise FileExistsError('Legacy audit is already complete')
original = Path('reports/biqwen-20261021/plain-quantity-schema')
baseline = loads((original / 'metrics.json').read_text())
previous = subprocess.check_output(['git', 'show', 'a7e801d:src/tractor/biqwen.py'], text=True)
assert hashlib.sha256(previous.encode()).hexdigest() == baseline['prediction_provenance']['adopted_backend_sha256']


def nodes(text):
    return {node.name: node for node in ast.parse(text).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


old, new = nodes(previous), nodes(Path('src/tractor/biqwen.py').read_text())
for name in ['verify_bundle', 'load_runtime', 'assign_occurrences', 'decode_multiple']:
    assert ast.dump(old[name]) == ast.dump(new[name]), name
old_methods = {n.name: n for n in old['BiQwenExtractor'].body if isinstance(n, ast.FunctionDef)}
new_methods = {n.name: n for n in new['BiQwenExtractor'].body if isinstance(n, ast.FunctionDef)}
for name in old_methods:
    changed = new_methods[name]
    if name == 'candidates':
        for node in ast.walk(changed):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'schema_for':
                assert ast.dump(node.args[1]) == ast.dump(ast.parse('self.schema_version', mode='eval').body)
                node.args.pop()
    assert ast.dump(old_methods[name]) == ast.dump(changed), name
namespace = {'field_ids': field_ids}
exec(compile(ast.Module(body=[old['schema_for']], type_ignores=[]), '<frozen-schema>', 'exec'), namespace)
rows = list(read_episodes('data/biqwen-full-de-en-v2/dev.episodes.jsonl'))
assert len(rows) == 4429 and not any(excluded_source(row) for row in rows)
assert sha(Path('data/biqwen-full-de-en-v2/dev.episodes.jsonl')) == baseline['input_sha256']
requests = 0
for row in rows:
    for request in typed_requests_for(row).values():
        assert namespace['schema_for'](request) == schema_for(request, LEGACY_SCHEMA)
        requests += 1

snapshot = Path('reports/biqwen-20261021/full-dev/pipeline-source')
changed_sources = {'pipeline_evaluation.py', 'document_eval.py', 'backends.py', 'biqwen.py'}
for name, digest in baseline['pipeline_sources'].items():
    if name not in changed_sources:
        assert sha(Path('src/tractor', name)) == digest, name
# The only old evaluator changes are early CORD rejection, verified above not
# to apply. All other top-level functions retain identical ASTs.
old_eval = nodes((snapshot / 'pipeline_evaluation.py').read_text())
new_eval = nodes(Path('src/tractor/pipeline_evaluation.py').read_text())
for name, node in old_eval.items():
    if name in {'typed_requests_for', 'evaluate_pipeline'}:
        node_new = new_eval[name]
        node_new.body = [n for n in node_new.body if not (
            isinstance(n, ast.ImportFrom) and n.module == 'source_policy')]
        node_new.body = [n for n in node_new.body if not (isinstance(n, ast.If)
                        and 'excluded_source' in ast.unparse(n.test))]
    assert ast.dump(node) == ast.dump(new_eval[name]), name
for name, digest in baseline['normalization_artifacts'].items():
    if name != 'uv.lock':
        assert sha(Path(name)) == digest, name
old_lock = tomllib.loads(subprocess.check_output(['git', 'show', 'a7e801d:uv.lock'], text=True))
new_lock = tomllib.loads(Path('uv.lock').read_text())
assert [p for p in old_lock['package'] if p['name'] != 'tractor'] == [
    p for p in new_lock['package'] if p['name'] != 'tractor']
lock = verify_bundle(Path('models/biqwen-extract-decide-4b'))
assert {k: v for k, v in lock['files'].items() if k.endswith('.safetensors')} == baseline['weights_sha256']
assert sha(original / 'predictions.jsonl') == baseline['prediction_provenance']['combined_prediction_sha256']
predictions = {v['id']: {k: x for k, x in v.items() if k != 'id'}
               for v in map(loads, (original / 'predictions.jsonl').open())}
assert set(predictions) == {row.id for row in rows}
policy = Path('configs/extraction-quality.json')
assert sha(policy) == baseline['policy_sha256']
report = evaluate_pipeline(rows, predictions, loads(policy.read_text()))
for section in ['scores', 'field_checks', 'rejection_checks']:
    assert loads(dumps(report[section])) == baseline[section], section
report.update(scope='Frozen full Dev predictions reused after inference/schema/dependency audit; rescored, no new inference.',
              source_predictions=str(original / 'predictions.jsonl'),
              source_predictions_sha256=sha(original / 'predictions.jsonl'),
              input_sha256=baseline['input_sha256'], schema_version=LEGACY_SCHEMA)
(root / 'metrics.json').write_text(dumps(report, indent=2) + '\n')
(root / 'verification.json').write_text(dumps({
    'status': 'passed', 'dev_documents': len(rows), 'identical_legacy_request_schemas': requests,
    'unchanged_inference_methods': list(old_methods), 'unchanged_decoder': True,
    'unchanged_dependency_packages': True, 'unchanged_weights': True,
    'scores_and_issue_checks_identical_after_rescoring': True, 'model_inference_performed': False,
}, indent=2) + '\n')
print(dumps({'status': 'passed', 'documents': len(rows), 'requests': requests}), flush=True)
