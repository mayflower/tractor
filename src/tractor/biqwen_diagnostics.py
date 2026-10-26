"""Offline span-stage observations; never used to select or normalize API values."""

from collections import Counter

from .api_training import supervised_requests
from .jsonio import dumps
from .model import field_ids
from .pipeline_evaluation import infer


class EvidenceTrace:
    """Observe proposal pools in the same production invocation, with no gold input."""

    def __init__(self, backend):
        self.backend = backend
        self.requests = {}
        self.current = None

    def __getattr__(self, key):
        return getattr(self.backend, key)

    def __enter__(self):
        self.original = self.backend.model._make_pool

        def observe(meta, field, *args):
            pool = self.original(meta, field, *args)
            if meta['gold'] is not None:
                raise ValueError('Inference trace must never receive gold')
            model = self.backend.model
            anchor = field.get('kind') == 'anchor'
            threshold = model.config.anchor_threshold if anchor else model.config.source_threshold
            temperature = model._temperature('anchor' if anchor else 'source')
            probabilities = (pool['source_logits'].detach().float() / temperature).sigmoid().tolist()
            pairs = [list(self.backend.encoding.token_span_to_chars(meta['text'], meta['offsets'], pair))
                     for pair in pool['pairs']]
            self.current['fields'][field['id']] = {
                'proposed': pairs, 'source_accepted': [pair for pair, p in zip(pairs, probabilities, strict=True)
                                                     if p >= threshold]}
            return pool

        self.backend.model._make_pool = observe
        return self

    def __exit__(self, *_):
        self.backend.model._make_pool = self.original

    def candidates(self, request):
        key = dumps(request.model_dump(), sort_keys=True)
        self.current = {'fields': {}, 'records': [], 'error': None}
        self.requests[key] = self.current
        try:
            result = self.backend.candidates(request)
            self.current['records'] = result
            return result
        except Exception as exc:
            self.current['error'] = getattr(exc, 'code', type(exc).__name__)
            raise


def diagnose(episode, traces):
    """Exact source-span diagnostics only; final-value evaluation is independent."""
    rows, rejected = supervised_requests(episode)
    counts, examples = Counter(), []
    counts['unscored_tasks'] = len(rejected)
    for task, request, gold_records in rows:
        trace = traces.get(dumps(request.model_dump(), sort_keys=True))
        if trace is None:
            counts['requests_not_reached'] += 1
            continue
        counts['requests'] += 1
        if trace['error']:
            counts['request_error:' + trace['error']] += 1
        ids = field_ids(request)
        selected = trace['records']
        for record_index, gold in enumerate(gold_records):
            if request.anchor:
                anchor = {(s['start'], s['end']) for s in gold[request.anchor]}
                owners = [row for row in selected if anchor & {
                    (s['start'], s['end']) for s in row.get(request.anchor, [])}]
            else:
                owners = selected
            for key, spans in gold.items():
                field = trace['fields'].get(ids[key], {})
                proposed = {tuple(pair) for pair in field.get('proposed', [])}
                accepted = {tuple(pair) for pair in field.get('source_accepted', [])}
                owned = {(s['start'], s['end']) for row in owners for s in row.get(key, [])}
                for span in spans:
                    pair = (span['start'], span['end'])
                    counts['gold_spans'] += 1
                    counts['proposal_hits'] += pair in proposed
                    counts['source_hits'] += pair in accepted
                    counts['owner_hits'] += pair in owned
                    stage = ('missing_proposal' if pair not in proposed else
                             'source_rejection' if pair not in accepted else
                             'anchor_missing_or_boundary_variant' if request.anchor and not owners else
                             'assignment_mismatch' if pair not in owned else 'selected_evidence')
                    counts[stage] += 1
                    if stage != 'selected_evidence' and len(examples) < 20:
                        examples.append({'task': task, 'record': record_index, 'field': key,
                                         'span': list(pair), 'stage': stage})
    return {'counts': dict(counts), 'examples': examples,
            'scope': 'Exact annotated spans only; boundary variants can be correct final values. Not a quality gate.'}


def infer_with_diagnostics(backend, episode):
    with EvidenceTrace(backend) as trace:
        try:
            result = {'prediction': infer(trace, episode)}
        except Exception as exc:
            result = {'error': type(exc).__name__, 'message': str(exc)[:400]}
    # Gold is consulted only after the normal production prediction is complete.
    return result, diagnose(episode, trace.requests)
