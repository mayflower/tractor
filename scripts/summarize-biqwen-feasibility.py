"""Compare completed diagnostic arms, retaining reference and source qualifications."""

import argparse
import json
from collections import Counter
from pathlib import Path


def aggregate(report, prefix='shape:'):
    counts = Counter()
    excluded = []
    for name, score in report['scores'].items():
        if not name.startswith(prefix):
            continue
        if not score.get('reference_complete', True):
            excluded.append(name)
            continue
        counts.update(score['counts'])
    denominator = 2 * counts['field_tp'] + counts['field_fp'] + counts['field_fn']
    return {'field_f1': 2 * counts['field_tp'] / denominator if denominator else None,
            'issue_accuracy': counts['issue_correct'] / counts['issue_fields'] if counts['issue_fields'] else None,
            'counts': dict(counts), 'excluded_incomplete_groups': excluded}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    arms = {name: json.loads((args.root / name / 'status.json').read_text()) for name in ['causal', 'hybrid']}
    for arm in arms.values():
        if arm['status'] != 'completed' or arm['full_training'] is not False:
            raise ValueError('Both bounded diagnostic arms must have completed')
    for key in ['initial_trainable_sha256', 'experiment_sha256']:
        if arms['causal'][key] != arms['hybrid'][key]:
            raise ValueError(f'Unmatched initial state or experiment: {key}')
    for key in ['schedule_sha256', 'request_presentations', 'unique_requests', 'steps']:
        if arms['causal']['training'][key] != arms['hybrid']['training'][key]:
            raise ValueError(f'Unmatched training budget: {key}')
    result = {'scope': 'bounded feasibility, not full training or release qualification',
              'matched_initial_state_and_schedule': True, 'arms': {}}
    for name, status in arms.items():
        phases = {}
        for phase in ['learn-before', 'learn-after', 'dev-before', 'dev-after']:
            report = json.loads((args.root / name / f'{phase}.json').read_text())
            phases[phase] = {'aggregate_complete_reference_groups': aggregate(report),
                             'scores': report['scores'], 'evaluation_contract': report['evaluation_contract'],
                             'rejection_checks': report['rejection_checks']}
        result['arms'][name] = {'phases': phases, 'training_requests_per_second': status['training']['requests_per_second']}
    (args.root / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: {phase: value['aggregate_complete_reference_groups']
                            for phase, value in arm['phases'].items()} for name, arm in result['arms'].items()}, indent=2))


if __name__ == '__main__':
    main()
