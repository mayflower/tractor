"""Per-area generalist quality floor; no compensation by an average or product score."""

import math
from decimal import Decimal


def quality_gate(scores, policy):
    threshold = policy['minimum_exclusive']
    if not isinstance(threshold, (int, float, Decimal)) or isinstance(threshold, bool) or not 0 <= threshold < 1:
        raise ValueError('Invalid quality threshold')
    if not policy['required_buckets']:
        raise ValueError('Quality policy must declare required areas')
    metrics = {f'bucket:{name}': scores.get('buckets', {}).get(name, {}).get(policy['bucket_metric'])
               for name in policy['required_buckets']}
    metrics.update({f'classification:{name}': scores.get('classification_tasks', {}).get(name)
                    for name in policy['required_classification_tasks']})
    checks = {name: {'score': value, 'passed': isinstance(value, (float, int, Decimal)) and not isinstance(value, bool)
                    and math.isfinite(value) and threshold < value <= 1}
              for name, value in metrics.items()}
    return {'passed': all(check['passed'] for check in checks.values()), 'minimum_exclusive': threshold,
            'checks': checks, 'failed_areas': [name for name, check in checks.items() if not check['passed']],
            'qualification': policy['qualification']}
