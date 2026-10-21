"""Render current pipeline Dev metrics: python scripts/plot-comparison.py."""

import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
report = json.loads((ROOT / 'release/metrics.json').read_text())
rows = [
    ('Single object · field F1', 'shape:single_object:synthetic/teacher:field_f1'),
    ('Entity list · field F1', 'shape:entity_list:synthetic/teacher:field_f1'),
    ('Repeated records · field F1', 'shape:repeated_records:synthetic/teacher:field_f1'),
    ('Complete records · record F1', 'shape:repeated_records:synthetic/teacher:record_f1'),
]
parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="380" viewBox="0 0 1100 380" role="img" aria-labelledby="title description">
<title id="title">tractor {escape(report['release_version'])}: synthetic Dev pipeline results</title>
<desc id="description">Field F1 and complete-record F1 after normalization. Dev selected the checkpoint. No independent test or latency claim.</desc>
<rect width="1100" height="380" rx="16" fill="#101c2b"/>
<g font-family="system-ui, sans-serif">
<text x="34" y="44" fill="#70efbe" font-size="26" font-weight="700">tractor · Typed extraction, measured</text>
<text x="34" y="75" fill="#cbd5e1" font-size="15">Synthetic Dev · full production pipeline after normalization · model {escape(report['release_version'])}</text>
''']
for i, (label, key) in enumerate(rows):
    value = report['checks'][key]['score'] * 100
    y = 111 + i * 48
    color = '#70efbe' if value > 70 else '#f6c86d'
    parts.append(f'''<text x="34" y="{y + 20}" fill="#ffffff" font-size="16">{escape(label)}</text>
<rect x="330" y="{y}" width="600" height="28" rx="5" fill="#203247"/>
<rect x="330" y="{y}" width="{value * 6:.2f}" height="28" rx="5" fill="{color}"/>
<text x="950" y="{y + 20}" fill="#ffffff" font-size="18" font-weight="700">{value:.2f}%</text>
''')
parts.append('''<text x="34" y="329" fill="#cbd5e1" font-size="14">Complete-record F1 remains below the 70% target. Real-document results are reported separately.</text>
<text x="34" y="354" fill="#96abc3" font-size="14">Checkpoint-selection Dev evidence, not held-out test accuracy. See docs/benchmarks.md.</text>
</g></svg>
''')
(ROOT / 'docs/brand/comparison.svg').write_text(''.join(parts))
