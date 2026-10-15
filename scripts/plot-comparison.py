"""Render the published comparison: uv run --no-project --with matplotlib python scripts/plot-comparison.py."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
report = json.loads((ROOT/'benchmarks/results/summary.json').read_text())
providers = report['providers']
names = ['tractor', 'haiku', 'luna']
labels = ['tractor · CPU', 'Haiku 5.5 · API', 'GPT-6 Luna · API']
colors = ['#70efbe', '#96abc3', '#96abc3']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'text.color': '#ffffff', 'axes.labelcolor': '#cbd5e1',
                     'xtick.color': '#cbd5e1', 'ytick.color': '#ffffff',
                     'svg.fonttype': 'none'})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={'wspace': .65})
fig.patch.set_facecolor('#101c2b')
for ax, key, title, limit in [
    (axes[0], 'accuracy', 'Complete responses correct · higher is better', 100),
    (axes[1], 'latency', 'Median response time · lower is better', None),
]:
    ax.set_facecolor('#101c2b')
    values = [providers[n]['rates']['exact']*100 if key == 'accuracy'
              else providers[n]['p50_ms'] for n in names]
    ax.barh(labels, values, color=colors, height=.52)
    ax.invert_yaxis()
    for index, value in enumerate(values):
        text = f'{value:.2f}%' if key == 'accuracy' else f'{value:,.0f} ms'
        ax.text(value + (2 if key == 'accuracy' else max(values)*.025), index,
                text, va='center', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 120 if limit else max(values)*1.38)
    ax.set_xticks([0, 25, 50, 75, 100] if limit else ax.get_xticks()[:-1])
    ax.set_title(title, loc='left', fontsize=10, color='#cbd5e1', pad=18)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='y', length=0, pad=8)
fig.suptitle('tractor · Specialized extraction, measured', x=.04, ha='left',
             fontsize=20, fontweight='bold', color='#70efbe')
fig.text(.04, .09, '680 synthetic requests · prompt-only LLM baselines · reasoning disabled · no retries',
         fontsize=10, color='#cbd5e1')
fig.text(.04, .035, 'CPU loopback vs remote cloud APIs; timings include network overhead. See docs/benchmarks.md.',
         fontsize=9, color='#96abc3')
fig.subplots_adjust(left=.16, right=.96, top=.73, bottom=.24)
fig.savefig(ROOT/'docs/brand/comparison.svg', facecolor=fig.get_facecolor())

# Keep the generated SVG clean in Git diffs without changing its geometry.
svg = ROOT/"docs/brand/comparison.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
