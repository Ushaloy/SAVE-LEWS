from common import out, TH, PR, DATA_TH
import os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
plt.rcParams.update({'font.size': 8})
fig, axs = plt.subplots(1, 3, figsize=(7.6, 6.0), sharey=True)
INK, MUT = '#1f1e1c', '#5c5b56'
SITES = [
 dict(t='(a) Thailand, Dev107/Dev108\nLoRaWAN nodes (low-cost)', base=None, fp=1.0,
      layers=[(0, 1.2, '#e9dcc6', 'residual soil\n[texture: to add]')], sm=[30], inp=30, sec=None, tgt=None, pz=[], lab=[],
      note='30 cm FDR probe rises 5–30 min\nafter ~5 mm of storm rain:\nfaster than matrix flow allows\n→ fast-pathway signal'),
 dict(t='(b) Utuado, Puerto Rico\n42°, coarse soil (SM)', base=0.92, fp=None,
      layers=[(0, 0.75, '#efe3c8', 'sandy soil'), (0.75, 1.05, '#d8c49e', 'less conductive\nlayer')],
      sm=[12, 27, 42, 57, 72], inp=27, tgt=72, sec=57, pz=[40, 55],
      lab=[(22, 'K$_s$ 1.0×10$^{-4}$'), (57, 'K$_s$ 4.1×10$^{-3}$'), (92, 'K$_s$ 3.6×10$^{-4}$')],
      note='57 cm: vertical response\n(median lag 1.1 h)\n72 cm: wets faster and further\nthan 57 cm in large storms'),
 dict(t='(c) Toro Negro, Puerto Rico\n45°, fine soil (CL/MH)', base=1.32, fp=None,
      layers=[(0, 1.4, '#dccfbf', 'volcani-\nclastic\nclayey\nsoil')], sm=[30, 50, 70, 90, 110], inp=30, tgt=90, sec=110, pz=[83, 126],
      lab=[(90, 'K$_s$ 3.6×10$^{-5}$'), (117, 'K$_s$ 7.0×10$^{-5}$')],
      note='30 cm input near saturation\n(rose in 5 of 84 storms);\n110 cm rose before 90 cm\nin 60% of storms'),
]
for ax, s in zip(axs, SITES):
    ax.set_xlim(0, 1.25); ax.set_ylim(1.78, -0.12); ax.set_xticks([]); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    for a, b, c, lab in s['layers']:
        ax.add_patch(Rectangle((0.3, a), 0.28, b - a, fc=c, ec='none'))
        ax.text(0.44, a + 0.1 if b - a < 0.4 else a + 0.62, lab, ha='center', va='center', fontsize=6.8, color=MUT)
    ax.plot([0.27, 0.61], [0, 0], color=INK, lw=1.2)
    ax.text(0.44, -0.06, 'rain gauge ▼', ha='center', fontsize=7, color='#2a78d6')
    for d in s['sm']:
        z = d / 100; role = 'input' if d == s['inp'] else ('primary target' if d == s.get('tgt') else ('secondary target' if d == s.get('sec') else ''))
        col = '#2a78d6' if role == 'input' else ('#b8452a' if role == 'primary target' else ('#eb6834' if role == 'secondary target' else '#3d3d3a'))
        ax.plot([0.58, 0.66], [z, z], color=col, lw=2.2); ax.plot(0.66, z, 's', color=col, ms=5)
        ax.text(0.7, z, f'{d} cm' + (f'  {role}' if role else ''), va='center', fontsize=6.8, color=col)
    for d in s['pz']:
        z = d / 100; ax.plot(0.54, z, 'v', color='#4a3aa7', ms=5); ax.text(0.51, z, f'P{d}', va='center', ha='right', fontsize=6, color='#4a3aa7')
    for d, t in s['lab']:
        ax.plot([0.28, 0.3], [d / 100, d / 100], color='#6b5a3a', lw=0.8); ax.text(0.27, d / 100, f'{d} cm: ' + t.replace('K$_s$ ', '') , fontsize=5.6, color='#6b5a3a', ha='right', va='center')
    if s['base']:
        ax.plot([0.3, 0.58], [s['base'], s['base']], color=INK, lw=1, ls='--'); ax.text(0.44, s['base'] + 0.02, f'pit base {int(s["base"]*100)} cm', fontsize=6.3, va='top', ha='center')
    if s['fp']:
        ax.plot([0.3, 0.58], [s['fp'], s['fp']], color='#b8452a', lw=1.2, ls='--'); ax.text(0.44, s['fp'] + 0.03, 'expected failure plane ≈1 m\n(not instrumented)', fontsize=6.5, va='top', ha='center', color='#b8452a')
    ax.text(0.62, 1.76, s['note'], ha='center', va='bottom', fontsize=6.6, style='italic', color='#4a3aa7',
            bbox=dict(boxstyle='round,pad=0.3', fc='#f5f3fb', ec='#c9c3ea', lw=0.6))
    ax.set_title(s['t'], fontsize=8.2, loc='left')
    for sp in ('top', 'right', 'bottom'): ax.spines[sp].set_visible(False)
axs[0].set_ylabel('depth below surface (m)')
for a in axs[1:]:
    a.spines['left'].set_visible(False); a.tick_params(left=False)
fig.text(0.5, 0.005, '▼ Pxx: piezometer at xx cm.   Brown labels: laboratory K$_s$ (drying curve, cm s$^{-1}$) of the sample at that depth.', ha='center', fontsize=6.8, color=MUT)
fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(out('Fig01.png'), dpi=300, bbox_inches='tight')
