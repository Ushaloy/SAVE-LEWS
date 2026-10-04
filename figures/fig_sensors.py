from common import out, TH, PR, DATA_TH
import os
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from data_pipeline import load_device
from pr_data import load
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 8.5})
S = {}
g8 = load_device(os.path.join(DATA_TH, 'pinn_108_new.csv')).loc['2025-11-01 10:40':]; g7 = load_device(os.path.join(DATA_TH, '107_pinn.csv'))
S['Thailand Dev108, 30 cm'] = g8.theta; S['Thailand Dev107, 30 cm'] = g7.theta
u = load('utuado'); t = load('toronegro')
S['Utuado, 27 cm'] = u['vwc_SP1_27cm_ccpercc']; S['Utuado, 72 cm'] = u['vwc_SP1_72cm_ccpercc']
S['Toro Negro, 30 cm'] = t['vwc_sp1_30cm_ccpercc']; S['Toro Negro, 90 cm'] = t['vwc_sp1_90cm_ccpercc']
col = {'Thailand': '#eb6834', 'Utuado': '#2a78d6', 'Toro Negro': '#4a3aa7'}
fig = plt.figure(figsize=(10, 4.4)); gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
ax = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])
names = list(S)
for i, nm in enumerate(names):
    s = S[nm]; c = col[nm.split(',')[0].split(' Dev')[0]]
    idx = s.index; ok = s.notna().values
    # daily availability
    daily = s.notna().resample('1D').mean()
    t0 = daily.index[0]
    x = (daily.index - t0).days
    for xi, v in zip(x, daily.values):
        ax.add_patch(plt.Rectangle((xi, i - 0.35), 1, 0.7, color=c if v > 0.5 else '#e6e5e0', lw=0))
    ax.text(-4, i, nm, ha='right', va='center', fontsize=7.8)
    ax.text(x[-1] + 6, i, f'{100 * (1 - s.notna().mean()):.0f}% missing', va='center', fontsize=7, color='#52514e')
ax.set_xlim(-2, 720); ax.set_ylim(len(names) - 0.4, -0.6); ax.set_yticks([]); ax.spines['left'].set_visible(False)
ax.set_xlabel('days since start of each record'); ax.set_title('(a) Data continuity (coloured: day with >50% of records)', fontsize=8.5, loc='left')
# (b) distributions of input probes
inp = [('Thailand Dev108, 30 cm', g8.theta), ('Thailand Dev107, 30 cm', g7.theta), ('Utuado, 27 cm', u['vwc_SP1_27cm_ccpercc']), ('Toro Negro, 30 cm', t['vwc_sp1_30cm_ccpercc'])]
for i, (nm, s) in enumerate(inp):
    v = s.dropna().values; c = col[nm.split(',')[0].split(' Dev')[0]]
    p1, p25, p50, p75, p99 = np.percentile(v, [1, 25, 50, 75, 99])
    ax2.plot([p1, p99], [i, i], color=c, lw=1.5); ax2.add_patch(plt.Rectangle((p25, i - 0.18), p75 - p25, 0.36, color=c, alpha=0.5, lw=0))
    ax2.plot(p50, i, '|', color='#1f1e1c', ms=10)
    ax2.text(p99 + 0.01, i, f'range {p99 - p1:.3f}', va='center', fontsize=7, color='#52514e')
ax2.set_yticks(range(len(inp))); ax2.set_yticklabels([n for n, _ in inp], fontsize=7.8); ax2.set_ylim(len(inp) - 0.5, -0.5)
ax2.set_xlabel('volumetric water content (m³ m⁻³)'); ax2.set_xlim(0.05, 0.68)
ax2.set_title('(b) Input-probe range (line: P1–P99; box: quartiles)', fontsize=8.5, loc='left')
fig.tight_layout(); fig.savefig(out('Fig02.png'), dpi=300)
