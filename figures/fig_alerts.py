from common import out, TH, PR, DATA_TH
import os
import json, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'font.size': 8.5})
T = json.load(open(os.path.join(TH, 'thai_alerts.json')))['0.01']
P = json.load(open(os.path.join(PR, 'out', 'pr_alerts_rise.json')))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw=dict(width_ratios=[1, 1.35]))
mods = ['PG-MLP', 'M0', 'M2', 'M2np', 'M3', 'B-TS']
lab = {'PG-MLP': 'PG-MLP', 'M0': 'M0\nRichards', 'M2': 'M2\nfast store', 'M2np': 'M2np\nno physics', 'M3': 'M3\nbounded', 'B-TS': 'B-TS\nBayesian'}
x = np.arange(len(mods)); w = 0.26
for j, (k, c, nm) in enumerate([('POD', '#2a78d6', 'POD'), ('FAR', '#eb6834', 'FAR'), ('CSI', '#1baf7a', 'CSI')]):
    v = [T[m][k] for m in mods]; ax[0].bar(x + (j - 1) * w, v, w, color=c, label=nm)
    for xi, vi in zip(x, v): ax[0].text(xi + (j - 1) * w, vi + 0.015, f'{vi:.2f}', ha='center', fontsize=6, rotation=90)
ax[0].set_xticks(x); ax[0].set_xticklabels([lab[m] for m in mods], fontsize=7.5); ax[0].set_ylim(0, 1.1)
ax[0].set_title('(a) Thailand: 60-min rapid-wetting alert (Δθ ≥ 0.01)', fontsize=8.5, loc='left', pad=22)
ax[0].legend(frameon=False, fontsize=7.5, ncol=3, loc='upper right')
groups = [('utuado', '57 cm', 'Utuado 57 cm'), ('utuado', '72 cm', 'Utuado 72 cm'), ('toronegro', '90 cm', 'Toro Negro 90 cm')]
models = [('R-cal', 'R-cal', '#eb6834'), ('R-cal-Se (exploratory)', 'R-cal-Se', '#4a3aa7'), ('PG-MLP', 'PG-MLP (3 seeds)', '#2a78d6')]
xx = 0; ticks = []; tl = []
for site, z, name in groups:
    R = P[site][z]['results']
    for mi, (key, ml, col) in enumerate(models):
        if key == 'PG-MLP':
            vals = {k: np.mean([R[f'PG-MLP s{s}'][k] for s in range(3)]) for k in ('POD', 'POD_timely', 'FAR')}
            lag = np.median([R[f'PG-MLP s{s}']['median_timing_h'] for s in range(3)])
        else:
            vals = {k: R[key][k] for k in ('POD', 'POD_timely', 'FAR')}; lag = R[key]['median_timing_h']
        xp = xx + mi * 0.9
        ax[1].bar(xp - 0.28, vals['POD'], 0.27, color=col, alpha=0.35, label='POD (any time)' if (xx == 0 and mi == 0) else None)
        ax[1].bar(xp, vals['POD_timely'], 0.27, color=col, label=None)
        ax[1].bar(xp + 0.28, vals['FAR'], 0.27, color='white', edgecolor=col, hatch='///', lw=0.8)
        ax[1].text(xp, 1.04, f'{lag:+.1f} h', ha='center', fontsize=6.5, color=col)
        ticks.append(xp); tl.append(ml.split(' ')[0])
    ax[1].text(xx + 0.9, -0.2, name, ha='center', fontsize=8, weight='bold', transform=ax[1].transData)
    xx += 3.4
ax[1].set_xticks(ticks); ax[1].set_xticklabels(tl, fontsize=6.8); ax[1].set_ylim(0, 1.12)
ax[1].set_title('(b) Puerto Rico: storm-level deep-wetting alert from the shallow probe', fontsize=8.5, loc='left', pad=22)
from matplotlib.patches import Patch
ax[1].legend(handles=[Patch(fc='#8c8a83', alpha=0.35, label='POD, any time'), Patch(fc='#8c8a83', label='POD, within ±2 h of onset'),
                      Patch(fc='white', ec='#8c8a83', hatch='///', label='FAR')], frameon=False, fontsize=7, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0))
fig.text(0.01, 0.005, 'Level 1 (moisture prediction) → Level 2 (hydrological alert state, shown here) → Level 3 (slope-failure warning, not evaluated).', fontsize=6.8, color='#b8452a')
fig.text(0.56, 0.028, 'Numbers above bars: median alert timing relative to observed onset (positive = late).', fontsize=6.8, color='#52514e')
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(out('Fig_alerts_levels.png'), dpi=300)
