from common import out, TH, PR
import os
import json, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'font.size': 8.5})
C = {'U': '#8c8a83', 'B': '#9fc3ec', 'PG': '#2a78d6', 'R': '#eb6834', 'Se': '#4a3aa7'}
fig, ax = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw=dict(width_ratios=[0.9, 1.6]))
# (a) Thailand
A = json.load(open(os.path.join(TH, 'ablation_summary.json')))
th = {k: [tuple(A[k]['oracle60']), tuple(A[k]['past60'])] for k in ['U', 'B', 'PG']}
lab = {'U': 'unconstrained MLP\n(data-driven)', 'B': 'storage bounds\nonly', 'PG': 'full constraints\n(PG-MLP)'}
for g, mode in enumerate(['oracle rain', 'past only']):
    for j, k in enumerate(['U', 'B', 'PG']):
        lo, hi = th[k][g]; x = g * 1.2 + (j - 1) * 0.3
        ax[0].bar(x, (lo + hi) / 2, 0.27, color=C[k], label=lab[k] if g == 0 else None)
        ax[0].errorbar(x, (lo + hi) / 2, yerr=[[(hi - lo) / 2], [(hi - lo) / 2]], color='#1f1e1c', lw=0.9, capsize=2)
ax[0].axhline(0, color='#1f1e1c', lw=0.8); ax[0].set_xticks([0, 1.2]); ax[0].set_xticklabels(['oracle rain', 'past only'])
ax[0].set_ylabel('held-out skill vs persistence (60 min)'); ax[0].set_ylim(-0.9, 0.95)
ax[0].set_title('(a) Thailand: adding physical constraints', fontsize=8.5, loc='left')
ax[0].legend(frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
ax[0].text(0.02, 0.98, 'rain-monotonicity violations: ' + ' → '.join(f"{100 * A[k]['oracle_viol_monotone']:.2g}%" for k in ['U', 'B', 'PG']) +
                '\nwetting without rain: ' + ' → '.join(f"{100 * A[k]['oracle_viol_wet_dry']:.0f}%" for k in ['U', 'B', 'PG']), transform=ax[0].transAxes, fontsize=6.8, color='#52514e', va='top')
# (b) Puerto Rico
MU = json.load(open(os.path.join(PR, 'out', 'mlpU_summary.json')))
NE = {st: json.load(open(os.path.join(PR, 'out', f'se_nested_{st}.json'))) for st in ['utuado', 'toronegro']}
def rng(sk, pre): v = [sk[f'{pre} s{i}'] for i in range(3)]; return (min(v), max(v))
tg = []
for nm, st, which in [('Utuado 57 cm', 'utuado', 'secondary'), ('Utuado 72 cm', 'utuado', 'primary'),
                      ('Toro Negro 90 cm', 'toronegro', 'primary'), ('Toro Negro 110 cm', 'toronegro', 'secondary')]:
    sk = MU[f'{st} {which}']['skill']
    tg.append((nm, rng(sk, 'U'), rng(sk, 'PG-MLP'), sk['R-cal'], NE[st][which]['skill']['Se nested']))
for g, (nm, u, pg, rc, se) in enumerate(tg):
    for j, (k, v, l) in enumerate([('U', u, 'unconstrained MLP (data-driven)'), ('PG', pg, 'PG-MLP (constraint-level physics)'),
                                    ('R', rc, 'Richards, calibrated (registered)'), ('Se', se, 'Richards, normalised sensors (post hoc, nested)')]):
        x = g * 1.4 + (j - 1.5) * 0.3
        if isinstance(v, tuple):
            m = (v[0] + v[1]) / 2; ax[1].bar(x, m, 0.27, color=C[k], label=l if g == 0 else None)
            ax[1].errorbar(x, m, yerr=[[(v[1] - v[0]) / 2], [(v[1] - v[0]) / 2]], color='#1f1e1c', lw=0.9, capsize=2)
        else:
            ax[1].bar(x, max(v, -0.6), 0.27, color=C[k], label=l if g == 0 else None)
            if abs(v) < 0.02: ax[1].text(x, 0.02, '0.00', ha='center', fontsize=6.5, color=C[k])
            if v < -0.6: ax[1].text(x, -0.57, f'{v:.2f}', rotation=90, fontsize=6.5, ha='center', va='bottom', color='white')
ax[1].axhline(0, color='#1f1e1c', lw=0.8); ax[1].set_ylim(-0.6, 0.95)
ax[1].set_xticks([g * 1.4 for g in range(4)]); ax[1].set_xticklabels([t[0] for t in tg])
ax[1].set_ylabel('held-out skill vs no-change'); ax[1].set_title('(b) Puerto Rico: virtual deep sensor from the shallow probe', fontsize=8.5, loc='left')
ax[1].legend(frameon=False, fontsize=7, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.1))
ax[1].axvspan(2.1, 5.0, color='#fbeee8', zorder=0); ax[1].text(3.55, 0.88, 'failed screening (Step I)', ha='center', fontsize=7, color='#b8452a')
fig.tight_layout(); fig.savefig(out('Fig08.png'), dpi=300)
