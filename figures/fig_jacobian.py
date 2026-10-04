"""Fig. 11 (revised manuscript): local identifiability from the Jacobian of the sensor prediction."""
from common import out, TH
import os, json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e5e0',
                     'grid.linewidth': 0.6, 'font.size': 8.5})
S = json.load(open(os.path.join(TH, 'res_rev_summary.json')))['jacobian']
order = [k for k in ['Thailand 30 cm', 'Utuado 57 cm', 'Utuado 72 cm', 'Toro Negro 90 cm', 'Toro Negro 110 cm'] if k in S]
COL = {'Thailand 30 cm': '#eb6834', 'Utuado 57 cm': '#4a3aa7', 'Utuado 72 cm': '#7a6cc4', 'Toro Negro 90 cm': '#8c8a83',
       'Toro Negro 110 cm': '#b3b1a8'}
fmt = lambda k: f'{k:.0f}' if k < 100 else f'{k:.0e}'.replace('e+0', '×10$^{') + '}$'
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
for a, (key, rk, cond, title) in enumerate([('sv_scaled', 'practical_rank', 'condition', 'a  all parameters'),
                                            ('matrix_sv', 'matrix_rank', 'matrix_condition', 'b  matrix parameters K$_s$, α, n only')]):
    ax = axs[a]
    for i, k in enumerate(order):
        v = S[k]; sv = np.array(v[key]); r = v['resid_rmse']
        x = i + np.linspace(-0.2, 0.2, len(sv))
        ax.scatter(x, sv, s=30, color=COL[k], edgecolor='white', lw=0.8, zorder=3)
        ax.plot([i - 0.32, i + 0.32], [r, r], color='#1f1e1c', lw=1.2, ls='--', zorder=2)
        n = v['n_params'] if a == 0 else 3
        ax.text(i, 0.6, f"{v[rk]}/{n}\nκ = {fmt(v[cond])}", ha='center', va='bottom', fontsize=6.8, color='#3d3d3a')
    ax.set_yscale('log'); ax.set_ylim(1e-5, 3)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([k.replace(' ', '\n', 1) if k.startswith('Toro') else k.replace(' ', '\n', 1) for k in order], fontsize=7.5)
    ax.set_title(title, loc='left', fontsize=8.8, weight='bold')
axs[0].set_ylabel('RMS change of the prediction per unit\nparameter change along each direction (m³ m⁻³)')
axs[0].plot([], [], color='#1f1e1c', ls='--', label='model residual error (RMSE)')
axs[0].legend(frameon=False, fontsize=7.2, loc='lower left')
fig.text(0.01, 0.005, 'Dots: singular values of J/√N. Units: factor e for K$_s$, α, n−1 and the interception and bypass-width parameters; unit logit for β; 0.1 m for bypass depth; '
         '0.05 m³ m⁻³ for θ$_s$, θ$_r$.\nNumbers: practical rank (directions above the residual error) / number of parameters, and condition number κ. '
         'Thailand: 8 parameters (Richards, interception, bypass); Puerto Rico: 5.', fontsize=6.3, color='#52514e')
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(out('Fig11_jacobian.png'), dpi=300)
print('ok')
