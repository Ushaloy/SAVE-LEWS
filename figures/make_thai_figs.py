from common import out, TH
import os
import json, glob, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e5e0',
                     'grid.linewidth': 0.6, 'font.size': 9})
# ---- Fig 3: physics consistency vs held-out skill (values from step-2 notebook summary table)
S2 = json.load(open(os.path.join(TH, 'step2_summary.json')))      # written by thailand/make_figs_thai.py
net = {t: (S2[t]['rho_R'], S2[t]['net_heldout_60']) for t in ['A_weak', 'B_strong', 'C_nophys', 'M0_s0', 'M0_s1', 'M1_s0', 'M1_s1']}
sol = {t: S2[t]['solver_heldout_60'] for t in ['M0_s0', 'M0_s1', 'M1_s0', 'M1_s1', 'apriori']}
FAM = {'A': '#2a78d6', 'B': '#eb6834', 'C': '#1baf7a', 'M0': '#4a3aa7', 'M1': '#e87ba4'}
LAB = {'A_weak': 'A: weak residual', 'B_strong': 'B: strong residual', 'C_nophys': 'C: no physics',
       'M0_s0': 'M0 (seed 0)', 'M0_s1': 'M0 (seed 1)', 'M1_s0': 'M1 (seed 0)', 'M1_s1': 'M1 (seed 1)'}
OFF = {'M0_s1': (6, -10), 'M0_s0': (6, 4), 'M1_s0': (6, 4), 'M1_s1': (8, -3)}
fig, ax = plt.subplots(figsize=(6.6, 4.3))
for t, (x, y) in net.items():
    ax.scatter(x, y, s=60, color=FAM[t.split('_')[0]], edgecolor='white', lw=1.5, zorder=3)
    ax.annotate(LAB[t], (x, y), xytext=OFF.get(t, (6, 4)), textcoords='offset points', fontsize=8, color='#3d3d3a')
sv = sorted(sol, key=lambda t: -sol[t])
for r, t in enumerate(sv):
    y = sol[t]; col = FAM.get(t.split('_')[0], '#8c8a83')
    ax.scatter(0, y, s=55, marker='D', color=col, edgecolor='white', lw=1.3, zorder=4)
    lab = 'solver only, a priori parameters' if t == 'apriori' else 'solver only, ' + LAB[t] + ' parameters'
    ax.annotate(lab, (0, y), xytext=(0.36, 0.145 - r * 0.03), textcoords='data', fontsize=7.5, color='#3d3d3a', va='center',
                ha='left', arrowprops=dict(arrowstyle='-', color='#c3c2b7', lw=0.6, relpos=(0, 0.5), shrinkA=2))
ax.axhline(0, color='#0b0b0b', lw=0.8); ax.set_xlim(-0.05, 0.9); ax.set_ylim(-0.02, 0.66)
ax.set_xlabel('residual-to-tendency ratio ρ$_R$ at 30 cm during wetting (0 = exact Richards)')
ax.set_ylabel('held-out storm skill, 60 min, oracle rain')
fig.tight_layout(); fig.savefig(out('Fig06.png'), dpi=300); plt.close(fig)

# ---- Fig 11: zero-shot transfer to Dev107 incl. B-TS
dates = ['10 Nov 2025', '19 Nov 2025', '25 Dec 2025', '28 Dec 2025']
cwd = os.getcwd(); os.chdir(TH)
from analyze_step3 import zeroshot
ZS = zeroshot(); ZS = ZS[ZS['mode'] == 'oracle']; os.chdir(cwd)
row = lambda m: [float(ZS[ZS.model == m][f's{i}'].iloc[0]) for i in range(4)]
Z = {'PG-MLP': [row(f'PG-MLP seed{s}') for s in range(3)], 'M0': [row('M0')], 'M2': [row('M2')], 'M3': [row('M3')]}
R4 = json.load(open(os.path.join(TH, 'bts_out', 'res4_all.json')))   # B-TS posterior from all Dev108 storms (NUTS)
Z['B-TS'] = [[1 - R4[f'd107s{i}_nuts_oracle']['storm_60']['mse'] / R4[f'd107s{i}_nuts_oracle']['storm_60']['msep'] for i in range(4)]]
cov = [R4[f'd107s{i}_nuts_oracle']['storm_60']['coverage'] for i in range(4)]
print('Fig15 zero-shot values:', {k: np.round(v, 2).tolist() for k, v in Z.items()}, 'coverage', np.round(cov, 2))
COL = {'PG-MLP': '#2a78d6', 'M0': '#eb6834', 'M2': '#1baf7a', 'M3': '#e87ba4', 'B-TS': '#4a3aa7'}
LBL = {'PG-MLP': 'PG-MLP (3 seeds)', 'M0': 'M0 Richards PINN', 'M2': 'M2 fast-store PINN', 'M3': 'M3 bounded fast store',
       'B-TS': 'B-TS Bayesian two-store (posterior median)'}
fig, ax = plt.subplots(figsize=(7.6, 4.0))
fams = list(Z)
for j, fm in enumerate(fams):
    dx = (j - 2) * 0.15
    for i in range(4):
        v = np.array([s[i] for s in Z[fm]])
        mk = 'D' if fm == 'B-TS' else 'o'
        ax.scatter(np.full(len(v), i + dx), np.clip(v, -1.5, 1), s=40 if fm != 'B-TS' else 48, marker=mk, color=COL[fm],
                   edgecolor='white', lw=1, label=LBL[fm] if i == 0 else None, zorder=3)
        cl = v[v < -1.5]
        if len(cl):
            lab = f'{cl.max():.1f}' if len(cl) == 1 else f'{cl.max():.1f} to {cl.min():.1f}'
            ax.text(i + dx, -1.40, lab, rotation=90, fontsize=6.5, ha='center', va='bottom')
        if fm == 'B-TS':
            ax.text(i + dx, v[0] + 0.1, f'cov. {cov[i]:.2f}', fontsize=6.5, ha='center', color=COL[fm])
ax.axhline(0, color='#0b0b0b', lw=0.8); ax.set_ylim(-1.55, 1.05)
ax.set_xticks(range(4)); ax.set_xticklabels(dates)
ax.set_ylabel('Dev107 storm skill vs persistence\n(60 min, oracle rain)')
ax.legend(frameon=False, fontsize=7.2, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(); fig.savefig(out('Fig15.png'), dpi=300); plt.close(fig)

# ---- Fig S1: parameter spread across LOSO folds
P = {m: [json.load(open(os.path.join(TH, f'res3_loso_{m}_f{k}.json')))['params'] for k in range(7)] for m in ['M0', 'M2', 'M3']}
pars = [('Ks_mph', 'K$_s$ (m h$^{-1}$)', True, 0.0043), ('alpha', 'α (m$^{-1}$)', False, 5.9), ('n', 'n', False, 1.48),
        ('c_mm', 'interception capacity c (mm)', False, 5.0), ('beta', 'bypass fraction β', False, None)]
C2 = {'M0': '#eb6834', 'M2': '#1baf7a', 'M3': '#e87ba4'}
fig, ax = plt.subplots(1, 5, figsize=(10.5, 3.2))
for j, (c, lab, lg, ref) in enumerate(pars):
    for i, m in enumerate(['M0', 'M2', 'M3']):
        d = [p[c] for p in P[m]]
        ax[j].scatter(np.full(7, i) + np.linspace(-0.14, 0.14, 7), d, s=26, color=C2[m], edgecolor='white', lw=0.8, zorder=3)
    if ref is not None: ax[j].axhline(ref, color='#52514e', lw=0.9, ls='--')
    if lg: ax[j].set_yscale('log')
    ax[j].set_xticks(range(3)); ax[j].set_xticklabels(['M0', 'M2', 'M3']); ax[j].set_title(lab, fontsize=8.5)
fig.text(0.01, 0.01, 'Dashed: a priori value (K$_s$, α, n) or observed ~5 mm rise threshold (c). One point per leave-one-storm-out fold.',
         fontsize=7, color='#52514e')
fig.tight_layout(); fig.subplots_adjust(bottom=0.17); fig.savefig(out('FigS1.png'), dpi=300); plt.close(fig)
print('ok')
