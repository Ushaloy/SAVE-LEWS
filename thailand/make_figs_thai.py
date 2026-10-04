"""Thai (Dev108/Dev107) figures and summary tables from the stored checkpoints and result files.

Run from the thailand/ folder:  python make_figs_thai.py
Writes figs/*.png and step2_summary.json. No training is performed; torch is used only for
forward passes of the stored development-stage checkpoints (Fig. 5 and rho_R of Fig. 6).
"""
import os, json, warnings
import numpy as np, pandas as pd, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore'); torch.set_num_threads(2)
plt.show = lambda *a, **k: plt.close('all')          # the figure blocks were written for notebooks
os.makedirs('figs', exist_ok=True)
DATA = '../data/thailand'

# ---------------------------------------------------------------- development stage (Figs 5, 6)
from data_pipeline import load_device, storm_catalogue
from richards_pinn import *          # Physics, Stepper, eff_rain, residual, I_OBS, DT, ...
from train_utils import *            # make_window, analysis, evaluate, HORIZONS, ...

g108 = load_device(f'{DATA}/pinn_108_new.csv').loc['2025-11-01 10:40':]; S108 = storm_catalogue(g108)
win0 = make_window(g108, S108.iloc[0])
models = {}
for tag in ['A_weak', 'B_strong', 'C_nophys']:
    m = Stepper(Physics()); m.load_state_dict(torch.load(f'checkpoints/{tag}.pt')); models[tag] = m.eval()


@torch.no_grad()
def budget(m, win):
    ph = m.phys; eff = eff_rain(ph, win['rain']); A = analysis(m, win, eff)
    th1, q0, S = m(A[:-1], eff[:-1]); R, dthdt, div, Ss = residual(ph, A[:-1], th1, q0, S)
    rise = (win['obs'][1:] - win['obs'][:-1]) > 0.002
    return dict(A=A, eff=eff, dthdt=dthdt, div=div, S=Ss, R=R, rise=rise)


# rho_R = mean|R| / mean|dtheta/dt| at the probe cell on wet-up steps of the training storm
rho = {}
for t, m in models.items():
    b = budget(m, win0); r = b['rise']
    rho[t] = float(b['R'][r, I_OBS].abs().mean() / b['dthdt'][r, I_OBS].abs().mean())
HELD = [1, 3, 4, 5, 6]
FILES = {'A_weak': 'res_pinn.json', 'B_strong': 'res_B_strong.json', 'C_nophys': 'res_C_nophys.json'}
summ = {}
for t, f in FILES.items():
    d = json.load(open(f))
    summ[t] = dict(rho_R=rho[t], net_heldout_60=float(np.mean([d[f's{k}_oracle']['skill_storm_60'] for k in HELD])))
for t in ['M0_s0', 'M0_s1', 'M1_s0', 'M1_s1']:
    d = json.load(open(f'res2_{t}.json')); s = json.load(open(f'solver_{t}.json'))
    summ[t] = dict(rho_R=d['budget_wetup']['ratio_R_dthdt'],
                   net_heldout_60=float(np.mean([d[f'd108_s{k}_oracle']['skill_storm_60'] for k in HELD])),
                   solver_heldout_60=float(np.mean([s[f's{k}_oracle_60'] for k in HELD])))
s = json.load(open('solver_apriori.json'))
summ['apriori'] = dict(solver_heldout_60=float(np.mean([s[f's{k}_oracle_60'] for k in HELD])))
json.dump(summ, open('step2_summary.json', 'w'), indent=1)
print(pd.DataFrame(summ).T.round(3))

# Fig 5 (manuscript): training storm, 60-min oracle forecasts and residual at the probe
COL = {'A_weak': '#2a78d6', 'B_strong': '#eb6834', 'C_nophys': '#1baf7a'}
LBL = {'A_weak': 'A weak residual', 'B_strong': 'B strong residual', 'C_nophys': 'C no physics'}
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e5e0',
                     'grid.linewidth': 0.6, 'font.size': 9})
w = win0; t = w['t']; sel = (t >= S108.start[0] - pd.Timedelta('3h')) & (t <= S108.end[0] + pd.Timedelta('6h'))
fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True, gridspec_kw=dict(height_ratios=[1, 2, 1.4]))
b = budget(models['B_strong'], w)
ax[0].bar(t[sel], w['rain'].numpy()[sel], width=0.003, color='#8c8a83', label='rain (5 min)')
ax[0].bar(t[sel], b['eff'][:, 0].numpy()[sel], width=0.003, color='#eb6834', label='effective rain (after store, B)')
ax[0].set_ylabel('mm / 5 min'); ax[0].legend(frameon=False, fontsize=8)
ax[1].plot(t[sel], w['obs'].numpy()[sel], color='#0b0b0b', lw=2, label='observed θ, 30 cm')
for tag, m in models.items():
    r = evaluate(m, w, oracle=True); idx = r['idx'].numpy(); pred = r['traj'][:, 11, I_OBS].numpy()
    tt = t[idx + 12]; ok = (tt >= t[sel][0]) & (tt <= t[sel][-1])
    ax[1].plot(tt[ok], pred[ok], color=COL[tag], lw=1.5, label=f'{LBL[tag]}: 60-min forecast (oracle rain)')
ax[1].set_ylabel('θ (m³/m³)'); ax[1].legend(frameon=False, fontsize=8)
for tag in ['A_weak', 'B_strong']:
    bb = budget(models[tag], w)
    ax[2].plot(t[1:][sel[1:]], bb['R'][:, I_OBS].numpy()[sel[1:]] * 1e6, color=COL[tag], lw=1.5, label=f'{LBL[tag]}: residual R at 30 cm')
ax[2].plot(t[1:][sel[1:]], np.diff(w['obs'].numpy())[sel[1:]] / DT * 1e6, color='#0b0b0b', lw=1, ls='--', label='observed dθ/dt')
ax[2].set_ylabel('×10⁻⁶ s⁻¹'); ax[2].legend(frameon=False, fontsize=8)
fig.suptitle('Training storm (Dev108, 6 Nov 2025, 61 mm)', x=0.01, ha='left', fontsize=10)
fig.tight_layout(); fig.savefig('figs/fig1_training_storm.png', dpi=300); plt.close(fig)

# ---------------------------------------------------------------- leave-one-storm-out (Figs 7, 9, S1)
from analyze_step3 import summary, diagnostics, params, zeroshot
g107 = load_device(f'{DATA}/107_pinn.csv'); S107 = storm_catalogue(g107)
SUM, PER = summary(); D = diagnostics(); PAR = params(); ZS = zeroshot()
SUM.to_csv('table_loso_summary.csv', index=False); ZS.to_csv('table_zeroshot_dev107.csv', index=False)
D.to_csv('table_attribution_diagnostics.csv', index=False)
exec(open('fig_step3.py').read())

# ---------------------------------------------------------------- Bayesian two-store and bands (Figs 14a, S2, S3)
import bts
from bts import *                    # PRIOR, pack, ...
from h2_compare import table, pooled, recalibrate, load_point
from train_loso import make_window
bts.set_nu(30.0)
exec(open('fig_step4.py').read())
print('Thai figures written to thailand/figs/')
