plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e5e0',
                     'grid.linewidth': 0.6, 'font.size': 9, 'figure.dpi': 110})
os.makedirs('figs', exist_ok=True)
C_PG, C_M2, C_BTS, C_REF = '#2a78d6', '#1baf7a', '#4a3aa7', '#8c8a83'
HS = [30, 60, 120]

# Fig 11: H2 — coverage and interval score, oracle, 6 regular storms (storm 2 excluded from evaluation)
T = {k: pooled(v, (2,)) for k, v in table('oracle').items()}
TG = {k: pooled(v, (2,)) for k, v in table('oracle', (2,)).items()}
series = [('B-TS (NUTS), raw posterior', T['B-TS nuts (raw)'], C_BTS, 'o', True),
          ('B-TS (NUTS) + conformal, storm 2 gated', TG['B-TS nuts + conformal'], C_BTS, 'o', False),
          ('M2 PINN + conformal', T['M2 + conformal'], C_M2, 's', True),
          ('PG-MLP + conformal', T['PGMLP + conformal'], C_PG, 'D', True)]
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
for lab, d, col, mk, filled in series:
    kw = dict(color=col, lw=1.8, marker=mk, ms=7, label=lab, mfc=col if filled else 'white', mec=col, mew=1.6)
    ax[0].plot(HS, d['coverage'].values, **kw); ax[1].plot(HS, d['interval score'].values, **kw)
pers = T['persistence + conformal']
ax[0].plot(HS, pers['coverage'].values, color=C_REF, lw=1.2, ls='--', label='persistence + conformal (reference)')
ax[1].plot(HS, pers['interval score'].values, color=C_REF, lw=1.2, ls='--')
ax[0].axhline(0.9, color='#0b0b0b', lw=0.8); ax[0].text(31, 0.905, 'nominal 90 %', fontsize=7.5, color='#52514e')
ax[0].set_ylim(0.7, 0.97); ax[0].set_ylabel('storm-time coverage of 90 % interval')
ax[1].set_ylabel('interval score (lower is better)')
for a in ax: a.set_xticks(HS); a.set_xlabel('horizon (min)')
ax[0].legend(frameon=False, fontsize=7.2, loc='lower left')
fig.text(0.01, 0.01, 'Leave-one-storm-out, oracle rain, 6 held-out storms (storm 2 excluded from evaluation). Conformal = cross-conformal from the other folds.',
         fontsize=7, color='#52514e')
fig.tight_layout(); fig.subplots_adjust(bottom=0.2); fig.savefig('figs/fig11_h2_calibration.png', dpi=300); plt.show()

# Fig 12: H3 — per-fold posterior 90 % intervals vs prior
fig, ax = plt.subplots(1, 4, figsize=(10, 3.2))
labs = {'c': 'interception c (mm)', 'tau_c': 'τ_c (h)', 'a': 'fast-store gain a (1/mm)', 'tau_F': 'τ_F (h)'}
for j, k in enumerate(['c', 'tau_c', 'a', 'tau_F']):
    m, s = PRIOR[k]
    ax[j].axhspan(np.exp(m - 1.645 * s), np.exp(m + 1.645 * s), color='#e6e5e0', zorder=0)
    for f in range(7):
        p = json.load(open(f'bts_out/res4_{f}.json'))['nuts']['post'][k]
        ax[j].plot([f, f], [p['q05'], p['q95']], color=C_BTS, lw=2.2, solid_capstyle='round')
        ax[j].plot(f, p['q50'], 'o', color=C_BTS, ms=5, mec='white')
    ax[j].set_yscale('log'); ax[j].set_xticks(range(7)); ax[j].set_xticklabels([f's{f}' for f in range(7)], fontsize=7.5)
    ax[j].set_title(labs[k], fontsize=8.5); ax[j].set_xlabel('held-out storm (fold)', fontsize=7.5)
ax[0].axhline(5.0, color='#52514e', lw=0.9, ls='--')
fig.text(0.01, 0.01, 'Shaded: prior 90 % range. Bars: NUTS posterior 90 % interval per fold. Dashed (c): observed ~5 mm storm-rain threshold.',
         fontsize=7, color='#52514e')
fig.tight_layout(); fig.subplots_adjust(bottom=0.24); fig.savefig('figs/fig12_h3_posteriors.png', dpi=300); plt.show()

# Fig 13: example held-out storm, 60-min forecasts with 90 % bands
f = 1; j = 1
z = np.load(f'bts_out/{f}_held_nuts_oracle.npz'); cz = np.load(f'conf_out/held_{f}.npz')
Rm2 = recalibrate(load_point('M2', 'oracle'), 'point')[f]
st = z['storm'] & z['ym'][:, j]
wf = make_window(g108, S108.iloc[f]); t_idx = wf['t'][pack([wf])['ki']]
t_idx = (t_idx - S108.start[f]).total_seconds() / 3600.0      # hours from storm start
fig, ax = plt.subplots(figsize=(8.5, 3.4))
tt = t_idx[st]
ax.fill_between(tt, Rm2['lo'][st, j], Rm2['hi'][st, j], color=C_M2, alpha=0.18, lw=0, label='M2 PINN + conformal, 90 %')
ax.fill_between(tt, z['lo'][st, j], z['hi'][st, j], color=C_BTS, alpha=0.22, lw=0, label='B-TS posterior predictive, 90 %')
ax.plot(tt, z['med'][st, j], color=C_BTS, lw=1.4, label='B-TS median')
ax.plot(tt, Rm2['med'][st, j], color=C_M2, lw=1.4, label='M2 PINN')
ax.plot(tt, z['y'][st, j], color='#0b0b0b', lw=1.6, label='observed')
ax.axhline(0, color='#0b0b0b', lw=0.6)
ax.set_ylabel('Δθ over next 60 min (m³/m³)'); ax.set_xlabel('hours from storm start (forecast origin)'); ax.legend(frameon=False, fontsize=7.5, ncol=2, loc='upper right')
ax.set_title(f'Held-out storm {S108.start[f].date()} (fold {f}), oracle rain', fontsize=9, loc='left')
fig.tight_layout(); fig.savefig('figs/fig13_example_storm.png', dpi=300); plt.show()
