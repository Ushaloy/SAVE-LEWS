plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e5e0',
                     'grid.linewidth': 0.6, 'font.size': 9, 'figure.dpi': 110})
os.makedirs('figs', exist_ok=True)
COL = {'PG-MLP': '#2a78d6', 'M0': '#eb6834', 'M2': '#1baf7a', 'M2np': '#eda100', 'M3': '#e87ba4'}
LBL = {'PG-MLP': 'PG-MLP (3 seeds)', 'M0': 'M0 1-D PINN', 'M2': 'M2 fast-store PINN', 'M2np': 'M2 no physics', 'M3': 'M3 bounded fast store'}
HS = [30, 60, 120]

# Fig 7: LOSO skill by horizon, excluding storm 2
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for k, mode in enumerate(['oracle', 'past']):
    s = SUM[SUM['mode'] == mode]
    pg = s[s.model.str.startswith('PG-MLP')]
    y = [pg[f'ex-s2 {h}'].mean() for h in HS]
    lo = [pg[f'ex-s2 {h}'].min() for h in HS]; hi = [pg[f'ex-s2 {h}'].max() for h in HS]
    ax[k].fill_between(HS, lo, hi, color=COL['PG-MLP'], alpha=0.15, lw=0)
    ax[k].plot(HS, y, '-o', color=COL['PG-MLP'], lw=2, ms=6, label=LBL['PG-MLP'])
    for mdl in ['M0', 'M2', 'M2np', 'M3']:
        r = s[s.model == mdl]
        if len(r):
            ax[k].plot(HS, [r[f'ex-s2 {h}'].iloc[0] for h in HS], '-o', color=COL[mdl], lw=2, ms=6, label=LBL[mdl])
    ax[k].axhline(0, color='#0b0b0b', lw=0.8); ax[k].set_xticks(HS); ax[k].set_xlabel('forecast horizon (min)')
    ax[k].set_title('oracle rain nowcast' if mode == 'oracle' else 'past data only', fontsize=9)
ax[0].set_ylabel('pooled storm skill vs persistence\n(6 held-out storms, storm 2 excluded)')
ax[0].legend(frameon=False, fontsize=7.5, loc='lower right')
fig.text(0.01, 0.01, 'Shaded: PG-MLP seed range. Storm 2 (244 mm, starts near saturation) is reported separately.', fontsize=7, color='#52514e')
fig.tight_layout(); fig.savefig('figs/fig7_loso_skill.png', dpi=300); plt.show()

# Fig 8: where the probe signal comes from
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
mods = [m for m in ['M0', 'M2', 'M3', 'M2np'] if m in set(D.model)]
for i, mdl in enumerate(mods):
    d = D[D.model == mdl]
    ax[0].scatter(np.full(len(d), i) + np.linspace(-0.15, 0.15, len(d)), d['matrix_share_wetup'].clip(lower=1e-3), s=36,
                  color=COL[mdl], edgecolor='white', lw=1, zorder=3)
    ax[1].scatter(np.full(len(d), i) + np.linspace(-0.15, 0.15, len(d)), d['fast_part_max'].fillna(0), s=36,
                  color=COL[mdl], edgecolor='white', lw=1, zorder=3)
for a in ax:
    a.set_xticks(range(len(mods))); a.set_xticklabels([LBL[m].replace(' ', '\n', 1) for m in mods], fontsize=8)
ax[0].set_yscale('log'); ax[0].axhline(1, color='#0b0b0b', lw=0.8)
ax[0].set_ylabel('Richards matrix share of observed\nprobe rise rate (1 = all of it)')
ax[1].axhline(0.05, color='#52514e', lw=0.8, ls='--'); ax[1].text(-0.4, 0.06, 'screening bound on macroporosity (0.05)', fontsize=7.5, color='#52514e')
ax[1].set_ylabel('max fast-store contribution\nto the reading (m³/m³)')
fig.suptitle('Held-out storms, one point per fold', x=0.01, ha='left', fontsize=10)
fig.tight_layout(); fig.savefig('figs/fig8_signal_source.png', dpi=300); plt.show()

# Fig 9: parameter spread across folds (H3)
P9 = PAR[PAR.fold != 'all']
pars = [('Ks_mph', 'Ks (m/h)', True, 0.0043), ('alpha', 'α (1/m)', False, 5.9), ('n', 'n', False, 1.48),
        ('c_mm', 'interception c (mm)', False, 5.0), ('beta', 'β (to bypass / fast)', False, None)]
fig, ax = plt.subplots(1, len(pars), figsize=(10, 3.2))
for j, (c, lab, lg, ref) in enumerate(pars):
    for i, mdl in enumerate(['M0', 'M2', 'M3']):
        d = P9[P9.model == mdl][c]
        ax[j].scatter(np.full(len(d), i) + np.linspace(-0.12, 0.12, len(d)), d, s=26, color=COL[mdl], edgecolor='white', lw=0.8, zorder=3)
    if ref is not None:
        ax[j].axhline(ref, color='#52514e', lw=0.9, ls='--')
    if lg:
        ax[j].set_yscale('log')
    ax[j].set_xticks(range(3)); ax[j].set_xticklabels(['M0', 'M2', 'M3']); ax[j].set_title(lab, fontsize=8.5)
fig.text(0.01, 0.01, 'Dashed: a priori value (Ks, α, n) or observed ~5 mm threshold (c). One point per LOSO fold.', fontsize=7, color='#52514e')
fig.tight_layout(); fig.savefig('figs/fig9_param_spread.png', dpi=300); plt.show()

# Fig 10: zero-shot Dev108 -> Dev107
z = ZS[ZS['mode'] == 'oracle'].copy(); z['fam'] = z.model.str.split(' ').str[0]
fig, ax = plt.subplots(figsize=(7.5, 3.2))
fams = ['PG-MLP', 'M0', 'M2', 'M3']
for j, fm in enumerate(fams):
    zz = z[z.fam == fm]
    for i in range(4):
        v = zz[f's{i}'].values
        ax.scatter(np.full(len(v), i + (j - 1.5) * 0.18), np.clip(v, -1.5, 1), s=34, color=COL[fm], edgecolor='white', lw=1,
                   label=LBL[fm] if i == 0 else None, zorder=3)
        clipped = v[v < -1.5]
        if len(clipped):
            lab = f'{clipped.max():.1f}' if len(clipped) == 1 else f'{clipped.max():.1f} to {clipped.min():.1f}'
            ax.text(i + (j - 1.5) * 0.18, -1.42, lab, rotation=90, fontsize=6.5, ha='center', va='bottom')
ax.axhline(0, color='#0b0b0b', lw=0.8); ax.set_ylim(-1.55, 1)
ax.set_xticks(range(4)); ax.set_xticklabels([f'{d.date()}' for d in S107.start])
ax.set_ylabel('Dev107 storm skill, 60 min, oracle'); ax.legend(frameon=False, fontsize=7.5, ncol=2, loc='lower left')
fig.text(0.01, 0.01, 'Models trained on all 7 Dev108 storms, applied unchanged; values below −1.5 clipped (printed).', fontsize=7, color='#52514e')
fig.tight_layout(); fig.savefig('figs/fig10_zeroshot.png', dpi=300); plt.show()
