import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from p1_physics import build
from eval_p1 import load_preds
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e5e0',
                     'grid.linewidth': 0.6, 'font.size': 9, 'figure.dpi': 110})
C = {'R-cal': '#eb6834', 'R-cal-Se (exploratory)': '#4a3aa7', 'PG-MLP': '#2a78d6', 'R-lab': '#8c8a83'}
rep = json.load(open('out/p1_report.json')); ex = json.load(open('out/explore_72.json')); p2 = json.load(open('res_p2.json'))

# Fig A: skill by depth and site
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
for a, site, depths in [(ax[0], 'utuado', ['57 cm (secondary)', '72 cm (primary)']), (ax[1], 'toronegro', ['90 cm (primary)', '110 cm (secondary)'])]:
    order = ['secondary', 'primary'] if site == 'utuado' else ['primary', 'secondary']
    for j, which in enumerate(order):
        sk = rep[site][which]['skill']
        items = [('R-lab', sk['R-lab']), ('R-cal', sk['R-cal']), ('R-cal-Se (exploratory)', sk.get('R-cal-Se (exploratory)', np.nan))]
        pg = [sk[f'PG-MLP s{s}'] for s in range(3)]
        for k, (nm, v) in enumerate(items):
            a.bar(j + (k - 1.5) * 0.2, max(v, -1.5), width=0.18, color=C[nm], label=nm if j == 0 else None)
            if v < -1.5: a.text(j + (k - 1.5) * 0.2, -1.45, f'{v:.1f}', rotation=90, fontsize=7, ha='center', va='bottom')
        a.bar(j + 1.5 * 0.2, np.mean(pg), width=0.18, color=C['PG-MLP'], label='PG-MLP (3 seeds)' if j == 0 else None)
        a.errorbar(j + 1.5 * 0.2, np.mean(pg), yerr=[[np.mean(pg) - min(pg)], [max(pg) - np.mean(pg)]], color='#0b0b0b', lw=1, capsize=2)
    a.axhline(0, color='#0b0b0b', lw=0.8); a.set_xticks([0, 1]); a.set_xticklabels(depths); a.set_ylim(-1.55, 1)
    a.set_title(f'{"Utuado (sandy, 42°)" if site == "utuado" else "Toro Negro (clayey, 45°)"} — {rep[site]["primary"]["n_storms"]} storms', fontsize=9, loc='left')
ax[0].set_ylabel('skill vs no-change, held-out storms'); ax[0].legend(frameon=False, fontsize=7.5, loc='lower left')
fig.text(0.01, 0.01, 'Virtual deep sensor: only the shallow probe (27 / 30 cm) and rain are inputs after the window start. Leave-one-block-out, 24 h purge. Bars below −1.5 clipped (value printed).', fontsize=7, color='#52514e')
fig.tight_layout(); fig.subplots_adjust(bottom=0.17); fig.savefig('figs/figA_skill_by_depth.png', dpi=300); plt.close(fig)

# Fig B: bootstrap differences (physics − PG-MLP) at Utuado
fig, ax = plt.subplots(figsize=(6.4, 2.8))
rows = []
for s in range(3):
    d = rep['utuado']['primary']['R-cal minus PG-MLP'][str(s)]; rows.append(('R-cal, 72 cm (registered P1)', s, d['diff'], d['ci95']))
    d = rep['utuado']['primary']['R-cal-Se (exploratory) minus PG-MLP'][str(s)]; rows.append(('R-cal-Se, 72 cm', s, d['diff'], d['ci95']))
    d = ex['R-cal minus PG-MLP @57cm'][str(s)]; rows.append(('R-cal, 57 cm', s, d[0], d[1]))
    d = ex['R-cal-Se (exploratory) minus PG-MLP @57cm'][str(s)]; rows.append(('R-cal-Se, 57 cm', s, d[0], d[1]))
labels = ['R-cal, 72 cm (registered P1)', 'R-cal-Se, 72 cm', 'R-cal, 57 cm', 'R-cal-Se, 57 cm']
for y, lab in enumerate(labels):
    for nm, s, dv, ci in rows:
        if nm != lab: continue
        yy = y + (s - 1) * 0.18; col = C['R-cal'] if lab.startswith('R-cal,') else C['R-cal-Se (exploratory)']
        ax.plot(ci, [yy, yy], color=col, lw=2, solid_capstyle='round'); ax.plot(dv, yy, 'o', color=col, ms=5, mec='white')
ax.axvline(0, color='#0b0b0b', lw=0.8); ax.set_yticks(range(4)); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis()
ax.set_xlabel('skill difference, physics − PG-MLP (95% storm-bootstrap interval; one line per MLP seed)')
fig.tight_layout(); fig.savefig('figs/figB_bootstrap.png', dpi=300); plt.close(fig)

# Fig C: example storm at Utuado — 57 and 72 cm
B = build('utuado'); P = load_preds('utuado', B); W = B['wins']
i = int(np.argmax([w['rain_mm'] if (not np.isnan(w['Y'][:, B['i_primary']]).all()) else 0 for w in W]))
w = W[i]; t = (w['t'] - w['start']).total_seconds() / 3600
fig, ax = plt.subplots(3, 1, figsize=(8, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[1, 2, 2]))
ax[0].bar(t, w['rain'], width=0.25, color='#8c8a83'); ax[0].set_ylabel('rain\n(mm / 15 min)')
ax[1].plot(t, w['top'], color='#0b0b0b', lw=1.2, ls='--', label='27 cm probe (input)')
for a, which, lab in [(ax[1], B['i_secondary'], '57 cm'), (ax[2], B['i_primary'], '72 cm')]:
    a.plot(t, w['Y'][:, which], color='#0b0b0b', lw=2, label=f'observed {lab}')
    a.plot(t, P['R-cal-Se (exploratory)'][i][:, which], color=C['R-cal-Se (exploratory)'], lw=1.6, label='R-cal-Se')
    a.plot(t, P['R-cal'][i][:, which], color=C['R-cal'], lw=1.6, label='R-cal')
    tk = 'primary' if which == B['i_primary'] else 'secondary'
    a.plot(t, P['PG-MLP s0'][1 if tk == 'primary' else 2][i], color=C['PG-MLP'], lw=1.6, label='PG-MLP (seed 0)')
    a.set_ylabel(f'θ at {lab} (m³/m³)'); a.legend(frameon=False, fontsize=7.5, ncol=2)
ax[2].set_xlabel('hours from storm start')
fig.suptitle(f'Utuado, largest held-out storm ({w["start"].date()}, {w["rain_mm"]:.0f} mm)', x=0.01, ha='left', fontsize=10)
fig.tight_layout(); fig.savefig('figs/figC_example_storm.png', dpi=300); plt.close(fig)

# Fig D: P2 lags
from pr_data import load, storms, SITES
fig, ax = plt.subplots(figsize=(6, 3))
d = load('utuado'); c = SITES['utuado']; S = storms(d); lags = {'57 cm': [], '72 cm': []}
for _, st in S.iterrows():
    pre = d.loc[st.start - pd.Timedelta(hours=24): st.start]; post = d.loc[st.start: st.start + pd.Timedelta(hours=48)]
    on = {}
    for key, col in [('sh', c['shallow']), ('57 cm', c['deep2']), ('72 cm', c['deep'])]:
        base = pre[col].mean(); hit = post[col] > base + 0.005
        on[key] = hit.idxmax() if (hit.any() and post[col].notna().mean() >= 0.8 and not np.isnan(base)) else None
    for k in ('57 cm', '72 cm'):
        if on['sh'] is not None and on[k] is not None: lags[k].append((on[k] - on['sh']).total_seconds() / 3600)
bins = np.arange(-6, 30, 1.0)
for k, col in [('57 cm', C['R-cal-Se (exploratory)']), ('72 cm', C['R-cal'])]:
    ax.hist(np.clip(lags[k], -6, 29.5), bins=bins, color=col, alpha=0.6, label=f'{k}: median {np.median(lags[k]):.1f} h (n = {len(lags[k])})')
ax.axvline(0.5, color='#52514e', lw=0.8, ls=':'); ax.axvline(2, color='#52514e', lw=0.8, ls='--')
ax.text(0.6, ax.get_ylim()[1] * 0.9, '30 min', fontsize=7, color='#52514e'); ax.text(2.1, ax.get_ylim()[1] * 0.8, '2 h (pre-registered)', fontsize=7, color='#52514e')
ax.set_xlabel('lag of deep rise after 27 cm rise (h); clipped at −6 and 30'); ax.set_ylabel('storms'); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig('figs/figD_p2_lags.png', dpi=300); plt.close(fig)
print('figs done', i)

# Fig E: P4 coverage vs interval score
import os
if os.path.exists('out/p4_report.json'):
    p4 = json.load(open('out/p4_report.json'))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    col = {'R-cal Laplace (registered)': C['R-cal'], 'R-cal-Se Laplace (exploratory)': C['R-cal-Se (exploratory)'], 'PG-MLP s0 + cross-conformal': C['PG-MLP']}
    for a, site in zip(ax, ['utuado', 'toronegro']):
        for k, v in p4[site].items():
            if k == 'n_windows': continue
            a.scatter(v['coverage'], v['interval_score'], s=70, color=col[k], edgecolor='white', lw=1.5, zorder=3, label=k)
            a.annotate(k.split(' (')[0].replace(' Laplace', '').replace(' s0 + cross-conformal', ' + conformal'), (v['coverage'], v['interval_score']),
                       xytext=(6, 4), textcoords='offset points', fontsize=7.5, color='#52514e')
        a.axvspan(0.85, 0.95, color='#e6e5e0', zorder=0); a.axvline(0.9, color='#0b0b0b', lw=0.8)
        a.set_xlabel('coverage of 90% interval (shaded: pre-registered 0.85–0.95)'); a.set_title(f'{site}, primary depth', fontsize=9, loc='left')
        a.set_xlim(min(0.5, a.get_xlim()[0]), 1.0)
    ax[0].set_ylabel('interval score (lower is better)')
    fig.tight_layout(); fig.savefig('figs/figE_p4.png', dpi=300); plt.close(fig)
