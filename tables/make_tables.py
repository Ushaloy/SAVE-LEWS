"""Write the numbers behind the tables of the revised manuscript as CSV files (tables/*.csv).

    python tables/make_tables.py        # from the repository root, < 1 min

Every value is read from the stored result files (thailand/*.json, puerto_rico/out/*.json) or computed
from the raw data. The CSVs are tidy (long) tables; the manuscript tables are formatted versions of them.
"""
import os, sys, json, contextlib
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
TH, PR, OUT = os.path.join(ROOT, 'thailand'), os.path.join(ROOT, 'puerto_rico'), os.path.join(ROOT, 'tables')
sys.path[:0] = [TH, PR]


@contextlib.contextmanager
def cd(p):
    old = os.getcwd(); os.chdir(p)
    try: yield
    finally: os.chdir(old)


def save(df, name):
    df.to_csv(os.path.join(OUT, name), index=False); print(f'{name:48s} {len(df):4d} rows')


def J(*p): return json.load(open(os.path.join(*p)))


# Table 2: laboratory van Genuchten-Mualem parameters (Puerto Rico)
from p1_physics import lab_params
save(pd.DataFrame([dict(site=s, **r) for s in ['utuado', 'toronegro'] for r in lab_params(s)]), 'TableS1_PR_lab_parameters.csv')

# Table 5 / Step I: lag test (P2) and screening metrics
rows = [dict(site=s, target=k, **{a: (str(b) if isinstance(b, list) else b) for a, b in v.items()})
        for s, d in J(PR, 'res_p2.json').items() for k, v in d.items() if isinstance(v, dict)]
save(pd.DataFrame(rows), 'Table05a_PR_lag_screening_P2.csv')
save(pd.DataFrame([dict(pair=k, **v) for k, v in J(PR, 'out', 'screening_metrics.json').items()]), 'Table05b_PR_depth_inversions.csv')

# Table S1: Thai water-balance bound d_max = P / dtheta for the largest 60-min rise in each storm
from data_pipeline import load_device, storm_catalogue
DT_ = os.path.join(ROOT, 'data', 'thailand')
rows = []
for dev, f, start in [(108, 'pinn_108_new.csv', '2025-11-01 10:40'), (107, '107_pinn.csv', None)]:
    g = load_device(os.path.join(DT_, f)); g = g.loc[start:] if start else g
    for i, st in storm_catalogue(g).iterrows():
        w = g.loc[st.start - pd.Timedelta('1h'): st.end + pd.Timedelta('3h')]
        roll = w.theta.shift(-12) - w.theta; t0 = roll.idxmax(); dth = roll.max()
        rr = w.rain.fillna(0).loc[t0: t0 + pd.Timedelta('60min')].sum()
        rows.append(dict(device=dev, storm=i, start=st.start.date(), total_mm=round(st.rain_mm, 1), theta_start=round(w.theta.loc[t0], 3),
                         max_60min_rise=round(dth, 3), rain_that_hour_mm=round(rr, 1), max_1D_wetted_thickness_cm=round(rr / 1000 / dth * 100, 1)))
save(pd.DataFrame(rows), 'TableS3_Thai_water_balance_bound.csv')

# Table 6: physics-guidance ladder
A = J(TH, 'ablation_summary.json'); rows = []
for v, name in [('U', 'unconstrained MLP'), ('B', 'bounds-only MLP'), ('PG', 'PG-MLP')]:
    for mode in ('oracle', 'past'):
        rows.append(dict(site='Thailand Dev108', model=name, mode=mode, **{f'skill_{h}_min_range': A[v][f'{mode}{h}'] for h in (30, 60, 120)},
                         storm2_60=A[v][f'{mode}_s2_60'], viol_monotone=A[v][f'{mode}_viol_monotone'],
                         viol_wet_without_rain=A[v][f'{mode}_viol_wet_dry'], viol_bounds=A[v][f'{mode}_viol_bounds']))
MU = J(PR, 'out', 'mlpU_summary.json')
for k, v in MU.items():
    for m, s in v['skill'].items():
        rows.append(dict(site=f'Puerto Rico {k}', model=m, mode='causal (past + current)', skill_60_min_range=round(s, 3)))
save(pd.DataFrame(rows), 'Table06_physics_guidance_ladder.csv')

# Table 7: Thai LOSO skill; Table 9 (Thai part): storm bootstrap of skill differences
with cd(TH):
    from analyze_step3 import summary
    S, _ = summary(); save(S.round(3), 'TableS6_Thai_LOSO_skill.csv')
    import bootstrap_table6 as bt
    rows = []
    for h in (30, 60, 120):
        for m1, m2 in [('PG0', 'M0'), ('PG1', 'M0'), ('PG2', 'M0'), ('M2np', 'M2'), ('M2', 'M3'), ('M2', 'PG0'), ('M2', 'PG1'), ('M2', 'PG2')]:
            for drop in (False, True):
                pt, ci, p = bt.boot(bt.fold_stats(m1, h), bt.fold_stats(m2, h), drop)
                rows.append(dict(horizon_min=h, a=m1, b=m2, storms='excl. storm 2' if drop else 'all', diff=round(pt, 3),
                                 ci95_lo=round(ci[0], 3), ci95_hi=round(ci[1], 3), P_diff_gt_0=round(p, 3)))
    save(pd.DataFrame(rows), 'Table08a_Thai_storm_bootstrap.csv')

# Tables 8 and 9 (Puerto Rico): pre-registered P1/P3a and storm bootstrap
R = J(PR, 'out', 'p1_report.json'); rows, brows = [], []
for site, d in R.items():
    for which in ('primary', 'secondary'):
        for m, s in d[which]['skill'].items():
            rows.append(dict(site=site, target=which, n_storms=d[which]['n_storms'], model=m, skill=round(s, 3)))
        for k, v in d[which].items():
            if k.endswith('minus PG-MLP'):
                for seed, x in v.items():
                    brows.append(dict(site=site, target=which, comparison=k, pgmlp_seed=seed, diff=round(x['diff'], 3),
                                      ci95_lo=round(x['ci95'][0], 3), ci95_hi=round(x['ci95'][1], 3)))
save(pd.DataFrame(rows), 'Table07_PR_preregistered_skill.csv'); save(pd.DataFrame(brows), 'Table08b_PR_storm_bootstrap.csv')
rows = []
for site, d in R.items():
    for tag, v in d['P3a'].items():
        if tag == 'lab_range': continue
        for p in ('Ks', 'alpha', 'n'):
            rows.append(dict(site=site, model=tag, parameter=p, fold_median=v['fold_median'][p], between_fold_sd_log=v['between_fold_sd_log'][p],
                             lab_min=d['P3a']['lab_range'][p][0], lab_max=d['P3a']['lab_range'][p][1], within_lab_bounds=v['within_lab_bounds'][p]))
save(pd.DataFrame(rows), 'Table07b_PR_P3a_parameters.csv')

# Table 10: robustness of the exploratory R-cal-Se result
rows = []
for which, d in J(PR, 'out', 'se_robust.json').items():
    for m, s in d['skill'].items(): rows.append(dict(analysis='alternative calibration/normalisation', target=f'utuado {which}', model=m, skill=round(s, 3)))
for site in ('utuado', 'toronegro'):
    N = J(PR, 'out', f'se_nested_{site}.json')
    for which in ('primary', 'secondary'):
        rows.append(dict(analysis='nested normalisation (training folds only)', target=f'{site} {which}', model='Se nested',
                         skill=round(N[which]['skill']['Se nested'], 3),
                         diff_vs_pgmlp=';'.join(f"s{k}:{v[0]:+.2f} [{v[1][0]:+.2f},{v[1][1]:+.2f}]" for k, v in N[which].get('diff', {}).items())))
for k, p in J(PR, 'out', 'se_holm.json').items(): rows.append(dict(analysis='Holm-corrected one-sided p', target=k, model='R-cal-Se', p_value=p))
save(pd.DataFrame(rows), 'TableS7_PR_robustness.csv')

# Table 11: hydrological alert states
rows = []
for thr, d in J(TH, 'thai_alerts.json').items():
    for m, v in d.items(): rows.append(dict(site='Thailand Dev108', target='30 cm', rise_threshold=thr, model=m, **v))
for site, d in J(PR, 'out', 'pr_alerts_rise.json').items():
    for tgt, v in d.items():
        for m, r in v['results'].items(): rows.append(dict(site=site, target=tgt, rise_threshold=round(v['threshold'], 4), model=m, **r))
save(pd.DataFrame(rows), 'Table09_alert_performance.csv')

# Table 12: warning bands (Thai: h2_compare; Puerto Rico: P4)
with cd(TH):
    from h2_compare import table, pooled
    rows = []
    for k, v in table('oracle').items():
        p = pooled(v, (2,)); p = p.reset_index().rename(columns={'index': 'horizon'}); p.insert(0, 'model', k); rows.append(p)
    save(pd.concat(rows).round(3), 'TableS9a_Thai_band_calibration.csv')
rows = [dict(site=s, model=m, **v) for s, d in J(PR, 'out', 'p4_report.json').items() for m, v in d.items() if isinstance(v, dict)]
save(pd.DataFrame(rows).round(3), 'TableS9b_PR_band_calibration_P4.csv')

# Nowcast degradation (Results, Step III)
save(pd.DataFrame([dict(scenario=k, skill_60_min=round(v, 3)) for k, v in J(TH, 'nowcast_degrade.json').items()]), 'Nowcast_degradation.csv')

# Revision analyses (Tables S4, S5, S8): summarised by thailand/summarize_revision.py
with cd(TH):
    import summarize_revision as SR
    S = dict(assimilation=SR.assim(), bounds=SR.bounds(), jacobian=SR.jacobian())
rows = [dict(variant=k.split('|')[0], model=k.split('|')[1], mode=k.split('|')[2], skill_30=v[0], skill_60=v[1], skill_120=v[2])
        for k, v in S['assimilation'].items() if k.count('|') == 2]
save(pd.DataFrame(rows), 'TableS5_assimilation_ablation.csv')
save(pd.DataFrame([dict(bound=k, **{kk: (str(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()}) for k, v in S['bounds'].items()]),
     'TableS4_fast_store_bound_sensitivity.csv')
save(pd.DataFrame([dict(configuration=k, **{kk: (json.dumps(vv) if isinstance(vv, (list, dict)) else vv) for kk, vv in v.items()})
                   for k, v in S['jacobian'].items()]), 'TableS8_jacobian_identifiability.csv')
