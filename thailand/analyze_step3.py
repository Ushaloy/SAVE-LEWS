"""Aggregate step-3 LOSO results into tables (printed + CSV)."""
import json, glob, os, numpy as np, pandas as pd
HZ = (30, 60, 120)

def pooled(rows, h, skip=()):
    num = sum(r[f'mse_storm_{h}'] * r[f'n_storm_{h}'] for i, r in rows.items() if i not in skip)
    den = sum(r[f'msep_storm_{h}'] * r[f'n_storm_{h}'] for i, r in rows.items() if i not in skip)
    return 1 - num / den

def load_net(model):
    out = {}
    for f in range(7):
        p = f'res3_loso_{model}_f{f}.json'
        if os.path.exists(p):
            out[f] = json.load(open(p))
    return out

def summary():
    rows = []; per = {}
    pg = json.load(open('res3_pgmlp.json'))
    for mode in ('oracle', 'past'):
        for s in range(3):
            d = {i: pg[f'seed{s}_{mode}_s{i}'] for i in range(7)}
            rows.append({'model': f'PG-MLP seed{s}', 'mode': mode, 'folds': 7,
                         **{f'pooled {h}': pooled(d, h) for h in HZ}, **{f'ex-s2 {h}': pooled(d, h, (2,)) for h in HZ},
                         'median 60': np.median([d[i]['skill_storm_60'] for i in d])})
            per[(f'PG-MLP seed{s}', mode)] = {i: d[i]['skill_storm_60'] for i in d}
    for model in ['M0', 'M2', 'M3', 'M2np']:
        R = load_net(model)
        if not R:
            continue
        for mode in ('oracle', 'past'):
            d = {f: R[f][f'held_{mode}'] for f in R}
            rows.append({'model': model, 'mode': mode, 'folds': len(d),
                         **{f'pooled {h}': pooled(d, h) for h in HZ},
                         **{f'ex-s2 {h}': pooled(d, h, (2,)) for h in HZ},
                         'median 60': np.median([d[i]['skill_storm_60'] for i in d])})
            per[(model, mode)] = {i: d[i]['skill_storm_60'] for i in d}
    return pd.DataFrame(rows), pd.DataFrame(per)

def diagnostics():
    rows = []
    for model in ['M0', 'M2', 'M3', 'M2np']:
        for f, r in load_net(model).items():
            h = r['held_oracle']
            rows.append({'model': model, 'fold': f, **{k: h.get(k, np.nan) for k in
                         ['matrix_share_wetup', 'fast_share_wetup', 'R_over_obs_wetup', 'matrix_share_falling',
                          'fast_share_falling', 'R_over_obs_falling', 'fast_part_max', 'viol_wet_without_rain',
                          'viol_above_theta_s']}})
    return pd.DataFrame(rows)

def params():
    rows = []
    for model in ['M0', 'M2', 'M3', 'M2np']:
        for f in list(range(7)) + ['all']:
            p = f'res3_loso_{model}_f{f}.json'
            if os.path.exists(p):
                rows.append({'model': model, 'fold': f, **json.load(open(p))['params']})
    return pd.DataFrame(rows)

def zeroshot():
    rows = []
    pg = json.load(open('res3_pgmlp_zeroshot.json'))
    for mode in ('oracle', 'past'):
        for s in range(3):
            rows.append({'model': f'PG-MLP seed{s}', 'mode': mode, **{f's{i}': pg[f'seed{s}_{mode}_s{i}']['skill_storm_60'] for i in range(4)}})
    for model in ['M0', 'M2', 'M3']:
        p = f'res3_loso_{model}_fall.json'
        if os.path.exists(p):
            r = json.load(open(p))
            for mode in ('oracle', 'past'):
                rows.append({'model': model, 'mode': mode, **{f's{i}': r[f'd107_s{i}_{mode}']['skill_storm_60'] for i in range(4)}})
    return pd.DataFrame(rows)

if __name__ == '__main__':
    pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
    S, P = summary(); print(S.round(3)); print(P.round(2))
    D = diagnostics(); print(D.groupby('model').median(numeric_only=True).round(3))
    print(params().round(4))
    if os.path.exists('res3_pgmlp_zeroshot.json'):
        print(zeroshot().round(2))
