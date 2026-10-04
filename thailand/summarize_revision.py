"""Summarise the revision analyses into res_rev_summary.json (and print tables).
Run from thailand/: python summarize_revision.py"""
import json, os, glob, numpy as np

REG = [0, 1, 3, 4, 5, 6]; HZ = (30, 60, 120)


def pooled(rows, h):
    se = sum(r[f'mse_storm_{h}'] * r[f'n_storm_{h}'] for r in rows)
    ref = sum(r[f'msep_storm_{h}'] * r[f'n_storm_{h}'] for r in rows)
    return float(1 - se / ref)


def assim():
    r = json.load(open('res_rev_nudgeeval.json')); out = {}
    for name in ['nudge L=0.08', 'nudge L=0.02', 'nudge L=0.16', 'open loop']:
        for mt in ['M0', 'M2', 'M3']:
            for mode in ('oracle', 'past'):
                rows = [r[f'{name}|{mt}|f{f}|{mode}'] for f in REG]
                out[f'{name}|{mt}|{mode}'] = [round(pooled(rows, h), 3) for h in HZ]
    fs = sorted(glob.glob('res_rev_noassim_f*.json'))
    if fs:
        R = {int(f.split('_f')[1].split('.')[0]): json.load(open(f)) for f in fs}
        folds = [f for f in REG if f in R]
        for mode in ('oracle', 'past'):
            out[f'retrained open loop|M0|{mode}'] = [round(pooled([R[f][f'held_{mode}'] for f in folds], h), 3) for h in HZ]
        out['retrained open loop folds'] = folds
    return out


def bounds():
    out = {}
    base = {f: json.load(open(f'res3_loso_M3_f{f}.json')) for f in range(7)}
    sets = {0.05: base}
    for f in glob.glob('res_rev_bound*_f*.json'):
        B = float(f.split('bound')[1].split('_f')[0]); k = int(f.split('_f')[1].split('.')[0])
        sets.setdefault(B, {})[k] = json.load(open(f))
    m2 = {f: json.load(open(f'res3_loso_M2_f{f}.json')) for f in range(7)}
    sets['M2 (unbounded)'] = m2
    for B, R in sorted(sets.items(), key=lambda kv: (isinstance(kv[0], str), kv[0])):
        folds = [f for f in REG if f in R]
        if not folds: continue
        row = dict(folds=folds, n_folds=len(folds))
        for mode in ('oracle', 'past'):
            row[mode] = [round(pooled([R[f][f'held_{mode}'] for f in folds], h), 3) for h in HZ]
        row['s_matrix_median'] = round(float(np.median([R[f]['held_oracle']['matrix_share_wetup'] for f in folds])), 4)
        row['fast_part_max_median'] = round(float(np.median([R[f]['held_oracle']['fast_part_max'] for f in folds])), 3)
        row['wet_without_rain'] = round(float(np.mean([R[f]['held_oracle']['viol_wet_without_rain'] for f in folds])), 3)
        if not isinstance(B, str):
            row['phi_f_fitted'] = [round(R[f]['params'].get('phi_f', np.nan), 4) for f in folds]
        out[str(B)] = row
    return out


SCALE = {'zb': 0.1, 'theta_s': 0.05, 'theta_r': 0.05}   # report per 0.1 m (bypass depth) or per 0.05 m3/m3 (theta_s, theta_r)
MATRIX = {'Ks_mph', 'alpha', 'n', 'log Ks', 'log alpha', 'log(n-1)'}


def jac_one(J, names, resid):
    J = np.asarray(J, dtype=float); ok = np.isfinite(J).all(1); J = J[ok]; N = len(J)
    J = J * np.array([SCALE.get(n, 1.0) for n in names])[None]
    s = np.linalg.svd(J, compute_uv=False) / np.sqrt(N)
    col = np.sqrt((J ** 2).mean(0))
    mi = [i for i, n in enumerate(names) if n in MATRIX]
    sm = np.linalg.svd(J[:, mi], compute_uv=False) / np.sqrt(N)
    C = np.corrcoef(J[:, mi].T)
    return dict(sv_scaled=[float(x) for x in s], condition=float(s[0] / s[-1]), resid_rmse=float(resid),
                practical_rank=int((s > resid).sum()), n_params=len(names), n_rows=int(N), n_dropped=int((~ok).sum()),
                col_rms={n: float(c) for n, c in zip(names, col)}, sv_over_resid=[float(x / resid) for x in s],
                matrix_sv=[float(x) for x in sm], matrix_rank=int((sm > resid).sum()), matrix_condition=float(sm[0] / sm[-1]),
                matrix_corr={f'{names[mi[a]]}~{names[mi[b]]}': float(C[a, b]) for a in range(len(mi)) for b in range(a + 1, len(mi))})


def jacobian():
    out = {}
    if os.path.exists('res_rev_jacobian_thai.json'):
        d = json.load(open('res_rev_jacobian_thai.json'))
        out['Thailand 30 cm'] = jac_one(d['J'], d['names'], d['resid_rmse'])
    p = '../puerto_rico/out/rev_jacobian_pr.json'
    if os.path.exists(p):
        D = json.load(open(p))
        lab = {'utuado secondary': 'Utuado 57 cm', 'utuado primary': 'Utuado 72 cm', 'toronegro primary': 'Toro Negro 90 cm',
               'toronegro secondary': 'Toro Negro 110 cm'}
        for k, v in D.items():
            o = jac_one(v['J'], v['names'], v['resid_rmse']); o['input_sensitivity_rms'] = v['input_sensitivity_rms']
            out[lab[k]] = o
    return out


if __name__ == '__main__':
    S = dict(assimilation=assim(), bounds=bounds(), jacobian=jacobian())
    json.dump(S, open('res_rev_summary.json', 'w'), indent=1)
    for k, v in S['assimilation'].items(): print(k, v)
    for k, v in S['bounds'].items(): print(k, v)
    for k, v in S['jacobian'].items():
        print(k, 'cond %.3g rank %d/%d' % (v['condition'], v['practical_rank'], v['n_params']), np.round(v['sv_over_resid'], 3), v['col_rms'])
