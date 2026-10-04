"""Evaluate P1 (and P3a) against the pre-registered rules.  usage: python eval_p1.py"""
import json, os, numpy as np, pandas as pd
from p1_physics import build, lab_params
from pr_data import SITES
rng = np.random.default_rng(0)

def load_preds(site, B):
    P = {}
    z = np.load(f'out/p1_physics_{site}.npz', allow_pickle=True); P['R-cal'] = z['cal']; P['R-lab'] = z['lab']
    if os.path.exists(f'out/p1_se_{site}.npz'):
        P['R-cal-Se (exploratory)'] = np.load(f'out/p1_se_{site}.npz', allow_pickle=True)['cal']
    if os.path.exists(f'out/p1_mlp_{site}.npz'):
        m = np.load(f'out/p1_mlp_{site}.npz', allow_pickle=True)
        for s in range(3):
            P[f'PG-MLP s{s}'] = ('mlp', m[f'primary_s{s}'], m[f'secondary_s{s}'])
    return P

def per_window(B, P, which):
    """per-window (sse, sse_ref, n) arrays for target 'primary' or 'secondary'."""
    k = B['i_primary'] if which == 'primary' else B['i_secondary']
    rows = {}
    for name, pr in P.items():
        sse, ref, nn_ = [], [], []
        for i, w in enumerate(B['wins']):
            y = w['Y'][:, k].copy(); y[~w['top_ok']] = np.nan
            if isinstance(pr, tuple):
                p = (pr[1] if which == 'primary' else pr[2])[i]
                if p is None: sse.append(np.nan); ref.append(np.nan); nn_.append(0); continue
            else:
                p = pr[i][:, k] if pr[i] is not None else None
            ok = ~np.isnan(y) & ~np.isnan(p)
            y0 = np.nanmean(y[:4]) if not np.isnan(y[:4]).all() else np.nan
            if ok.sum() == 0 or np.isnan(y0): sse.append(np.nan); ref.append(np.nan); nn_.append(0); continue
            sse.append(np.sum((p[ok] - y[ok]) ** 2)); ref.append(np.sum((y0 - y[ok]) ** 2)); nn_.append(int(ok.sum()))
        rows[name] = (np.array(sse), np.array(ref), np.array(nn_))
    return rows

def skill(sse, ref, idx):
    return 1 - np.nansum(sse[idx]) / np.nansum(ref[idx])

def boot_diff(a, b, common, B=10000):
    d = []
    for _ in range(B):
        k = rng.choice(common, len(common), replace=True)
        d.append(skill(*a[:2], k) - skill(*b[:2], k))
    d = np.array(d); return float(np.quantile(d, .025)), float(np.quantile(d, .975))

def main():
    report = {}
    for site in SITES:
        B = build(site); P = load_preds(site, B); report[site] = {}
        for which in ('primary', 'secondary'):
            rows = per_window(B, P, which)
            valid = [np.where(~np.isnan(r[0]) & (r[2] > 0))[0] for r in rows.values()]
            common = np.array(sorted(set.intersection(*[set(v) for v in valid])))
            sk = {name: skill(r[0], r[1], common) for name, r in rows.items()}
            out = dict(n_storms=int(len(common)), skill=sk)
            if which == 'primary' and any(k.startswith('PG-MLP') for k in rows):
                for phys in [k for k in rows if k.startswith('R-')]:
                    out[f'{phys} minus PG-MLP'] = {s: dict(diff=sk[phys] - sk[f'PG-MLP s{s}'], ci95=boot_diff(rows[phys], rows[f'PG-MLP s{s}'], common)) for s in range(3)}
            report[site][which] = out
        # P3a: calibrated vs lab
        J = json.load(open(f'out/p1_physics_{site}.json')); labs = lab_params(site)
        lab = dict(Ks=[min(min(l['Ks_d'], l['Ks_w']) for l in labs), max(max(l['Ks_d'], l['Ks_w']) for l in labs)],
                   alpha=[min(min(l['a_d'], l['a_w']) for l in labs), max(max(l['a_d'], l['a_w']) for l in labs)],
                   n=[min(min(l['n_d'], l['n_w']) for l in labs), max(max(l['n_d'], l['n_w']) for l in labs)])
        p3 = {}
        for tag, fn in [('R-cal', f'out/p1_physics_{site}.json'), ('R-cal-Se (exploratory)', f'out/p1_se_{site}.json')]:
            if not os.path.exists(fn): continue
            F = json.load(open(fn))['folds']
            med = {k: float(np.median([f['params'][k] for f in F])) for k in ('Ks', 'alpha', 'n')}
            sdlog = {k: float(np.std(np.log([f['params'][k] for f in F]))) for k in ('Ks', 'alpha', 'n')}
            ok = dict(Ks=lab['Ks'][0] / 3 <= med['Ks'] <= lab['Ks'][1] * 3, alpha=lab['alpha'][0] / 3 <= med['alpha'] <= lab['alpha'][1] * 3,
                      n=lab['n'][0] - 0.25 <= med['n'] <= lab['n'][1] + 0.25)
            p3[tag] = dict(fold_median=med, between_fold_sd_log=sdlog, within_lab_bounds=ok)
        report[site]['P3a'] = dict(lab_range=lab, **p3)
    json.dump(report, open('out/p1_report.json', 'w'), indent=1, default=float)
    print(json.dumps(report, indent=1, default=float))


if __name__ == '__main__':
    main()
