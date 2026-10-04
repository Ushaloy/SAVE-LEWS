"""H2: interval calibration on held-out storms. B-TS (VI / NUTS / VI without discrepancy), raw and
cross-conformal-recalibrated, vs persistence / M2 PINN / PG-MLP with cross-conformal intervals.
Cross-conformal: fold k is calibrated only on the other folds' held-out residuals (Mondrian storm/quiet, per horizon)."""
import os, json, numpy as np, pandas as pd
ALPHA = 0.10; HZ = [30, 60, 120]; FOLDS = range(7)

def load_bts(method, mode):
    out = {}
    for f in FOLDS:
        p = f'bts_out/{f}_held_{method}_{mode}.npz'
        if os.path.exists(p):
            out[f] = dict(np.load(p))
    return out

def load_point(model, mode):
    out = {}
    for f in FOLDS:
        z = np.load(f'conf_out/held_{f}.npz')
        pred = z[f'{model}_{mode}']
        half = np.full_like(pred, np.nan)
        out[f] = dict(y=z['y'], ym=z['ym'], storm=z['storm'], med=pred, lo=pred, hi=pred)
    return out

def conf_quantile(scores):
    s = np.sort(scores[~np.isnan(scores)]); n = len(s)
    if n == 0: return np.nan
    k = int(np.ceil((1 - ALPHA) * (n + 1))) - 1
    return s[min(k, n - 1)]

def recalibrate(R, kind, exclude_calib=()):
    """kind 'point': interval = med +/- q(|y - med|); kind 'scale': med +/- q(|y-med|/halfwidth) * halfwidth."""
    new = {}
    for f in R:
        calib = [g for g in R if g != f and g not in exclude_calib]
        d = R[f]; lo = np.array(d['lo'], float); hi = np.array(d['hi'], float); med = d['med']
        for j in range(3):
            for reg in (False, True):
                sc = []
                for g_ in calib:
                    e = R[g_]; sel = e['ym'][:, j] & (e['storm'] == reg) & ~np.isnan(e['med'][:, j])
                    r = np.abs(e['y'][sel, j] - e['med'][sel, j])
                    if kind == 'scale':
                        r = r / np.maximum((e['hi'][sel, j] - e['lo'][sel, j]) / 2, 1e-6)
                    sc.append(r)
                q = conf_quantile(np.concatenate(sc))
                rows = d['storm'] == reg
                if kind == 'point':
                    lo[rows, j] = med[rows, j] - q; hi[rows, j] = med[rows, j] + q
                else:
                    hw = (d['hi'][rows, j] - d['lo'][rows, j]) / 2
                    lo[rows, j] = med[rows, j] - q * hw; hi[rows, j] = med[rows, j] + q * hw
        new[f] = dict(d, lo=lo, hi=hi)
    return new

def metrics(R):
    rows = []
    for f, d in R.items():
        for j, h in enumerate(HZ):
            sel = d['ym'][:, j] & d['storm'] & ~np.isnan(d['med'][:, j])
            y, lo, hi, med = d['y'][sel, j], d['lo'][sel, j], d['hi'][sel, j], d['med'][sel, j]
            wink = (hi - lo) + 2 / ALPHA * (lo - y) * (y < lo) + 2 / ALPHA * (y - hi) * (y > hi)
            rows.append(dict(fold=f, h=h, n=int(sel.sum()), cover=np.sum((y >= lo) & (y <= hi)), width=np.sum(hi - lo),
                             wink=np.sum(wink), se=np.sum((med - y) ** 2), sep=np.sum(y ** 2)))
    return pd.DataFrame(rows)

def pooled(M, drop=()):
    M = M[~M.fold.isin(drop)]
    G = M.groupby('h')[['n', 'cover', 'width', 'wink', 'se', 'sep']].sum()
    return pd.DataFrame({'coverage': G.cover / G.n, 'mean width': G.width / G.n, 'interval score': G.wink / G.n,
                         'skill': 1 - G.se / G.sep})

def table(mode, exclude_calib=()):
    res = {}
    for m in ['vi', 'nuts', 'vi_nodisc']:
        R = load_bts(m, mode)
        if len(R) == 7:
            res[f'B-TS {m} (raw)'] = metrics(R)
            res[f'B-TS {m} + conformal'] = metrics(recalibrate(R, 'scale', exclude_calib))
    for m in ['persistence', 'M2', 'PGMLP']:
        res[f'{m} + conformal'] = metrics(recalibrate(load_point(m, mode), 'point', exclude_calib))
    return res

if __name__ == '__main__':
    pd.set_option('display.width', 200)
    import sys
    excl = (2,) if 'gated' in sys.argv else ()
    for mode in ['oracle', 'past']:
        res = table(mode, excl)
        for drop, lab in [((), 'all 7 storms'), ((2,), 'excluding storm 2')]:
            T = pd.concat({k: pooled(v, drop) for k, v in res.items()}, axis=0)
            print(f'\n=== {mode}, storm-time, {lab} ===\n', T.round(4).unstack('h').to_string())
