"""Evaluate P4: coverage, width, interval score of 90% intervals on the primary target (held-out windows)."""
import json, os, numpy as np
from p1_physics import build
from pr_data import SITES
BINS = np.array([0, 6, 24, 48, 1e9]) * 4
ALPHA = 0.1

def truth(B, i):
    w = B['wins'][i]; y = w['Y'][:, B['i_primary']].copy(); y[~w['top_ok']] = np.nan; return y

def score(B, lo, hi, med, idx):
    cov = wid = win = se = ref = 0.0; N = 0
    for i in idx:
        if lo[i] is None: continue
        y = truth(B, i); ok = ~np.isnan(y) & ~np.isnan(lo[i])
        if not ok.any(): continue
        yy, l, h, m = y[ok], lo[i][ok], hi[i][ok], med[i][ok]; y0 = np.nanmean(y[:4])
        cov += np.sum((yy >= l) & (yy <= h)); wid += np.sum(h - l)
        win += np.sum((h - l) + 2 / ALPHA * (l - yy) * (yy < l) + 2 / ALPHA * (yy - h) * (yy > h))
        se += np.sum((m - yy) ** 2); ref += np.sum((y0 - yy) ** 2); N += ok.sum()
    return dict(coverage=cov / N, width=wid / N, interval_score=win / N, skill=1 - se / ref, n=int(N))

def mlp_conformal(B, site, blocks):
    m = np.load(f'out/p1_mlp_{site}.npz', allow_pickle=True)['primary_s0']
    n = len(B['wins']); lo, hi, med = [None] * n, [None] * n, [None] * n
    bin_of = lambda L: np.clip(np.searchsorted(BINS, np.arange(L), side='right') - 1, 0, 3)
    resid = {k: {b: [] for b in range(4)} for k in range(len(blocks))}
    for k, blk in enumerate(blocks):
        for i in blk:
            if m[i] is None: continue
            y = truth(B, i); p = m[i]; bb = bin_of(len(y)); ok = ~np.isnan(y) & ~np.isnan(p)
            for b in range(4): resid[k][b] += list(np.abs(y - p)[ok & (bb == b)])
    for k, blk in enumerate(blocks):
        q = []
        for b in range(4):
            s = np.sort(np.concatenate([np.array(resid[j][b]) for j in range(len(blocks)) if j != k] + [np.array([])]))
            q.append(s[min(int(np.ceil((1 - ALPHA) * (len(s) + 1))) - 1, len(s) - 1)] if len(s) else np.nan)
        q = np.array(q)
        for i in blk:
            if m[i] is None: continue
            p = m[i]; qq = q[bin_of(len(p))]; lo[i], hi[i], med[i] = p - qq, p + qq, p
    return lo, hi, med

rep = {}
for site in SITES:
    B = build(site); n = len(B['wins'])
    blocks = json.load(open(f'out/p1_physics_{site}.json'))['blocks']
    R = {}
    for model, tag in [('cal', 'R-cal Laplace (registered)'), ('se', 'R-cal-Se Laplace (exploratory)')]:
        f = f'out/p4_{model}_{site}.npz'
        if os.path.exists(f):
            z = np.load(f, allow_pickle=True); R[tag] = (z['lo'], z['hi'], z['med'])
    R['PG-MLP s0 + cross-conformal'] = mlp_conformal(B, site, blocks)
    valid = [set(i for i in range(n) if v[0][i] is not None) for v in R.values()]
    common = sorted(set.intersection(*valid))
    rep[site] = {k: score(B, *v, common) for k, v in R.items()}
    rep[site]['n_windows'] = len(common)
json.dump(rep, open('out/p4_report.json', 'w'), indent=1, default=float)
print(json.dumps(rep, indent=1, default=float))
