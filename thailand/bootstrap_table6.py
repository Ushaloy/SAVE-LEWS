"""Table VI: storm-level bootstrap of pooled LOSO skill differences (10 000 replicates, seed 0)."""
import json, numpy as np
rng = np.random.default_rng(0)

def fold_stats(model, h=60, mode='oracle'):
    if model.startswith('PG'):
        s = int(model[-1]); r = json.load(open('res3_pgmlp.json'))
        return [(r[f'seed{s}_{mode}_s{i}'][f'mse_storm_{h}'] * r[f'seed{s}_{mode}_s{i}'][f'n_storm_{h}'],
                 r[f'seed{s}_{mode}_s{i}'][f'msep_storm_{h}'] * r[f'seed{s}_{mode}_s{i}'][f'n_storm_{h}']) for i in range(7)]
    out = []
    for f in range(7):
        x = json.load(open(f'res3_loso_{model}_f{f}.json'))[f'held_{mode}']
        out.append((x[f'mse_storm_{h}'] * x[f'n_storm_{h}'], x[f'msep_storm_{h}'] * x[f'n_storm_{h}']))
    return out

def boot(a, b, drop2=False, B=10000):
    idx = [i for i in range(7) if not (drop2 and i == 2)]
    A = np.array(a)[idx]; Bm = np.array(b)[idx]; n = len(idx); d = []
    for _ in range(B):
        k = rng.integers(0, n, n)
        d.append((1 - A[k, 0].sum() / A[k, 1].sum()) - (1 - Bm[k, 0].sum() / Bm[k, 1].sum()))
    d = np.array(d); pt = (1 - A[:, 0].sum() / A[:, 1].sum()) - (1 - Bm[:, 0].sum() / Bm[:, 1].sum())
    return pt, np.quantile(d, [.025, .975]), (d > 0).mean()

if __name__ == '__main__':
    for h in (30, 60, 120):
        for m1, m2 in [('PG0', 'M0'), ('PG1', 'M0'), ('PG2', 'M0'), ('M2np', 'M2'), ('M2', 'M3'), ('M2', 'PG0'), ('M2', 'PG1'), ('M2', 'PG2')]:
            for drop in (False, True):
                pt, ci, p = boot(fold_stats(m1, h), fold_stats(m2, h), drop)
                print(f"h{h} {m1}-{m2} {'ex-s2' if drop else 'all  '}: diff {pt:+.2f} 95%CI [{ci[0]:+.2f},{ci[1]:+.2f}] P(diff>0)={p:.2f}")
