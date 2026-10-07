"""Ablation: unconstrained MLP (U) vs PG-MLP vs physics on the common held-out storms. Writes out/mlpU_summary.json."""
import json, numpy as np
from p1_physics import build
from pr_data import SITES
from eval_p1 import load_preds, per_window, skill

def main():
    res = {}
    for site in SITES:
        B = build(site); P = load_preds(site, B)
        m = np.load(f'out/p1_mlpU_{site}.npz', allow_pickle=True)
        for s in range(3):
            P[f'U s{s}'] = ('mlp', m[f'primary_s{s}'], m[f'secondary_s{s}'])
        for which in ('primary', 'secondary'):
            rows = per_window(B, P, which)
            valid = [np.where(~np.isnan(r[0]) & (r[2] > 0))[0] for r in rows.values()]
            common = np.array(sorted(set.intersection(*[set(v) for v in valid])))
            res[f'{site} {which}'] = dict(n_storms=int(len(common)), skill={k: round(float(skill(r[0], r[1], common)), 3) for k, r in rows.items()})
    json.dump(res, open('out/mlpU_summary.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))

if __name__ == '__main__':
    main()
