"""Summarise the physics-guidance ablation (U = unconstrained MLP, B = bounds only, PG = full PG-MLP).
Reads res_ablation_{U,B,PG}.json (written by pg_ablation.py) and writes ablation_summary.json.
Skill = 1 - sum(SE)/sum(SE_persistence) pooled over the six regular held-out storms (storm 2 reported separately);
ranges are over three seeds."""
import json, numpy as np
HZ = (30, 60, 120); REG = [0, 1, 3, 4, 5, 6]

def pooled(d, mode, seed, h, storms):
    se = sum(d[f'loso_{mode}_seed{seed}_s{i}'][f'se_{h}'] for i in storms)
    ref = sum(d[f'loso_{mode}_seed{seed}_s{i}'][f'ref_{h}'] for i in storms)
    return 1 - se / ref

def one(d, key, mode, seed, i, h=60):
    r = d[f'{key}_{mode}_seed{seed}_s{i}']; return 1 - r[f'se_{h}'] / r[f'ref_{h}']

def rate(d, mode, name):
    """violation rate: unweighted mean over all forecast sets evaluated (7 LOSO folds incl. storm 2
    + 4 zero-shot Dev107 storms) and 3 seeds, as reported in the manuscript"""
    ks = [f'loso_{mode}_seed{s}_s{i}' for s in range(3) for i in range(7)] + \
         [f'zs_{mode}_seed{s}_s{i}' for s in range(3) for i in range(4)]
    v = [d[k].get(name) for k in ks]
    if any(x is None for x in v): return None
    return round(float(np.mean(v)), 4)

def main():
    out = {}
    for V in ['U', 'B', 'PG']:
        d = json.load(open(f'res_ablation_{V}.json')); o = {}
        for mode in ('oracle', 'past'):
            for h in HZ:
                v = [pooled(d, mode, s, h, REG) for s in range(3)]; o[f'{mode}{h}'] = [round(min(v), 2), round(max(v), 2)]
            o[f'{mode}_s2_60'] = [round(one(d, 'loso', mode, s, 2), 2) for s in range(3)]
            o[f'zs_{mode}'] = [[round(one(d, 'zs', mode, s, i), 2) for s in range(3)] for i in range(4)]
            for name in ('viol_bounds', 'viol_wet_dry', 'viol_monotone'):
                o[f'{mode}_{name}'] = rate(d, mode, name)
        out[V] = o
    json.dump(out, open('ablation_summary.json', 'w'), indent=1)
    return out

if __name__ == '__main__':
    print(json.dumps(main(), indent=1)[:3000])
