import numpy as np, pandas as pd, json
from p1_physics import build
from eval_p1 import per_window, load_preds, skill, boot_diff
B = build('utuado'); d = B['d']; W = B['wins']
P = load_preds('utuado', B); rows = per_window(B, P, 'secondary')
common = np.array([i for i in range(len(W)) if all(~np.isnan(r[0][i]) and r[2][i] > 0 for r in rows.values())])
out = {}
for phys in ['R-cal', 'R-cal-Se (exploratory)']:
    out[phys + ' minus PG-MLP @57cm'] = {s: [round(skill(*rows[phys][:2], common) - skill(*rows[f'PG-MLP s{s}'][:2], common), 3), [round(c, 3) for c in boot_diff(rows[phys], rows[f'PG-MLP s{s}'], common)]] for s in range(3)}
# positive pore pressure (perched saturation) at 40/55 cm during storm windows
pos = {}
for col in ['pwp_40cm_kPa', 'pwp_55cm_kPa']:
    frac = [float((d.loc[w['a']:w['b'], col] > 0.5).any()) for w in W if d.loc[w['a']:w['b'], col].notna().mean() > 0.5]
    pos[col] = dict(windows=len(frac), frac_windows_with_positive_pressure=round(float(np.mean(frac)), 3))
out['perched_saturation'] = pos
# where does R-cal miss at 72 cm? peak rise predicted vs observed per window
k = B['i_primary']; cal = P['R-cal-Se (exploratory)']
ratios = []
for i, w in enumerate(W):
    y = w['Y'][:, k]; p = cal[i][:, k]
    if np.isnan(y[:4]).all() or np.isnan(y).mean() > 0.5: continue
    y0 = np.nanmean(y[:4]); dy, dp = np.nanmax(y) - y0, np.nanmax(p) - p[0]
    if dy > 0.02: ratios.append(dp / dy)
out['72cm_peak_rise_model_over_obs'] = dict(windows=len(ratios), median=round(float(np.median(ratios)), 2), q25_q75=[round(float(np.quantile(ratios, q)), 2) for q in (.25, .75)])
json.dump(out, open('out/explore_72.json', 'w'), indent=1); print(json.dumps(out, indent=1))
