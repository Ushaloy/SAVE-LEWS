import pandas as pd, numpy as np, json
from pr_data import *
res = {}
for site in SITES:
    d = load(site); c = SITES[site]; S = storms(d)
    rows = []
    for i, st in S.iterrows():
        pre = d.loc[st.start - pd.Timedelta(hours=24): st.start]
        post = d.loc[st.start: st.start + pd.Timedelta(hours=48)]
        on = {}
        for key in ['shallow', 'deep', 'deep2']:
            col = c[key]; base = pre[col].mean()
            if np.isnan(base) or post[col].notna().mean() < 0.8:
                on[key] = None; continue
            hit = post[col] > base + 0.005
            on[key] = hit.idxmax() if hit.any() else pd.NaT
        rows.append(dict(storm=i, start=st.start, rain=st.rain_mm, **{f'on_{k}': v for k, v in on.items()}))
    R = pd.DataFrame(rows)
    out = {}
    for key in ['deep', 'deep2']:
        both = R[R.on_shallow.notna() & R[f'on_{key}'].notna() & R.on_shallow.apply(lambda x: x is not None and not pd.isna(x)) & R[f'on_{key}'].apply(lambda x: x is not None and not pd.isna(x))]
        lag = (pd.to_datetime(both[f'on_{key}']) - pd.to_datetime(both.on_shallow)).dt.total_seconds() / 3600
        valid_storms = R[R.on_shallow.apply(lambda x: x is not None) & R[f'on_{key}'].apply(lambda x: x is not None)]
        n_sh_rise = valid_storms.on_shallow.apply(lambda x: not pd.isna(x)).sum()
        n_deep_rise = valid_storms[f'on_{key}'].apply(lambda x: not pd.isna(x)).sum()
        out[key] = dict(z=c[f'z_{key}'], storms_with_data=int(len(valid_storms)), shallow_rises=int(n_sh_rise), deep_rises=int(n_deep_rise),
                        both_rise=int(len(lag)), median_lag_h=float(np.median(lag)) if len(lag) else None,
                        iqr_h=[float(np.quantile(lag, .25)), float(np.quantile(lag, .75))] if len(lag) else None,
                        frac_le_30min=float((lag <= 0.5).mean()) if len(lag) else None,
                        frac_deep_before_shallow=float((lag < 0).mean()) if len(lag) else None)
    res[site] = out
    print(site, json.dumps(out, indent=1))
json.dump(res, open('res_p2.json', 'w'), indent=1)
