import os
from common import ROOT
OUT_T = os.path.join(ROOT, 'tables', 'x')
from common import out, TH, PR, DATA_TH
import os
import pandas as pd, numpy as np, sys
out = {}
def metrics(th, rain, dt_min, temp=None, name=''):
    th = th.astype(float); rain = rain.fillna(0).astype(float)
    r24 = rain.rolling(int(24*60/dt_min), min_periods=1).sum()
    quiet = (r24 == 0) & th.notna()
    d = th.diff()
    # noise: std of residual from 1-h centred moving average during quiet periods
    ma = th.rolling(max(3, int(60/dt_min)), center=True, min_periods=2).mean()
    noise = float((th - ma)[quiet].std())
    nz = np.abs(d[d != 0].dropna()); res = float(np.nanpercentile(nz, 5)) if len(nz) else np.nan
    rng = (float(np.nanpercentile(th, 1)), float(np.nanpercentile(th, 99)))
    miss = float(th.isna().mean())
    tsens = np.nan
    if temp is not None:
        q = quiet & temp.notna()
        x = temp[q] - temp[q].rolling(int(24*60/dt_min), center=True, min_periods=10).mean()
        y = th[q] - th[q].rolling(int(24*60/dt_min), center=True, min_periods=10).mean()
        ok = x.notna() & y.notna()
        if ok.sum() > 100: tsens = float(np.polyfit(x[ok], y[ok], 1)[0])
    return dict(noise=noise, resolution=res, p1=rng[0], p99=rng[1], range=rng[1]-rng[0], missing=miss, temp_sens=tsens,
                snr=(rng[1]-rng[0])/noise if noise > 0 else np.nan)
from data_pipeline import load_device
for dev, f in [('Dev108', 'pinn_108_new.csv'), ('Dev107', '107_pinn.csv')]:
    g = load_device(os.path.join(DATA_TH, f))
    if dev == 'Dev108': g = g.loc['2025-11-01 10:40':]
    raw = pd.read_csv(os.path.join(DATA_TH, f))
    s = raw['soil'].values; runs = np.diff(np.flatnonzero(np.r_[True, s[1:] != s[:-1], True]))
    m = metrics(g.theta, g.rain, 5, g.temp, dev); m['median_run_min'] = float(np.median(runs))
    out[dev] = m
from pr_data import load, SITES
for site, cols in [('utuado', ['vwc_SP1_27cm_ccpercc', 'vwc_SP1_57cm_ccpercc', 'vwc_SP1_72cm_ccpercc']),
                   ('toronegro', ['vwc_sp1_30cm_ccpercc', 'vwc_sp1_90cm_ccpercc', 'vwc_sp1_110cm_ccpercc'])]:
    d = load(site)
    for c in cols:
        out[f'{site} {c.split("_")[2]}'] = metrics(d[c], d['rain'], 15, d['airTemperature_degC'])
    out[f'{site} rain'] = dict(tip=float(np.nanmin(d['rain'][d['rain'] > 0])))
df = pd.DataFrame(out).T
pd.set_option('display.width', 200); print(df.round(4).to_string())
df.round(4).to_csv(os.path.join(os.path.dirname(OUT_T), 'Table02_sensor_comparison.csv'))
