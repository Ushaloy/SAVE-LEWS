"""Puerto Rico data pipeline: load, QC, storm catalogue (pre-registered rules)."""
import os, numpy as np, pandas as pd

# Raw USGS files live in <repo>/data/puerto_rico (see data/puerto_rico/README.md)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'puerto_rico')

SITES = {
    'utuado': dict(file='utuado_15min.csv', shallow='vwc_SP1_27cm_ccpercc', deep='vwc_SP1_72cm_ccpercc',
                   deep2='vwc_SP1_57cm_ccpercc', z_shallow=0.27, z_deep=0.72, z_deep2=0.57, z_base=0.92,
                   profile={0.12: 'vwc_SP1_12cm_ccpercc', 0.27: 'vwc_SP1_27cm_ccpercc', 0.42: 'vwc_SP1_42cm_ccpercc',
                            0.57: 'vwc_SP1_57cm_ccpercc', 0.72: 'vwc_SP1_72cm_ccpercc'}),
    'toronegro': dict(file='toroNegro_15min.csv', shallow='vwc_sp1_30cm_ccpercc', deep='vwc_sp1_90cm_ccpercc',
                      deep2='vwc_sp1_110cm_ccpercc', z_shallow=0.30, z_deep=0.90, z_deep2=1.10, z_base=1.32,
                      profile={0.30: 'vwc_sp1_30cm_ccpercc', 0.50: 'vwc_sp1_50cm_ccpercc', 0.70: 'vwc_sp1_70cm_ccpercc',
                               0.90: 'vwc_sp1_90cm_ccpercc', 1.10: 'vwc_sp1_110cm_ccpercc'}),
}

def load(site):
    c = SITES[site]
    d = pd.read_csv(os.path.join(DATA_DIR, c['file']))
    d['t'] = pd.to_datetime(d.timestamp_UTC)
    d = d.set_index('t').sort_index()
    d = d[~d.index.duplicated()]
    d = d.asfreq('15min')                      # gaps become NaN rows
    d['rain'] = d.precipitation_mm
    for col in list(c['profile'].values()):
        v = d[col]
        d.loc[(v <= 0) | (v > 0.7), col] = np.nan  # physically impossible readings
    return d

def storms(d, min_total=10.0, dry_gap_h=6.0):
    r = d.rain.fillna(0)
    wet = r[r > 0].index
    out, start, last, tot = [], None, None, 0.0
    for t in wet:
        if start is None or (t - last) > pd.Timedelta(hours=dry_gap_h):
            if start is not None: out.append((start, last, tot))
            start, tot = t, 0.0
        tot += r[t]; last = t
    if start is not None: out.append((start, last, tot))
    s = pd.DataFrame(out, columns=['start', 'end', 'rain_mm'])
    return s[s.rain_mm >= min_total].reset_index(drop=True)

def window(d, st, pre_h=24, post_h=72):
    return d.loc[st.start - pd.Timedelta(hours=pre_h): st.end + pd.Timedelta(hours=post_h)]
