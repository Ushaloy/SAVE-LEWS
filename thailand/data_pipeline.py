"""Shared data pipeline: Dev107/108 -> 5-min grid with reset-aware rain increments."""
import numpy as np, pandas as pd
TIP_MM = 0.2794

def load_device(path, temp_coef=-0.0009, tref=None):
    d = pd.read_csv(path)
    d['t'] = pd.to_datetime(d.timestamp, format='%m/%d/%Y %H:%M')
    d = d.sort_values('t', kind='stable').groupby('t').agg(
        rain=('rain', 'last'), soil=('soil', 'mean'), temp=('temp', 'mean'))
    c = d.rain.values
    dif = np.diff(c, prepend=c[0])
    inc = np.clip(dif, 0, None)
    # a descent = interpolated hourly reset; the true counter restarted at 0,
    # so the bottom value of the descent is already new-hour rain
    bottom = (dif < 0) & (np.r_[dif[1:], 0] >= 0)
    inc = inc + c * bottom
    d['rain_inc'] = inc
    g = pd.DataFrame({
        'rain': d.rain_inc.resample('5min').sum(min_count=1),   # empty bin -> NaN (missing, not dry)
        'theta_raw': d.soil.resample('5min').mean(),
        'temp': d.temp.resample('5min').mean(),
        'n_raw': d.soil.resample('5min').count()})
    tref = g.temp.mean() if tref is None else tref
    # sensor reads low when warm: remove the artifact
    g['theta'] = g.theta_raw - temp_coef * (g.temp - tref)
    return g

def storm_catalogue(g, min_total=10.0, dry_gap_h=6.0):
    """Storms = rain clusters separated by >= dry_gap_h hours with no rain; keep total >= min_total."""
    r = g.rain.fillna(0)
    wet = r[r > 0].index
    storms, start, last, tot = [], None, None, 0.0
    for t in wet:
        if start is None or (t - last) > pd.Timedelta(hours=dry_gap_h):
            if start is not None: storms.append((start, last, tot))
            start, tot = t, 0.0
        tot += r[t]; last = t
    if start is not None: storms.append((start, last, tot))
    s = pd.DataFrame(storms, columns=['start', 'end', 'rain_mm'])
    return s[s.rain_mm >= min_total].reset_index(drop=True)
