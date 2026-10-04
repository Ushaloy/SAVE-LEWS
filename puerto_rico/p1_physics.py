"""P1 pipeline: storm windows -> padded arrays; R-cal calibration (Adam through the solver); R-lab; evaluation."""
import numpy as np, pandas as pd, jax, jax.numpy as jnp, optax, json, time, openpyxl
import os
from pr_data import SITES, load, storms, window, DATA_DIR
from richards_jax import simulate, pack, unpack

DZ = 2.5
PRE_H, POST_H, PURGE_H = 24, 72, 24

def lab_params(site):
    wb = openpyxl.load_workbook(os.path.join(DATA_DIR, 'PuertoRico_TestingData.xlsx'), data_only=True)['Summary']
    rows = [r for r in wb.iter_rows(min_row=3, max_row=9, values_only=True)]
    key = 'UTU-1' if site == 'utuado' else 'ELT-1'
    out = []
    for r in rows:
        if r[0] and r[0].startswith(key) and isinstance(r[11], (int, float)):
            out.append(dict(sample=r[0], depth=r[1], ts_d=r[9], tr_d=r[10], a_d=r[11], n_d=r[12], Ks_d=r[13],
                            ts_w=r[14], tr_w=r[15], a_w=r[16], n_w=r[17], Ks_w=r[18]))
    return out

def build(site):
    c = SITES[site]; d = load(site); S = storms(d)
    zs, zb = c['z_shallow'] * 100, c['z_base'] * 100
    z = np.arange(zs, zb + 1e-6, DZ)
    prof = {k * 100: v for k, v in c['profile'].items() if k * 100 >= zs}
    tz = [k for k in sorted(prof) if k > zs]                      # target depths below the shallow probe
    obs_idx = np.array([np.argmin(abs(z - q)) for q in tz])
    wins = []
    for i, st in S.iterrows():
        w = window(d, st, PRE_H, POST_H)
        top_raw = w[c['shallow']]
        if top_raw.notna().mean() < 0.8 or np.isnan(top_raw.iloc[:4]).all():
            continue
        top = top_raw.interpolate(limit_direction='both').values
        row0 = w.iloc[:4].mean(numeric_only=True)
        pz = np.array([q for q in sorted(prof) if not np.isnan(row0[prof[q]])])
        if len(pz) < 2 or c['deep'] not in w or np.isnan(row0[c['deep']]):
            continue
        th0 = np.interp(z, pz, [row0[prof[q]] for q in pz])
        Y = np.stack([w[prof[q]].values for q in tz], 1)          # (T, n_targets), NaN = missing
        in_storm = ((w.index >= st.start - pd.Timedelta(hours=1)) & (w.index <= st.end + pd.Timedelta(hours=6)))
        wins.append(dict(storm=i, start=st.start, end=st.end, rain_mm=st.rain_mm, t=w.index, top=top, top_ok=top_raw.notna().values,
                         th0=th0, Y=Y, rain=w.rain.fillna(0).values, in_storm=in_storm,
                         a=w.index[0], b=w.index[-1]))
    return dict(site=site, z=z, tz=tz, obs_idx=obs_idx, wins=wins, S=S, d=d,
                i_primary=tz.index(c['z_deep'] * 100), i_secondary=tz.index(c['z_deep2'] * 100))

def pad(wins, keys=('top', 'Y')):
    T = max(len(w['top']) for w in wins); n = len(wins)
    top = np.zeros((n, T)); Y = np.full((n, T, wins[0]['Y'].shape[1]), np.nan); th0 = np.stack([w['th0'] for w in wins])
    for i, w in enumerate(wins):
        L = len(w['top']); top[i, :L] = w['top']; top[i, L:] = w['top'][-1]
        Y[i, :L] = w['Y']
        Y[i, :L][~w['top_ok']] = np.nan                            # exclude rows whose input was filled
    return jnp.array(top), jnp.array(th0), jnp.array(Y)

def train_folds(B, held):
    """indices of training windows for held-out window index `held` (purge +/- 24 h)."""
    h = B['wins'][held]
    p0, p1 = h['a'] - pd.Timedelta(hours=PURGE_H), h['b'] + pd.Timedelta(hours=PURGE_H)
    return [j for j, w in enumerate(B['wins']) if j != held and (w['b'] < p0 or w['a'] > p1)]

def make_loss(B):
    z, oi = B['z'], B['obs_idx']
    sim = jax.vmap(lambda p, top, th0: simulate(p, top, th0, z, oi), in_axes=(None, 0, 0))
    def loss(p, top, th0, Y):
        pred = sim(p, top, th0)
        m = ~jnp.isnan(Y)
        return jnp.sum(jnp.where(m, (pred - jnp.nan_to_num(Y)) ** 2, 0.0)) / jnp.sum(m)
    return jax.jit(loss), jax.jit(jax.value_and_grad(loss)), jax.jit(sim)

def calibrate(vg, top, th0, Y, starts, iters=250, lr=0.05):
    best = None
    for p0 in starts:
        p = jnp.array(p0); opt = optax.adam(lr); st = opt.init(p)
        for it in range(iters):
            L, g = vg(p, top, th0, Y)
            g = jnp.nan_to_num(g)
            up, st = opt.update(g, st, p); p = optax.apply_updates(p, up)
        L = float(vg(p, top, th0, Y)[0])
        if best is None or L < best[1]:
            best = (np.array(p), L)
    return best
