"""Leave-one-storm-out harness: multi-window training with purge, probe observation operator
(fast domain), physics residual, evaluation on a held-out storm window."""
import numpy as np, pandas as pd, torch, time
from richards_pinn import Physics, Stepper, residual, DT, NZ, Z, I_OBS, I_FAIL
from data_pipeline import load_device, storm_catalogue

HORIZONS = (6, 12, 24)
H = 24
PRE_H, POST_H, PURGE_H = 48, 24, 24

def window_bounds(storm):
    return storm.start - pd.Timedelta(hours=PRE_H), storm.end + pd.Timedelta(hours=POST_H)

def make_window(g, storm, purge=None):
    """storm window; if purge=(p0,p1) overlaps it, cut the window back to the side holding the storm."""
    a, b = window_bounds(storm)
    if purge is not None:
        p0, p1 = purge
        if p1 > a and p0 < b:
            if p1 <= storm.start:
                a = max(a, p1)
            elif p0 >= storm.end:
                b = min(b, p0)
            else:
                return None           # purge covers the storm itself
    w = g.loc[a:b]
    rain = w.rain.values.astype(np.float32); obs = w.theta.values.astype(np.float32)
    in_storm = (w.index >= storm.start - pd.Timedelta(hours=1)) & (w.index <= storm.end + pd.Timedelta(hours=6))
    return dict(t=w.index, rain=torch.tensor(np.nan_to_num(rain)), obs=torch.tensor(np.nan_to_num(obs)),
                obs_ok=torch.tensor(~np.isnan(obs)), in_storm=torch.tensor(in_storm),
                name=str(storm.start.date()), rain_mm=float(storm.rain_mm))

def loso_windows(g, S, held):
    """training windows for fold `held` (index into S), purging held-out window +/- PURGE_H."""
    a, b = window_bounds(S.iloc[held])
    purge = (a - pd.Timedelta(hours=PURGE_H), b + pd.Timedelta(hours=PURGE_H))
    wins = [make_window(g, S.iloc[i], purge) for i in range(len(S)) if i != held]
    return [w for w in wins if w is not None and int(w['obs_ok'].sum()) > H + 12]

def stack(wins):
    """pad windows to (W, Tmax)."""
    W = len(wins); T = max(len(w['obs']) for w in wins)
    P = dict(rain=torch.zeros(W, T), obs=torch.zeros(W, T), ok=torch.zeros(W, T, dtype=torch.bool),
             storm=torch.zeros(W, T, dtype=torch.bool), T=torch.tensor([len(w['obs']) for w in wins]))
    for i, w in enumerate(wins):
        n = len(w['obs'])
        P['rain'][i, :n] = w['rain']; P['obs'][i, :n] = w['obs']
        P['ok'][i, :n] = w['obs_ok']; P['storm'][i, :n] = w['in_storm']
    return P

def nudge_gain(L=0.08):
    return torch.exp(-0.5 * ((Z - Z[I_OBS]) / L) ** 2)

@torch.no_grad()
def analysis(model, P, drv):
    """real-time assimilation for all windows in parallel. drv (W,T,C). returns A (W,T,NZ)."""
    ph = model.phys; W, T = P['obs'].shape; G = nudge_gain()
    first = torch.stack([P['obs'][i][P['ok'][i]][0] for i in range(W)])
    th = first[:, None].expand(W, NZ).clone(); out = [th]
    fast = ph.fast
    for k in range(T - 1):
        th, _, _ = model(th, drv[:, k])
        ok = P['ok'][:, k + 1]
        if ok.any():
            tgt = ph.sensor_inverse(P['obs'][:, k + 1], drv[:, k + 1, 3]) if fast else P['obs'][:, k + 1]
            upd = (th + G[None] * (tgt - th[:, I_OBS])[:, None]).clamp(ph.tr + 1e-3, ph.ts - 1e-3)
            th = torch.where(ok[:, None], upd, th)
        out.append(th)
    return torch.stack(out, 1)

def future_drive(ph, drv, wi, ki, oracle):
    """(B,H,C) drive after origins (window wi, step ki). Past-only: no new rain; the fast store decays."""
    steps = torch.arange(1, H + 1)
    if oracle:
        return drv[wi[:, None], (ki[:, None] + steps[None] - 1).clamp(max=drv.shape[1] - 1)]
    C = drv.shape[-1]
    fd = torch.zeros(len(wi), H, C)
    if ph.fast:
        fdec, fx = ph.fast_decay()
        F0 = drv[wi, ki, 3]
        j = torch.arange(H, dtype=torch.float32)
        fd[:, :, 2] = F0[:, None] * fdec ** j[None] * (1 - fdec) * fx      # exchange during step
        fd[:, :, 3] = F0[:, None] * fdec ** (j[None] + 1)                   # store after step
    return fd

def rollout(model, th0, fdrv):
    th = th0; traj = []; res = []
    for j in range(fdrv.shape[1]):
        th_new, q0, S = model(th, fdrv[:, j])
        R, *_ = residual(model.phys, th, th_new, q0, S)
        traj.append(th_new); res.append(R); th = th_new
    return torch.stack(traj, 1), torch.stack(res, 1)

def origins(P):
    W, T = P['obs'].shape
    k = torch.arange(T)[None].expand(W, T)
    valid = P['ok'] & (k < (P['T'][:, None] - H))
    wi, ki = torch.nonzero(valid, as_tuple=True)
    return wi, ki

def pred_sensor_delta(ph, traj, fdrv, obs0, h):
    return ph.sensor(traj[:, h - 1, I_OBS], fdrv[:, h - 1, 3] if ph.fast else None) - obs0

def train(wins, lam=1.0, r_scale=1e-6, ramp=0.4, event_weight=20.0, lateral=False, fast=False, bounded=False,
          iters=1500, lr=3e-3, seed=0, refresh=25, batch=64, log_every=250, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    phys = Physics(lateral=lateral, fast=fast, bounded=bounded); model = Stepper(phys)
    opt = torch.optim.Adam([{'params': model.net.parameters(), 'lr': lr},
                            {'params': phys.parameters(), 'lr': lr * 3}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    P = stack(wins); wi, ki = origins(P)
    obs, ok = P['obs'], P['ok']
    pers = []
    for h in HORIZONS:
        m = ok[wi, ki + h]
        pers.append(((obs[wi, ki + h] - obs[wi, ki])[m] ** 2).mean())
    pers = torch.stack(pers).clamp_min(1e-8)
    w_orig = torch.where(P['storm'][wi, ki], 4.0, 1.0)
    # residual sampling: storm x event weights over all (w,k) with k < T-1
    W, T = obs.shape
    dob = torch.zeros(W, T); dob[:, :-1] = torch.where(ok[:, 1:] & ok[:, :-1], (obs[:, 1:] - obs[:, :-1]).abs(), 0.)
    kk = torch.arange(T)[None].expand(W, T)
    wres = torch.where(P['storm'], 4.0, 1.0) * (1 + event_weight * (dob / 0.002).clamp(max=5.0))
    wres = torch.where(kk < P['T'][:, None] - 1, wres, 0.).flatten()
    t0 = time.time(); hist = []
    for it in range(iters):
        drv = phys.drive(P['rain'])
        if it % refresh == 0:
            A = analysis(model, P, drv.detach())
        sel = torch.multinomial(w_orig, batch, replacement=True); bw, bk = wi[sel], ki[sel]
        fd = future_drive(phys, drv, bw, bk, oracle=True)
        traj, R = rollout(model, A[bw, bk], fd)
        l_data = 0
        for j, h in enumerate(HORIZONS):
            m = ok[bw, bk + h]
            pred = pred_sensor_delta(phys, traj, fd, obs[bw, bk], h)
            l_data = l_data + ((pred - (obs[bw, bk + h] - obs[bw, bk]))[m] ** 2).mean() / pers[j]
        l_data = l_data / len(HORIZONS)
        r = torch.multinomial(wres, batch, replacement=True); rw, rk = r // T, r % T
        th1, q0, S = model(A[rw, rk], drv[rw, rk])
        R1, *_ = residual(phys, A[rw, rk], th1, q0, S)
        l_phys = 0.5 * ((R1 / r_scale) ** 2).mean() + 0.5 * ((R / r_scale) ** 2).mean()
        lam_it = lam * (min(1.0, 0.01 + it / max(1, ramp * iters)) if ramp > 0 else 1.0)
        loss = l_data + lam_it * l_phys
        opt.zero_grad()
        if not torch.isfinite(loss):
            continue
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            opt.zero_grad(); continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        hist.append((float(l_data.detach()), float(l_phys.detach())))
        if verbose and (it % log_every == 0 or it == iters - 1):
            pv = phys.values()
            print(f"it {it:5d} data {hist[-1][0]:.4f} phys {hist[-1][1]:.4f} | " +
                  ' '.join(f"{k} {v:.3g}" for k, v in pv.items()) + f" | {time.time()-t0:.0f}s", flush=True)
    return model, np.array(hist)

@torch.no_grad()
def evaluate(model, win, oracle=True, keep=False):
    ph = model.phys; P = stack([win])
    drv = ph.drive(P['rain']); A = analysis(model, P, drv)
    wi, ki = origins(P)
    fd = future_drive(ph, drv, wi, ki, oracle)
    traj, R = rollout(model, A[wi, ki], fd)
    obs, ok, st = P['obs'], P['ok'], P['storm']
    out = {}
    for h in HORIZONS:
        m = ok[wi, ki + h]
        pred = pred_sensor_delta(ph, traj, fd, obs[wi, ki], h); targ = obs[wi, ki + h] - obs[wi, ki]
        for tag, s in [('all', m), ('storm', m & st[wi, ki])]:
            out[f'skill_{tag}_{h*5}'] = float(1 - ((pred - targ)[s] ** 2).mean() / (targ[s] ** 2).mean())
            out[f'mse_{tag}_{h*5}'] = float(((pred - targ)[s] ** 2).mean())
            out[f'msep_{tag}_{h*5}'] = float((targ[s] ** 2).mean())
            out[f'n_{tag}_{h*5}'] = int(s.sum())
    # wetting at the probe with no rain in the preceding 3 h (history + fed future rain)
    rain = P['rain'][0]; csum = torch.cat([torch.zeros(1), torch.cumsum(rain, 0)])
    sens = torch.stack([ph.sensor(traj[:, j, I_OBS], fd[:, j, 3] if ph.fast else None) for j in range(H)], 1)
    prev = torch.cat([obs[wi, ki][:, None], sens[:, :-1]], 1)
    tstep = ki[:, None] + torch.arange(1, H + 1)[None]
    if oracle:
        r3 = csum[(tstep + 1).clamp(max=len(rain))] - csum[(tstep - 35).clamp_min(0)]
    else:
        r3 = (csum[ki + 1] - csum[(ki - 35).clamp_min(0)])[:, None].expand(-1, H)
    out['viol_wet_without_rain'] = float((((sens - prev) > 1e-4) & (r3 <= 0)).float().mean())
    out['viol_above_theta_s'] = float((sens > ph.ts + 1e-6).float().mean())
    # physics diagnostics at the probe on wet-up / falling steps (network one-step proposals)
    th1, q0, S = model(A[0, :-1], drv[0, :-1]); Rr, dthdt, div, Ss = residual(ph, A[0, :-1], th1, q0, S)
    d = obs[0, 1:] - obs[0, :-1]; okp = ok[0, 1:] & ok[0, :-1]
    sens = ph.sensor(A[0, :, I_OBS], drv[0, :, 3] if ph.fast else None)
    fastpart = sens - A[0, :, I_OBS]
    for nm, s in [('wetup', okp & (d > 0.002)), ('falling', okp & (d < -0.002))]:
        out[f'n_{nm}'] = int(s.sum())
        if s.any():
            orate = (d[s] / DT).abs().mean()
            out[f'ratio_{nm}'] = float(Rr[s, I_OBS].abs().mean() / dthdt[s, I_OBS].abs().mean().clamp_min(1e-20))
            # sensor-level: fraction of the observed probe rate carried by the Richards matrix / the fast store,
            # and the residual relative to the observed rate
            out[f'matrix_share_{nm}'] = float(dthdt[s, I_OBS].abs().mean() / orate)
            out[f'fast_share_{nm}'] = float(((fastpart[1:] - fastpart[:-1])[s] / DT).abs().mean() / orate)
            out[f'R_over_obs_{nm}'] = float(Rr[s, I_OBS].abs().mean() / orate)
    out['fast_part_max'] = float(fastpart.max())
    if keep:
        out.update(A=A[0], drv=drv[0], wi=wi, ki=ki, traj=traj, fd=fd)
    return out
