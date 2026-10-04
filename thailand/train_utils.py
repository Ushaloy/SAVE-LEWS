"""Training / evaluation utilities for the step-1 Richards profile PINN."""
import numpy as np, pandas as pd, torch, time
from richards_pinn import Physics, Stepper, residual, DT, NZ, Z, I_OBS, I_FAIL
from data_pipeline import load_device, storm_catalogue

HORIZONS = (6, 12, 24)        # 30 / 60 / 120 min
R_SCALE = 1e-5                # s^-1, scale for the Richards residual (storm dtheta/dt ~ 1e-5..1e-4)

def make_window(g, storm, pre_h=48, post_h=24):
    w = g.loc[storm.start - pd.Timedelta(hours=pre_h): storm.end + pd.Timedelta(hours=post_h)]
    rain = w.rain.values.astype(np.float32)
    obs = w.theta.values.astype(np.float32)
    in_storm = ((w.index >= storm.start - pd.Timedelta(hours=1)) &
                (w.index <= storm.end + pd.Timedelta(hours=6)))
    return dict(t=w.index, rain=torch.tensor(np.nan_to_num(rain)), rain_missing=np.isnan(rain),
                obs=torch.tensor(obs), obs_ok=torch.tensor(~np.isnan(obs)),
                in_storm=torch.tensor(in_storm), name=str(storm.start.date()),
                rain_mm=float(storm.rain_mm))

def nudge_gain(L=0.08):
    return torch.exp(-0.5 * ((Z - Z[I_OBS]) / L) ** 2)

def eff_rain(phys, rain):
    """drive series (T,2): [effective rain mm, lateral inflow mm]."""
    return phys.drive(rain)

@torch.no_grad()
def analysis(model, win, eff):
    """Real-time assimilation pass: step the profile, nudge the 30 cm neighbourhood to obs."""
    obs, ok = win['obs'], win['obs_ok']
    first = obs[ok][0]
    th = torch.full((1, NZ), float(first))
    G = nudge_gain()
    states = [th]
    for k in range(len(obs) - 1):
        th, _, _ = model(th, eff[k:k + 1])
        if ok[k + 1]:
            th = th + G * (obs[k + 1] - th[:, I_OBS])
            th = th.clamp(model.phys.tr + 1e-3, model.phys.ts - 1e-3)
        states.append(th)
    return torch.cat(states)          # (T, NZ): analysis state at each time

def forecast(model, th0, eff_seq, H):
    """open-loop rollouts. th0 (B,NZ); eff_seq (B,H) effective rain. returns list of states."""
    th = th0; traj = []; res = []
    for j in range(H):
        th_new, q0, S = model(th, eff_seq[:, j])
        R, *_ = residual(model.phys, th, th_new, q0, S)
        res.append(R); traj.append(th_new); th = th_new
    return torch.stack(traj, 1), torch.stack(res, 1)

def origins_and_targets(win, H=24, stride=1):
    obs, ok = win['obs'], win['obs_ok']
    T = len(obs)
    idx = [k for k in range(0, T - H, stride) if ok[k]]
    return torch.tensor(idx)

def future_eff(eff, idx, H, oracle=True):
    if oracle:
        return torch.stack([eff[i:i + H] for i in idx.tolist()])
    return torch.zeros(len(idx), H, eff.shape[-1])

def train(win, lam=1.0, r_scale=R_SCALE, ramp=0.4, storm_weight_residual=True, event_weight=0.0, lateral=False, iters=1500, lr=3e-3, seed=0, refresh=25, batch=64, log_every=250,
          init=None, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    phys = Physics(init=init, lateral=lateral); model = Stepper(phys)
    opt = torch.optim.Adam([{'params': model.net.parameters(), 'lr': lr},
                            {'params': phys.parameters(), 'lr': lr * 3}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    H = max(HORIZONS)
    idx_all = origins_and_targets(win, H)
    obs, ok = win['obs'], win['obs_ok']
    # persistence MSE per horizon (normaliser) on training origins
    pers = []
    for h in HORIZONS:
        m = ok[idx_all + h]
        pers.append(((obs[idx_all + h] - obs[idx_all])[m] ** 2).mean())
    pers = torch.stack(pers).clamp_min(1e-8)
    # storm-heavy sampling: most of the information is in the storm
    wts = torch.where(win['in_storm'][idx_all], 4.0, 1.0)
    hist = []
    t0 = time.time()
    for it in range(iters):
        eff = eff_rain(phys, win['rain'])
        if it % refresh == 0:
            A = analysis(model, win, eff.detach())
        b = idx_all[torch.multinomial(wts, batch, replacement=True)]
        traj, R = forecast(model, A[b], future_eff(eff, b, H), H)
        l_data = 0
        for j, h in enumerate(HORIZONS):
            m = ok[b + h]
            pred = traj[:, h - 1, I_OBS] - A[b, I_OBS]
            targ = obs[b + h] - obs[b]
            l_data = l_data + ((pred - targ)[m] ** 2).mean() / pers[j]
        l_data = l_data / len(HORIZONS)
        # residual on analysis states (one-step) + along forecasts
        wk = torch.where(win['in_storm'][:-1], 4.0 if storm_weight_residual else 1.0, 1.0)
        if event_weight > 0:   # emphasise steps where the sensor actually moves
            dob = torch.nan_to_num((win['obs'][1:] - win['obs'][:-1]).abs(), 0.0)
            wk = wk * (1 + event_weight * (dob / 0.002).clamp(max=5.0))
        k = torch.multinomial(wk, batch, replacement=True)
        th1, q0, S = model(A[k], eff[k])
        R1, *_ = residual(phys, A[k], th1, q0, S)
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
            print(f"it {it:5d} data {float(l_data):.4f} phys {float(l_phys):.4f} | "
                  f"Ks {pv['Ks_mph']:.4g} a {pv['alpha']:.3g} n {pv['n']:.3g} c {pv['c_mm']:.2f} "
                  f"beta {pv['beta']:.2f} zb {pv['zb']:.2f} | {time.time()-t0:.0f}s", flush=True)
    return model, np.array(hist)

@torch.no_grad()
def evaluate(model, win, oracle=True):
    phys = model.phys
    H = max(HORIZONS)
    eff = eff_rain(phys, win['rain'])
    A = analysis(model, win, eff)
    idx = origins_and_targets(win, H)
    traj, R = forecast(model, A[idx], future_eff(eff, idx, H, oracle), H)
    obs, ok, st = win['obs'], win['obs_ok'], win['in_storm']
    out = {}
    for h in HORIZONS:
        m = ok[idx + h]
        pred = traj[:, h - 1, I_OBS] - A[idx, I_OBS]
        targ = obs[idx + h] - obs[idx]
        for tag, sel in [('all', m), ('storm', m & st[idx])]:
            mse = ((pred - targ)[sel] ** 2).mean(); mp = (targ[sel] ** 2).mean()
            out[f'skill_{tag}_{h*5}'] = float(1 - mse / mp)
    # constraint violations on forecasts
    dth = traj[:, :, I_OBS] - torch.cat([A[idx, None, I_OBS], traj[:, :-1, I_OBS]], 1)
    # rain (as fed to the model) in the preceding 3 h at each forecast step
    rain_in = win['rain'].clone()
    if not oracle:
        pass  # future rain is zero in past-only mode; history still observed
    csum = torch.cat([torch.zeros(1), torch.cumsum(rain_in, 0)])
    tstep = idx[:, None] + torch.arange(1, H + 1)[None, :]
    fut = torch.where(torch.tensor(oracle), torch.ones(()), (tstep <= idx[:, None]).float())
    r3h = csum[tstep + 1] - csum[(tstep - 35).clamp_min(0)]
    dry = r3h <= 0 if oracle else ((csum[idx + 1] - csum[(idx - 35).clamp_min(0)])[:, None] <= 0)
    out['viol_wet_without_rain'] = float(((dth > 1e-4) & dry).float().mean())
    out['viol_above_theta_s'] = float((traj > phys.ts + 1e-6).float().mean())
    # physics magnitudes along the analysis trajectory (network one-step proposals)
    th1, q0, S = model(A[:-1], eff[:-1])
    Rr, dthdt, div, Ssrc = residual(phys, A[:-1], th1, q0, S)
    stm = st[:-1]
    for tag, sel in [('storm', stm), ('quiet', ~stm)]:
        for nm, v in [('dthdt', dthdt), ('div', div), ('S', Ssrc), ('R', Rr)]:
            a = v[sel].abs()
            out[f'{nm}_{tag}_med'] = float(a.median())
            out[f'{nm}_{tag}_p95'] = float(torch.quantile(a.flatten()[:200000], 0.95))
        out[f'R_over_dthdt_{tag}'] = float(Rr[sel].abs().mean() / dthdt[sel].abs().mean().clamp_min(1e-20))
    out['eff'] = eff; out['A'] = A; out['idx'] = idx; out['traj'] = traj
    return out
