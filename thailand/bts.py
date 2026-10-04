"""Bayesian two-store (B-TS) probe model: interception store -> fast (preferential-flow) store -> probe.
Posterior over physical/conceptual parameters (c, tau_c, a, tau_F) + a Bayesian discrepancy last layer,
Student-t likelihood with storm/quiet noise per horizon. VI (full-rank Gaussian) and NUTS."""
import numpy as np, jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoMultivariateNormal
jax.config.update('jax_enable_x64', False)

DT_H = 5 / 60.0
HOR = np.array([6, 12, 24])            # 30 / 60 / 120 min
NU = 4.0
def set_nu(v):
    global NU
    NU = v
PRIOR = dict(c=(np.log(5.0), 0.6), tau_c=(np.log(6.0), 1.0), a=(np.log(0.03), 1.0), tau_F=(np.log(8.0), 1.0))

# ------------------------------------------------------------------ data
def pack(wins, stride=1, storm_only=False, storm_stride=None):
    """windows (train_loso.make_window dicts) -> padded arrays + origin table."""
    W = len(wins); T = max(len(w['obs']) for w in wins)
    rain = np.zeros((W, T), np.float32); obs = np.zeros((W, T), np.float32)
    ok = np.zeros((W, T), bool); st = np.zeros((W, T), bool); Tw = np.zeros(W, int)
    for i, w in enumerate(wins):
        n = len(w['obs']); Tw[i] = n
        rain[i, :n] = w['rain'].numpy(); obs[i, :n] = w['obs'].numpy()
        ok[i, :n] = w['obs_ok'].numpy(); st[i, :n] = w['in_storm'].numpy()
    wi, ki = [], []
    for i in range(W):
        ss = storm_stride or stride
        for k in range(0, Tw[i] - HOR.max()):
            step = ss if st[i, k] else stride
            if k % step == 0 and ok[i, k] and (not storm_only or st[i, k]):
                wi.append(i); ki.append(k)
    wi, ki = np.array(wi), np.array(ki)
    y = np.stack([obs[wi, ki + h] - obs[wi, ki] for h in HOR], 1)
    ym = np.stack([ok[wi, ki + h] for h in HOR], 1)
    csum = np.concatenate([np.zeros((W, 1)), np.cumsum(rain, 1)], 1)
    fut = np.stack([csum[wi, ki + h + 1] - csum[wi, ki + 1] for h in HOR], 1)        # rain in (k, k+h]
    past1h = csum[wi, ki + 1] - csum[wi, np.maximum(ki - 11, 0)]
    d15 = obs[wi, ki] - obs[wi, np.maximum(ki - 3, 0)]
    return dict(rain=rain, obs=obs, wi=wi, ki=ki, y=np.nan_to_num(y), ym=ym, storm=st[wi, ki],
                obs0=obs[wi, ki], fut=fut, past1h=past1h, d15=d15, W=W, T=T)

def features(D, ts, tr, oracle=True):
    """discrepancy features (N, 3, P): [1, Se0, future rain/10mm, past-1h rain/10mm, 15-min trend x10]."""
    se0 = (D['obs0'] - tr) / (ts - tr)
    fut = D['fut'] if oracle else np.zeros_like(D['fut'])
    P = np.stack([np.ones_like(fut), np.repeat(se0[:, None], 3, 1), fut / 10.0,
                  np.repeat(D['past1h'][:, None] / 10.0, 3, 1), np.repeat(D['d15'][:, None] * 10.0, 3, 1)], -1)
    return P.astype(np.float32)

# ------------------------------------------------------------------ physics
def stores(rain, c, tau_c, tau_F):
    """rain (W,T) mm/5min -> fast store F (W,T) after each step."""
    dc = jnp.exp(-DT_H / tau_c); dF = jnp.exp(-DT_H / tau_F)
    def step(carry, r):
        s, F = carry
        s = s * dc + r
        eff = jax.nn.softplus(20.0 * (s - c)) / 20.0
        s = s - eff
        F = F * dF + eff
        return (s, F), F
    W = rain.shape[0]
    _, Fs = jax.lax.scan(step, (jnp.zeros(W), jnp.zeros(W)), rain.T)
    return Fs.T

def delta_ts(F, D, a, tau_F, ts, tr, oracle=True):
    """two-store forecast of probe change (N,3), closed form after assimilating obs at the origin."""
    F0 = F[D['wi'], D['ki']]
    Fh = jnp.stack([F[D['wi'], D['ki'] + h] if oracle else F0 * jnp.exp(-h * DT_H / tau_F) for h in HOR], 1)
    t0 = jnp.minimum(jnp.tanh(a * F0), 0.995); th = jnp.tanh(a * Fh)
    obs0 = D['obs0'][:, None]
    d = (ts - obs0) * (th - t0[:, None]) / (1 - t0[:, None])
    return jnp.maximum(d, tr - obs0)                           # reading cannot fall below theta_r

# ------------------------------------------------------------------ model
def model(D, Phi, ts, tr, oracle=True, discrepancy=True, y=None):
    c = numpyro.sample('c', dist.LogNormal(*PRIOR['c']))
    tau_c = numpyro.sample('tau_c', dist.LogNormal(*PRIOR['tau_c']))
    a = numpyro.sample('a', dist.LogNormal(*PRIOR['a']))
    tau_F = numpyro.sample('tau_F', dist.LogNormal(*PRIOR['tau_F']))
    sig = numpyro.sample('sigma', dist.HalfNormal(0.02).expand([2, 3]).to_event(2))   # [quiet/storm, horizon]
    F = stores(jnp.asarray(D['rain']), c, tau_c, tau_F)
    mu = delta_ts(F, D, a, tau_F, ts, tr, oracle)
    if discrepancy:
        s_w = numpyro.sample('s_w', dist.HalfNormal(0.005))
        w = numpyro.sample('w', dist.Normal(0, 1).expand([3, Phi.shape[-1]]).to_event(2))
        mu = mu + jnp.einsum('nhp,hp->nh', jnp.asarray(Phi), w * s_w)
    numpyro.deterministic('mu', mu)
    scale = sig[jnp.asarray(D['storm']).astype(int)]                                  # (N,3)
    with numpyro.handlers.mask(mask=jnp.asarray(D['ym'])):
        numpyro.sample('obs', dist.StudentT(NU, mu, scale), obs=None if y is None else jnp.asarray(y))

SITES = ['c', 'tau_c', 'a', 'tau_F', 'sigma', 's_w', 'w']

def fit_vi(D, Phi, ts, tr, discrepancy=True, steps=6000, seed=0):
    guide = AutoMultivariateNormal(lambda *a, **k: model(*a, **k))
    svi = SVI(model, guide, numpyro.optim.ClippedAdam(step_size=0.01, clip_norm=10.0), Trace_ELBO())
    res = svi.run(jax.random.PRNGKey(seed), steps, D, Phi, ts, tr, True, discrepancy, D['y'], progress_bar=False)
    post = guide.sample_posterior(jax.random.PRNGKey(seed + 1), res.params, sample_shape=(1000,))
    return {k: np.asarray(v) for k, v in post.items() if k in SITES}, np.asarray(res.losses)

def fit_nuts(D, Phi, ts, tr, discrepancy=True, warmup=600, samples=600, chains=2, seed=0):
    mcmc = MCMC(NUTS(model, target_accept_prob=0.9, max_tree_depth=8), num_warmup=warmup, num_samples=samples,
                num_chains=chains, chain_method='sequential', progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), D, Phi, ts, tr, True, discrepancy, D['y'])
    s = mcmc.get_samples(group_by_chain=True)
    post = {k: np.asarray(v).reshape(-1, *np.asarray(v).shape[2:]) for k, v in s.items() if k in SITES}
    from numpyro.diagnostics import summary
    summ = summary({k: np.asarray(v) for k, v in s.items() if k in ['c', 'tau_c', 'a', 'tau_F']})
    diag = {k: dict(r_hat=float(v['r_hat']), n_eff=float(v['n_eff'])) for k, v in summ.items()}
    extra = mcmc.get_extra_fields()
    diag['divergences'] = int(np.asarray(extra['diverging']).sum())
    return post, diag

# ------------------------------------------------------------------ predictive
def predictive(post, D, ts, tr, oracle=True, discrepancy=True, n=400, seed=0):
    """posterior predictive draws (n, N, 3) including Student-t noise; plus noise-free mean draws."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(post['c']), n, replace=False)
    Phi = jnp.asarray(features(D, ts, tr, oracle))
    storm = jnp.asarray(D['storm']).astype(int)
    rain = jnp.asarray(D['rain'])
    def one(c, tau_c, a, tau_F, sig, s_w, w, key):
        F = stores(rain, c, tau_c, tau_F)
        mu = delta_ts(F, D, a, tau_F, ts, tr, oracle)
        if discrepancy:
            mu = mu + jnp.einsum('nhp,hp->nh', Phi, w * s_w)
        eps = jax.random.t(key, NU, mu.shape) * sig[storm]
        return mu, mu + eps
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    sw = post['s_w'][idx] if discrepancy else np.zeros(n)
    ww = post['w'][idx] if discrepancy else np.zeros((n, 3, 5))
    mu, yy = jax.vmap(one)(post['c'][idx], post['tau_c'][idx], post['a'][idx], post['tau_F'][idx],
                           post['sigma'][idx], sw, ww, keys)
    return np.asarray(mu), np.asarray(yy)

def interval_metrics(yy, D, alpha=0.1, scale=None):
    """storm-time and all-time coverage, width, Winkler interval score, point skill of predictive median."""
    lo = np.quantile(yy, alpha / 2, 0); hi = np.quantile(yy, 1 - alpha / 2, 0); med = np.median(yy, 0)
    if scale is not None:                     # conformal rescaling of the half-width around the median
        lo = med - (med - lo) * scale; hi = med + (hi - med) * scale
    y, m, st = D['y'], D['ym'], D['storm']
    out = {}
    for j, h in enumerate(HOR * 5):
        for tag, sel in [('storm', m[:, j] & st), ('all', m[:, j])]:
            yt, l, u, p = y[sel, j], lo[sel, j], hi[sel, j], med[sel, j]
            cov = np.mean((yt >= l) & (yt <= u)); wid = np.mean(u - l)
            ws = np.mean((u - l) + 2 / alpha * (l - yt) * (yt < l) + 2 / alpha * (yt - u) * (yt > u))
            out[f'{tag}_{h}'] = dict(coverage=float(cov), width=float(wid), winkler=float(ws), n=int(sel.sum()),
                                     mse=float(np.mean((p - yt) ** 2)), msep=float(np.mean(yt ** 2)))
    return out, dict(lo=lo, hi=hi, med=med)
