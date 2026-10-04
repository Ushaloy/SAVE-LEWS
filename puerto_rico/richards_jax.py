"""1-D Richards solver (h-form, modified Picard, mass-conservative, Celia et al. 1990) in JAX.
Top node: Dirichlet from the observed shallow probe. Bottom: free drainage. z positive downward (cm)."""
import jax, jax.numpy as jnp, numpy as np
jax.config.update('jax_enable_x64', True)

def unpack(p):
    """unconstrained vector -> physical params dict (cm, s units)."""
    s = jax.nn.sigmoid
    return dict(Ks=jnp.exp(jnp.log(1e-6) + s(p[0]) * (jnp.log(1e-1) - jnp.log(1e-6))),     # cm/s
                alpha=jnp.exp(jnp.log(1e-3) + s(p[1]) * (jnp.log(0.5) - jnp.log(1e-3))),     # 1/cm
                n=1.05 + s(p[2]) * 1.95,
                ts=0.30 + s(p[3]) * 0.30,
                tr=0.0 + s(p[4]) * 0.08)

def pack(Ks, alpha, n, ts, tr):
    lg = lambda v: np.log(v / (1 - v))
    return np.array([lg((np.log(Ks) - np.log(1e-6)) / (np.log(1e-1) - np.log(1e-6))),
                     lg((np.log(alpha) - np.log(1e-3)) / (np.log(0.5) - np.log(1e-3))),
                     lg((n - 1.05) / 1.95), lg((ts - 0.30) / 0.30), lg(np.clip(tr / 0.08, 1e-3, 1 - 1e-3))])

def _hn(h):
    return jnp.minimum(h, -1e-6)          # safe negative head: keeps the unselected where-branch finite for gradients

def theta_of_h(h, P):
    m = 1 - 1 / P['n']
    se = jnp.where(h < 0, (1 + (P['alpha'] * (-_hn(h))) ** P['n']) ** (-m), 1.0)
    return P['tr'] + (P['ts'] - P['tr']) * se

def h_of_theta(th, P):
    m = 1 - 1 / P['n']
    se = jnp.clip((th - P['tr']) / (P['ts'] - P['tr']), 1e-4, 0.9995)
    return -(1 / P['alpha']) * (se ** (-1 / m) - 1) ** (1 / P['n'])

def K_of_h(h, P):
    m = 1 - 1 / P['n']
    se = jnp.where(h < 0, (1 + (P['alpha'] * (-_hn(h))) ** P['n']) ** (-m), 1.0)
    se = jnp.minimum(se, 1 - 1e-9)
    return P['Ks'] * jnp.sqrt(se) * (1 - (1 - se ** (1 / m)) ** m) ** 2

def C_of_h(h, P):
    m = 1 - 1 / P['n']; a, n = P['alpha'], P['n']
    ah = a * (-_hn(h))
    c = (P['ts'] - P['tr']) * a * n * m * ah ** (n - 1) * (1 + ah ** n) ** (-m - 1)
    return jnp.where(h < 0, c, 1e-7) + 1e-9

def make_step(dz, dt, n_sub=3, n_iter=8):
    """returns step(h, h_top_new, P) -> h_new; nodes 0..N, node 0 Dirichlet."""
    dts = dt / n_sub
    def picard(h_old, h_top, P):
        th_old = theta_of_h(h_old, P)
        def body(h, _):
            h = h.at[0].set(h_top)
            K = K_of_h(h, P); C = C_of_h(h, P); th = theta_of_h(h, P)
            Kf = 0.5 * (K[1:] + K[:-1])                          # interfaces i+1/2
            N = h.shape[0]
            # unknowns nodes 1..N-1; node N-1 bottom with free drainage out
            # interior node i: C/dt (h_i - h^m_i) + (th^m - th_old)/dt = [Kf_i ((h_{i+1}-h_i)/dz - 1) - Kf_{i-1} ((h_i-h_{i-1})/dz - 1)] / dz
            Kl = Kf; Kr = jnp.concatenate([Kf[1:], jnp.zeros(1)])
            idx = jnp.arange(1, N)
            Ci = C[idx]
            a = -Kl / dz ** 2                                     # coef h_{i-1}
            c = jnp.where(idx < N - 1, -Kr / dz ** 2, 0.0)         # coef h_{i+1}
            b = Ci / dts - a - c
            rhs = Ci / dts * h[idx] - (th[idx] - th_old[idx]) / dts \
                  + (-(Kr) + Kl) / dz                             # gravity terms: -Kr*(-1)/dz ... see below
            # gravity: d/dz[K(dh/dz - 1)] contributes (-Kr + Kl)/dz ; bottom free drainage: flux out = K_N (unit gradient)
            rhs = rhs.at[-1].add(-K[N - 1] / dz + Kr[-1] / dz)     # replace missing right interface by free drainage
            rhs = rhs.at[0].add(-a[0] * h_top)                     # Dirichlet contribution
            a0 = a.at[0].set(0.0)
            sol = jax.lax.linalg.tridiagonal_solve(a0, b, c, rhs[:, None])[:, 0]
            h_new = jnp.concatenate([h_top[None], sol])
            h_new = jnp.clip(h_new, -1e9, 50.0)
            return h_new, None
        h, _ = jax.lax.scan(body, h_old, None, length=n_iter)
        return h
    def step(h, h_top, P):
        for _ in range(n_sub):
            h = picard(h, h_top, P)
        return h
    return step

def simulate(p, th_top_series, th0_profile, z_nodes, obs_idx, dt=900.0):
    """th_top_series (T,), th0_profile (N+1,) on nodes; returns theta at obs_idx nodes (T, len(obs_idx))."""
    P = unpack(p)
    dz = float(z_nodes[1] - z_nodes[0])
    step = make_step(dz, dt)
    h0 = h_of_theta(th0_profile, P)
    @jax.checkpoint
    def f(h, th_top):
        h = step(h, h_of_theta(th_top, P), P)
        return h, theta_of_h(h, P)[obs_idx]
    _, out = jax.lax.scan(f, h0, th_top_series)
    return out
