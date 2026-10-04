"""
Step 1: discrete-time Richards profile PINN stepped in real time (dt = 5 min).

State: theta(z) on a 30-cell column, dz = 4 cm, 0-1.2 m (cell centres 2,6,...,118 cm;
the 30 cm sensor is cell 7, the 1.0 m failure plane is the mean of cells 24/25).

Per step k -> k+1 the network proposes the whole profile theta^{k+1}. It is trained with
  (a) data loss on the 30 cm cell (multi-step open-loop forecasts, 30/60/120 min), and
  (b) the implicit (backward-Euler) Richards residual of its own proposal
      R_i = (theta_i^{k+1} - theta_i^k)/dt - [(q_{i-1/2} - q_{i+1/2})/dz + S_i]
      with Darcy fluxes q from van Genuchten K(theta^{k+1}), h(theta^{k+1}) (z down, q down +),
      surface flux = matrix share of effective rain, free drainage at 1.2 m,
      S_i = macropore (bypass) delivery of effective rain around depth z_b.
Effective rain passes an interception/initial-abstraction store of capacity c (learned),
which empties with time constant tau between storms.
Learned physical parameters: Ks, alpha, n, c, tau, bypass fraction beta, z_b, bypass width.
theta_r, theta_s fixed (Dev108: 0.05 / 0.55).
"""
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

DT = 300.0                       # s
DZ = 0.04                        # m
NZ = 30
Z = (torch.arange(NZ) + 0.5) * DZ  # cell centres, m
I_OBS = 7                        # z = 0.30 m
I_FAIL = (24, 25)                # z = 0.98 / 1.02 -> 1.0 m

def bounded(raw, lo, hi, log=False):
    s = torch.sigmoid(raw)
    if log:
        return torch.exp(math.log(lo) + s * (math.log(hi) - math.log(lo)))
    return lo + s * (hi - lo)

def inv_bounded(v, lo, hi, log=False):
    if log:
        s = (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))
    else:
        s = (v - lo) / (hi - lo)
    s = min(max(s, 1e-4), 1 - 1e-4)
    return math.log(s / (1 - s))

# (name, lo, hi, log, init)   a priori: Ks 0.0043 m/h, alpha 5.9 1/m, n 1.48
PARAM_SPEC = [
    ('Ks_mph', 1e-4, 1.0, True, 0.0043),
    ('alpha',  0.5, 20.0, True, 5.9),
    ('n',      1.1, 3.0, False, 1.48),
    ('c_mm',   0.0, 15.0, False, 5.0),    # interception / initial-abstraction capacity
    ('tau_h',  1.0, 72.0, True, 6.0),     # store emptying time constant
    ('beta',   0.0, 1.0, False, 0.5),     # bypass fraction of effective rain
    ('zb',     0.05, 1.0, False, 0.30),   # bypass delivery depth
    ('wb',     0.02, 0.5, True, 0.10),    # bypass delivery width
]

# convergent lateral inflow (linear reservoir fed by gamma x effective rain, released around z_l)
# and lateral drainage sink lam_out * K(theta)  [1/m * m/s = 1/s]
LATERAL_SPEC = [
    ('gamma',     0.0, 5.0, False, 0.5),
    ('tau_lat_h', 0.05, 24.0, True, 0.5),
    ('zl',        0.05, 1.0, False, 0.30),
    ('wl',        0.02, 0.5, True, 0.08),
    ('lam_out',   0.01, 1e4, True, 1.0),
]

# fast (macropore) domain at the probe: store F (mm) fed by beta x effective rain,
# drains out of the column (tau_fd) and exchanges into the matrix around z_b (tau_fx);
# the probe reads theta_m + (theta_s - theta_m) * tanh(kappa * F)
FAST_SPEC = [
    ('kappa',    1e-3, 1.0, True, 0.03),
    ('tau_fd_h', 0.05, 12.0, True, 0.5),
    ('tau_fx_h', 0.05, 48.0, True, 2.0),
]

# bounded fast domain: macropore water can add at most phi_f (macropore porosity, <= 5 %) to the probe reading
BOUNDED_SPEC = [('phi_f', 0.005, 0.05, False, 0.02)]

class Physics(nn.Module):
    def __init__(self, theta_r=0.05, theta_s=0.55, init=None, lateral=False, fast=False, bounded=False):
        super().__init__()
        self.tr, self.ts = theta_r, theta_s
        self.lateral, self.fast, self.bounded = lateral, fast, bounded
        spec = (PARAM_SPEC + (LATERAL_SPEC if lateral else []) + (FAST_SPEC if fast else [])
                + (BOUNDED_SPEC if bounded else []))
        init = init or {}
        self.raw = nn.ParameterDict({
            n: nn.Parameter(torch.tensor(inv_bounded(init.get(n, v0), lo, hi, lg)))
            for n, lo, hi, lg, v0 in spec})
        self.spec = {n: (lo, hi, lg) for n, lo, hi, lg, _ in spec}

    def p(self, name):
        lo, hi, lg = self.spec[name]
        return bounded(self.raw[name], lo, hi, lg)

    def values(self):
        return {n: float(self.p(n)) for n in self.spec}

    # van Genuchten-Mualem
    def Se(self, th):
        return ((th - self.tr) / (self.ts - self.tr)).clamp(1e-3, 0.999)

    def h(self, th):                       # pressure head, m (negative)
        n = self.p('n'); m = 1 - 1 / n; a = self.p('alpha')
        lg = (-(1 / m) * torch.log(self.Se(th))).clamp(max=25.0)   # overflow-safe
        return -(1 / a) * torch.expm1(lg).clamp_min(1e-12) ** (1 / n)

    def K(self, th):                       # m/s
        n = self.p('n'); m = 1 - 1 / n; se = self.Se(th)
        Ks = self.p('Ks_mph') / 3600.0
        return Ks * se.sqrt() * (1 - (1 - se ** (1 / m)) ** m) ** 2

    def bypass_weights(self):
        w = torch.exp(-0.5 * ((Z - self.p('zb')) / self.p('wb')) ** 2)
        return w / w.sum()

    def interception(self, store, rain_mm):
        """store (B,), rain_mm (B,) for one 5-min step -> (new store, effective rain mm)."""
        c = self.p('c_mm')
        s = store * torch.exp(-DT / 3600.0 / self.p('tau_h')) + rain_mm
        eff = F.softplus(s - c, beta=20.0)          # smooth overflow
        return s - eff, eff

    def gauss(self, zc, w):
        g = torch.exp(-0.5 * ((Z - zc) / w) ** 2)
        return g / g.sum()

    def drive(self, rain):
        """rain (T,) or (W,T) mm/5min -> drive (T,C) or (W,T,C):
        [effective rain, lateral inflow, (fast exchange, fast store F)] in mm per 5-min step."""
        if rain.dim() == 2:
            return self._drive(rain)
        return self._drive(rain[None])[0]

    def fast_decay(self):
        kd, kx = 1 / self.p('tau_fd_h'), 1 / self.p('tau_fx_h')
        return torch.exp(-DT / 3600.0 * (kd + kx)), kx / (kd + kx)

    def _drive(self, rain):
        W = rain.shape[0]
        store = torch.zeros(W); L = torch.zeros(W); Fs = torch.zeros(W); out = []
        if self.lateral:
            dec = torch.exp(-DT / 3600.0 / self.p('tau_lat_h')); gam = self.p('gamma')
        if self.fast:
            fdec, fx = self.fast_decay(); beta = self.p('beta')
        for k in range(rain.shape[1]):
            store, e = self.interception(store, rain[:, k])
            if self.lateral:
                rel = L * (1 - dec); L = L - rel + gam * e
            else:
                rel = torch.zeros(W)
            if self.fast:
                lost = Fs * (1 - fdec); exch = lost * fx; Fs = Fs - lost + beta * e
                out.append(torch.stack([e, rel, exch, Fs], -1))
            else:
                out.append(torch.stack([e, rel], -1))
        return torch.stack(out, 1)

    def sensor(self, th_obs_cell, F_mm):
        """probe reading from matrix theta at the probe cell and fast-domain store (identity if no fast domain)."""
        if not self.fast:
            return th_obs_cell
        if self.bounded:
            return (th_obs_cell + self.p('phi_f') * torch.tanh(self.p('kappa') * F_mm)).clamp(max=self.ts)
        return th_obs_cell + (self.ts - th_obs_cell) * torch.tanh(self.p('kappa') * F_mm)

    def sensor_inverse(self, obs, F_mm):
        """matrix theta at the probe consistent with an observation."""
        if not self.fast:
            return obs
        if self.bounded:
            return obs - self.p('phi_f') * torch.tanh(self.p('kappa') * F_mm)
        t = torch.tanh(self.p('kappa') * F_mm)
        return (obs - self.ts * t) / (1 - t).clamp_min(1e-3)

    def forcing(self, th, drive):
        """surface flux q0 (B,) and internal source S (B,NZ) in m/s, limited by free storage.
        drive (B,2) = [effective rain mm, lateral inflow mm] per 5-min step (legacy: (B,) rain only)."""
        if drive.dim() == 1:
            drive = torch.stack([drive, torch.zeros_like(drive)], 1)
        eff_mm, lat_mm = drive[:, 0], drive[:, 1]
        p = eff_mm / 1000.0 / DT                              # m/s
        room = 1 - self.Se(th)
        avail = room / (room + 0.02)                          # ~1 unless near saturation; excess -> runoff
        beta = self.p('beta')
        q0 = (1 - beta) * p * avail[:, 0]
        if self.fast:   # macropore water enters the matrix by delayed exchange, not instantly
            S = (drive[:, 2] / 1000.0 / DT)[:, None] * self.bypass_weights()[None, :] * avail / DZ
        else:
            S = beta * p[:, None] * self.bypass_weights()[None, :] * avail / DZ
        if self.lateral:
            S = S + (lat_mm / 1000.0 / DT)[:, None] * self.gauss(self.p('zl'), self.p('wl'))[None, :] * avail / DZ
        return q0, S

    def rhs(self, th, q0, S):
        """d theta/dt (B,NZ) plus flux divergence term, evaluated at profile th."""
        K = self.K(th); h = self.h(th)
        Kf = 0.5 * (K[:, 1:] + K[:, :-1])
        q_int = Kf * (1 - (h[:, 1:] - h[:, :-1]) / DZ)       # downward positive
        q = torch.cat([q0[:, None], q_int, K[:, -1:]], dim=1) # (B, NZ+1), free drainage at bottom
        div = (q[:, :-1] - q[:, 1:]) / DZ                      # -dq/dz
        f = div + S
        if self.lateral:
            f = f - self.p('lam_out') * K                      # lateral (slope-parallel) drainage
        return f, div, q

class Stepper(nn.Module):
    """1-D conv net proposing theta^{k+1} from theta^k and forcing; storage-bounded output."""
    def __init__(self, phys, width=32):
        super().__init__()
        self.phys = phys
        cin = 6
        self.net = nn.Sequential(
            nn.Conv1d(cin, width, 5, padding=2), nn.SiLU(),
            nn.Conv1d(width, width, 5, padding=2), nn.SiLU(),
            nn.Conv1d(width, width, 5, padding=2), nn.SiLU(),
            nn.Conv1d(width, 2, 1))
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.fill_(-6.0)

    def forward(self, th, drive):
        ph = self.phys
        q0, S = ph.forcing(th, drive)
        f_expl, _, _ = ph.rhs(th, q0, S)
        B = th.shape[0]
        se = ph.Se(th)
        x = torch.stack([
            se,
            torch.log10(-ph.h(th) + 1e-4) / 3,
            Z[None, :].expand(B, -1),
            (q0[:, None] * DT / DZ * 10).expand(-1, NZ),     # surface input, scaled
            S * DT * 10,
            torch.tanh(f_expl * DT * 20),                   # explicit physics hint (bounded)
        ], dim=1)
        a = self.net(x)
        tr, ts = ph.tr, ph.ts
        th_new = th + (ts - th) * torch.sigmoid(a[:, 0]) - (th - tr) * torch.sigmoid(a[:, 1])
        return th_new, q0, S

def residual(phys, th_old, th_new, q0, S):
    f, div, q = phys.rhs(th_new, q0, S)
    dthdt = (th_new - th_old) / DT
    return dthdt - f, dthdt, div, S
