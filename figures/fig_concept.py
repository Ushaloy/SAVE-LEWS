"""Fig. 1 (revised manuscript): when the physics matches, and mismatches, what the sensor observes."""
from common import out
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch

INK, MUT = '#1f1e1c', '#5c5b56'
SOIL, SOIL2, ROCK = '#efe4d2', '#e3d3ba', '#c9c3b8'
BLUE, ORANGE, PURPLE, GREEN = '#2a78d6', '#eb6834', '#4a3aa7', '#1baf7a'
fig = plt.figure(figsize=(7.4, 7.6))


def column(ax, title, sub):
    ax.set_xlim(0, 10); ax.set_ylim(-11.2, 2.4); ax.axis('off')
    ax.add_patch(Rectangle((0.6, -8.0), 5.2, 8.0, fc=SOIL, ec='none'))
    ax.add_patch(Rectangle((0.6, -8.0), 5.2, 1.3, fc=SOIL2, ec='none'))
    ax.add_patch(Rectangle((0.6, -9.2), 5.2, 1.2, fc=ROCK, ec='none', hatch='///', lw=0))
    ax.plot([0.6, 5.8], [0, 0], color='#6b5a3e', lw=1.2)
    ax.text(0.0, 2.2, title, fontsize=8.6, weight='bold', color=INK, va='top')
    ax.text(0.0, 1.35, sub, fontsize=7.0, color=MUT, va='top')
    ax.text(5.95, -7.35, 'hypothesised\nfailure-plane depth', fontsize=5.8, color=MUT, va='center')
    ax.plot([0.6, 5.8], [-6.7, -6.7], color='#b8452a', lw=0.9, ls=(0, (4, 2)))
    ax.text(5.95, -8.6, 'saprolite /\nbedrock', fontsize=5.8, color=MUT, va='center')
    for x in np.linspace(1.2, 5.2, 6):
        ax.annotate('', xy=(x, 0.15), xytext=(x - 0.25, 1.0), arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=0.8))


def probe(ax, y, lab, c=INK, below=False):
    ax.add_patch(Rectangle((2.9, y - 0.12), 1.2, 0.24, fc='#3d3d3a', ec='none'))
    if below: ax.text(3.5, y - 0.3, lab, fontsize=6.2, color=c, va='top', ha='center', bbox=dict(fc=SOIL, ec='none', pad=0.5))
    else: ax.text(4.25, y, lab, fontsize=6.2, color=c, va='center')


def trace(ax, x0, y0, w, h, curves, lab):
    ax.add_patch(Rectangle((x0, y0), w, h, fc='white', ec='#c3c2b7', lw=0.6))
    t = np.linspace(0, 1, 200)
    for f, c, ls in curves:
        ax.plot(x0 + t * w, y0 + 0.08 * h + 0.84 * h * f(t), color=c, lw=1.1, ls=ls)
    ax.text(x0 + w / 2, y0 - 0.25, lab, fontsize=5.8, color=MUT, ha='center', va='top')


sig = lambda t, t0, k: 1 / (1 + np.exp(-(t - t0) * k))

# ---------------------------------------------------------------- panel a: matrix-connected observation
ax = fig.add_axes([0.02, 0.44, 0.47, 0.54])
column(ax, 'a  Matrix-connected observation', 'vertical wetting front reaches each probe in turn')
ax.add_patch(Rectangle((0.6, -2.9), 5.2, 2.9, fc='#d4e5f8', ec='none'))
ax.plot([0.6, 5.8], [-2.9, -2.9], color=BLUE, lw=0.8, ls=':')
ax.text(5.7, -2.75, 'wetting front', fontsize=5.8, color=BLUE, ha='right', va='bottom')
ax.annotate('', xy=(1.5, -5.6), xytext=(1.5, -0.6), arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=1.6))
ax.text(1.3, -4.4, 'matrix\nflow', fontsize=6.2, color=BLUE, ha='right', va='center')
probe(ax, -1.6, '≈25 cm'); probe(ax, -3.9, '≈55 cm'); probe(ax, -6.3, 'probe at failure-plane depth')
trace(ax, 6.6, -3.2, 3.2, 2.2, [(lambda t: sig(t, 0.25, 30), '#3d3d3a', '-'), (lambda t: 0.8 * sig(t, 0.45, 18), '#8c8a83', '-'),
                                (lambda t: 0.6 * sig(t, 0.65, 12), '#b8452a', '-')], 'probe response: delayed,\nsmoothed with depth')
ax.text(0.0, -9.8, 'y(t) ≈ θ$_m$(z$_s$, t): the observation operator is\nclose to the identity → Richards equation testable',
        fontsize=6.6, color=PURPLE, va='top', weight='bold')

# ---------------------------------------------------------------- panel b: fast-pathway observation
ax = fig.add_axes([0.51, 0.44, 0.47, 0.54])
column(ax, 'b  Fast-pathway observation', 'water reaches the probe around the matrix')
for x, y1 in [(2.1, -3.0), (3.4, -2.2), (4.6, -3.6)]:
    xs = x + 0.12 * np.sin(np.linspace(0, 9, 40)); ax.plot(xs, np.linspace(0, y1, 40), color=ORANGE, lw=1.4)
ax.add_patch(Rectangle((0.6, -0.7), 5.2, 0.7, fc='#d4e5f8', ec='none'))
ax.annotate('', xy=(5.6, -6.3), xytext=(0.9, -5.7), arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.4, ls='--'))
ax.text(3.2, -5.45, 'lateral / perched flow above contact', fontsize=5.8, color=ORANGE, ha='center')
probe(ax, -2.4, 'single probe (30 cm)', below=True)
ax.text(1.9, -1.4, 'macropores,\nroot channels', fontsize=5.8, color=ORANGE, ha='right', va='center')
trace(ax, 6.6, -3.2, 3.2, 2.2, [(lambda t: sig(t, 0.12, 60), '#3d3d3a', '-'), (lambda t: 0.9 * sig(t, 0.55, 10), '#8c8a83', '--')],
      'observed: 5–30 min rise;\ndashed: matrix-only response')
ax.text(0.0, -9.8, 'y(t) = $\\mathcal{H}$[θ$_m$, F]: the probe reads the fast store F →\nmatrix equation incomplete; physics can be inert',
        fontsize=6.6, color='#b8452a', va='top', weight='bold')

# ---------------------------------------------------------------- panel c: SAVE and the physics ladder, and the warning chain
ax = fig.add_axes([0.02, 0.0, 0.96, 0.42]); ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis('off')
ax.text(0, 43.5, 'c  SAVE selects the level of physics the sensor configuration can support', fontsize=8.6, weight='bold', color=INK, va='top')
steps = [('I  Screening', 'lag, storage bound,\ncomparability', '#8c8a83'), ('II  Attribution', 'is the physics\nactive?', PURPLE),
         ('III  Validation', 'storm-blocked,\nvs strong baseline', GREEN), ('IV  Hydrological\nalerts', 'POD, FAR, timing,\ncalibrated bands', BLUE)]
for i, (a, b, c) in enumerate(steps):
    x = 1 + i * 25
    ax.add_patch(FancyBboxPatch((x, 28), 21, 10.5, boxstyle='round,pad=0.3,rounding_size=1.2', fc='white', ec=c, lw=1.2))
    ax.text(x + 10.5, 37.2, a, ha='center', va='top', fontsize=7.4, weight='bold', color=INK, linespacing=1.1)
    ax.text(x + 10.5, 33.6 if '\n' not in a else 32.0, b, ha='center', va='top', fontsize=6.4, color=MUT, linespacing=1.2)
    if i < 3:
        ax.annotate('', xy=(x + 24.3, 33.2), xytext=(x + 21.7, 33.2), arrowprops=dict(arrowstyle='-|>', color='#3d3d3a', lw=1))
lad = [('constraint-level', 'bounds, monotone rain response,\nno wetting without rain', '#9fc3ec', 'single probe (a or b)'),
       ('process-level', 'conceptual fast/slow stores\nof the observed pathway', '#6aa3e2', 'pathway identified (b)'),
       ('equation-level', 'Richards PINN or\ncalibrated solver', PURPLE, 'passes I and II (a)')]
ax.text(1, 25.5, 'physics ladder (each rung benchmarked against persistence and an unconstrained MLP)', fontsize=6.6, color=MUT, va='top')
for i, (a, b, c, need) in enumerate(lad):
    x = 1 + i * 33
    ax.add_patch(Rectangle((x, 14.0), 31, 9.0, fc=c, ec='none', alpha=0.25 if i < 2 else 0.18))
    ax.text(x + 1, 22.2, a, fontsize=7.2, weight='bold', color=INK, va='top')
    ax.text(x + 1, 19.4, b, fontsize=6.2, color=MUT, va='top', linespacing=1.2)
    ax.text(x + 30, 14.4, 'needs: ' + need, fontsize=5.8, color=c if i == 2 else '#2a5ea8', ha='right', va='bottom', style='italic')
    if i < 2:
        ax.annotate('', xy=(x + 33.4, 18.2), xytext=(x + 31.3, 18.2), arrowprops=dict(arrowstyle='-|>', color='#3d3d3a', lw=0.9))
chain = [('rainfall', True), ('infiltration /\nfast flow', True), ('subsurface\nwetting', True), ('pore pressure', False),
         ('effective-stress\nreduction', False), ('slope failure', False)]
ax.text(1, 11.6, 'landslide chain', fontsize=6.6, color=MUT, va='top')
for i, (s, ev) in enumerate(chain):
    x = 1 + i * 16.6
    ax.add_patch(FancyBboxPatch((x, 3.0), 14.2, 6.0, boxstyle='round,pad=0.2,rounding_size=1', fc='#eef7f2' if ev else '#f4f3ef',
                                ec=GREEN if ev else '#b3b1a8', lw=1, ls='-' if ev else '--'))
    ax.text(x + 7.1, 6.0, s, ha='center', va='center', fontsize=6.3, color=INK if ev else MUT, linespacing=1.1)
    if i < 5:
        ax.annotate('', xy=(x + 16.4, 6.0), xytext=(x + 14.6, 6.0), arrowprops=dict(arrowstyle='-|>', color='#3d3d3a', lw=0.8))
ax.text(25.9, 0.2, 'evaluated here (hydrological precursor states)', fontsize=5.9, color=GREEN, ha='center', va='bottom')
ax.text(75.7, 0.2, 'not evaluated (needs shear strength and slope stability)', fontsize=5.9, color=MUT, ha='center', va='bottom')
fig.savefig(out('Fig01_concept.png'), dpi=300, bbox_inches='tight', pad_inches=0.05)
print('ok')
