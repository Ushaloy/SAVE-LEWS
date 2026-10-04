from common import out, TH, PR, DATA_TH
import os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
fig, ax = plt.subplots(figsize=(7.2, 9.0)); ax.set_xlim(0, 100); ax.set_ylim(0, 131); ax.axis('off')
INK, MUT = '#1f1e1c', '#5c5b56'
def box(x, y, w, h, title, body='', fc='#f4f3ef', ec='#8c8a83', fs=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.6', fc=fc, ec=ec, lw=1))
    if body:
        ax.text(x + w / 2, y + h - 2.0, title, ha='center', va='top', fontsize=fs, weight='bold', color=INK)
        ax.text(x + w / 2, y + h - 6.0, body, ha='center', va='top', fontsize=7.4, color=MUT, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center', fontsize=fs, weight='bold', color=INK)
def arrow(x0, y0, x1, y1, lab=None, lx=0, ly=0, c='#3d3d3a'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=c, lw=1.1, shrinkA=0, shrinkB=0))
    if lab: ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + ly, lab, fontsize=7.6, color=c, ha='center', va='center', weight='bold')
def note(x, y, s, ha='left'):
    ax.text(x, y, s, fontsize=6.8, color='#4a3aa7', ha=ha, va='top', style='italic', linespacing=1.3)

box(15, 120, 70, 9, 'IoT sensor configuration', 'shallow probe(s) + rain gauge (+ deep probe or piezometer)', fc='#eef4fc', ec='#2a78d6')
ax.add_patch(FancyBboxPatch((2, 93), 96, 21, boxstyle='round,pad=0.4,rounding_size=1.6', fc='white', ec='#8c8a83', lw=1, ls='--'))
ax.text(4, 113, 'Step I  Screening (sensor data only, before any model)', fontsize=8.6, weight='bold', color=INK, va='top')
box(5, 95, 28, 12.5, 'Lag test', 'shallow → deep onset lag\nvs Green–Ampt time t$_m$')
box(36, 95, 28, 12.5, 'Water-balance bound', 'd$_{max}$ = P/Δθ vs depth;\ndeeper sensor first?')
box(67, 95, 28, 12.5, 'Sensor comparability', 'site calibration or\nnormalisation to S$_e$')
arrow(50, 119.6, 50, 114.6)
ax.add_patch(Polygon([(50, 89.5), (73, 80), (50, 70.5), (27, 80)], closed=True, fc='#fff8e6', ec='#eda100', lw=1.1))
ax.text(50, 80, 't$_{obs}$ ≳ t$_m$, d$_{max}$ ≥ depth,\nno depth inversions, and\nsensors comparable?', ha='center', va='center', fontsize=7.7, color=INK, weight='bold', linespacing=1.3)
arrow(50, 92.6, 50, 89.5)
box(3, 50, 40, 13, 'Step II-a  Physics path', 'Richards PINN or calibrated solver\n(structural prior on propagation)', fc='#f1eefb', ec='#4a3aa7')
box(3, 29, 40, 15, 'Attribution diagnostics', 'pass if ρ$_R$ ≲ 0.3, s$_{matrix}$ ≳ 0.3 (indicative);\nindependent solver reproduces skill;\nno-physics ablation loses skill', fc='#f1eefb', ec='#4a3aa7')
arrow(27, 80, 20, 63.6, 'yes', lx=4, ly=0)
arrow(23, 50, 23, 44.6)
box(57, 50, 40, 13, 'Step II-b  Conceptual path', 'two-store (fast/slow) model or PG-MLP\nof what the sensor actually sees', fc='#fdf0ea', ec='#eb6834')
box(57, 29, 40, 15, 'Report the sensing gap', 'recommend a probe or piezometer\nat the failure plane', fc='#fdf0ea', ec='#eb6834')
arrow(73, 80, 80, 63.6, 'no', lx=-4, ly=0)
arrow(77, 50, 77, 44.6)
arrow(43.4, 38, 56.6, 52, 'fail', lx=-1.5, ly=3, c='#b8452a')
box(12, 8, 76, 12, 'Step III  Storm-blocked validation', 'retain a model only if the storm-bootstrap 95% interval of its skill\ndifference from the constraint-level baseline lies above zero', fc='#eef7f2', ec='#1baf7a')
arrow(33, 28.6, 38, 20.6, 'pass', lx=-4.5, ly=0.5)
arrow(67, 28.6, 62, 20.6)
box(4, 0.4, 92, 5, 'Step IV  Hydrological alerts: POD, FAR, timely detection, calibrated 90% band', fc='#eef4fc', ec='#2a78d6', fs=7.6)
arrow(50, 7.6, 50, 5.8)
note(1, 76, 'Utuado 57 cm\n(after S$_e$\nnormalisation)')
note(99, 77, 'Thai 30 cm (fails\nlag and storage);\nToro Negro (satur-\nated input, inversions)', ha='right')
note(1, 27, 'Thai PINN: ρ$_R$ = 0.68,\ns$_{matrix}$ ≈ 0.01, ablation +0.02')
note(99, 27, 'Utuado 72 cm: screening ambiguous;\nregistered model fails validation', ha='right')
fig.savefig(out('Fig04_SAVE_decision.png'), dpi=300, bbox_inches='tight', pad_inches=0.05)
