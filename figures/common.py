"""Shared paths for the manuscript figure scripts (run from anywhere)."""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
TH = os.path.join(ROOT, 'thailand')
PR = os.path.join(ROOT, 'puerto_rico')
DATA_TH = os.path.join(ROOT, 'data', 'thailand')
OUT = os.path.join(ROOT, 'figures', 'parts')          # intermediate panels
FINAL = os.path.join(ROOT, 'figures', 'output')       # figures under their manuscript numbers
for p in (TH, PR):
    if p not in sys.path:
        sys.path.insert(0, p)

def out(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)
