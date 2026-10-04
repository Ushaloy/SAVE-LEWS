"""Regenerate every figure of the revised manuscript (Fig. 1-11, S1-S9) from the data and stored results.

    python figures/make_all_figures.py          # from the repository root (about 1-2 min on CPU)

Output: figures/output/Fig01.png ... Fig11.png, FigS1.png ... FigS9.png (300 dpi), numbered as in the revised
manuscript; intermediate panels are written to figures/parts/. No model is trained here; see notebooks/ and the README for the full re-runs.
"""
import os, sys, shutil, subprocess
from common import ROOT, TH, PR, OUT, FINAL, out

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, cwd):
    print(f'>> {os.path.relpath(os.path.join(cwd, script), ROOT)}')
    subprocess.run([PY, script], cwd=cwd, check=True)


def compose(top, bottom, title_top, title_bottom, dst):
    """stack two panels with bold (a)/(b) titles, as in the manuscript"""
    from PIL import Image, ImageDraw, ImageFont
    A, B = Image.open(top).convert('RGB'), Image.open(bottom).convert('RGB')
    W = max(A.width, B.width)
    A = A.resize((W, round(A.height * W / A.width))); B = B.resize((W, round(B.height * W / B.width)))
    try:
        import matplotlib
        font = ImageFont.truetype(os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSans-Bold.ttf'), 44)
    except Exception:
        font = ImageFont.load_default()
    band = 90
    img = Image.new('RGB', (W, A.height + B.height + 2 * band), 'white'); d = ImageDraw.Draw(img)
    d.text((20, 20), title_top, fill='black', font=font); img.paste(A, (0, band))
    d.text((20, band + A.height + 20), title_bottom, fill='black', font=font); img.paste(B, (0, 2 * band + A.height))
    img.save(dst, dpi=(300, 300))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for d in (TH, PR): os.makedirs(os.path.join(d, 'figs'), exist_ok=True)
    # 1. model-based panels written inside the analysis folders
    run('make_figs_thai.py', TH)                      # Thai figs + thailand/step2_summary.json
    run('figs_pr.py', PR); run('figs_final_pr.py', PR)
    # 2. framework, sensor, ladder, alert and summary figures
    run('summarize_revision.py', TH)                  # revision analyses -> thailand/res_rev_summary.json
    for s in ['fig_concept.py', 'fig_sites.py', 'fig_sensors.py', 'fig1_save.py', 'fig_ladder.py', 'fig_alerts.py',
              'make_thai_figs.py', 'fig_jacobian.py']:
        run(s, HERE)
    # 3. copy model-based panels under their manuscript numbers
    COPY = {'Fig04.png': (PR, 'figs/figD_p2_lags.png'), 'Fig05.png': (TH, 'figs/fig1_training_storm.png'),
            'Fig07.png': (TH, 'figs/fig8_signal_source.png'), 'Fig09.png': (TH, 'figs/fig7_loso_skill.png'),
            'Fig10.png': (PR, 'figs/figA_skill_by_depth.png'), 'Fig11.png': (PR, 'figs/figB_bootstrap.png'),
            'Fig12.png': (PR, 'figs/figC_example_storm.png'), 'FigS2.png': (TH, 'figs/fig12_h3_posteriors.png'),
            'FigS3.png': (TH, 'figs/fig13_example_storm.png')}
    for dst, (base, src) in COPY.items():
        shutil.copy(os.path.join(base, src), out(dst))
    compose(os.path.join(TH, 'figs/fig11_h2_calibration.png'), os.path.join(PR, 'figs/figE_p4.png'),
            '(a) Thailand, Dev108: held-out storms by forecast horizon', '(b) Puerto Rico: primary targets', out('Fig14.png'))
    # 4. final numbering of the revised manuscript
    FINAL_MAP = {'Fig01': 'Fig01_concept.png', 'Fig02': 'Fig01.png', 'Fig03': 'Fig02.png', 'Fig04': 'Fig04_SAVE_decision.png',
                 'Fig05': 'Fig05.png', 'Fig06': 'Fig07.png', 'Fig07': 'Fig08.png', 'Fig08': 'Fig10.png', 'Fig09': 'Fig12.png',
                 'Fig10': 'Fig_alerts_levels.png', 'Fig11': 'Fig11_jacobian.png',
                 'FigS1': 'Fig04.png', 'FigS2': 'Fig06.png', 'FigS3': 'Fig09.png', 'FigS4': 'Fig11.png', 'FigS5': 'FigS1.png',
                 'FigS6': 'FigS2.png', 'FigS7': 'Fig14.png', 'FigS8': 'FigS3.png', 'FigS9': 'Fig15.png'}
    os.makedirs(FINAL, exist_ok=True)
    for dst, src in FINAL_MAP.items():
        shutil.copy(os.path.join(OUT, src), os.path.join(FINAL, dst + '.png'))
    got = sorted(f for f in os.listdir(FINAL) if f.endswith('.png'))
    print(f'{len(got)} figures in figures/output:', ', '.join(got))
