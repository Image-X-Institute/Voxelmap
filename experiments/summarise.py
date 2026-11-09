"""Per-case and mean results of run directories, from test.py's <ckpt>_eval.json and the
stage-1 time in train.py's <ckpt>_loss.npz.

    python experiments/summarise.py runs/hyper runs/lsc
"""
import glob
import json
import os
import re
import sys

import numpy as np

for d in sys.argv[1:]:
    rows = []
    for f in sorted(glob.glob(os.path.join(d, 'case*_s0_eval.json')),
                    key=lambda p: int(re.search(r'case(\d+)_', p).group(1))):
        e = json.load(open(f))
        case = int(re.search(r'case(\d+)_', f).group(1))
        loss = f.replace('_eval.json', '_loss.npz')
        t1 = float(np.load(loss)['t_stage1']) / 60 if os.path.exists(loss) else float('nan')
        rows.append((case, e['arc']['tre'], e['arc']['ssim_body'], e['arc']['psnr_body'], t1))
    if not rows:
        print(f'{d}: no evaluations')
        continue
    print(f'{d}\n  case   TRE (mm)   SSIM   PSNR   stage 1 (min)')
    for r in rows:
        print(f'  {r[0]:>4}   {r[1]:8.2f}   {r[2]:.3f}  {r[3]:5.1f}   {r[4]:6.2f}')
    m = np.mean([r[1:] for r in rows], 0)
    print(f'  mean   {m[0]:8.2f}   {m[1]:.3f}  {m[2]:5.1f}   {m[3]:6.2f}   ({len(rows)} cases)')
