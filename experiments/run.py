"""Train and evaluate one regularizer at its frozen setting (experiments/frozen.json).

    python experiments/run.py hyper                          # held-out cases, 8.5-min stage-1 budget
    python experiments/run.py lsc --cases 1 8                # development cases
    python experiments/run.py hyper --budget 2               # fixed budget: 2, 4.25, 8.5 or 17 min
    python experiments/run.py hyper --outers 14              # fixed iterations: exactly 14, no cap
    python experiments/run.py hyper --stride 30 --budget 1 --c-steps 14   # 23 solved views
    python experiments/run.py tv --lam 3e-4 --c-accept 1 --cases 1 8   # one point of the development sweep

Each case trains with train.py, then test.py writes <ckpt>_eval.json (TRE,
SSIM, PSNR). Runs go to runs/<reg>_b<budget>/ (or _o<outers>, plus any overrides) unless --out is given. Cases run one after another on one GPU.
At stride 30 the paper selected K (--c-steps) per budget on the development cases: 23 at
2, 4.25 and 8.5 min, and 14 at 1 min.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FROZEN = json.load(open(os.path.join(HERE, 'frozen.json')))
DEV, HELD_OUT = [1, 8], [2, 3, 4, 5, 6, 7, 9, 10]

P = argparse.ArgumentParser()
P.add_argument('reg', choices=sorted(FROZEN))
P.add_argument('--cases', type=int, nargs='+', default=HELD_OUT)
g = P.add_mutually_exclusive_group()
g.add_argument('--budget', type=float, default=8.5, help='stage-1 wall-clock cap, minutes')
g.add_argument('--outers', type=int, help='exactly this many outer iterations, no cap')
P.add_argument('--stride', type=int, default=12, help='solve on every k-th of the 680 views')
P.add_argument('--c-steps', type=int, default=23, help='basis steps per outer iteration (K)')
P.add_argument('--lam', help='override the frozen weight (development sweep)')
P.add_argument('--c-accept', type=int, choices=(0, 1),
               help='override the frozen basis-update rule (the sweep used 1 for every regularizer)')
P.add_argument('--gpu', type=int, default=0)
P.add_argument('--out')
A = P.parse_args()

f = FROZEN[A.reg]
flags = f['flags']
if A.lam:
    if 'lam' not in f:
        sys.exit(f'{A.reg} has no weight')
    flags = flags.replace(f['lam'], A.lam)
flags += f' --outer {A.outers} --stage1-minutes 0' if A.outers else f' --outer 200 --stage1-minutes {A.budget:g}'
if A.c_accept is not None:
    flags = flags.replace(' --c-accept 1', '') + f' --c-accept {A.c_accept}'
flags += f' --train-stride {A.stride} --c-steps {A.c_steps}'
tag = A.reg + (f'_lam{A.lam}' if A.lam else '') + (f'_acc{A.c_accept}' if A.c_accept is not None else '')
tag += f'_o{A.outers}' if A.outers else f'_b{A.budget:g}'
tag += f'_s{A.stride}_k{A.c_steps}' if (A.stride, A.c_steps) != (12, 23) else ''
out = A.out or os.path.join('runs', tag)

for case in A.cases:
    sh = lambda c: subprocess.check_call(c, shell=True, cwd=REPO)
    sh(f'{sys.executable} -u train.py --case {case} --gpu {A.gpu} --out {out} {flags}')
    sh(f'{sys.executable} -u test.py --ckpt {out}/case{case}_s0.pt --gpu {A.gpu}')
