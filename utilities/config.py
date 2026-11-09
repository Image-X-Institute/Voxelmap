"""Run settings. The defaults are the VM-DS baseline (self-consistency, 14 outer iterations).
experiments/frozen.json gives the flags of every regularizer at its frozen setting."""
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # acquisition: the ventilated arc
    case: int = 8
    n_breaths: int = 1            # machine-paced cycles across the arc
    train_stride: int = 12        # solve on every k-th of the 680 views (57 views)
    noise_i0: float = 0.0         # Poisson quantum noise, photons per pixel; 0 = noiseless
    bin: int = 2                  # solve at 2 mm voxels

    # motion model
    levels: int = 3
    base_grid: int = 6

    # stage 1: alternating least squares, sized to an 8.5-minute window
    outer: int = 14
    w_iters: int = 1              # Gauss-Newton iterations per w-step
    lm_init: float = 1e-3
    eps: float = 1e-3             # finite-difference step on the scores
    c_steps: int = 23
    c_lr: float = 3e-3
    batch: int = 32
    sc_lambda: float = 1e3        # spatial consistency weight
    sc_iters: int = 3             # fixed-point iterations of the inverse field
    sc_vox: int = 100_000         # voxel subset for L_SC in the w-step
    # hyperelastic: weight on mean psi(det J), the hyperelastic volume
    # term on every 2nd voxel; 0 = off (published)
    hyper_lambda: float = 0.0
    # comparator regularizer: one of '', 'bending', 'diffusion', 'tv', 'selfcon', 'jneg',
    # 'logj', weighted by reg_lambda (see utilities/regularisers.py)
    reg: str = ''
    reg_lambda: float = 0.0
    # svf: > 0 makes the basis field a stationary velocity, phi = exp(v) by
    # this many squarings (VoxelMap: 10, no smoothness term); 0 = direct displacement
    svf_steps: int = 0
    # c-step gradient micro-batch (memory only: gradients accumulate, so the objective
    # and its step are unchanged); 0 = batch
    c_micro: int = 0
    # 'composed' samples I_ref once at the composed coordinates instead of warping the
    # warped image back ('image', the VM-DS form); used by analysis/lsc_decompose.py
    sc_mode: str = 'image'
    # control-point bound: after each c-step, cap every (level, axis) at this multiple of the
    # Choi-Lee injectivity bound, 0.40 x knot spacing (1.0 = strict); 0 = off
    ctrl_bound: float = 0.0
    # 1 undoes any c-step that raises the c-step objective
    c_accept: int = 0
    # stage-1 budget: > 0 starts no outer that the last one's duration
    # says would end past this many minutes; 0 runs exactly `outer`
    stage1_minutes: float = 0.0

    # stage 2: encoder on known-DVF synthesised pairs
    synth_pairs: int = 4096
    synth_epochs: int = 100
    enc_lr: float = 1e-3
    enc_max_ch: int = 32

    seed: int = 0

    def as_dict(self):
        return asdict(self)
