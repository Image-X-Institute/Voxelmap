"""Run settings. The defaults are the configuration the paper reports."""
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

    # stage 2: encoder on known-DVF synthesised pairs
    synth_pairs: int = 4096
    synth_epochs: int = 100
    enc_lr: float = 1e-3
    enc_max_ch: int = 32

    seed: int = 0

    def as_dict(self):
        return asdict(self)
