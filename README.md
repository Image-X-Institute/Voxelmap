# Hyperelasticity as an efficient regularizer for 2D–3D thoracic motion models

This builds on
[VM-DS](https://arxiv.org/abs/2610.34792), which learns a patient-specific motion model on the day of
an intervention and then infers a 3D volume from each 2D x-ray projection in a single forward pass.
Here the image-space self-consistency regularizer of VM-DS is replaced by a hyperelastic volume
penalty on the deformation. We also include the other regularizers explored in our ablation study.

## What this does

A breath-hold CBCT is the reference volume. A second, motion-inclusive CBCT arc is the training
data. Training has two stages:

1. **Motion model.** A low-rank B-spline motion model, `DVF(p) = sum_l w_l(p) e_l`, is fit to the
   arc projections by alternating least squares. The w-step solves nine scores per projection by
   damped Gauss–Newton. The c-step descends the basis control points. The regularizer enters both
   steps, so every step descends one objective.
2. **Encoder.** A CNN is trained to map one projection to the nine scores, on projection/score
   pairs synthesised from the frozen model, so the labels are exact.

Inference is one forward pass of the encoder and the basis.

The hyperelastic term is the volume part of hyperelastic regularization (Burger, Modersitzki and
Ruthotto 2013), `psi(v) = (v - 1)^4 / v^2` on `v = det J`, averaged over every second voxel. It is
blind to rigid motion and shear and grows without bound as a voxel's volume goes to zero. It is
paired with a descent-only basis update (`--c-accept 1`), which undoes any c-step that raises the
objective.

Evaluation uses DIR-Lab 4DCT. T00 stands in for the breath-hold reference, and the arc is simulated
by rendering each projection from the phase volume at that gantry angle. Phase labels are used only
to render the simulation and to score it. No part of the method reads them.

## Installation

```bash
git clone -b hyper https://github.com/Image-X-Institute/Voxelmap.git voxelmap-hyper && cd voxelmap-hyper
pip install -e .
conda install -c astra-toolbox -c nvidia astra-toolbox        # the cone-beam projector
```

A CUDA GPU is required.

## Data

DIR-Lab 4DCT is registration-gated and is not redistributed here. Obtain it from
the [DIR-Lab website](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html) and point the code at it:

```bash
export VOXELMAP_DIRLAB=/path/to/DIR-LAB
python -m utilities.data --case 8        # once per case: crop, resample, normalisation
```

## Training one case

```bash
# hyperelastic, frozen setting, 8.5-min stage-1 budget
python train.py --case 8 --gpu 0 --out runs/hyper \
    --sc-lambda 0 --hyper-lambda 1e-1 --c-accept 1 --outer 200 --stage1-minutes 8.5
python test.py --ckpt runs/hyper/case8_s0.pt --gpu 0
```

With no flags, `train.py` runs the VM-DS baseline (self-consistency, 14 outer iterations). Every
field of `utilities/config.py` is also a flag. The output is a checkpoint (motion model, encoder,
solved scores) and `_loss.npz`, the per-component loss against outer iteration and wall clock.
`test.py` writes `_eval.json`: landmark TRE at the held-out arc views, and SSIM and PSNR of the
predicted volumes.

## Reproducing our results

`experiments/frozen.json` holds every regularizer's frozen weight, basis-update rule and the weight
grid it was selected from. `experiments/run.py` trains and evaluates one regularizer over a set of
cases:

| Paper | Command |
|---|---|
| Development sweep (cases 1, 8) | `python experiments/run.py tv --cases 1 8 --lam 3e-4 --c-accept 1` for each value in `lambda_grid` |
| Basis-update rule (cases 1, 8) | `python experiments/run.py <reg> --cases 1 8 --c-accept 0` (or `1`, the other rule) |
| Held-out comparison (8.5 min) | `python experiments/run.py <reg>` for each regularizer with `held_out: true` |
| Fixed budget | `python experiments/run.py <reg> --budget 2` (also 4.25, 8.5, 17) for `lsc`, `hyper`, `selfcon`, `none` |
| Fixed iterations | `python experiments/run.py <reg> --outers 14` for the same four |
| Fewer projections | `python experiments/run.py hyper --stride 30 --budget 2` (also 4.25, 8.5; `--budget 1 --c-steps 14`) |

Regularizer keys: `lsc` (self-consistency, the baseline), `none`, `svf`, `bending`, `diffusion`,
`selfcon` (first-order inverse consistency), `tv`, `jneg` (negative-Jacobian hinge), `logj`,
`cbound` (control-point injectivity bound) and `hyper`.

```bash
python experiments/summarise.py runs/hyper_b8.5 runs/lsc_b8.5         # per-case and mean TRE, SSIM, PSNR, time
python analysis/encoder_folding.py --gpu 0 --out runs/folding.json runs/hyper_b8.5/case*_s0.pt
python analysis/lsc_decompose.py --gpu 0 --hyper runs/hyper_b8.5 --baseline runs/lsc_b8.5
```

`encoder_folding.py` counts folded voxels in encoder-predicted fields at every held-out view.
`lsc_decompose.py` separates self-consistency into double-resampling error and inverse error. It reads
cases 8 and 1, so its directories must hold development runs (`--cases 1 8`).

Stage-1 budgets are wall-clock caps, measured on an RTX PRO 4000 Blackwell. On other hardware the
same cap completes a different number of outer iterations. The fixed-iteration runs do not depend on
hardware. Reruns agree to about 1e-3, not bitwise, because the ALS solve is not deterministic on GPU.
All results use seed 0.

## Layout

```
train.py               stage 1 then stage 2, one case
test.py                held-out TRE, SSIM and PSNR of a checkpoint
utilities/
  config.py            run settings; every field is a train.py flag
  data.py              DIR-Lab reading and cropping; the simulated arc
  projector.py         differentiable cone-beam projection (ASTRA)
  model.py             B-spline motion model, SVF, encoder, warp, inference
  objective.py         stage-1 objective: projection rows, self-consistency, hyperelastic term
  regularisers.py      comparator regularizers (bending, diffusion, TV, inverse consistency,
                       negative-J hinge, |log J|)
  als.py               stage 1: Gauss-Newton w-step, basis c-step, control-point bound, outer loop
  synthesis.py         stage 2: known-DVF pairs and the encoder fit
  landmarks.py         DIR-Lab landmarks and TRE
experiments/           frozen settings, run and summary scripts
analysis/              folding of deployed fields; decomposition of self-consistency
```

## Citation

```bibtex
@article{hindley2026hyper,
  author  = {Hindley, Nicholas},
  title   = {Hyperelasticity as an efficient regularizer for {2D--3D} thoracic motion models},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## Licence

Source code is MIT (see `LICENSE`). **Patent rights are reserved separately: see `NOTICE`.** The
underlying methods are protected by US Patent Application US20250285300A1. This repository grants no
patent licence, and commercial use requires a separate licence.
