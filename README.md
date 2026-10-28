# VM-DS

Real-time volumetric respiratory motion from a single x-ray projection for
image-guided interventions.

## What this does

During an interventional procedure a breath-hold (DIBH) CBCT is already
acquired as standard of care. VM-DS takes that volume as the reference, and a second CBCT arc acquired while the
patient is anaesthetised and ventilated, and learns in two stages:

1. **Motion model.** A low-rank B-spline motion model, `DVF(p) = sum_l w_l(p) e_l`,
   is fit to the ventilated arc projections by alternating least squares.
   The w-step solves nine scores per projection by damped Gauss-Newton; the
   c-step descends the basis control points.
2. **Encoder.** A CNN is trained to map one projection to the nine scores, on
   projection/score pairs synthesised from the frozen model, so the labels are
   exact.

Inference is one forward pass of the encoder and the basis, with no
per-frame optimisation. Training takes about 10 minutes per case on one GPU and inference takes about 1.5 milliseconds.

Evaluation uses DIR-Lab 4DCT. T00 stands in for the DIBH reference,
and the ventilated arc is simulated by rendering each projection from the
phase volume the patient is in at that gantry angle. Phase labels are used only
to render the simulation. No part of the method reads them.

## Installation

```bash
git clone <repo-url> && cd voxelmap
pip install -e .
conda install -c astra-toolbox -c nvidia astra-toolbox   # the cone-beam projector
```

A CUDA GPU is required.

## Data

DIR-Lab 4DCT is registration-gated and is not redistributed here. Obtain it
from [dir-lab.com](https://www.dir-lab.com/) and point VM-DS at it:

```bash
export VOXELMAP_DIRLAB=/path/to/DIR-LAB
python -m voxelmap.data --case 8        # once per case: crop, resample, normalisation
```

## Training

```bash
python train.py --case 8 --gpu 0
```

The defaults are the configuration the paper reports, i.e. 57 solved views (every
12th of 680), 14 outer iterations of 23 basis steps, then 4096 synthesised
pairs for 100 epochs. Every field of `voxelmap/config.py` is also a flag. These hyperparameters may require retuning for other datasets.

The output, under `runs/`, is a checkpoint holding the motion model, the
encoder, and the solved scores, plus the stage-1 loss curve.

## Layout

```
train.py               stage 1 then stage 2, one case
voxelmap/
  config.py            run settings
  data.py              DIR-Lab reading and cropping; the simulated ventilated arc
  projector.py         differentiable cone-beam projection (ASTRA)
  model.py             B-spline motion model, encoder, warp, inference
  objective.py         the stage-1 objective and its residual rows
  als.py               stage 1: Gauss-Newton w-step, basis c-step, outer loop
  synthesis.py         stage 2: known-DVF pairs and the encoder fit
```

## Licence

Source code is MIT (see `LICENSE`). **Patent rights are reserved separately:
see `NOTICE`.** The underlying methods are protected by US Patent Application
US20250285300A1. This repository grants no patent licence, and commercial use
requires a separate licence.
