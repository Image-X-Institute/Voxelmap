# Motion-Guided Volume Refinement

A patient-specific 2D→3D registration and volumetric imaging model. Given a real-time
2D projection acquired during a procedure and a source 3D image (e.g. planning CT), the
network estimates a 3D deformation vector field (DVF) and produces an updated 3D image.

The updated image is formed by **sequential refinement** on top of the published
spatial-transform slot:

```
updated 3D image = warp(source, DVF) + bounded residual correction
```

The DVF carries the primary physical explanation (anatomy moves); the residual arm only
corrects what pure warping cannot capture — interpolation artefacts, local intensity
mismatch, imperfect deformation, small non-deformable anatomical change. The residual is
bounded (`tanh * residual_scale`) so the model degrades gracefully to pure warping and
the DVF remains the dominant term rather than the residual quietly taking over
reconstruction.

## Files

| File | Contents |
|------|----------|
| `networks.py` | Encoder, DVF decoder, scaling-and-squaring integration, spatial transform, residual arm, full `VoxelMapRefine` model. |
| `losses.py`   | Supervised DVF MSE, unsupervised image + smoothness, cycle consistency, residual–DVF consistency. |
| `train.py`    | Config, train/val loop, checkpointing, CLI for the experiment matrix. |

## Architecture

Same inputs and outputs across all configurations:

- **Inputs:** 2D projection pair `(B, in_ch, 2^n, 2^n)`, source volume `(B, 1, D, H, W)`.
- **Outputs:** 3D DVF `(B, 3, D, H, W)` and updated 3D image `(B, 1, D, H, W)`.

Encoding arm: `n` downsampling 2D residual blocks (4×4 s2 → 3×3 s1, BN, ReLU),
channels doubling each block, reducing `2^n × 2^n` to a `1×1` latent. The latent is
reshaped to a 3D tensor and passed through `n` upsampling 3D residual blocks; two final
3×3×3 convs (Tanh then linear) produce a stationary velocity field, integrated to a
diffeomorphic DVF by scaling-and-squaring. The spatial transform module warps the source
volume by the DVF; the residual arm refines the result. `n` is derived from `vol_size`
(`n = log2(vol_size)`), so larger volumes use proportionally larger networks.

## The two switches

### `coupling` — how the residual arm relates to the motion estimator

- **`decoupled`** — the residual arm reads only the warped volume. No pressure on the
  latent; clean separation of motion vs intensity correction. Risk: nothing ties the
  residual to the predicted motion, so a good-looking volume can sit on a slightly wrong
  DVF.
- **`shared_latent`** — the residual decoder conditions on the *same* latent as the DVF
  decoder. Regularises the residual toward motion-consistency, but forces one latent to
  serve two genuinely different objectives (motion + intensity), which can degrade DVF
  accuracy if the residual's true causes are motion-independent (scatter, beam hardening).
- **`consistency`** — architecturally identical to `decoupled`, but a consistency loss
  penalises the residual for encoding displacement the DVF could have explained. Aims for
  the regularisation benefit of coupling without the latent serving two masters.

`consistency` and `decoupled` share architecture and differ only in the loss, making them
a clean A/B for whether tying the residual to the DVF helps or hurts **DVF error
specifically**.

### `proj_mode` — how the DVF is predicted

- **`single`** — one acquired projection pair predicts the DVF.
- **`cycle`** — two routes (separate encoder/decoder pairs, different input projections)
  predict the *same* DVF label; a cycle-consistency loss penalises disagreement. This is
  *earned* convergence pressure: both routes describe the same physical motion, so forcing
  agreement shapes the latent toward route-invariant motion structure.

> Note the contrast with `shared_latent`: both squeeze through one bottleneck, but
> `proj_mode='cycle'` squeezes **two views of one thing** (agreement is physically correct),
> whereas `shared_latent` squeezes **two different things** (agreement isn't required by
> physics — just capacity competition). Same mechanism, opposite verdict.

Set `use_residual=False` (`--no-residual`) for the pure-warp base embodiment.

## Usage

```bash
python train.py \
  --proj-mode cycle \
  --coupling consistency \
  --supervised \
  --vol-size 128 --in-ch 2 \
  --epochs 50 --batch-size 2 --lr 1e-5 \
  --alpha 1e-5 --lambda-cycle 1.0 --lambda-consistency 1.0 \
  --out-dir ./runs --tag p1
```

Key flags: `--proj-mode {single,cycle}`, `--coupling {decoupled,shared_latent,consistency}`,
`--supervised` (DVF MSE; omit for unsupervised image + smoothness), `--no-residual`,
`--residual-scale` (residual bound), `--integrate-steps` (scaling-and-squaring).
