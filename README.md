Reconstruct a 3D volume at a target respiratory phase from a **single 2D
projection**, in (near) real time, by deforming a known source-phase volume.
A shared-weight recurrent **Iterative Refinement Block (IRB)** progressively
sharpens the deformation, optionally accompanied by a bounded photometric
correction.

---

## The idea

A single 3D encoder consumes the source volume plus a broadcast embedding of
the target projection and produces a bottleneck. An initial decoder predicts a
velocity field, which is integrated (scaling-and-squaring) into a displacement
field and used to warp the source. A shared-weight IRB is then applied `N`
times; each application:

1. re-warps the source by the current flow and re-embeds it (sees residual error),
2. updates a ConvGRU hidden state,
3. emits a **velocity update**, integrated and **composed** onto the running
   flow (diffeomorphic; keeps the field physically valid),
4. warps clean source by the composed field.

Depth comes from *repeated application of the same operator*, not extra
parameters. The flow fed back as input is detached (RAFT-style).

**Dual (multimodal) variants** additionally emit a photometric correction:
raw image increments **accumulate** across steps (paralleling the flow
accumulation), and the **accumulator** is ε-bounded (`ε·tanh`, ε=0.05) so the
total image budget is fixed regardless of `N`. The image residual rides on the
output only — it never enters the next warp — so the DVF stays clean.

---

## Variants

| Mode       | `--num_irb` | Description                                                        |
|------------|-------------|--------------------------------------------------------------------|
| `baseline` | 0           | Initial decoder only — no refinement.                              |
| `dvf`      | 1, 2, 3     | + N shared-weight DVF-only IRBs.                                   |
| `dual`     | 1, 2, 3     | + N shared-weight dual IRBs (DVF + bounded photometric residual).  |
| `bigfly`   | 0           | Deeper one-shot decoder, **no recurrence** — capacity/FLOP control.|

`bigfly` exists to separate "recurrence helps" from "more parameters help": it
must have ≥ the params of the recurrent variants while *not* showing the same
iterative gains for the recurrence claim to hold.

---

## Loss

Two weighting axes, each with the appropriate tool:

- **Step axis — deep supervision, fixed γ prior.** Every intermediate output
  `y_t` is scored against the target with weight `γ^(T−t)` (γ=0.8, RAFT
  default). Steps are an ordered refinement *sequence* we want to improve
  monotonically, so a fixed ordering prior is correct; uncertainty weighting
  would wrongly down-weight early steps for having higher expected loss.
- **Term axis — uncertainty weighting (learned).** For dual variants the two
  final reconstructions — clean motion (`warp(source, flow_final)`) and motion
  + bounded image (`y_final`) — are genuinely competing observations of the
  same target, so each gets a learned `log_var` and is combined as
  `0.5·exp(−lv)·L1 + 0.5·lv`.

All reconstruction L1s are computed **inside the thoracic mask**. There is **no
smoothness penalty**: scaling-and-squaring integration yields diffeomorphic
(non-folding) fields, so it is redundant. Dual variants freeze the image head
for the first `IMG_FREEZE_EPOCHS` (5) so the DVF learns the geometry first.

---
