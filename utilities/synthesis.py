"""Stage 2: train the encoder on known-DVF synthesised pairs.

With the basis frozen, any score vector w defines a DVF, a warped reference and
its projection at any arc angle, so (projection, w) pairs come with EXACT
labels. Scores are drawn between solved views adjacent in acquisition order --
no phase label is read -- and rendered at angles over the whole arc. Only the
geometry and the solved model enter; no held-out measurement is read.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from voxelmap.data import N_ANGLES


def sample_scores(w, views, n, rng):
    """Interpolate between a solved view and the next in acquisition order."""
    v = np.sort(np.asarray(views))
    i = rng.integers(0, len(v) - 1, n)
    t = torch.tensor(rng.random(n), device=w.device, dtype=torch.float32).view(-1, 1, 1)
    rng.random(n)   # keeps the stream of the published runs, which drew an amplitude here
    return (1 - t) * w[v[i]] + t * w[v[i + 1]]


def build_bank(ops, w, views, n_pairs, batch, rng, log=print):
    """Render n_pairs (projection, score) pairs; projections kept on the CPU in fp16."""
    X, W = [], []
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n_pairs, batch):
            nb = min(batch, n_pairs - s)
            ws = sample_scores(w, views, nb, rng)
            ang = rng.integers(0, N_ANGLES, nb)
            vol = ops.warp(ops.iref, ops.deform(ws))
            X.append(ops.project(vol, ang).half().cpu())
            W.append(ws.cpu())
    X, W = torch.cat(X), torch.cat(W)
    log(f'  bank: {len(X)} pairs, {X.numel() * 2 / 2**30:.2f} GiB, {time.time() - t0:.0f} s')
    return X, W


def fit_encoder(enc, w, views, ops, n_pairs, epochs, batch, lr, seed=0, log=print):
    """Render the bank once, then regress the encoder onto its exact scores."""
    dev = ops.iref.device
    X, W = build_bank(ops, w, views, n_pairs, batch, np.random.default_rng(seed), log)
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        perm = rng.permutation(len(X))
        tot = 0.0
        for s in range(0, len(X), batch):
            k = perm[s:s + batch]
            opt.zero_grad()
            loss = F.mse_loss(enc(X[k].to(dev).float()), W[k].to(dev))
            loss.backward()
            opt.step()
            tot += float(loss) * len(k)
        if (ep + 1) % max(1, epochs // 10) == 0:
            log(f'  synth epoch {ep + 1}/{epochs}  loss {tot / len(X):.5g}')
    return enc
