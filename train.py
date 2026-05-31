"""
Training entry point for VoxelMap motion-guided refinement.

Runs the 2x2 matrix: proj_mode in {single, cycle} x coupling in {fixed, multisrc}
generalised here to the residual coupling switch. Example:

    python train.py --proj-mode cycle --coupling consistency --supervised \
        --vol-size 128 --epochs 50 --batch-size 2 --lr 1e-5

The dataset is intentionally abstracted behind `build_dataloaders`; wire it to your
existing DRR-augmentation / disk-cache pipeline. Each batch is a dict:
    proj_a       : (B, in_ch, H, W)
    proj_b       : (B, in_ch, H, W)   # only needed for proj_mode='cycle'
    source_vol   : (B, 1, D, H, W)
    target_vol   : (B, 1, D, H, W)
    dvf_true     : (B, 3, D, H, W)    # supervised only
    thorax_mask  : (B, 1, D, H, W)    # optional
"""

from __future__ import annotations
import argparse
import os
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from utilities.networks import VoxelMapRefine
from utilities.losses import compute_loss


# --------------------------------------------------------------------------- #
# Dataset hook — replace with your DRR-augmentation / disk-cache dataset.
# --------------------------------------------------------------------------- #
def build_dataloaders(cfg):
    """
    Return (train_loader, val_loader). Stubbed with a tiny synthetic dataset so the
    script runs end-to-end; swap in your real Dataset here.
    """
    from torch.utils.data import Dataset

    class _Synthetic(Dataset):
        def __init__(self, n, cfg):
            self.n = n
            self.cfg = cfg

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            v = self.cfg.vol_size
            s = self.cfg.proj_size
            item = {
                "proj_a": torch.randn(self.cfg.in_ch, s, s),
                "proj_b": torch.randn(self.cfg.in_ch, s, s),
                "source_vol": torch.randn(1, v, v, v),
                "target_vol": torch.randn(1, v, v, v),
                "dvf_true": torch.zeros(3, v, v, v),
                "thorax_mask": torch.ones(1, v, v, v),
            }
            return item

    train = _Synthetic(8, cfg)
    val = _Synthetic(4, cfg)
    return (
        DataLoader(train, batch_size=cfg.batch_size, shuffle=True, num_workers=0),
        DataLoader(val, batch_size=cfg.batch_size, shuffle=False, num_workers=0),
    )


# --------------------------------------------------------------------------- #
def move_batch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def run_epoch(model, loader, cfg, device, optim=None):
    train = optim is not None
    model.train(train)
    agg = {}
    n = 0
    for batch in loader:
        batch = move_batch(batch, device)
        proj_b = batch["proj_a"] if cfg.proj_mode == "single" else batch["proj_b"]
        with torch.set_grad_enabled(train):
            out = model(batch["proj_a"], batch["source_vol"],
                        proj_b=proj_b if cfg.proj_mode == "cycle" else None)
            loss, logs = compute_loss(out, batch, cfg)
        if train:
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
        bs = batch["proj_a"].shape[0]
        n += bs
        for k, val in logs.items():
            agg[k] = agg.get(k, 0.0) + val * bs
    return {k: v / max(n, 1) for k, v in agg.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--proj-mode", choices=["single", "cycle"], default="single")
    p.add_argument("--coupling",
                choices=["decoupled", "shared_latent", "consistency"],
                default="decoupled")
    p.add_argument("--no-residual", action="store_true",
                help="pure-warp base embodiment (no refinement arm)")
    p.add_argument("--supervised", action="store_true")
    p.add_argument("--vol-size", type=int, default=128)
    p.add_argument("--proj-size", type=int, default=128)
    p.add_argument("--in-ch", type=int, default=2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--alpha", type=float, default=1e-5, help="DVF smoothness weight")
    p.add_argument("--lambda-cycle", type=float, default=1.0)
    p.add_argument("--lambda-consistency", type=float, default=1.0)
    p.add_argument("--residual-scale", type=float, default=0.1)
    p.add_argument("--integrate-steps", type=int, default=7)
    p.add_argument("--out-dir", default="./runs")
    p.add_argument("--tag", default="exp")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    model = VoxelMapRefine(
        vol_size=args.vol_size, in_ch=args.in_ch, proj_mode=args.proj_mode,
        coupling=args.coupling, use_residual=not args.no_residual,
        integrate_steps=args.integrate_steps, residual_scale=args.residual_scale,
    ).to(device)

    cfg = SimpleNamespace(
        supervised=args.supervised, alpha=args.alpha,
        lambda_cycle=args.lambda_cycle, lambda_consistency=args.lambda_consistency,
        coupling=args.coupling, proj_mode=args.proj_mode,
        vol_size=args.vol_size, proj_size=args.proj_size, in_ch=args.in_ch,
        batch_size=args.batch_size, _transform=model.transform,
    )

    train_loader, val_loader = build_dataloaders(cfg)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_name = f"{args.tag}_{args.proj_mode}_{args.coupling}" \
            f"_{'sup' if args.supervised else 'unsup'}"
    best_val = float("inf")

    for epoch in range(args.epochs):
        tr = run_epoch(model, train_loader, cfg, device, optim)
        va = run_epoch(model, val_loader, cfg, device, optim=None)
        msg = f"[{run_name}] epoch {epoch+1:3d}/{args.epochs} " \
            f"train {tr.get('total', 0):.4e} val {va.get('total', 0):.4e}"
        extras = {k: va[k] for k in ("dvf_mse", "img_mse", "cycle", "consistency")
                if k in va}
        if extras:
            msg += " | " + " ".join(f"{k}={v:.3e}" for k, v in extras.items())
        print(msg, flush=True)

        if va.get("total", float("inf")) < best_val:
            best_val = va["total"]
            ckpt = os.path.join(args.out_dir, f"{run_name}_best.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "cfg": vars(args), "val": va}, ckpt)

    print(f"done. best val {best_val:.4e}")


if __name__ == "__main__":
    main()
