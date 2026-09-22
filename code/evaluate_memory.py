"""
Memory-bounded on-device training: end-to-end backprop vs. layer-wise local
("firebreak") training.

A depth-configurable encoder is trained two ways with a VICReg-style
variance/covariance objective:

  * end-to-end : one global objective on the final embedding; the whole
    network's activation graph is retained for backprop, so peak activation
    memory grows with depth.

  * layer-wise local ("firebreak") : each block has its own local objective and
    is optimized in isolation, with the block output detached before the next
    block. Only one block's activation graph is live at a time, so peak
    activation memory is (approximately) independent of depth. This is the
    on-device training scheme the thesis motivates.

For each depth we report peak training memory (real ``torch.cuda`` peak on GPU;
a portable analytical activation-memory estimate otherwise) and the effective
rank of the learned final embedding (to confirm the local scheme still learns
structured, non-collapsed features).

Run on a Colab GPU for the real peak-memory numbers; it also runs on CPU/laptop
(reporting the analytical estimate).
"""

import json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEPTHS = [2, 4, 8, 16, 24]
C = 64          # channels per block
HW = 32         # spatial size kept constant across blocks
BATCH = 64
STEPS = 30
LATENT = 128


def make_blocks(depth):
    stem = nn.Sequential(nn.Conv2d(3, C, 3, stride=2, padding=1), nn.BatchNorm2d(C), nn.ReLU())
    blocks = nn.ModuleList([
        nn.Sequential(nn.Conv2d(C, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU())
        for _ in range(depth)
    ])
    head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(C, LATENT))
    return stem, blocks, head


def vicreg(z, gamma=1.0):
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(gamma - std))
    zc = z - z.mean(0, keepdim=True)
    cov = (zc.T @ zc) / (z.size(0) - 1)
    off = (cov * (1 - torch.eye(cov.size(0), device=z.device))).pow(2).sum() / cov.size(0)
    return var_loss + off


def local_heads(depth, device):
    return nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                        nn.Linear(C, LATENT)) for _ in range(depth)]).to(device)


def activation_bytes(depth, mode):
    """Portable estimate of retained activation memory (float32) for one step."""
    per_block = BATCH * C * HW * HW * 4
    return (depth * per_block) if mode == "e2e" else per_block  # e2e retains all; local one


def run(depth, mode, device):
    torch.manual_seed(0)
    stem, blocks, head = make_blocks(depth)
    stem, blocks, head = stem.to(device), blocks.to(device), head.to(device)
    params = list(stem.parameters()) + list(blocks.parameters()) + list(head.parameters())
    lheads = local_heads(depth, device) if mode == "local" else None
    if mode == "local":
        params += list(lheads.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for _ in range(STEPS):
        x = torch.rand(BATCH, 3, 64, 64, device=device)
        opt.zero_grad()
        h = stem(x)
        if mode == "e2e":
            for b in blocks:
                h = b(h)
            z = head(h)
            loss = vicreg(z)
            loss.backward()
        else:  # layer-wise local firebreak
            for i, b in enumerate(blocks):
                h = b(h)
                z = lheads[i](h)
                vicreg(z).backward()      # confined to this block
                h = h.detach()            # firebreak: cut the graph
            z = head(h.detach())          # final embedding for evaluation only
        opt.step()

    # Final embedding effective rank (eval)
    with torch.no_grad():
        h = stem(torch.rand(256, 3, 64, 64, device=device))
        for b in blocks:
            h = b(h)
        z = head(h)
        zc = z - z.mean(0, keepdim=True)
        cov = (zc.T @ zc) / (z.size(0) - 1)
        eig = torch.linalg.eigvalsh(cov).clamp(min=0)
        rank = (eig.sum() ** 2 / (eig ** 2).sum()).item()

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        metric = "peak_gpu_mb"
    else:
        peak_mb = activation_bytes(depth, mode) / (1024 ** 2)
        metric = "est_activation_mb"
    return peak_mb, metric, rank


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}\n", flush=True)
    results = {}
    for depth in DEPTHS:
        row = {}
        for mode in ("e2e", "local"):
            try:
                peak, metric, rank = run(depth, mode, device)
                row[mode] = {metric: round(peak, 1), "eff_rank": round(rank, 2)}
                print(f"depth={depth:2d} {mode:5s}  {metric}={peak:8.1f}  eff_rank={rank:.2f}", flush=True)
            except RuntimeError as e:  # e.g. CUDA OOM for deep e2e
                row[mode] = {"error": str(e)[:80]}
                print(f"depth={depth:2d} {mode:5s}  ERROR: {str(e)[:60]}", flush=True)
        results[depth] = row
    with open("memory_bounded.json", "w") as f:
        json.dump({"channels": C, "hw": HW, "batch": BATCH, "results": results}, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
