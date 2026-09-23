"""
Low-power training study: end-to-end backprop vs. layer-wise local
("firebreak") training, measuring REAL peak training memory vs. encoder depth.

Two training schemes for a depth-configurable encoder, both using a
VICReg-style variance+covariance objective with the two penalties CO-LOCATED
on the same embedding (the collapse-resistant placement identified in the
main results):

  * end-to-end ("e2e"): one global objective on the final embedding; the whole
    network's activation graph is retained for backprop, so peak activation
    memory grows with depth.

  * layer-wise local ("local", the firebreak scheme): each block has its own
    local objective and is optimized in isolation, with the block output
    detached before the next block. Only one block's activation graph is live
    at a time, so peak activation memory is (approximately) independent of
    depth. This is the on-device training scheme the thesis motivates.

For each depth and scheme we report the REAL peak training memory
(``torch.cuda.max_memory_allocated`` on GPU) and the effective rank of the
learned final embedding (to confirm the local scheme still learns structured,
non-collapsed features). Results are averaged over several seeds and written to
JSON, and a two-panel figure (peak memory vs. depth; effective rank vs. depth)
is saved for direct inclusion in the thesis.

RUN ON A COLAB GPU (Runtime -> Change runtime type -> GPU, e.g. L4/T4) for the
real peak-memory numbers. It also runs on CPU, but then peak memory falls back
to a portable analytical estimate (clearly labelled as such).
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- configuration -----------------------------
DEPTHS = [2, 4, 8, 16, 24, 32]   # encoder depth (number of blocks)
SEEDS = [0, 1, 2]                # repeated runs for mean +/- std
C = 128                          # channels per block
HW = 32                          # spatial size (kept constant across blocks)
BATCH = 128                      # training batch size
STEPS = 30                       # optimization steps per run
LATENT = 128                     # embedding dimension
GAMMA = 1.0                      # VICReg variance hinge threshold


def make_blocks(depth):
    stem = nn.Sequential(nn.Conv2d(3, C, 3, stride=2, padding=1),
                         nn.BatchNorm2d(C), nn.ReLU())
    blocks = nn.ModuleList([
        nn.Sequential(nn.Conv2d(C, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU())
        for _ in range(depth)
    ])
    head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(C, LATENT))
    return stem, blocks, head


def local_heads(depth):
    return nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                        nn.Linear(C, LATENT)) for _ in range(depth)])


def vicreg(z, gamma=GAMMA):
    """Variance + covariance penalties CO-LOCATED on the same embedding z."""
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(gamma - std))
    zc = z - z.mean(0, keepdim=True)
    cov = (zc.T @ zc) / (z.size(0) - 1)
    off = (cov * (1 - torch.eye(cov.size(0), device=z.device))).pow(2).sum() / cov.size(0)
    return var_loss + off


def effective_rank(z):
    zc = z - z.mean(0, keepdim=True)
    cov = (zc.T @ zc) / (z.size(0) - 1)
    eig = torch.linalg.eigvalsh(cov).clamp(min=0)
    return (eig.sum() ** 2 / (eig ** 2).sum()).item()


def activation_estimate_mb(depth, mode):
    """Portable float32 activation-memory estimate (CPU fallback only)."""
    per_block = BATCH * C * HW * HW * 4
    total = (depth * per_block) if mode == "e2e" else per_block
    return total / (1024 ** 2)


def run(depth, mode, device, seed):
    torch.manual_seed(seed)
    stem, blocks, head = make_blocks(depth)
    stem, blocks, head = stem.to(device), blocks.to(device), head.to(device)
    params = list(stem.parameters()) + list(blocks.parameters()) + list(head.parameters())
    lheads = None
    if mode == "local":
        lheads = local_heads(depth).to(device)
        params += list(lheads.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(STEPS):
        x = torch.rand(BATCH, 3, 64, 64, device=device)
        opt.zero_grad()
        h = stem(x)
        if mode == "e2e":
            for b in blocks:
                h = b(h)
            z = head(h)
            vicreg(z).backward()          # full-depth graph retained
        else:                              # layer-wise local firebreak
            for i, b in enumerate(blocks):
                h = b(h)
                z = lheads[i](h)
                vicreg(z).backward()       # confined to this block
                h = h.detach()             # firebreak: cut the graph
            _ = head(h.detach())           # final embedding for eval only
        opt.step()

    # Final-embedding effective rank (eval only, no grad)
    with torch.no_grad():
        h = stem(torch.rand(256, 3, 64, 64, device=device))
        for b in blocks:
            h = b(h)
        rank = effective_rank(head(h))

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        measured = True
    else:
        peak_mb = activation_estimate_mb(depth, mode)
        measured = False
    return peak_mb, rank, measured


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = device.type == "cuda"
    print(f"device={device}"
          f"{' ('+torch.cuda.get_device_name(0)+')' if gpu else ''}\n", flush=True)
    if not gpu:
        print("WARNING: no GPU -> peak memory is an ANALYTICAL ESTIMATE, not a "
              "measurement. Use a Colab GPU runtime for real numbers.\n", flush=True)

    results = {}
    for depth in DEPTHS:
        results[depth] = {}
        for mode in ("e2e", "local"):
            peaks, ranks, oom, measured = [], [], False, True
            for seed in SEEDS:
                try:
                    peak, rank, m = run(depth, mode, device, seed)
                    peaks.append(peak); ranks.append(rank); measured = m
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        oom = True
                        if gpu:
                            torch.cuda.empty_cache()
                        break
                    raise
            if oom or not peaks:
                results[depth][mode] = {"oom": True}
                print(f"depth={depth:2d} {mode:5s}  OOM", flush=True)
            else:
                row = {
                    "peak_mb_mean": round(float(np.mean(peaks)), 1),
                    "peak_mb_std": round(float(np.std(peaks)), 1),
                    "eff_rank_mean": round(float(np.mean(ranks)), 2),
                    "eff_rank_std": round(float(np.std(ranks)), 2),
                    "measured": measured,
                }
                results[depth][mode] = row
                print(f"depth={depth:2d} {mode:5s}  peak={row['peak_mb_mean']:8.1f}"
                      f" +/- {row['peak_mb_std']:<5.1f} MB   "
                      f"eff_rank={row['eff_rank_mean']:6.2f} +/- {row['eff_rank_std']:.2f}",
                      flush=True)

    payload = {"config": {"channels": C, "hw": HW, "batch": BATCH, "steps": STEPS,
                          "latent": LATENT, "seeds": SEEDS, "depths": DEPTHS,
                          "device": str(device), "measured_peak": gpu},
               "results": results}
    with open("memory_bounded.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("\nsaved memory_bounded.json", flush=True)

    _plot(results, gpu)
    print("saved memory_bounded.png", flush=True)
    print("\n--- PASTE memory_bounded.json BACK TO CLAUDE ---", flush=True)


def _plot(results, gpu):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depths = DEPTHS
    def series(mode, key):
        return [results[d][mode].get(key, np.nan) for d in depths]

    e2e_mem, loc_mem = series("e2e", "peak_mb_mean"), series("local", "peak_mb_mean")
    e2e_rk, loc_rk = series("e2e", "eff_rank_mean"), series("local", "eff_rank_mean")
    e2e_rk_s, loc_rk_s = series("e2e", "eff_rank_std"), series("local", "eff_rank_std")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    C_E2E, C_LOC = "#D55E00", "#0072B2"  # colorblind-safe

    ax1.plot(depths, e2e_mem, "o-", color=C_E2E, label="End-to-end")
    ax1.plot(depths, loc_mem, "s-", color=C_LOC, label="Layer-wise local")
    ax1.set_xlabel("Encoder depth (blocks)")
    ax1.set_ylabel("Peak training memory (MB)"
                   + ("" if gpu else ", estimated"))
    ax1.set_title("(a) Peak training memory vs. depth")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.errorbar(depths, e2e_rk, yerr=e2e_rk_s, fmt="o-", color=C_E2E, label="End-to-end", capsize=3)
    ax2.errorbar(depths, loc_rk, yerr=loc_rk_s, fmt="s-", color=C_LOC, label="Layer-wise local", capsize=3)
    ax2.axhline(1.0, ls="--", color="gray", lw=1, label="collapse (rank 1)")
    ax2.set_xlabel("Encoder depth (blocks)")
    ax2.set_ylabel("Effective rank of final embedding")
    ax2.set_title("(b) Representation quality vs. depth")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("memory_bounded.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
