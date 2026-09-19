"""
Evaluation protocol for EdgeVerify.

Runs the collapse ablation (full VICReg-style objective vs. an unregularized
alpha=beta=0 baseline) on each available dataset, and measures:

  RQ1 (collapse) : mean per-dimension embedding std (sigma_bar) and the
                   effective rank (participation ratio) of the context-embedding
                   covariance. Higher is healthier; collapse drives both to ~0/1.
  RQ2 (efficiency): parameter count, parameter memory, single-sample inference
                   latency (mean +/- std), and peak process RSS during training.
  RQ3 (convergence): per-epoch loss components.

Outputs a results JSON and PNG figures for inclusion in the thesis.
"""

import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import VisionJEPA, compute_stable_local_loss
from dataset import get_dataset


def _rss_mb():
    """Resident set size of this process in MB, read from /proc/self/status."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0  # kB -> MB
    except FileNotFoundError:
        pass
    return float("nan")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_model(dataset, epochs, alpha, beta, batch_size=64, lr=1e-3, wd=1e-4, seed=0):
    set_seed(seed)
    device = torch.device("cpu")
    model = VisionJEPA().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    history = []
    peak_rss = _rss_mb()
    for epoch in range(epochs):
        agg = {"distill": 0.0, "var": 0.0, "cov": 0.0, "total": 0.0}
        n = 0
        for partial, full, action in loader:
            partial, full, action = partial.to(device), full.to(device), action.to(device)
            optimizer.zero_grad()
            s_pred, s_tgt, s_t = model(partial, full, action)
            loss, comps = compute_stable_local_loss(
                s_pred, s_tgt.detach(), s_t,
                variance_threshold=1.0, alpha=alpha, beta=beta, return_components=True)
            loss.backward()
            optimizer.step()
            model.update_target_encoder()
            for k in agg:
                agg[k] += comps[k]
            n += 1
            peak_rss = max(peak_rss, _rss_mb())
        history.append({k: agg[k] / max(n, 1) for k in agg})
    return model, history, peak_rss


@torch.no_grad()
def collapse_metrics(model, dataset, max_samples=512):
    model.eval()
    device = torch.device("cpu")
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    embs = []
    seen = 0
    for partial, full, action in loader:
        s_t = model.context_encoder(partial.to(device))
        embs.append(s_t)
        seen += s_t.size(0)
        if seen >= max_samples:
            break
    E = torch.cat(embs, dim=0)[:max_samples]                 # (M, d)
    sigma_bar = torch.sqrt(E.var(dim=0) + 1e-8).mean().item()
    Ec = E - E.mean(dim=0, keepdim=True)
    cov = (Ec.T @ Ec) / (E.size(0) - 1)
    eig = torch.linalg.eigvalsh(cov).clamp(min=0)
    s1, s2 = eig.sum().item(), (eig ** 2).sum().item()
    eff_rank = (s1 ** 2) / s2 if s2 > 0 else 0.0             # participation ratio
    return {"sigma_bar": sigma_bar, "effective_rank": eff_rank, "latent_dim": E.size(1)}


@torch.no_grad()
def efficiency_metrics(model, n_runs=50, warmup=10):
    model.eval()
    device = torch.device("cpu")
    n_params = sum(p.numel() for p in model.parameters())
    param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    partial = torch.rand(1, 3, 64, 64)
    full = torch.rand(1, 3, 64, 64)
    action = torch.rand(1, 4)
    for _ in range(warmup):
        model(partial, full, action)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model(partial, full, action)
        times.append((time.perf_counter() - t0) * 1000.0)     # ms
    times = np.array(times)
    return {
        "params": int(n_params),
        "param_memory_mb": round(param_mb, 3),
        "latency_ms_mean": round(float(times.mean()), 3),
        "latency_ms_std": round(float(times.std()), 3),
    }


def run_dataset(name, epochs, ds_kwargs, out_img_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset = get_dataset(name, **ds_kwargs)
    results = {}
    curves = {}
    for tag, (alpha, beta) in {"full": (1.0, 0.01), "unregularized": (0.0, 0.0)}.items():
        model, history, peak_rss = train_model(dataset, epochs, alpha, beta, seed=0)
        col = collapse_metrics(model, dataset)
        eff = efficiency_metrics(model)
        results[tag] = {
            "alpha": alpha, "beta": beta,
            "final_loss": history[-1],
            "collapse": col,
            "efficiency": eff,
            "peak_rss_mb": round(peak_rss, 1),
        }
        curves[tag] = history

    # Loss-curve figure (total loss per epoch, both configs).
    plt.figure(figsize=(6, 4))
    for tag in curves:
        plt.plot([h["total"] for h in curves[tag]], label=tag, linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel(r"$\mathcal{L}_{total}$")
    plt.title(f"Convergence on {name} data"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(out_img_dir, f"loss_curves_{name}.png")
    plt.savefig(loss_path, dpi=150); plt.close()

    return results, loss_path


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.abspath(os.path.join(here, "..", "latex_report", "images"))
    res_dir = os.path.abspath(os.path.join(here, "..", "results"))
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    all_results = {}
    plans = [
        ("structured", 40, {"n_samples": 768, "seed": 0}),
        ("digits", 25, {"seed": 0}),
    ]
    for name, epochs, kw in plans:
        print(f"\n=== Running {name} (epochs={epochs}) ===", flush=True)
        res, _ = run_dataset(name, epochs, kw, img_dir)
        all_results[name] = res
        for tag, r in res.items():
            c, e = r["collapse"], r["efficiency"]
            print(f"  [{tag:13s}] total={r['final_loss']['total']:.4f} "
                  f"sigma_bar={c['sigma_bar']:.4f} eff_rank={c['effective_rank']:.2f}/"
                  f"{c['latent_dim']} lat={e['latency_ms_mean']:.2f}ms "
                  f"params={e['params']:,} peakRSS={r['peak_rss_mb']:.0f}MB", flush=True)

    # Collapse comparison bar chart (structured dataset).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name in all_results:
        r = all_results[name]
        fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
        tags = ["unregularized", "full"]
        ax[0].bar(tags, [r[t]["collapse"]["sigma_bar"] for t in tags],
                  color=["#c0504d", "#4f81bd"])
        ax[0].set_title(r"Mean embedding std $\bar{\sigma}$")
        ax[1].bar(tags, [r[t]["collapse"]["effective_rank"] for t in tags],
                  color=["#c0504d", "#4f81bd"])
        ax[1].set_title("Effective rank")
        for a in ax:
            a.grid(alpha=0.3, axis="y")
        fig.suptitle(f"Collapse indicators ({name} data)")
        fig.tight_layout()
        fig.savefig(os.path.join(img_dir, f"collapse_bars_{name}.png"), dpi=150)
        plt.close(fig)

    with open(os.path.join(res_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results.json and figures to {res_dir} and {img_dir}", flush=True)


if __name__ == "__main__":
    main()
