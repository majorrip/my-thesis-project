"""
Multi-seed robustness evaluation (addresses the single-seed threat to validity).

For each dataset and each configuration (full objective vs. unregularized),
the model is trained under several random training seeds. Data content is held
fixed (data seed = 0); the varied seed controls weight initialization and
minibatch ordering, isolating training stochasticity. We report the mean and
standard deviation of the collapse indicators across seeds.
"""

import json
import os

import numpy as np
import torch

from model import VisionJEPA  # noqa: F401  (ensures import path is valid)
from dataset import get_dataset
from evaluate import train_model, collapse_metrics

SEEDS = [0, 1, 2, 3, 4]
PLANS = [
    ("structured", 40, {"n_samples": 768, "seed": 0}),
    ("digits", 25, {"seed": 0}),
]
CONFIGS = {"full": (1.0, 0.01), "unregularized": (0.0, 0.0)}


def agg(values):
    a = np.array(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std()), "n": len(a), "values": [round(v, 4) for v in values]}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.abspath(os.path.join(here, "..", "results"))
    img_dir = os.path.abspath(os.path.join(here, "..", "latex_report", "images"))
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    results = {}
    for name, epochs, ds_kwargs in PLANS:
        dataset = get_dataset(name, **ds_kwargs)
        results[name] = {}
        for tag, (alpha, beta) in CONFIGS.items():
            sig, rank, tot = [], [], []
            for seed in SEEDS:
                model, history, _ = train_model(dataset, epochs, alpha, beta, seed=seed)
                col = collapse_metrics(model, dataset)
                sig.append(col["sigma_bar"])
                rank.append(col["effective_rank"])
                tot.append(history[-1]["total"])
                print(f"[{name:10s}/{tag:13s}] seed={seed} "
                      f"sigma_bar={col['sigma_bar']:.4f} eff_rank={col['effective_rank']:.3f} "
                      f"total={history[-1]['total']:.4f}", flush=True)
            results[name][tag] = {
                "sigma_bar": agg(sig),
                "effective_rank": agg(rank),
                "total_loss": agg(tot),
            }

    with open(os.path.join(res_dir, "multiseed.json"), "w") as f:
        json.dump({"seeds": SEEDS, "results": results}, f, indent=2)

    # Figure: mean +/- std bars for effective rank and sigma_bar.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name in results:
        fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
        tags = ["unregularized", "full"]
        colors = ["#c0504d", "#4f81bd"]
        for j, metric, title in [(0, "sigma_bar", r"Mean embedding std $\bar{\sigma}$"),
                                 (1, "effective_rank", "Effective rank")]:
            means = [results[name][t][metric]["mean"] for t in tags]
            stds = [results[name][t][metric]["std"] for t in tags]
            ax[j].bar(tags, means, yerr=stds, capsize=6, color=colors)
            ax[j].set_title(title)
            ax[j].grid(alpha=0.3, axis="y")
        fig.suptitle(f"Robustness over {len(SEEDS)} seeds ({name} data): mean $\\pm$ std")
        fig.tight_layout()
        fig.savefig(os.path.join(img_dir, f"multiseed_{name}.png"), dpi=150)
        plt.close(fig)

    print("\nSUMMARY (mean +/- std over seeds):", flush=True)
    for name in results:
        for tag in CONFIGS:
            r = results[name][tag]
            print(f"  {name:10s} {tag:13s}  "
                  f"sigma_bar={r['sigma_bar']['mean']:.3f}+/-{r['sigma_bar']['std']:.3f}  "
                  f"eff_rank={r['effective_rank']['mean']:.2f}+/-{r['effective_rank']['std']:.2f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
