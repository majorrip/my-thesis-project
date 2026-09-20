"""
Evaluates the two experimental fixes:

  1. Shared-embedding variant: apply the covariance penalty to the context
     embedding s_t (same tensor as the variance penalty) instead of to the
     prediction s_pred, testing whether it restores effective rank.
  2. Linear probe: freeze the trained context encoder and fit a linear
     classifier on labels (digits), measuring whether the representation is
     actually useful for a downstream task -- a ground-truth signal the
     intrinsic collapse metrics lack.

Three configurations are compared on structured and digits data:
  - unregularized (alpha=beta=0)
  - full          (original wiring: variance on s_t, covariance on s_pred)
  - shared        (variance and covariance both on s_t)
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from dataset import get_dataset
from evaluate import train_model, collapse_metrics

CONFIGS = {
    "unregularized": dict(alpha=0.0, beta=0.0, shared_cov=False),
    "full":          dict(alpha=1.0, beta=0.01, shared_cov=False),
    "shared":        dict(alpha=1.0, beta=1.0, shared_cov=True),
}
PLANS = [
    ("structured", 40, {"n_samples": 768, "seed": 0}),
    ("digits", 25, {"seed": 0}),
]


def digits_features_and_labels(size=64):
    from sklearn.datasets import load_digits
    d = load_digits()
    imgs = torch.from_numpy(d.images.astype("float32") / 16.0).unsqueeze(1)
    imgs = F.interpolate(imgs, size=(size, size), mode="bilinear", align_corners=False)
    imgs = imgs.repeat(1, 3, 1, 1)
    return imgs, d.target


@torch.no_grad()
def linear_probe(model, imgs, labels, seed=0):
    """Freeze encoder, extract context features on full images, fit a linear classifier."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    model.eval()
    feats = model.context_encoder(imgs).numpy()
    Xtr, Xte, ytr, yte = train_test_split(feats, labels, test_size=0.3,
                                          random_state=seed, stratify=labels)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000)
    clf.fit(scaler.transform(Xtr), ytr)
    return float(clf.score(scaler.transform(Xte), yte))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.abspath(os.path.join(here, "..", "results"))
    os.makedirs(res_dir, exist_ok=True)

    digit_imgs, digit_labels = digits_features_and_labels()
    results = {}
    for name, epochs, ds_kwargs in PLANS:
        dataset = get_dataset(name, **ds_kwargs)
        results[name] = {}
        for tag, cfg in CONFIGS.items():
            model, history, _ = train_model(dataset, epochs, cfg["alpha"], cfg["beta"],
                                            seed=0, shared_cov=cfg["shared_cov"])
            col = collapse_metrics(model, dataset)
            probe = linear_probe(model, digit_imgs, digit_labels) if name == "digits" else None
            results[name][tag] = {
                "config": cfg,
                "sigma_bar": round(col["sigma_bar"], 4),
                "effective_rank": round(col["effective_rank"], 3),
                "probe_accuracy": (round(probe, 4) if probe is not None else None),
            }
            msg = (f"[{name:10s}/{tag:13s}] sigma_bar={col['sigma_bar']:.3f} "
                   f"eff_rank={col['effective_rank']:.2f}/256")
            if probe is not None:
                msg += f"  probe_acc={probe:.3f}"
            print(msg, flush=True)

    with open(os.path.join(res_dir, "fixes.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
