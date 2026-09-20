"""
Regularizer-placement study.

The variance and covariance penalties can each be applied to the context
embedding s_t ("context") or the predictor output s_pred ("pred"). This script
sweeps the full 2x2 grid of placements at fixed weights (alpha=beta=1.0) and
reports, for the resulting frozen context encoder: mean embedding std, effective
rank, the final variance-loss term (to show whether the variance criterion is
"satisfied"), and -- on digits -- linear-probe accuracy.

The aim is to determine which placement is responsible for collapse resistance,
turning the earlier "shared-embedding" observation into a systematic result.
"""

import json
import os

import torch

from dataset import get_dataset
from evaluate import train_model, collapse_metrics
from evaluate_fixes import digits_features_and_labels, linear_probe

GRID = [
    ("var s_t, cov s_pred (original)", "context", "pred"),
    ("var s_t, cov s_t (shared)",      "context", "context"),
    ("var s_pred, cov s_pred",         "pred",    "pred"),
    ("var s_pred, cov s_t (swapped)",  "pred",    "context"),
]
PLANS = [
    ("structured", 40, {"n_samples": 768, "seed": 0}),
    ("digits", 25, {"seed": 0}),
]
ALPHA = BETA = 1.0


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.abspath(os.path.join(here, "..", "results"))
    os.makedirs(res_dir, exist_ok=True)
    digit_imgs, digit_labels = digits_features_and_labels()

    results = {}
    for name, epochs, kw in PLANS:
        dataset = get_dataset(name, **kw)
        results[name] = {}
        for label, von, con in GRID:
            model, history, _ = train_model(dataset, epochs, ALPHA, BETA, seed=0,
                                            var_on=von, cov_on=con)
            col = collapse_metrics(model, dataset)
            probe = linear_probe(model, digit_imgs, digit_labels) if name == "digits" else None
            results[name][label] = {
                "var_on": von, "cov_on": con,
                "sigma_bar": round(col["sigma_bar"], 4),
                "effective_rank": round(col["effective_rank"], 3),
                "final_var_loss": round(history[-1]["var"], 4),
                "probe_accuracy": (round(probe, 4) if probe is not None else None),
            }
            msg = (f"[{name:10s}] {label:32s} sigma_bar={col['sigma_bar']:.3f} "
                   f"eff_rank={col['effective_rank']:6.2f}/256 var_loss={history[-1]['var']:.3f}")
            if probe is not None:
                msg += f" probe_acc={probe:.3f}"
            print(msg, flush=True)

    with open(os.path.join(res_dir, "placement.json"), "w") as f:
        json.dump({"alpha": ALPHA, "beta": BETA, "results": results}, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
