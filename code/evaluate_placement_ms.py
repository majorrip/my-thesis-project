"""Multi-seed regularizer-placement study (5 seeds). Aggregates effective rank
and (digits) linear-probe accuracy as mean +/- std across seeds."""
import json, os
import numpy as np
from dataset import get_dataset
from evaluate import train_model, collapse_metrics
from evaluate_fixes import digits_features_and_labels, linear_probe

GRID = [
    ("var s_t, cov s_pred (original)", "context", "pred"),
    ("var s_t, cov s_t (shared)",      "context", "context"),
    ("var s_pred, cov s_pred",         "pred",    "pred"),
    ("var s_pred, cov s_t (swapped)",  "pred",    "context"),
]
PLANS = [("structured", 40, {"n_samples": 768, "seed": 0}), ("digits", 25, {"seed": 0})]
SEEDS = [0, 1, 2, 3, 4]
A = B = 1.0


def agg(v):
    a = np.array(v, float)
    return {"mean": round(float(a.mean()), 3), "std": round(float(a.std()), 3),
            "values": [round(x, 3) for x in v]}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.abspath(os.path.join(here, "..", "results"))
    di, dl = digits_features_and_labels()
    results = {}
    for name, ep, kw in PLANS:
        ds = get_dataset(name, **kw)
        results[name] = {}
        for label, von, con in GRID:
            ranks, accs = [], []
            for s in SEEDS:
                m, h, _ = train_model(ds, ep, A, B, seed=s, var_on=von, cov_on=con)
                ranks.append(collapse_metrics(m, ds)["effective_rank"])
                if name == "digits":
                    accs.append(linear_probe(m, di, dl))
            results[name][label] = {"effective_rank": agg(ranks),
                                    "probe_accuracy": (agg(accs) if name == "digits" else None)}
            r = results[name][label]
            msg = f"[{name:10s}] {label:32s} eff_rank={r['effective_rank']['mean']:6.2f}+/-{r['effective_rank']['std']:.2f}"
            if accs:
                msg += f" probe={r['probe_accuracy']['mean']:.3f}+/-{r['probe_accuracy']['std']:.3f}"
            print(msg, flush=True)
    with open(os.path.join(res_dir, "placement_ms.json"), "w") as f:
        json.dump({"seeds": SEEDS, "alpha": A, "beta": B, "results": results}, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
