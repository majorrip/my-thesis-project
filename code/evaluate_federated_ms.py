"""Multi-seed (3 seeds) version of the decentralized study. Aggregates
effective rank and (digits) probe accuracy as mean +/- std across seeds for
centralized-shared, federated-shared, and federated-original."""
import json, os
import numpy as np

from dataset import get_dataset
from evaluate import train_model, collapse_metrics
from evaluate_fixes import digits_features_and_labels, linear_probe
from evaluate_federated import federated_train

SEEDS = [0, 1, 2]
SHARED = {"var_on": "context", "cov_on": "context"}
ORIGINAL = {"var_on": "context", "cov_on": "pred"}
# (name, centralized_epochs, rounds, local_epochs, ds_kwargs)
PLANS = [
    ("structured", 36, 6, 6, {"n_samples": 768, "seed": 0}),
    ("digits", 24, 6, 4, {"seed": 0}),
]


def agg(v):
    a = np.array(v, float)
    return {"mean": round(float(a.mean()), 3), "std": round(float(a.std()), 3),
            "values": [round(x, 3) for x in v]}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.abspath(os.path.join(here, "..", "results"))
    di, dl = digits_features_and_labels()
    results = {}
    for name, cen_ep, rounds, local_ep, kw in PLANS:
        ds = get_dataset(name, **kw)
        results[name] = {}

        def run(tag, builder):
            ranks, accs = [], []
            for s in SEEDS:
                model = builder(s)
                ranks.append(collapse_metrics(model, ds)["effective_rank"])
                if name == "digits":
                    accs.append(linear_probe(model, di, dl))
            results[name][tag] = {"effective_rank": agg(ranks),
                                  "probe_accuracy": (agg(accs) if name == "digits" else None)}
            r = results[name][tag]
            msg = f"[{name:10s}/{tag:20s}] eff_rank={r['effective_rank']['mean']:6.2f}+/-{r['effective_rank']['std']:.2f}"
            if accs:
                msg += f" probe={r['probe_accuracy']['mean']:.3f}+/-{r['probe_accuracy']['std']:.3f}"
            print(msg, flush=True)

        run("centralized-shared", lambda s: train_model(ds, cen_ep, 1.0, 1.0, seed=s,
                                                        var_on="context", cov_on="context")[0])
        run("federated-shared", lambda s: federated_train(ds, 2, rounds, local_ep, 1.0, 1.0, SHARED, seed=s))
        run("federated-original", lambda s: federated_train(ds, 2, rounds, local_ep, 1.0, 1.0, ORIGINAL, seed=s))

    with open(os.path.join(res_dir, "federated_ms.json"), "w") as f:
        json.dump({"seeds": SEEDS, "results": results}, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
