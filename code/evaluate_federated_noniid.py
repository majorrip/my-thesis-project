"""
Non-IID federated extension: many clients with Dirichlet label-skewed shards.

Extends the two-node IID FedAvg study to N clients whose local data is a
label-skewed partition drawn from a Dirichlet(alpha) distribution (small alpha
= strongly non-IID, large alpha ~ IID), with size-weighted FedAvg aggregation.
The question: does the shared-embedding placement remain collapse-resistant as
decentralization becomes harder (more clients, more label skew)?

Reported on the labeled digits set (offline). A ``BACKBONE`` / CIFAR variant is
straightforward to add on Colab by swapping the dataset and backbone.
"""

import copy, json, os
import numpy as np
import torch
from torch.utils.data import Subset

from model import VisionJEPA
from dataset import get_dataset
from evaluate import set_seed
from evaluate_fixes import digits_features_and_labels, linear_probe
from evaluate_federated import _get_online, _set_online, _local_train, ONLINE

SHARED = {"var_on": "context", "cov_on": "context"}
ORIGINAL = {"var_on": "context", "cov_on": "pred"}
# (n_clients, dirichlet_alpha, label)
SETTINGS = [
    (5, 100.0, "5 clients, near-IID"),
    (5, 0.5, "5 clients, moderate non-IID"),
    (10, 0.1, "10 clients, strong non-IID"),
]
ROUNDS, LOCAL_EPOCHS, BATCH = 6, 3, 32


def dirichlet_partition(labels, n_clients, alpha, seed=0):
    rng = np.random.default_rng(seed)
    n_classes = int(labels.max()) + 1
    client_idx = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        props = rng.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
        for cli, part in enumerate(np.split(idx_c, cuts)):
            client_idx[cli].extend(part.tolist())
    return [np.array(x) for x in client_idx]


def _average_weighted(states, sizes):
    total = float(sum(sizes))
    avg = {}
    for n in ONLINE:
        avg[n] = {}
        for k in states[0][n].keys():
            avg[n][k] = sum(sizes[i] * states[i][n][k].float() for i in range(len(states))) / total
    return avg


def federated_train_partitioned(dataset, client_indices, rounds, local_epochs,
                                alpha, beta, cfg, seed=0, backbone="cnn", batch_size=BATCH):
    set_seed(seed)
    clients = [Subset(dataset, idx.tolist()) for idx in client_indices if len(idx) >= batch_size]
    global_model = VisionJEPA(backbone=backbone)
    gstate = _get_online(global_model)
    for _ in range(rounds):
        states, sizes = [], []
        for sub in clients:
            node = VisionJEPA(backbone=backbone)
            _set_online(node, gstate)
            node.target_encoder = copy.deepcopy(node.context_encoder)
            for p in node.target_encoder.parameters():
                p.requires_grad = False
            _local_train(node, sub, local_epochs, alpha, beta, cfg, batch_size=batch_size)
            states.append(_get_online(node)); sizes.append(len(sub))
        gstate = _average_weighted(states, sizes)
    _set_online(global_model, gstate)
    return global_model


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.abspath(os.path.join(here, "..", "results"))
    from sklearn.datasets import load_digits
    ds = get_dataset("digits", seed=0)
    labels = np.array(load_digits().target)
    di, dl = digits_features_and_labels()

    results = {}
    for n_clients, alpha_d, label in SETTINGS:
        parts = dirichlet_partition(labels, n_clients, alpha_d, seed=0)
        sizes = [len(p) for p in parts]
        results[label] = {"n_clients": n_clients, "dirichlet_alpha": alpha_d,
                          "client_sizes": sizes}
        for tag, cfg in {"shared": SHARED, "original": ORIGINAL}.items():
            m = federated_train_partitioned(ds, parts, ROUNDS, LOCAL_EPOCHS, 1.0, 1.0, cfg, seed=0)
            from evaluate import collapse_metrics
            rank = collapse_metrics(m, ds)["effective_rank"]
            acc = linear_probe(m, di, dl)
            results[label][tag] = {"effective_rank": round(rank, 3), "probe_accuracy": round(acc, 4)}
            print(f"[{label:28s}] {tag:9s} eff_rank={rank:6.2f}/256 probe_acc={acc:.3f}", flush=True)

    with open(os.path.join(res_dir, "federated_noniid.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
