"""
Decentralized (multi-agent) training via FedAvg.

Two nodes hold disjoint shards of the data and train locally; after each
communication round their online parameters (context encoder, action encoder,
predictor -- including BatchNorm statistics) are averaged into a global model
and broadcast back. Target encoders are re-synchronised to the global context
encoder at the start of each round and updated by EMA locally.

The question: does the collapse behaviour observed in the centralized setting
survive decentralized averaging? We compare, for the frozen global context
encoder, effective rank and (on digits) linear-probe accuracy across:
  - centralized, shared-embedding objective (reference)
  - federated (2 nodes), shared-embedding objective
  - federated (2 nodes), original wiring
"""

import copy
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from model import VisionJEPA, compute_stable_local_loss
from dataset import get_dataset
from evaluate import collapse_metrics, set_seed, train_model
from evaluate_fixes import digits_features_and_labels, linear_probe

ONLINE = ["context_encoder", "action_encoder", "predictor"]


def _get_online(model):
    return {n: copy.deepcopy(getattr(model, n).state_dict()) for n in ONLINE}


def _set_online(model, states):
    for n in ONLINE:
        getattr(model, n).load_state_dict(states[n])


def _average(states_list):
    avg = {}
    for n in ONLINE:
        avg[n] = {}
        for k in states_list[0][n].keys():
            avg[n][k] = sum(s[n][k].float() for s in states_list) / len(states_list)
    return avg


def _local_train(model, subset, epochs, alpha, beta, cfg, batch_size=64, lr=1e-3, wd=1e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True)
    for _ in range(epochs):
        for partial, full, action in loader:
            opt.zero_grad()
            s_pred, s_tgt, s_t = model(partial, full, action)
            loss = compute_stable_local_loss(s_pred, s_tgt.detach(), s_t,
                                             alpha=alpha, beta=beta,
                                             var_on=cfg["var_on"], cov_on=cfg["cov_on"])
            loss.backward()
            opt.step()
            model.update_target_encoder()
    return model


def federated_train(dataset, n_nodes, rounds, local_epochs, alpha, beta, cfg, seed=0):
    set_seed(seed)
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)
    shards = [Subset(dataset, idx[i::n_nodes].tolist()) for i in range(n_nodes)]
    global_model = VisionJEPA()
    global_states = _get_online(global_model)
    for _ in range(rounds):
        node_states = []
        for shard in shards:
            node = VisionJEPA()
            _set_online(node, global_states)
            node.target_encoder = copy.deepcopy(node.context_encoder)
            for p in node.target_encoder.parameters():
                p.requires_grad = False
            _local_train(node, shard, local_epochs, alpha, beta, cfg)
            node_states.append(_get_online(node))
        global_states = _average(node_states)
    _set_online(global_model, global_states)
    return global_model


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.abspath(os.path.join(here, "..", "results"))
    os.makedirs(res_dir, exist_ok=True)
    digit_imgs, digit_labels = digits_features_and_labels()

    shared = {"var_on": "context", "cov_on": "context"}
    original = {"var_on": "context", "cov_on": "pred"}

    # (name, epochs_centralized, rounds, local_epochs, ds_kwargs)
    plans = [
        ("structured", 36, 6, 6, {"n_samples": 768, "seed": 0}),
        ("digits", 24, 6, 4, {"seed": 0}),
    ]
    results = {}
    for name, cen_ep, rounds, local_ep, kw in plans:
        ds = get_dataset(name, **kw)
        results[name] = {}

        def record(tag, model):
            col = collapse_metrics(model, ds)
            probe = linear_probe(model, digit_imgs, digit_labels) if name == "digits" else None
            results[name][tag] = {"effective_rank": round(col["effective_rank"], 3),
                                  "sigma_bar": round(col["sigma_bar"], 4),
                                  "probe_accuracy": (round(probe, 4) if probe is not None else None)}
            msg = f"[{name:10s}/{tag:22s}] eff_rank={col['effective_rank']:6.2f}/256 sigma_bar={col['sigma_bar']:.3f}"
            if probe is not None:
                msg += f" probe_acc={probe:.3f}"
            print(msg, flush=True)

        m_cen, _, _ = train_model(ds, cen_ep, 1.0, 1.0, seed=0,
                                  var_on="context", cov_on="context")
        record("centralized-shared", m_cen)
        record("federated-shared", federated_train(ds, 2, rounds, local_ep, 1.0, 1.0, shared, seed=0))
        record("federated-original", federated_train(ds, 2, rounds, local_ep, 1.0, 1.0, original, seed=0))

    with open(os.path.join(res_dir, "federated.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
