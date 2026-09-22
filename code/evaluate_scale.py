"""
Larger-scale study: ResNet-18 backbone on the full CIFAR-10 benchmark.

Runs the regularizer-placement grid (plus an unregularized baseline) with a
ResNet-18 context encoder on all 50k CIFAR-10 training images, then evaluates
each configuration with a linear probe on the CIFAR-10 test set. Uses a GPU
when available. This is written for Colab; it is not run in the offline
environment (CIFAR-10 cannot be downloaded there).

Tune the knobs at the top: EPOCHS, SEEDS, BACKBONE. Defaults are modest for a
first pass -- increase EPOCHS/SEEDS for publication-grade numbers.
"""

import json, os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import VisionJEPA, compute_stable_local_loss
from dataset import _apply_mask

# ---- knobs ----
EPOCHS = 20
SEEDS = [0]
BACKBONE = "resnet18"          # or "cnn" to compare
BATCH = 256
SIZE = 64
PROBE_TRAIN, PROBE_TEST = 5000, 2000
CONFIGS = {
    "unregularized":       dict(alpha=0.0, beta=0.0, var_on="context", cov_on="pred"),
    "original(var s_t,cov s_pred)": dict(alpha=1.0, beta=1.0, var_on="context", cov_on="pred"),
    "shared(var s_t,cov s_t)":      dict(alpha=1.0, beta=1.0, var_on="context", cov_on="context"),
    "both s_pred":         dict(alpha=1.0, beta=1.0, var_on="pred", cov_on="pred"),
    "swapped(var s_pred,cov s_t)":  dict(alpha=1.0, beta=1.0, var_on="pred", cov_on="context"),
}


class CIFARScaleDataset(Dataset):
    """Lazy full-CIFAR JEPA dataset: masks on the fly to avoid a 2GB precompute."""
    def __init__(self, train=True, size=SIZE, mask_seed=0):
        import torchvision
        tv = torchvision.datasets.CIFAR10(root="./cifar", train=train, download=True)
        self.data = tv.data                      # uint8 (N,32,32,3)
        self.labels = np.array(tv.targets)
        self.size = size
        self.mask_seed = mask_seed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img = torch.from_numpy(self.data[i]).float().permute(2, 0, 1) / 255.0
        img = F.interpolate(img.unsqueeze(0), size=(self.size, self.size),
                            mode="bilinear", align_corners=False)[0]
        rng = np.random.default_rng(self.mask_seed * 100003 + i)
        partial, action = _apply_mask(img, rng)
        return partial, img, self.labels[i], action


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)


def train_jepa(dataset, epochs, cfg, backbone, device, seed=0):
    set_seed(seed)
    model = VisionJEPA(backbone=backbone).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, drop_last=True, num_workers=2)
    for ep in range(epochs):
        for partial, full, _y, action in loader:
            partial, full, action = partial.to(device), full.to(device), action.to(device)
            opt.zero_grad()
            s_pred, s_tgt, s_t = model(partial, full, action)
            loss = compute_stable_local_loss(s_pred, s_tgt.detach(), s_t,
                                             alpha=cfg["alpha"], beta=cfg["beta"],
                                             var_on=cfg["var_on"], cov_on=cfg["cov_on"])
            loss.backward(); opt.step(); model.update_target_encoder()
    return model


@torch.no_grad()
def features(model, dataset, device, n):
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
    feats, labs, seen = [], [], 0
    for partial, full, y, action in loader:
        feats.append(model.context_encoder(full.to(device)).cpu())
        labs.append(y)
        seen += full.size(0)
        if seen >= n:
            break
    return torch.cat(feats)[:n], torch.cat(labs)[:n].numpy()


def effective_rank(F_):
    Fc = F_ - F_.mean(0, keepdim=True)
    cov = (Fc.T @ Fc) / (F_.size(0) - 1)
    eig = torch.linalg.eigvalsh(cov).clamp(min=0)
    s1, s2 = eig.sum().item(), (eig ** 2).sum().item()
    return (s1 ** 2) / s2 if s2 > 0 else 0.0


def probe(train_feats, train_labels, test_feats, test_labels):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(train_feats.numpy())
    clf = LogisticRegression(max_iter=3000).fit(sc.transform(train_feats.numpy()), train_labels)
    return float(clf.score(sc.transform(test_feats.numpy()), test_labels))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, backbone={BACKBONE}, epochs={EPOCHS}, seeds={SEEDS}\n", flush=True)
    train_ds = CIFARScaleDataset(train=True)
    test_ds = CIFARScaleDataset(train=False)

    results = {}
    for tag, cfg in CONFIGS.items():
        ranks, accs = [], []
        for s in SEEDS:
            model = train_jepa(train_ds, EPOCHS, cfg, BACKBONE, device, seed=s)
            tf, tl = features(model, train_ds, device, PROBE_TRAIN)
            ef, el = features(model, test_ds, device, PROBE_TEST)
            ranks.append(effective_rank(tf))
            accs.append(probe(tf, tl, ef, el))
        results[tag] = {"eff_rank_mean": round(float(np.mean(ranks)), 3),
                        "eff_rank_std": round(float(np.std(ranks)), 3),
                        "probe_mean": round(float(np.mean(accs)), 4),
                        "probe_std": round(float(np.std(accs)), 4)}
        r = results[tag]
        print(f"[{tag:30s}] eff_rank={r['eff_rank_mean']:6.2f}+/-{r['eff_rank_std']:.2f} "
              f"probe_acc={r['probe_mean']:.3f}+/-{r['probe_std']:.3f}", flush=True)

    with open("scale_cifar_resnet18.json", "w") as f:
        json.dump({"epochs": EPOCHS, "seeds": SEEDS, "backbone": BACKBONE, "results": results}, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
