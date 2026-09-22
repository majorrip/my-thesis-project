# EdgeVerify: Preventing Representation Collapse in JEPAs for Edge Devices

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **Master's Thesis Research**
> **Author:** Atif Karim (Student ID: 1000055796)
> **Department:** Computer Science and Engineering, BRAC University
> **Supervisor:** Annajiat Alim Rasel

---

## 📌 Overview

**EdgeVerify** is a lightweight, action-conditioned Joint-Embedding Predictive
Architecture (JEPA) for on-device ("Dew"/Edge) self-supervised learning.
Rather than reconstructing raw pixels, it predicts context representations in
an abstract joint-embedding space using an EMA target encoder and an
action-conditioned predictor, with VICReg-style variance/covariance
regularization intended to resist representation collapse.

This repository is the proof-of-concept implementation and experiment suite
for the thesis. Its central finding is a **regularizer-placement principle**:
applying the variance penalty to an embedding *without* a co-located
covariance penalty on the same embedding induces dimensional (rank) collapse
that is invisible to both the training loss and the variance criterion —
and correct placement is what enables collapse-resistant learning, including
under decentralized (FedAvg) training.

---

## ✨ Key results

* **Placement matters:** across the 2×2 ways to place the variance/covariance
  penalties, only *variance on `s_t` without a co-located covariance term*
  collapses (effective rank ≈ 1–2); every other placement preserves rank.
* **Silent collapse:** in the collapsing case the variance loss is `0` and the
  mean embedding std is the *highest* of any configuration, yet the
  representation is rank-1 and least useful on a linear probe.
* **The fix:** co-locating both penalties on the context embedding restores
  effective rank by more than 10× and recovers linear-probe accuracy to the
  unregularized level (digits 0.56 → 0.94; CIFAR-10 0.31 → 0.41).
* **Decentralized transfer:** the fix's collapse resistance survives two-node
  FedAvg training; the flawed placement collapses in both regimes.
* Reproduced across two independent environments and five random seeds.

> Note: measurements are CPU-only; the earlier claim of "drastically reduced
> VRAM" is *not* evaluated here (see the thesis Threats to Validity).

---

## 🛠️ Repository Structure

```text
.
├── code/
│   ├── model.py                # Vision-JEPA architecture + configurable variance/covariance loss
│   ├── dataset.py              # Structured, sklearn-digits, and CIFAR-10 masking loaders
│   ├── train.py                # Minimal training smoke test
│   ├── evaluate.py             # Core protocol: convergence, collapse metrics, efficiency
│   ├── evaluate_multiseed.py   # 5-seed robustness study
│   ├── evaluate_fixes.py       # Shared-embedding fix + linear probe
│   ├── evaluate_placement.py   # Regularizer-placement grid (central finding)
│   └── evaluate_federated.py   # Two-node FedAvg (decentralized) study
├── latex_report/               # LaTeX thesis sources, figures, and compiled main.pdf
├── results/                    # Saved metrics (JSON)
├── notebooks/                  # Colab reproduction notebook
├── requirements.txt            # Python dependency manifest
├── LICENSE                     # MIT License
└── README.md                   # Project documentation
```
