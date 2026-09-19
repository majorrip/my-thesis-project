# EdgeVerify: A Decentralized Multi-Agent JEPA Framework for Localized Edge Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **Bachelor's Thesis Research**  
> **Author:** Atif Karim (Student ID: 1000055796)  
> **Department:** Computer Science and Engineering, BRAC University  
> **Supervisor:** Annajiat Alim Rasel  

---

## 📌 Overview

**EdgeVerify** is a lightweight, decentralized Joint-Embedding Predictive Architecture (I-JEPA) optimized for on-device ("Dew" and Edge computing) execution. 

Unlike traditional generative edge models that reconstruct high-dimensional raw inputs (e.g., pixel-by-pixel or token-by-token) or rely on cloud backbones, EdgeVerify operates entirely inside an abstract **joint-embedding latent space**. By combining **local layer-wise gradient isolation (`.detach()`)** with **VicReg-inspired variance/covariance regularization**, EdgeVerify prevents latent representation collapse while drastically reducing peak VRAM overhead.

This repository serves as the official proof-of-concept (POC) implementation, benchmark suite, and documentation for the thesis.

---

## ✨ Key Features

* **Pure Latent-Space Prediction:** Simulates visual/contextual transformations inside a compact vector space without costly decoding steps.
* **On-Device Data Sovereignty:** Keeps raw telemetry local, eliminating third-party cloud dependence and reducing the network attack surface.
* **Anti-Collapse Shield:** Integrates dynamic hinge-loss variance thresholds and feature decorrelation constraints to preserve embedding entropy without negative sample pairing.
* **Layer-Wise Distillation:** Isolates local gradient graphs to enable multi-layer training on memory-constrained micro-architectures.

---

## 🛠️ Repository Structure

```text
.
├── code/
│   ├── model.py            # Vision JEPA architecture & loss routines
│   ├── dataset.py          # Custom dataset generators & masking loaders
│   ├── train.py            # Multi-epoch execution & validation loop
│   └── evaluate.py         # Benchmarks (latency, memory, loss curves)
├── latex_report/           # Complete LaTeX thesis write-up source files
├── docs/                   # Architectural diagrams & evaluation plots
├── requirements.txt        # Python dependency manifest
├── LICENSE                 # MIT License
└── README.md               # Project documentation
