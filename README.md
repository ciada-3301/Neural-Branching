# 🌿 Neural Branching

> *A framework for hierarchical neural network specialization via hidden-layer activation branching.*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![CIFAR-100](https://img.shields.io/badge/Dataset-CIFAR--100-green.svg)](https://www.cs.toronto.edu/~kriz/cifar.html)
[![Torchly](https://img.shields.io/badge/Framework-Torchly-orange.svg)](https://github.com/ciada-3301/Torchly)

---

## What is Neural Branching?

Traditional neural networks use the entire model to classify every single input — even when you only care about a narrow subclass. That's computationally wasteful and architecturally inelegant.

**Neural Branching** is a lightweight architecture pattern where *specialized child networks* branch off from a hidden layer of a trained parent network. Instead of retraining from scratch or fine-tuning the whole model, branch networks take the **intermediate activations** of the parent as their input — inheriting the parent's learned feature space and specializing further on top of it.

```
Input Image
     │
     ▼
┌─────────────┐
│   PARENT    │  [3072 → 1024 → 512 → 6]
│   NETWORK   │  (learns shared features)
└──────┬──────┘
       │  hidden layer activations (512-dim)
   ┌───┴───┐
   ▼       ▼
┌──────┐ ┌──────────┐
│FISH  │ │ FLOWER   │
│ NET  │ │   NET    │
│[512→ │ │  [512→   │
│256→3]│ │  256→3]  │
└──────┘ └──────────┘
  crab     sunflower
  fish     poppy
  shark    tulip
```

---

## Architecture

### Parent Network
- **Input:** Flattened CIFAR-100 images (32×32×3 = 3072)
- **Architecture:** `[3072 → 1024 → 512 → 6]`
- **Role:** Learns a generalized feature representation across all 6 classes. The parent's accuracy isn't the goal — its *hidden activations* are.
- **Branching point:** 2nd hidden layer (512 neurons) — low-level enough to be general, high-level enough to be meaningful.

### Branch Networks
Both branches share the same architecture: `[512 → 256 → 3]`

| Branch | Classes | Model File |
|---|---|---|
| **Fishnet** | Crab, Aquarium Fish, Shark | `models/fishnet_2.pt` |
| **Flowernet** | Sunflower, Poppy, Tulip | `models/flowernet_2.pt` |

Branches are trained on activations extracted from the parent's layer 1 (512-dim), with the parent frozen.

---

## Results

### Fishnet
| Metric | Value |
|---|---|
| Total Test Samples | 300 |
| Accepted Predictions | 250 (83.3%) |
| Overall Accuracy | 75.33% |
| Accuracy on Accepted | **80.00%** |

### Flowernet
| Metric | Value |
|---|---|
| Total Test Samples | 300 |
| Accepted Predictions | 224 (74.7%) |
| Overall Accuracy | 67.33% |
| Accuracy on Accepted | **74.11%** |

> Flowernet underperformed relative to Fishnet — likely due to low visual diversity in flower images at CIFAR-100's 32×32 resolution, where the dominant blue/green hues across flower classes reduce discriminability.

---

## Dataset

**CIFAR-100** — 2 superclasses, 3 subclasses each:

- 🌸 **Flowers:** Sunflower, Poppy, Tulip
- 🐟 **Fish:** Crab, Aquarium Fish, Shark

Only these 6 subclasses are used. Images are normalized using the parent's saved mean and standard deviation (`parent_X_mean.npy`, `parent_X_std.npy`).

---

## Project Structure

```
Neural-Branching/
├── c10/                    # CIFAR-100 dataset (train, test, meta)
├── models/
│   ├── parent_2.pt         # Trained parent network
│   ├── fishnet_2.pt        # Trained fish branch
│   └── flowernet_2.pt      # Trained flower branch
├── parent_branch.py        # Train the parent network
├── parent_test.py          # Evaluate parent network
├── fish_branch.py          # Train Fishnet (branch from parent)
├── fishtest.py             # Evaluate Fishnet on test set
├── flower_branch.py        # Train Flowernet (branch from parent)
├── flowertest.py           # Evaluate Flowernet on test set
├── parent_X_mean.npy       # Normalization mean (saved from parent training)
├── parent_X_std.npy        # Normalization std (saved from parent training)
└── torchly.py              # Torchly framework (local copy)
```

---

## Getting Started

### Prerequisites
```bash
pip install torch numpy scikit-learn
```

Or install [Torchly](https://github.com/ciada-3301/Torchly) directly.

### Training Pipeline

Run in order:

**Step 1 — Train the parent:**
```bash
python parent_branch.py
```
This trains the parent network on all 6 classes and saves `parent_X_mean.npy`, `parent_X_std.npy`, and `models/parent_2.pt`.

**Step 2 — Train the branches:**
```bash
python fish_branch.py
python flower_branch.py
```
Each branch script loads the frozen parent, extracts layer-1 activations from the relevant subset, and trains a small specialized network on top.

**Step 3 — Evaluate:**
```bash
python fishtest.py
python flowertest.py
```

---

## How It Works — The Key Idea

```python
# 1. Extract activations from parent's hidden layer
activations = parent.get_activations([X_norm], layer=1)  # (N, 512)

# 2. Train branch network on those activations
branch = Model([512, 256, 3], activation="relu", dropout=0.5)
# ... train branch using activations as inputs ...

# 3. At inference: pipe input through parent → branch
act = parent.get_activations([image_norm], layer=1)
prediction = branch.predict([act])
```

The branch never sees raw pixels — only the parent's *interpretation* of them.

---

## Built With

- **[Torchly](https://github.com/ciada-3301/Torchly)** — A minimal PyTorch wrapper for rapid neural network prototyping, built for this project.
- **PyTorch** — Core deep learning backend
- **CIFAR-100** — Benchmark dataset with natural hierarchical class structure

---

## Key Observations

1. **A neural network can learn from another network's interpretation** — branches don't need the parent's output, just its hidden-layer activations.
2. **Feature reuse without fine-tuning** — branch networks are small and cheap to train since the heavy lifting of feature extraction is already done.
3. **Composable specialization** — this pattern extends naturally: branches can themselves be branched for tasks requiring deeper hierarchy (e.g., *what crop? → what disease? → where?*).

---

## Future Work

- More efficient feature extraction from parent layers
- Extending the architecture to transformers — branches emerging from intermediate transformer layers for things like self-evaluating answer quality
- Dynamic routing: automatically selecting which branch handles a given input

---

## Author

**Arkadyuti Maiti**

---

## License

MIT
