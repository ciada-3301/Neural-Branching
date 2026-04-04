import pickle
import numpy as np
import warnings
from torchly import Model  # your library

# ── Load test data ─────────────────────────────────────────────────────────────
def load_cifar100(filepath):
    with open(filepath, 'rb') as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            data = pickle.load(f, encoding='bytes')
    return data


test_data  = load_cifar100('c10/test')
meta       = load_cifar100('c10/meta')

test_images  = np.array(test_data[b'data'])
test_fine    = np.array(test_data[b'fine_labels'])
test_coarse  = np.array(test_data[b'coarse_labels'])
fine_names   = meta[b'fine_label_names']
coarse_names = meta[b'coarse_label_names']

# ── Filter ─────────────────────────────────────────────────────────────────────
flower_id = coarse_names.index(b'flowers')
fish_id   = coarse_names.index(b'fish')

your_subclasses = [b'orchid', b'poppy', b'tulip', b'ray', b'shark', b'shark']
subclass_ids    = [fine_names.index(n) for n in your_subclasses]

mask      = np.isin(test_fine, subclass_ids)
X_test    = test_images[mask].astype(np.float32) / 255.0
y_coarse  = test_coarse[mask]

coarse_map = {flower_id: 0, fish_id: 1}
y_test     = np.array([[coarse_map[c]] for c in y_coarse], dtype=np.float32)

# ── Load normalizer & normalize ────────────────────────────────────────────────
X_mean = np.load('parent_X_mean.npy')
X_std  = np.load('parent_X_std.npy')
X_test_norm = (X_test - X_mean) / X_std

# ── Load model ─────────────────────────────────────────────────────────────────
parent = Model.load("parent.pt")

# ── Evaluate ───────────────────────────────────────────────────────────────────
loss, metrics = parent.evaluate(
    [X_test_norm.tolist()],
    y_test.tolist(),
    metrics=['mse', 'mae']
)

print(f"Test Loss (BCE): {loss:.4f}")
print(f"MSE:             {metrics['mse']:.4f}")
print(f"MAE:             {metrics['mae']:.4f}")

# ── Manual accuracy (evaluate() doesn't support binary accuracy yet) ───────────
import torch

preds_raw = parent.predict([X_test_norm.tolist()])        # raw logits (N, 1)
preds_prob = torch.sigmoid(torch.FloatTensor(preds_raw))  # → probabilities
preds_class = (preds_prob > 0.5).numpy().astype(int)      # → 0 or 1

y_true = y_test.astype(int)
accuracy = np.mean(preds_class == y_true)

print(f"Accuracy:        {accuracy * 100:.2f}%")

# ── Per class breakdown ────────────────────────────────────────────────────────
flower_mask = y_true.flatten() == 0
fish_mask   = y_true.flatten() == 1

flower_acc = np.mean(preds_class[flower_mask] == y_true[flower_mask])
fish_acc   = np.mean(preds_class[fish_mask]   == y_true[fish_mask])

print(f"Flower accuracy: {flower_acc * 100:.2f}%")
print(f"Fish accuracy:   {fish_acc   * 100:.2f}%")