import pickle
import numpy as np
import warnings
from torchly import Model

# ── Load data ──────────────────────────────────────────────────────────────────
def load_cifar100(filepath):
    with open(filepath, 'rb') as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            data = pickle.load(f, encoding='bytes')
    return data

train_data = load_cifar100('c10/train')
test_data  = load_cifar100('c10/test')
meta       = load_cifar100('c10/meta')

images       = np.array(train_data[b'data'])
fine         = np.array(train_data[b'fine_labels'])
coarse       = np.array(train_data[b'coarse_labels'])
fine_names   = meta[b'fine_label_names']
coarse_names = meta[b'coarse_label_names']

# ── Filter to flowers & fish only ─────────────────────────────────────────────
flower_id = coarse_names.index(b'flowers')
fish_id   = coarse_names.index(b'fish')

your_subclasses = [b'sunflower', b'poppy', b'tulip', b'crab', b'aquarium_fish', b'shark']
subclass_ids    = [fine_names.index(n) for n in your_subclasses]

mask     = np.isin(fine, subclass_ids)
X        = images[mask].astype(np.float32) / 255.0   # (N, 3072)
subclass_map = {
    fine_names.index(b'sunflower'):     0,
    fine_names.index(b'poppy'):          1,
    fine_names.index(b'tulip'):         2,
    fine_names.index(b'crab'):          3,
    fine_names.index(b'aquarium_fish'): 4,
    fine_names.index(b'shark'):      5
}

y_fine = np.array([subclass_map[f] for f in fine[mask]], dtype=np.int64)
# ── Normalize inputs ───────────────────────────────────────────────────────────
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_std[X_std == 0] = 1                  # avoid division by zero
X_norm = (X - X_mean) / X_std

# Save normalizer params —  need these when running the full branch later
np.save('parent_X_mean.npy', X_mean)
np.save('parent_X_std.npy',  X_std)

# ── Train parent network ───────────────────────────────────────────────────────
parent = Model(
    [3072, 1024, 512, 6],
    activation="relu",
    dropout=0.3,
    lr=0.001
)

parent.train(
    [X_norm.tolist()],
    y_fine,
    epochs=200,
    batch_size=64,
    loss="cross_entropy",      # binary classification — flower vs fish
    verbose=1
)

parent.save("parent_2.pt")
print("Parent branch saved.")