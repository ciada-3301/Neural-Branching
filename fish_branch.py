import pickle
import numpy as np
import warnings
from torchly import Model
import torch

# ── Load data ──────────────────────────────────────────────────────────────────
def load_cifar100(filepath):
    with open(filepath, 'rb') as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            data = pickle.load(f, encoding='bytes')
    return data

train_data = load_cifar100('c10/train')
meta       = load_cifar100('c10/meta')

images    = np.array(train_data[b'data'])
fine      = np.array(train_data[b'fine_labels'])
fine_names = meta[b'fine_label_names']

# ── Filter to fish subclasses only ────────────────────────────────────────────
fish_subclasses = [b'crab', b'aquarium_fish', b'shark']
subclass_ids    = [fine_names.index(n) for n in fish_subclasses]

subclass_map = {
    fine_names.index(b'shark'):      0,
    fine_names.index(b'aquarium_fish'):    1,
    fine_names.index(b'crab'): 2
}

mask   = np.isin(fine, subclass_ids)
X      = images[mask].astype(np.float32) / 255.0
y_fine = np.array([subclass_map[f] for f in fine[mask]], dtype=np.int64).flatten()

print(f"Total samples: {len(y_fine)}")
print(f"Class distribution: {np.bincount(y_fine)}")

# ── IMPROVEMENT 1: Train/Validation Split (80/20) ──────────────────────────────
from sklearn.model_selection import train_test_split

# Stratified split to maintain class balance
X_train, X_val, y_train, y_val = train_test_split(
    X, y_fine, test_size=0.2, random_state=42, stratify=y_fine
)

print(f"\nTraining samples: {len(y_train)}")
print(f"Validation samples: {len(y_val)}")
print(f"Train class distribution: {np.bincount(y_train)}")
print(f"Val class distribution: {np.bincount(y_val)}")

# ── Normalize using parent's saved normalizer ──────────────────────────────────
X_mean = np.load('parent_X_mean.npy')
X_std  = np.load('parent_X_std.npy')

X_train_norm = (X_train - X_mean) / X_std
X_val_norm = (X_val - X_mean) / X_std

# ── Load parent and extract layer 0 activations ────────────────────────────────
parent = Model.load("models/parent_2.pt")

print("\nExtracting activations from parent layer 0...")
train_activations = parent.get_activations([X_train_norm], layer=1)
train_activations = list(train_activations.values())[0]  # (N_train, 1024)

val_activations = parent.get_activations([X_val_norm], layer=1)
val_activations = list(val_activations.values())[0]  # (N_val, 1024)

print(f"Train activations shape: {train_activations.shape}")
print(f"Val activations shape: {val_activations.shape}")

# ── Create fishnet model ───────────────────────────────────────────────────────
print("\nCreating fishnet model with anti-overfitting measures...")
fishnet = Model(
    [512, 256, 3],
    activation="relu",
    dropout=0.5,          # less dropout — limited samples
    lr=0.001
)

# ── WORKAROUND: Manual training loop with proper validation ────────────────────
print("\nTraining fishnet with manual validation monitoring...")

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Prepare data
X_train_tensor = torch.FloatTensor(train_activations).to(fishnet.device)
y_train_tensor = torch.LongTensor(y_train).to(fishnet.device)

X_val_tensor = torch.FloatTensor(val_activations).to(fishnet.device)
y_val_tensor = torch.LongTensor(y_val).to(fishnet.device)

dataset = TensorDataset(X_train_tensor, y_train_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

criterion = nn.CrossEntropyLoss()
# Note: L2 regularization is applied via weight_decay in optimizer
optimizer = optim.Adam(fishnet.network.parameters(), lr=0.0003, weight_decay=0.01)

# Training loop with early stopping
epochs = 500
patience = 50
best_val_loss = float('inf')
patience_counter = 0
best_epoch = 0

print(f"\n{'Epoch':<10} {'Train Loss':<15} {'Val Loss':<15} {'Train Acc':<15} {'Val Acc':<15}")
print("-" * 70)

for epoch in range(epochs):
    # Training phase
    fishnet.network.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = fishnet.network(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_X.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += batch_y.size(0)
        train_correct += (predicted == batch_y).sum().item()
    
    train_loss = train_loss / train_total
    train_acc = 100.0 * train_correct / train_total
    
    # Validation phase
    fishnet.network.eval()
    with torch.no_grad():
        val_outputs = fishnet.network(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor).item()
        _, val_predicted = torch.max(val_outputs.data, 1)
        val_correct = (val_predicted == y_val_tensor).sum().item()
        val_acc = 100.0 * val_correct / len(y_val)
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"{epoch+1:<10} {train_loss:<15.4f} {val_loss:<15.4f} {train_acc:<15.2f}% {val_acc:<15.2f}%")
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        patience_counter = 0
        # Save best model state
        best_model_state = fishnet.network.state_dict().copy()
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
        print(f"✓ Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
        # Restore best model
        fishnet.network.load_state_dict(best_model_state)
        break

# Final evaluation
print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

# Training set
fishnet.network.eval()
with torch.no_grad():
    train_outputs = fishnet.network(X_train_tensor)
    _, train_predicted = torch.max(train_outputs.data, 1)
    final_train_acc = 100.0 * (train_predicted == y_train_tensor).sum().item() / len(y_train)

# Validation set
with torch.no_grad():
    val_outputs = fishnet.network(X_val_tensor)
    _, val_predicted = torch.max(val_outputs.data, 1)
    final_val_acc = 100.0 * (val_predicted == y_val_tensor).sum().item() / len(y_val)

print(f"\nFinal Training Accuracy:   {final_train_acc:.2f}%")
print(f"Final Validation Accuracy: {final_val_acc:.2f}%")

gap = final_train_acc - final_val_acc
print(f"\nTrain-Val Gap: {gap:.2f}%")

if gap > 10:
    print("⚠️  Warning: Large train-val gap indicates overfitting")
elif gap > 5:
    print("⚠️  Moderate train-val gap, model might be slightly overfit")
else:
    print("✓ Good generalization!")

# Save model
fishnet.save("models/fishnet_2.pt")
print("\n✓ Fishnet saved successfully!")

# ── Per-class validation accuracy ──────────────────────────────────────────────
print("\n" + "="*70)
print("PER-CLASS VALIDATION PERFORMANCE")
print("="*70)

label_names = {0: "crab", 1: "aquarium_fish", 2: "shark"}
print(f"\n{'Class':<12} {'Correct':<10} {'Total':<10} {'Accuracy'}")
print("-" * 45)

val_predicted_np = val_predicted.cpu().numpy()
y_val_np = y_val_tensor.cpu().numpy()

for class_idx, class_name in label_names.items():
    mask = (y_val_np == class_idx)
    if np.sum(mask) > 0:
        class_correct = np.sum(val_predicted_np[mask] == y_val_np[mask])
        class_total = np.sum(mask)
        class_acc = (class_correct / class_total * 100)
        print(f"{class_name:<12} {class_correct:<10} {class_total:<10} {class_acc:.2f}%")

print("="*70)