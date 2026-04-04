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

test_data  = load_cifar100('c10/test')
meta       = load_cifar100('c10/meta')

test_images = np.array(test_data[b'data'])
test_fine   = np.array(test_data[b'fine_labels'])
fine_names  = meta[b'fine_label_names']

# ── Filter to flower subclasses only ────────────────────────────────────────────
flower_subclasses = [b'sunflower', b'poppy', b'tulip']
subclass_ids    = [fine_names.index(n) for n in flower_subclasses]

subclass_map = {
    fine_names.index(b'sunflower'):      0,
    fine_names.index(b'poppy'):    1,
    fine_names.index(b'tulip'): 2
}

label_map = {0: "sunflower", 1: "poppy", 2: "tulip"}

mask   = np.isin(test_fine, subclass_ids)
X_test = test_images[mask].astype(np.float32) / 255.0
y_test = np.array([subclass_map[f] for f in test_fine[mask]], dtype=np.int64)

print(f"Test samples: {len(y_test)}")
print(f"Class distribution: {np.bincount(y_test)}")

# ── Normalize using parent's saved normalizer ──────────────────────────────────
X_mean = np.load('parent_X_mean.npy')
X_std  = np.load('parent_X_std.npy')
X_test_norm = (X_test - X_mean) / X_std

# ── Load models ────────────────────────────────────────────────────────────────
print("\nLoading models...")
parent  = Model.load("models/parent_2.pt")
flowernet = Model.load("models/flowernet_2.pt")
print("Models loaded successfully!")
parent_layer = 1

# ── Predict function ───────────────────────────────────────────────────────────
def predict_flower(raw_image_norm, threshold=96.0, temperature=3.0):
    if raw_image_norm.ndim == 1:
        raw_image_norm = raw_image_norm.reshape(1, -1)

    act         = parent.get_activations([raw_image_norm], layer=parent_layer)
    neuron_vals = list(act.values())[0]

    logits      = flowernet.predict([neuron_vals])

    exp_logits    = np.exp(logits[0] - np.max(logits[0]))
    probabilities = exp_logits / np.sum(exp_logits)

    class_idx  = np.argmax(probabilities)
    confidence = round(float(probabilities[class_idx]) * 100, 2)

    if confidence < threshold:
        return "uncertain", confidence   # ← rejected

    return label_map[class_idx], confidence

# ── BATCH PREDICTION (more efficient) ──────────────────────────────────────────
print("\n" + "="*70)
print("TEST SET EVALUATION")
print("="*70 + "\n")

print("Extracting activations for all test samples...")
all_test_activations = parent.get_activations([X_test_norm], layer=parent_layer)
all_test_activations = list(all_test_activations.values())[0]

print("Running batch prediction...")
all_logits = flowernet.predict([all_test_activations])
all_predicted_classes = np.argmax(all_logits, axis=1)

# Overall accuracy
test_accuracy = np.mean(all_predicted_classes == y_test) * 100
print(f"\n{'Overall Test Accuracy:':<25} {test_accuracy:.2f}%")

# Per-class accuracy
print(f"\n{'Class':<12} {'Correct':<10} {'Total':<10} {'Accuracy'}")
print("-" * 45)
for class_idx, class_name in label_map.items():
    mask = (y_test == class_idx)
    class_acc = np.mean(all_predicted_classes[mask] == y_test[mask]) * 100
    class_total = np.sum(mask)
    class_correct = np.sum(all_predicted_classes[mask] == y_test[mask])
    print(f"{class_name:<12} {class_correct:<10} {class_total:<10} {class_acc:.2f}%")

# ── Confusion Matrix ───────────────────────────────────────────────────────────
print("\n" + "="*70)
print("CONFUSION MATRIX")
print("="*70)

confusion = np.zeros((3, 3), dtype=int)
for true_idx in range(3):
    for pred_idx in range(3):
        mask = (y_test == true_idx)
        confusion[true_idx, pred_idx] = np.sum(all_predicted_classes[mask] == pred_idx)

print(f"\n{'':>12}", end="")
for class_name in ["sunflower", "poppy", "tulip"]:
    print(f"{class_name:>10}", end="")
print("\n" + "-" * 45)

for true_idx, true_class in enumerate(["sunflower", "poppy", "tulip"]):
    print(f"{true_class:>12}", end="")
    for pred_idx in range(3):
        print(f"{confusion[true_idx, pred_idx]:>10}", end="")
    print()

# ── Confidence Analysis ────────────────────────────────────────────────────────
print("\n" + "="*70)
print("CONFIDENCE ANALYSIS")
print("="*70)

# Calculate softmax probabilities for all predictions
exp_logits = np.exp(all_logits - np.max(all_logits, axis=1, keepdims=True))
all_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
max_confidences = np.max(all_probs, axis=1)

correct_mask = (all_predicted_classes == y_test)
correct_confidences = max_confidences[correct_mask]
incorrect_confidences = max_confidences[~correct_mask]

print(f"\nAverage confidence (correct predictions): {np.mean(correct_confidences) * 100:.2f}%")
print(f"Average confidence (incorrect predictions): {np.mean(incorrect_confidences) * 100:.2f}%")
print(f"Overall average confidence: {np.mean(max_confidences) * 100:.2f}%")

# ── Sample predictions ─────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SAMPLE PREDICTIONS (first 15 test samples)")
print("="*70)
print(f"{'#':<5} {'Actual':<12} {'Predicted':<12} {'Confidence':<12} {'Match'}")
print("-" * 55)

for i in range(min(15, len(X_test_norm))):
    pred_label, conf = predict_flower(X_test_norm[i])
    actual = label_map[y_test[i]]

    if pred_label == "uncertain":
        pass
    else:
        match = "✓" if pred_label == actual else "✗"
        print(f"{i+1:<5} {actual:<12} {pred_label:<12} {conf:<12.2f}% {match}")

print("="*70)

accepted  = 0
correct   = 0

for i in range(len(X_test_norm)):
    pred_label, conf = predict_flower(X_test_norm[i])
    actual = label_map[y_test[i]]

    if pred_label == "uncertain":
        continue                          # skip — don't penalize

    accepted += 1
    if pred_label == actual:
        correct += 1

print(f"Accepted predictions : {accepted}/{len(y_test)} ({accepted/len(y_test)*100:.1f}%)")
print(f"Accuracy on accepted : {correct/accepted*100:.2f}%")