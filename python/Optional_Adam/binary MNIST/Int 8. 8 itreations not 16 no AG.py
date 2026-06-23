# Binary-MNIST + ReCA features (NO augmentation) with full reproducibility and timing
# >>> Version with 8 CA iterations <<<

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

import os, random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.quantization import quantize_dynamic

# -----------------------------
# 🔧 Global switch for iterations
# -----------------------------
NUM_ITER = 8  # changed from 16 → 8

# -----------------------------
# Reproducibility switches
# -----------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # CUDA GEMM determinism hint
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

# -----------------------------
# ⏱ Timing helpers
# -----------------------------
times = {}
def _tick(): return time.time()
def _tock(start): return time.time() - start

# -----------------------------
# ReCA feature pipeline for Binary MNIST (1-bit/pixel)
# -----------------------------
def to_bitplanes(img):
    # Convert grayscale [0..255] to pure black/white {0,1} using threshold 128
    return (img > 128).astype(np.uint8)[None, :, :]  # (1,28,28)

def rule90_step(arr, axis):
    # zero-pad shift + XOR neighbors (Rule 90)
    if axis == 0:
        left  = np.zeros_like(arr); left[1: ,:]  = arr[:-1,:]
        right = np.zeros_like(arr); right[:-1,:] = arr[1: ,:]
    else:
        left  = np.zeros_like(arr); left[:,1: ]  = arr[:,:-1]
        right = np.zeros_like(arr); right[:,:-1] = arr[:,1: ]
    return np.bitwise_xor(left, right)

def evolve_bitplanes(bitplanes, steps=NUM_ITER):
    evolved = []
    current = bitplanes.copy()
    for _ in range(steps):
        row_evol = np.array([rule90_step(bp, axis=1) for bp in current])
        col_evol = np.array([rule90_step(bp, axis=0) for bp in current])
        x_k = np.bitwise_xor(row_evol, col_evol)
        evolved.append(x_k)
        current = x_k
    return evolved  # list of `steps` arrays (1,28,28)

def grayscale_reform(evolved_layers):
    # Only one bitplane -> just take it
    return [layer[0] for layer in evolved_layers]  # list of `steps` arrays (28,28)

def max_pool(image, size=2, stride=2):
    h, w = image.shape
    out = np.zeros((h//stride, w//stride), dtype=image.dtype)
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            out[i//stride, j//stride] = image[i:i+size, j:j+size].max()
    return out

def reca_features(img):
    bps = to_bitplanes(img)                 # (1,28,28)
    evo = evolve_bitplanes(bps)             # list of NUM_ITER arrays (1,28,28)
    gs  = grayscale_reform(evo)             # list of NUM_ITER arrays (28,28)
    pp  = [max_pool(g) for g in gs]         # list of NUM_ITER arrays (14,14)
    return np.concatenate([p.ravel() for p in pp]).astype(np.float32)  # (NUM_ITER*14*14,)

# -----------------------------
# 1) Load MNIST
# -----------------------------
print("1) Loading MNIST…")
_t = _tick()
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_imgs = mnist['data'].reshape(-1,28,28).astype(np.uint8)
y_all  = mnist['target'].astype(int)
times["1_load_mnist_s"] = _tock(_t)
print("   ✅ Done.")
print(f"   ⏱ Step 1 time: {times['1_load_mnist_s']:.3f} s\n")

# -----------------------------
# 2) Extract ReCA features (on binary images)
# -----------------------------
print(f"2) Extracting binary ReCA features with {NUM_ITER} iterations…")
_t = _tick()
N = len(X_imgs)
FEAT_DIM = NUM_ITER * 14 * 14  # 8*14*14 = 1568
X_feat = np.empty((N, FEAT_DIM), dtype=np.float32)
for i, img in enumerate(tqdm(X_imgs, desc="   features")):
    X_feat[i] = reca_features(img)
times["2_feature_extract_s"] = _tock(_t)
print("   ✅ Done.")
print(f"   ⏱ Step 2 time: {times['2_feature_extract_s']:.3f} s\n")

# -----------------------------
# 3) Split into train/val/test
# -----------------------------
print("3) Splitting data…")
_t = _tick()
X_trainval, y_trainval = X_feat[:60000], y_all[:60000]
X_test,     y_test     = X_feat[60000:], y_all[60000:]
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=5000, random_state=42, stratify=y_trainval
)
times["3_split_s"] = _tock(_t)
print("   ✅ Done.")
print(f"   ⏱ Step 3 time: {times['3_split_s']:.3f} s\n")

# -----------------------------
# 4) Build DataLoaders
# -----------------------------
print("4) Building DataLoader…")
_t = _tick()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def to_tensor(x,y=None):
    tx = torch.from_numpy(x)
    return (tx, torch.from_numpy(y).long()) if y is not None else tx

X_tr, y_tr = X_train, y_train
X_tr_t, y_tr_t = to_tensor(X_tr, y_tr)
X_val_t, y_val_t = to_tensor(X_val, y_val)
X_te_t, y_te_t = to_tensor(X_test, y_test)

train_ds = TensorDataset(X_tr_t, y_tr_t)
g = torch.Generator().manual_seed(SEED)  # fixed shuffle order
train_loader = DataLoader(train_ds, batch_size=17000, shuffle=True, generator=g)
times["4_build_loader_s"] = _tock(_t)
print("   ✅ Done.")
print(f"   ⏱ Step 4 time: {times['4_build_loader_s']:.3f} s\n")

# -----------------------------
# 5) Define model & optimizer
# -----------------------------
print("5) Setting up model…")
_t = _tick()
class SoftmaxLogistic(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)
    def forward(self, x): return self.linear(x)

model = SoftmaxLogistic(FEAT_DIM, 10).to(device)  # input dim adjusted to 8*14*14
optimizer = optim.Adam(model.parameters(), lr=0.008, weight_decay=0.00012)
criterion = nn.CrossEntropyLoss()
times["5_setup_model_s"] = _tock(_t)
print("   ✅ Done.")
print(f"   ⏱ Step 5 time: {times['5_setup_model_s']:.3f} s\n")

# -----------------------------
# 6) Training loop (NO augmentation)
# -----------------------------
print("6) Training loop…")
t_train_total = _tick()
epoch_times = []
num_epochs = 36
for epoch in trange(1, num_epochs+1, desc="   epochs"):
    t_ep = _tick()
    model.train()
    running = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward(); optimizer.step()
        running += loss.item()*Xb.size(0)
    # validate
    model.eval()
    with torch.no_grad():
        Xv, yv = X_val_t.to(device), y_val_t.to(device)
        val_logits = model(Xv)
        val_acc = (val_logits.argmax(1)==yv).float().mean().item()*100
    ep_time = _tock(t_ep)
    epoch_times.append(ep_time)
    tqdm.write(f"   Epoch {epoch}/{num_epochs}  "
               f"Train Loss={running/len(train_loader.dataset):.4f}  "
               f"Val Acc={val_acc:.2f}%  ⏱ {ep_time:.3f}s")
times["6_train_total_s"] = _tock(t_train_total)
print("   ✅ Training complete.")
print(f"   ⏱ Step 6 total time: {times['6_train_total_s']:.3f} s")
print(f"   ⏱ Step 6 avg epoch time: {np.mean(epoch_times):.3f} s\n")

# -----------------------------
# 7) Post‐training 8-bit quantization
# -----------------------------
print("7) Quantizing weights to int8…")
_t = _tick()
qmodel = quantize_dynamic(model.cpu(), {nn.Linear}, dtype=torch.qint8)  # CPU int8
qmodel.eval()
times["7_quantize_s"] = _tock(_t)
print("   ✅ Quantization done.")
print(f"   ⏱ Step 7 time: {times['7_quantize_s']:.3f} s\n")

# -----------------------------
# 8) Final evaluation on test set (quantized, CPU)
# -----------------------------
print("8) Evaluating quantized model on CPU…")
start_time = time.time()
with torch.no_grad():
    Xte, yte = X_te_t.cpu(), y_te_t.cpu()
    logits = qmodel(Xte)
    preds  = logits.argmax(1)
    test_acc = (preds==yte).float().mean().item()*100
end_time = time.time()
elapsed = end_time - start_time
times["8_eval_cpu_s"] = elapsed
print(f"   Test Accuracy (int8 weights): {test_acc:.2f}%")
print(f"   ⏱ Step 8 time (CPU inference): {times['8_eval_cpu_s']:.3f} s\n")

# -----------------------------
# 9) ROC & Confusion Matrix
# -----------------------------
print("9) Computing ROC & Confusion Matrix…")
_t = _tick()
probs = nn.functional.softmax(logits, dim=1).cpu().numpy()
y_test_bin = label_binarize(y_test, classes=np.arange(10))
y_pred_np  = preds.cpu().numpy()

# ROC Curves
plt.figure(figsize=(8, 6))
for i in range(10):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
    plt.plot(fpr, tpr, label=f"{i}")
plt.plot([0,1],[0,1],'k--')
plt.title(f"ROC Curves (Binary-MNIST ReCA, {NUM_ITER} iters, int8)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(title="Digit", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_np)
fig, ax = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues', ax=ax, colorbar=True)
plt.title(f"Confusion Matrix (Binary-MNIST ReCA, {NUM_ITER} iters, int8)")
plt.tight_layout()
plt.show()

times["9_metrics_plots_s"] = _tock(_t)
print(f"   ⏱ Step 9 time: {times['9_metrics_plots_s']:.3f} s\n")

# -----------------------------
# Summary
# -----------------------------
print("===== Timing Summary (seconds) =====")
for k in [
    "1_load_mnist_s",
    "2_feature_extract_s",
    "3_split_s",
    "4_build_loader_s",
    "5_setup_model_s",
    "6_train_total_s",
    "7_quantize_s",
    "8_eval_cpu_s",
    "9_metrics_plots_s",
]:
    print(f"{k:>26}: {times.get(k, float('nan')):8.3f}")
if "6_train_total_s" in times:
    print(f"{'6_train_avg_epoch_s':>26}: {np.mean(epoch_times):8.3f}")
print("====================================\n")
