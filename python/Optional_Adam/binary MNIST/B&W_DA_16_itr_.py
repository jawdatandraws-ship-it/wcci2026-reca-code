# Binary-MNIST + ReCA features with full reproducibility and timing tooling

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from scipy.ndimage import map_coordinates, gaussian_filter

import os, random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.quantization import quantize_dynamic

# -----------------------------
#  Reproducibility switches
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
#  Timing helpers
# -----------------------------
times = {}
def _tick(): return time.time()
def _tock(start): return time.time() - start

# -----------------------------
# (Re)define ReCA feature pipeline for binary MNIST (1-bit/pixel)
# -----------------------------
def to_bitplanes(img):
    # Convert to pure black/white, 0 or 1 (threshold at 128)
    return (img > 128).astype(np.uint8)[None, :, :]  # Shape: (1,28,28)

def rule90_step(arr, axis):
    # zero-pad shift
    if axis == 0:
        left  = np.zeros_like(arr); left[:,1:]  = arr[:,:-1]
        right = np.zeros_like(arr); right[:,:-1]= arr[:,1:]
    else:
        left  = np.zeros_like(arr); left[1:,:]  = arr[:-1,:]
        right = np.zeros_like(arr); right[:-1,:]= arr[1:,:]
    return np.bitwise_xor(left, right)

def evolve_bitplanes(bitplanes, steps=16):
    evolved = []
    current = bitplanes.copy()
    for _ in range(steps):
        # rows then columns, zero-pad
        row_evol = np.array([rule90_step(bp, axis=1) for bp in current])
        col_evol = np.array([rule90_step(bp, axis=0) for bp in current])
        x_k = np.bitwise_xor(row_evol, col_evol)
        evolved.append(x_k)
        current = x_k
    return evolved

def grayscale_reform(evolved_layers):
    # Only one bitplane: just return it as is
    return [layer[0] for layer in evolved_layers]

def max_pool(image, size=2, stride=2):
    h, w = image.shape
    out = np.zeros((h//stride, w//stride), dtype=image.dtype)
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            out[i//stride, j//stride] = image[i:i+size, j:j+size].max()
    return out

def reca_features(img):
    bps = to_bitplanes(img)                 # (1,28,28)
    evo = evolve_bitplanes(bps)             # list of 16 arrays (1,28,28)
    gs  = grayscale_reform(evo)             # list of 16 arrays (28,28)
    pp  = [max_pool(g) for g in gs]         # list of 16 arrays (14,14)
    return np.concatenate([p.ravel() for p in pp])  # (3136,)

# -----------------------------
# Elastic-distortion augmentation (deterministic)
# -----------------------------
def elastic_distort(image, alpha=36, sigma=6, rng=None):
    rs = rng or np.random.RandomState(SEED)
    shape = image.shape
    dx = gaussian_filter((rs.rand(*shape)*2 - 1), sigma) * alpha
    dy = gaussian_filter((rs.rand(*shape)*2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    coords = (y + dy).reshape(-1), (x + dx).reshape(-1)
    return map_coordinates(image, coords, order=1, mode='reflect').reshape(shape)

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
print("2) Extracting binary ReCA features…")
_t = _tick()
N = len(X_imgs)
X_feat = np.empty((N, 16*14*14), dtype=np.float32)
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
# 4) Precompute elastic distortions (on binary images)
# -----------------------------
print("4) Precomputing distortions…")
_t_total = _tick()
orig_imgs, orig_lbls = X_imgs[:60000], y_all[:60000]
aug_sets = []
rng_aug = np.random.RandomState(SEED)
per_set_times = []
for s in range(2):
    t_set = _tick()
    Xa = np.empty((55000,16*14*14), dtype=np.float32)
    ya = np.empty(55000, dtype=int)
    for j in tqdm(range(55000), desc=f"   set {s+1}/2"):
        idx = rng_aug.randint(60000)
        # Distort and re-binarize (deterministic rng)
        distorted = elastic_distort(orig_imgs[idx], rng=rng_aug)
        distorted_bin = (distorted > 128).astype(np.uint8)
        Xa[j] = reca_features(distorted_bin)
        ya[j] = orig_lbls[idx]
    aug_sets.append((Xa, ya))
    per_set_times.append(_tock(t_set))
times["4_precompute_total_s"] = _tock(_t_total)
print("   ✅ Done.")
for i, t in enumerate(per_set_times, 1):
    print(f"   ⏱ Step 4 set {i} time: {t:.3f} s")
print(f"   ⏱ Step 4 total time: {times['4_precompute_total_s']:.3f} s\n")

# -----------------------------
# 5) Build DataLoaders (initial)
# -----------------------------
print("5) Building DataLoader…")
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
g = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(train_ds, batch_size=17000, shuffle=True, generator=g)
times["5_build_loader_s"] = _tock(_t)
print("   ✅ Done.")
print(f"   ⏱ Step 5 time: {times['5_build_loader_s']:.3f} s\n")

# -----------------------------
# 6) Define model & optimizer
# -----------------------------
print("6) Setting up model…")
_t = _tick()
class SoftmaxLogistic(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)
    def forward(self, x): return self.linear(x)

model = SoftmaxLogistic(16*14*14, 10).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.008, weight_decay=0.00012)
criterion = nn.CrossEntropyLoss()
times["6_setup_model_s"] = _tock(_t)
print("   ✅ Done.")
print(f"   ⏱ Step 6 time: {times['6_setup_model_s']:.3f} s\n")

# -----------------------------
# 7) Training with DA at epochs 11 & 14
# -----------------------------
print("7) Training loop…")
t_train_total = _tick()
epoch_times = []
num_epochs = 36
for epoch in trange(1, num_epochs+1, desc="   epochs"):
    t_ep = _tick()
    if epoch in (11,14):
        idx = 0 if epoch==11 else 1
        Xa, ya = aug_sets[idx]
        X_tr = np.vstack([X_tr, Xa]); y_tr = np.hstack([y_tr, ya])
        X_tr_t, y_tr_t = to_tensor(X_tr, y_tr)
        train_ds = TensorDataset(X_tr_t, y_tr_t)
        train_loader = DataLoader(train_ds, batch_size=17000, shuffle=True, generator=g)
        tqdm.write(f"   Added 55k distorted samples at epoch {epoch}")
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
    tqdm.write(f"   Epoch {epoch}/{num_epochs}  Train Loss={running/len(train_loader.dataset):.4f}  Val Acc={val_acc:.2f}%  ⏱ {ep_time:.3f}s")
times["7_train_total_s"] = _tock(t_train_total)
print("   ✅ Training complete.")
print(f"   ⏱ Step 7 total time: {times['7_train_total_s']:.3f} s")
print(f"   ⏱ Step 7 avg epoch time: {np.mean(epoch_times):.3f} s\n")

# -----------------------------
# 8) Post‐training 8-bit quantization
# -----------------------------
print("8) Quantizing weights to int8…")
_t = _tick()
qmodel = quantize_dynamic(model.cpu(), {nn.Linear}, dtype=torch.qint8)  # force CPU for int8
qmodel.eval()
times["8_quantize_s"] = _tock(_t)
print("   ✅ Quantization done.")
print(f"   ⏱ Step 8 time: {times['8_quantize_s']:.3f} s\n")

# -----------------------------
# 9) Final evaluation on test set (quantized, CPU)
# -----------------------------
print("9) Evaluating quantized model on CPU…")
start_time = time.time()
with torch.no_grad():
    Xte, yte = X_te_t.cpu(), y_te_t.cpu()  # CPU tensors for quantized model
    logits = qmodel(Xte)
    preds  = logits.argmax(1)
    test_acc = (preds==yte).float().mean().item()*100
end_time = time.time()
elapsed = end_time - start_time
times["9_eval_cpu_s"] = elapsed
print(f"   Test Accuracy (int8 weights): {test_acc:.2f}%")
print(f"   ⏱ Step 9 time (CPU inference): {times['9_eval_cpu_s']:.3f} s\n")

# -----------------------------
# 10) ROC & confusion (improved visualization)
# -----------------------------
print("10) Computing ROC & Confusion Matrix…")
_t = _tick()
probs = nn.functional.softmax(logits, dim=1).cpu().numpy()
y_test_bin = label_binarize(y_test, classes=np.arange(10))
y_pred_np  = preds.cpu().numpy()

# ---- ROC Curve
plt.figure(figsize=(8, 6))
for i in range(10):
    fpr, tpr, _ = roc_curve(y_test_bin[:,i], probs[:,i])
    plt.plot(fpr, tpr, label=f"{i}")
plt.plot([0,1],[0,1],'k--')
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(title="Digit", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.show()

# ---- Confusion Matrix
cm = confusion_matrix(y_test, y_pred_np)
fig, ax = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues', ax=ax, colorbar=True)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

times["10_metrics_plots_s"] = _tock(_t)
print(f"   ⏱ Step 10 time: {times['10_metrics_plots_s']:.3f} s\n")

# -----------------------------
# Summary
# -----------------------------
print("===== Timing Summary (seconds) =====")
for k in [
    "1_load_mnist_s",
    "2_feature_extract_s",
    "3_split_s",
    "4_precompute_total_s",
    "5_build_loader_s",
    "6_setup_model_s",
    "7_train_total_s",
    "8_quantize_s",
    "9_eval_cpu_s",
    "10_metrics_plots_s",
]:
    print(f"{k:>26}: {times.get(k, float('nan')):8.3f}")
if "4_precompute_total_s" in times:
    for i, t in enumerate(per_set_times, 1):
        print(f"   4_precompute_set_{i}_s: {t:8.3f}")
if "7_train_total_s" in times:
    print(f"{'7_train_avg_epoch_s':>26}: {np.mean(epoch_times):8.3f}")
print("====================================\n")
