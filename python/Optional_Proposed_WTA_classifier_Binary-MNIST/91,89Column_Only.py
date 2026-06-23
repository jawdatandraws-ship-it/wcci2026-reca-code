# Binary-MNIST + ReCA features (no augmentation)
# One-pass uint6 perceptron-style training with tunable step ±x and int6 inference
# NOTE: This version uses COLUMNS-ONLY CA (no row/horizontal processing)

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Reproducibility & timing
# -----------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # for CUDA matmul determinism

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.is_available() and torch.cuda.manual_seed_all(SEED)

# Disallow nondeterministic kernels
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Avoid TF32 variability on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

times = {}
def _tick(): return time.time()
def _tock(t): return time.time() - t

# -----------------------------
# ReCA features (8 iterations)
# COLUMNS-ONLY CA: up/down neighbors only
# -----------------------------
def to_bitplanes(img):
    return (img > 128).astype(np.uint8)[None,:,:]

def rule90_step(arr, axis):
    if axis==0:
        # left/right neighbors (horizontal)
        L=np.zeros_like(arr); L[:,1:]=arr[:,:-1]
        R=np.zeros_like(arr); R[:,:-1]=arr[:,1:]
    else:
        # up/down neighbors (vertical)
        L=np.zeros_like(arr); L[1:,:]=arr[:-1,:]
        R=np.zeros_like(arr); R[:-1,:]=arr[1:,:]
    return np.bitwise_xor(L,R)

def evolve_bitplanes(bitplanes, steps=8):
    # COLUMNS-ONLY: apply rule90 only along columns (up/down), no horizontal processing
    out=[]; cur=bitplanes.copy()
    for _ in range(steps):
        x = np.array([rule90_step(bp, 1) for bp in cur])   # axis=1 => up/down
        out.append(x); cur=x
    return out

def grayscale_reform(e):
    return [layer[0] for layer in e]

def max_pool(im, size=2, stride=2):
    h,w=im.shape; o=np.zeros((h//stride,w//stride),dtype=im.dtype)
    for i in range(0,h,stride):
        for j in range(0,w,stride):
            o[i//stride,j//stride]=im[i:i+size,j:j+size].max()
    return o

def reca_features(img):
    bps=to_bitplanes(img); evo=evolve_bitplanes(bps)
    gs=grayscale_reform(evo); pp=[max_pool(g) for g in gs]
    return np.concatenate([p.ravel() for p in pp])  # 8*14*14=1568

# -----------------------------
# 1) Load MNIST
# -----------------------------
print("1) Loading MNIST…"); _t=_tick()
mnist=fetch_openml('mnist_784',version=1,as_frame=False)
X_imgs=mnist['data'].reshape(-1,28,28).astype(np.uint8)
y_all =mnist['target'].astype(int)
times["1"]=_tock(_t); print("   ✅ Done.", f"\n   ⏱ {times['1']:.3f} s\n")

# -----------------------------
# 2) Extract features
# -----------------------------
print("2) Extracting binary ReCA features (COLUMNS-ONLY CA)…"); _t=_tick()
N=len(X_imgs); X_feat=np.empty((N,8*14*14),dtype=np.float32)
for i,img in enumerate(tqdm(X_imgs,desc="   features")):
    X_feat[i]=reca_features(img)
times["2"]=_tock(_t); print("   ✅ Done.", f"\n   ⏱ {times['2']:.3f} s\n")

# -----------------------------
# 3) Split data
# -----------------------------
print("3) Splitting data…"); _t=_tick()
X_trainval,y_trainval=X_feat[:60000],y_all[:60000]
X_test,y_test=X_feat[60000:],y_all[60000:]
X_train,X_val,y_train,y_val=train_test_split(
    X_trainval,y_trainval,test_size=5000,random_state=SEED,stratify=y_trainval)
times["3"]=_tock(_t); print("   ✅ Done.", f"\n   ⏱ {times['3']:.3f} s\n")

# -----------------------------
# 5) Prepare tensors (no DataLoader)
# -----------------------------
print("5) Preparing tensors (no DataLoader)…"); _t=_tick()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def to_tensor(x,y=None):
    tx=torch.from_numpy(x); return (tx, torch.from_numpy(y).long()) if y is not None else tx
X_tr_t,y_tr_t=to_tensor(X_train,y_train)
X_val_t,y_val_t=to_tensor(X_val,y_val)
X_te_t,y_te_t=to_tensor(X_test,y_test)
times["5"]=_tock(_t); print("   ✅ Done.", f"\n   ⏱ {times['5']:.3f} s\n")

# -----------------------------
# 6) Model (functional): uint6 weights + perceptron-style updates with step ±x
# -----------------------------
print("6) Setting up model…"); _t=_tick()

UPDATE_STEP = 1   # any positive integer (kept fixed for reproducibility)

def create_int6_perceptron(in_dim, num_classes, init_val=32, step=1, use_bias=True, device="cpu"):
    W = torch.full((num_classes, in_dim), init_val, dtype=torch.uint8, device=device)   # [C,D]
    state = {"W": W, "step": int(step), "use_bias": bool(use_bias), "device": device}
    if use_bias:
        b0 = int(max(0, min(64, init_val)))
        state["b"] = torch.full((num_classes,), b0, dtype=torch.uint8, device=device)   # [C]
    return state

def forward_logits(state, x):
    Wf = state["W"].float().to(x.device)
    bf = state["b"].float().to(x.device) if state.get("use_bias", False) else None
    return F.linear(x, Wf, bf)

@torch.no_grad()
def update_batch(state, Xb, yb):
    dev = state["device"]
    W = state["W"]
    step = state["step"]
    use_bias = state["use_bias"]

    x = Xb.to(dev).to(torch.uint8).clamp_(0,1)[0]  # [D] {0,1}
    y = int(yb.item())

    logits = (W.float() @ x.float())
    if use_bias:
        logits += state["b"].float()
    y_hat = int(torch.argmax(logits).item())
    if y_hat == y:
        return

    inc = (x.to(torch.int16) * step)               # [D]

    Wy = W[y].to(torch.int16); Wy.add_(inc).clamp_(0,63)
    Wh = W[y_hat].to(torch.int16); Wh.sub_(inc).clamp_(0,63)
    W[y].copy_(Wy.to(torch.uint8))
    W[y_hat].copy_(Wh.to(torch.uint8))

    if use_bias:
        by = state["b"][y].to(torch.int16)
        bh = state["b"][y_hat].to(torch.int16)
        by = torch.clamp(by + step, 0, 64)
        bh = torch.clamp(bh - step, 0, 64)
        state["b"][y]     = by.to(torch.uint8)
        state["b"][y_hat] = bh.to(torch.uint8)

@torch.no_grad()
def deterministic_bias_offset(state):
    if state.get("use_bias", False):
        C = state["b"].numel()
        pattern = torch.arange(C, device=state["device"], dtype=torch.uint8) % 3
        state["b"].add_(pattern).clamp_(0, 64)

@torch.no_grad()
def accuracy(state, X, y):
    logits = forward_logits(state, X.float().to(state["device"]))
    preds = logits.argmax(1).cpu()
    return (preds == y).float().mean().item()*100.0, logits

model = create_int6_perceptron(
    in_dim=8*14*14, num_classes=10, init_val=32,
    step=UPDATE_STEP, use_bias=True, device=device
)
deterministic_bias_offset(model)

times["6"]=_tock(_t); print("   ✅ Done.", f"\n   ⏱ {times['6']:.3f} s\n")

# -----------------------------
# 7) One-pass training (shuffle once, per-input updates)
# -----------------------------
print("7) One-pass training (per-input updates, shuffled once)…"); _t=_tick()

g_cpu = torch.Generator(device='cpu').manual_seed(SEED)
perm = torch.randperm(X_tr_t.size(0), generator=g_cpu)
X_tr_t, y_tr_t = X_tr_t[perm], y_tr_t[perm]

for i in trange(X_tr_t.size(0), desc="   samples"):
    update_batch(model, X_tr_t[i:i+1], y_tr_t[i:i+1])

with torch.no_grad():
    val_acc, val_logits = accuracy(model, X_val_t, y_val_t)
print(f"   Validation after single pass: {val_acc:.2f}%")
times["7"]=_tock(_t); print("   ✅ Training complete.", f"\n   ⏱ {times['7']:.3f} s\n")

# -----------------------------
# 8) Inference using final int6 weights (no packing)
# -----------------------------
print("9) Evaluating final int6-trained weights on test set…")
Xte, yte = X_te_t.float().to(device), y_te_t.to(device)
t = time.time()
with torch.no_grad():
    logits = forward_logits(model, Xte)
    preds  = logits.argmax(1)
    acc    = (preds == yte).float().mean().item()*100
t = time.time() - t
print(f"   Test Accuracy (final int6 weights): {acc:.2f}%   ⏱ {t:.3f}s\n")

# -----------------------------
# 10) ROC & Confusion Matrix
# -----------------------------
print("10) Computing ROC & Confusion Matrix…")
probs = F.softmax(logits, dim=1).detach().cpu().numpy()
y_test_bin = label_binarize(y_test, classes=np.arange(10))
plt.figure(figsize=(8,6))
for i in range(10):
    fpr,tpr,_ = roc_curve(y_test_bin[:,i], probs[:,i])
    plt.plot(fpr,tpr,label=f"{i}")
plt.plot([0,1],[0,1],'k--'); plt.title("ROC Curves (Final int6-trained weights) — COLUMNS-ONLY CA features")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend(title="Digit", bbox_to_anchor=(1.05,1), loc='upper left'); plt.tight_layout(); plt.show()

cm = confusion_matrix(y_test, preds.detach().cpu().numpy())
fig,ax = plt.subplots(figsize=(8,8))
ConfusionMatrixDisplay(cm).plot(cmap='Blues', ax=ax, colorbar=True)
plt.title("Confusion Matrix (Final int6-trained weights) — COLUMNS-ONLY CA features")
plt.tight_layout(); plt.show()
