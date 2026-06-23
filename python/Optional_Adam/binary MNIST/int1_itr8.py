# Binary-MNIST + ReCA features (no augmentation) with timing
# Post-training INT1 weight-only quantization with true packing + cached fast path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from scipy.ndimage import map_coordinates, gaussian_filter  # (elastic not used here)

import os, random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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
# ReCA feature pipeline for binary MNIST (1-bit/pixel)
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

def evolve_bitplanes(bitplanes, steps=8):  # ReCA = 8 iterations
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
    evo = evolve_bitplanes(bps)             # list of 8 arrays (1,28,28)
    gs  = grayscale_reform(evo)             # list of 8 arrays (28,28)
    pp  = [max_pool(g) for g in gs]         # list of 8 arrays (14,14)
    return np.concatenate([p.ravel() for p in pp])  # (1568,)

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
X_feat = np.empty((N, 8*14*14), dtype=np.float32)
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
# 4) (Removed) No augmentation — pure MNIST only
# -----------------------------

# -----------------------------
# 5) Build DataLoaders
# -----------------------------
print("5) Building DataLoader…")
_t = _tick()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def to_tensor(x,y=None):
    tx = torch.from_numpy(x)
    return (tx, torch.from_numpy(y).long()) if y is not None else tx

X_tr_t, y_tr_t = to_tensor(X_train, y_train)
X_val_t, y_val_t = to_tensor(X_val, y_val)
X_te_t,  y_te_t  = to_tensor(X_test, y_test)

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

model = SoftmaxLogistic(8*14*14, 10).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.008, weight_decay=0.00012)
criterion = nn.CrossEntropyLoss()
times["6_setup_model_s"] = _tock(_t)
print("   ✅ Done.")
print(f"   ⏱ Step 6 time: {times['6_setup_model_s']:.3f} s\n")

# -----------------------------
# 7) Training loop (no augmentation)
# -----------------------------
print("7) Training loop (no augmentation, pure MNIST)…")
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
    tqdm.write(f"   Epoch {epoch}/{num_epochs}  Train Loss={running/len(train_loader.dataset):.4f}  "
               f"Val Acc={val_acc:.2f}%  ⏱ {ep_time:.3f}s")
times["7_train_total_s"] = _tock(t_train_total)
print("   ✅ Training complete.")
print(f"   ⏱ Step 7 total time: {times['7_train_total_s']:.3f} s")
print(f"   ⏱ Step 7 avg epoch time: {np.mean(epoch_times):.3f} s\n")

# -----------------------------
# 8) Post‐training INT1 weight-only quantization
#     with TRUE bit packing + cached dequant fast path
# -----------------------------
print("8) Quantizing weights to INT1 (per-channel, symmetric) with true bit packing…")
_t = _tick()

def _signed_range(bits: int):
    # ✅ minimal change: allow bits=1
    assert 1 <= bits <= 7, "Supported bits: 1..7"
    if bits == 1:
        # signed 1-bit: {-1, +1}
        return -1, 1
    qmin = -(1 << (bits - 1))
    qmax =  (1 << (bits - 1)) - 1
    return qmin, qmax

def quantize_linear_weights_to_intN(W: torch.Tensor, bits: int):
    assert W.dim() == 2, "Expected [out_features, in_features]"
    qmin, qmax = _signed_range(bits)
    max_abs = W.abs().amax(dim=1)  # [out]
    scales = torch.where(max_abs > 0, max_abs / qmax, torch.ones_like(max_abs))
    q = torch.round(W / scales.unsqueeze(1)).clamp(qmin, qmax).to(torch.int8)
    return q, scales.to(torch.float32)

def pack_intN_rows(qW_int8: torch.Tensor, bits: int):
    """
    Bitstream pack a [out,in] int8 matrix with 'bits' per value into uint8 bytes.
    """
    assert qW_int8.dim() == 2
    qmin, qmax = _signed_range(bits)
    offset = -qmin
    out_features, in_features = qW_int8.shape
    qU = (qW_int8.to(torch.int32) + offset)  # 0..(2^bits-1)
    out_bytes = (in_features * bits + 7) // 8
    packed = torch.empty((out_features, out_bytes), dtype=torch.uint8)

    mask = (1 << bits) - 1
    for r in range(out_features):
        row = qU[r]
        bitbuf = 0
        bitcnt = 0
        byte_idx = 0
        for v in row:
            v = int(v.item()) & mask
            bitbuf |= (v << bitcnt)
            bitcnt += bits
            while bitcnt >= 8:
                packed[r, byte_idx] = torch.tensor(bitbuf & 0xFF, dtype=torch.uint8)
                bitbuf >>= 8
                bitcnt -= 8
                byte_idx += 1
        if bitcnt > 0:
            packed[r, byte_idx] = torch.tensor(bitbuf & 0xFF, dtype=torch.uint8)
    return packed, in_features

def unpack_intN_rows(packed: torch.Tensor, in_features: int, bits: int) -> torch.Tensor:
    """
    Unpack bitstream bytes [out, nbytes] -> int8 [out, in_features] in [qmin,qmax].
    """
    qmin, qmax = _signed_range(bits)
    offset = -qmin
    packed = packed.cpu()
    out_features, nbytes = packed.shape
    out = torch.empty((out_features, in_features), dtype=torch.int8)

    mask = (1 << bits) - 1
    for r in range(out_features):
        bitbuf = 0
        bitcnt = 0
        byte_idx = 0
        vals = []
        while len(vals) < in_features:
            while bitcnt < bits and byte_idx < nbytes:
                bitbuf |= int(packed[r, byte_idx].item()) << bitcnt
                bitcnt += 8
                byte_idx += 1
            if bitcnt < bits:
                bitbuf |= 0 << bitcnt
                bitcnt = bits
            v = bitbuf & mask
            bitbuf >>= bits
            bitcnt  -= bits
            vals.append(v)
        row_int8 = (torch.tensor(vals, dtype=torch.int32) - offset).to(torch.int8)
        out[r] = row_int8
    return out

class PackedIntNLinear(nn.Module):
    """
    nn.Linear replacement with:
      - Per-out-channel symmetric int{bits} quantization
      - TRUE packed storage (bitstream in uint8)
      - Optional cached dequantized FP32 weight for speed
    """
    def __init__(self, float_linear: nn.Linear, bits: int, cache_dequant: bool = True):
        super().__init__()
        # ✅ minimal change: allow bits=1
        assert 1 <= bits <= 7, "Supported bits: 1..7"
        self.bits = int(bits)
        with torch.no_grad():
            W = float_linear.weight.detach().cpu()
            qW, scales = quantize_linear_weights_to_intN(W, self.bits)
            packed, in_features = pack_intN_rows(qW, self.bits)
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales.to(torch.float32))
        self.in_features = int(in_features)
        self.out_features = int(scales.numel())
        self.cache_dequant = bool(cache_dequant)

        if float_linear.bias is not None:
            self.bias = nn.Parameter(float_linear.bias.detach().cpu().clone())
        else:
            self.bias = None

        self.register_buffer("_cached_W_deq", None, persistent=False)
        self._cached_device = None

    def _get_W_deq(self, device: torch.device) -> torch.Tensor:
        if self.cache_dequant and self._cached_W_deq is not None and self._cached_device == device:
            return self._cached_W_deq
        qW_int8 = unpack_intN_rows(self.packed, self.in_features, self.bits)
        W_deq = (qW_int8.to(torch.float32) * self.scales.unsqueeze(1)).to(device)
        if self.cache_dequant:
            self._cached_W_deq = W_deq
            self._cached_device = device
        return W_deq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        if self.bias is not None and self.bias.device != device:
            self.bias = self.bias.to(device)
        W_deq = self._get_W_deq(device)
        return F.linear(x, W_deq, self.bias)

def quantize_model_to_packed_intN(model_fp32: nn.Module, bits: int, cache_dequant: bool = True) -> nn.Module:
    model_cpu = model_fp32.cpu()
    if isinstance(model_cpu, nn.Linear):
        return PackedIntNLinear(model_cpu, bits=bits, cache_dequant=cache_dequant)
    for name, child in list(model_cpu.named_children()):
        setattr(model_cpu, name, quantize_model_to_packed_intN(child, bits, cache_dequant))
    return model_cpu

qmodel1 = quantize_model_to_packed_intN(model, bits=1, cache_dequant=True).eval()
times["8_quantize_s"] = _tock(_t)
print("   ✅ Quantization + packing done for INT1.")
print(qmodel1)
print(f"   ⏱ Step 8 time: {times['8_quantize_s']:.3f} s\n")

# -----------------------------
# 9) Final evaluation on test set (INT1, CPU)
# -----------------------------
print("9) Evaluating INT1-packed model on CPU…")
Xte_cpu, yte_cpu = X_te_t.cpu(), y_te_t.cpu()

t1 = time.time()
with torch.no_grad():
    logits1 = qmodel1(Xte_cpu)
    preds1  = logits1.argmax(1)
    acc1 = (preds1 == yte_cpu).float().mean().item() * 100
t1 = time.time() - t1

times["9_eval_cpu_s"] = t1
print(f"   Test Accuracy (INT1-packed): {acc1:.2f}%   ⏱ {t1:.3f}s\n")

logits = logits1
preds  = preds1

# -----------------------------
# 10) ROC & confusion (visualization)
# -----------------------------
print("10) Computing ROC & Confusion Matrix…")
_t = _tick()
probs = nn.functional.softmax(logits, dim=1).cpu().numpy()
y_test_bin = label_binarize(y_test, classes=np.arange(10))
y_pred_np  = preds.cpu().numpy()

plt.figure(figsize=(8, 6))
for i in range(10):
    fpr, tpr, _ = roc_curve(y_test_bin[:,i], probs[:,i])
    plt.plot(fpr, tpr, label=f"{i}")
plt.plot([0,1],[0,1],'k--')
plt.title("ROC Curves (INT1-packed)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(title="Digit", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test, y_pred_np)
fig, ax = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues', ax=ax, colorbar=True)
plt.title("Confusion Matrix (INT1-packed)")
plt.tight_layout()
plt.show()

times["10_metrics_plots_s"] = _tock(_t)
print(f"   ⏱ Step 10 time: {times['10_metrics_plots_s']:.3f} s\n")
