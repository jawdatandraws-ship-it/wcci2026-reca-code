# Binary-MNIST + ReCA features with reproducibility, timing
# FIXED to match CODE 3 feature pipeline exactly

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange

import os, random, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.quantization import quantize_dynamic

# -----------------------------
#  Reproducibility switches (same as code 3)
# -----------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# ReCA feature pipeline (MATCH CODE 3)
# -----------------------------
def to_bitplanes(img):
    return (img > 128).astype(np.uint8)[None, :, :]  # (1,28,28)

def rule90_step(arr, axis):
    # EXACTLY like code 3
    if axis == 0:
        left  = np.zeros_like(arr); left[1: ,:]  = arr[:-1,:]   # shift up/down
        right = np.zeros_like(arr); right[:-1,:] = arr[1: ,:]
    else:
        left  = np.zeros_like(arr); left[:,1: ]  = arr[:,:-1]   # shift left/right
        right = np.zeros_like(arr); right[:,:-1] = arr[:,1: ]
    return np.bitwise_xor(left, right)

def evolve_bitplanes(bitplanes, steps):
    evolved = []
    current = bitplanes.copy()
    for _ in range(steps):
        row_evol = np.array([rule90_step(bp, axis=1) for bp in current])
        col_evol = np.array([rule90_step(bp, axis=0) for bp in current])
        x_k = np.bitwise_xor(row_evol, col_evol)
        evolved.append(x_k)
        current = x_k
    return evolved  # list length = steps, each is (1,28,28)

def grayscale_reform(evolved_layers):
    return [layer[0] for layer in evolved_layers]  # list of (28,28)

def max_pool(image, size=2, stride=2):
    h, w = image.shape
    out = np.zeros((h//stride, w//stride), dtype=image.dtype)
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            out[i//stride, j//stride] = image[i:i+size, j:j+size].max()
    return out

def reca_features(img, num_steps):
    bps = to_bitplanes(img)
    evo = evolve_bitplanes(bps, steps=num_steps)   # EXACTLY num_steps
    gs  = grayscale_reform(evo)
    pp  = [max_pool(g) for g in gs]
    return np.concatenate([p.ravel() for p in pp]).astype(np.float32)

# -----------------------------
# Load MNIST ONCE (same data as code 3)
# -----------------------------
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X_imgs = mnist["data"].reshape(-1, 28, 28).astype(np.uint8)
y_all  = mnist["target"].astype(int)

# -----------------------------
# Core experiment function
# -----------------------------
class SoftmaxLogistic(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        return self.linear(x)

def run_experiment(num_steps):
    print(f"\n===== Running with {num_steps} CA iterations kept =====")
    feat_dim = num_steps * 14 * 14

    # 1) Features (match code 3)
    N = len(X_imgs)
    X_feat = np.empty((N, feat_dim), dtype=np.float32)
    for i, img in enumerate(tqdm(X_imgs, desc="   features")):
        X_feat[i] = reca_features(img, num_steps)

    # 2) Split (match code 3)
    X_trainval, y_trainval = X_feat[:60000], y_all[:60000]
    X_test,     y_test     = X_feat[60000:], y_all[60000:]

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=5000,
        random_state=42,
        stratify=y_trainval
    )

    # 3) Loader (match code 3)
    X_tr_t  = torch.from_numpy(X_train)
    y_tr_t  = torch.from_numpy(y_train).long()
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val).long()
    X_te_t  = torch.from_numpy(X_test)
    y_te_t  = torch.from_numpy(y_test).long()

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=17000, shuffle=True, generator=g)

    # 4) Model + optimizer (match code 3)
    model = SoftmaxLogistic(feat_dim, 10).to(device)
    opt = optim.Adam(model.parameters(), lr=0.008, weight_decay=0.00012)
    criterion = nn.CrossEntropyLoss()

    # 5) Train (match code 3: 36 epochs)
    for epoch in trange(1, 36 + 1, desc="   epochs"):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t.to(device))
            val_acc = (val_logits.argmax(1) == y_val_t.to(device)).float().mean().item() * 100
        tqdm.write(f"   Epoch {epoch} Val Acc={val_acc:.2f}%")

    # 6) Quantize + test (match code 3)
    qmodel = quantize_dynamic(model.cpu(), {nn.Linear}, dtype=torch.qint8)
    qmodel.eval()
    with torch.no_grad():
        logits = qmodel(X_te_t.cpu())
        preds = logits.argmax(1)
        test_acc = (preds == y_te_t.cpu()).float().mean().item() * 100

    print(f"   Final Test Accuracy ({num_steps} steps): {test_acc:.2f}%")
    return test_acc

# -----------------------------
# Run experiments
# -----------------------------
results = {}
for k in [17,22]:
    results[k] = run_experiment(k)

print("\n===== Summary =====")
for k, acc in results.items():
    print(f"Steps kept={k:2d} → Test Accuracy={acc:.2f}%")
