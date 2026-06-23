# Binary-MNIST + ReCA features with reproducibility, timing
# Runs experiment for multiple CA iteration counts [1_16]

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from scipy.ndimage import map_coordinates, gaussian_filter

import os, random, time
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
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try: torch.use_deterministic_algorithms(True)
except Exception: pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# Helpers
# -----------------------------
def _tick(): return time.time()
def _tock(start): return time.time() - start

def to_bitplanes(img):
    return (img > 128).astype(np.uint8)[None, :, :]  # (1,28,28)

def rule90_step(arr, axis):
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
        row_evol = np.array([rule90_step(bp, axis=1) for bp in current])
        col_evol = np.array([rule90_step(bp, axis=0) for bp in current])
        x_k = np.bitwise_xor(row_evol, col_evol)
        evolved.append(x_k)
        current = x_k
    return evolved

def grayscale_reform(evolved_layers):
    return [layer[0] for layer in evolved_layers]

def max_pool(image, size=2, stride=2):
    h, w = image.shape
    out = np.zeros((h//stride, w//stride), dtype=image.dtype)
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            out[i//stride, j//stride] = image[i:i+size, j:j+size].max()
    return out

def reca_features(img, use_steps):
    bps = to_bitplanes(img)
    evo_all = evolve_bitplanes(bps, steps=max(use_steps)+1)  # evolve up to max step
    selected = [evo_all[k] for k in use_steps]
    gs  = grayscale_reform(selected)
    pp  = [max_pool(g) for g in gs]
    return np.concatenate([p.ravel() for p in pp])

def elastic_distort(image, alpha=36, sigma=6, rng=None):
    rs = rng or np.random.RandomState(SEED)
    shape = image.shape
    dx = gaussian_filter((rs.rand(*shape)*2 - 1), sigma) * alpha
    dy = gaussian_filter((rs.rand(*shape)*2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    coords = (y + dy).reshape(-1), (x + dx).reshape(-1)
    return map_coordinates(image, coords, order=1, mode='reflect').reshape(shape)

# -----------------------------
# Core experiment function
# -----------------------------
def run_experiment(num_steps):
    print(f"\n===== Running with {num_steps} CA iterations kept =====")
    use_steps = list(range(num_steps))   # keep first k iterations
    feat_dim = len(use_steps)*14*14

    times = {}
    # 1) Load MNIST
    _t=_tick()
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X_imgs = mnist['data'].reshape(-1,28,28).astype(np.uint8)
    y_all  = mnist['target'].astype(int)
    times["load"]=_tock(_t)

    # 2) Features
    _t=_tick()
    N=len(X_imgs)
    X_feat=np.empty((N, feat_dim),dtype=np.float32)
    for i,img in enumerate(tqdm(X_imgs,desc="   features")):
        X_feat[i]=reca_features(img,use_steps)
    times["features"]=_tock(_t)

    # 3) Split
    X_trainval,y_trainval=X_feat[:60000],y_all[:60000]
    X_test,y_test=X_feat[60000:],y_all[60000:]
    X_train,X_val,y_train,y_val=train_test_split(
        X_trainval,y_trainval,test_size=5000,random_state=42,stratify=y_trainval
    )

    # 4) Build loader
    def to_tensor(x,y=None):
        tx=torch.from_numpy(x)
        return (tx,torch.from_numpy(y).long()) if y is not None else tx
    X_tr_t,y_tr_t=to_tensor(X_train,y_train)
    X_val_t,y_val_t=to_tensor(X_val,y_val)
    X_te_t,y_te_t=to_tensor(X_test,y_test)
    train_ds=TensorDataset(X_tr_t,y_tr_t)
    g=torch.Generator().manual_seed(SEED)
    train_loader=DataLoader(train_ds,batch_size=17000,shuffle=True,generator=g)

    # 5) Model
    class SoftmaxLogistic(nn.Module):
        def __init__(self,in_dim,num_classes):
            super().__init__()
            self.linear=nn.Linear(in_dim,num_classes)
        def forward(self,x): return self.linear(x)
    model=SoftmaxLogistic(feat_dim,10).to(device)
    opt=optim.Adam(model.parameters(),lr=0.008,weight_decay=0.00012)
    criterion=nn.CrossEntropyLoss()

    # 6) Train
    for epoch in trange(36,desc="   epochs"):  # fewer epochs for demo; change to 36
        model.train()
        for Xb,yb in train_loader:
            Xb,yb=Xb.to(device),yb.to(device)
            opt.zero_grad()
            loss=criterion(model(Xb),yb)
            loss.backward(); opt.step()
        # validate
        model.eval()
        with torch.no_grad():
            val_logits=model(X_val_t.to(device))
            val_acc=(val_logits.argmax(1)==y_val_t.to(device)).float().mean().item()*100
        tqdm.write(f"   Epoch {epoch+1} Val Acc={val_acc:.2f}%")

    # 7) Quantize & test
    qmodel=quantize_dynamic(model.cpu(),{nn.Linear},dtype=torch.qint8)
    qmodel.eval()
    with torch.no_grad():
        logits=qmodel(X_te_t)
        preds=logits.argmax(1)
        test_acc=(preds==y_te_t).float().mean().item()*100

    print(f"   Final Test Accuracy ({num_steps} steps): {test_acc:.2f}%")
    return test_acc

# -----------------------------
# Run experiments for steps [1,4,8,12,16]
# -----------------------------
results={}
for k in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,30]:
    results[k]=run_experiment(k)

print("\n===== Summary =====")
for k,acc in results.items():
    print(f"Steps kept={k:2d} → Test Accuracy={acc:.2f}%")
