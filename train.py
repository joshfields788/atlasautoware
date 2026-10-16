"""Behavior cloning: fit LidarPolicy to the recorded expert data."""
import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from model import LidarPolicy

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/ftg.npz")
p.add_argument("--out", default="policy.pt")
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--lr", type=float, default=1e-3)
a = p.parse_args()

d = np.load(a.data)
ds = TensorDataset(torch.from_numpy(d["X"]), torch.from_numpy(d["Y"]))
n_val = int(0.1 * len(ds))
tr, va = random_split(ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(0))
tl, vl = DataLoader(tr, batch_size=256, shuffle=True), DataLoader(va, batch_size=1024)

net, opt = LidarPolicy(), None
opt = torch.optim.Adam(net.parameters(), lr=a.lr)
w = torch.tensor([2.0, 1.0])            # weight steering errors more than speed errors
best = 1e9
for ep in range(a.epochs):
    net.train()
    for x, y in tl:
        loss = (((net(x) - y) ** 2) * w).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        v = sum((((net(x) - y) ** 2) * w).mean().item() * len(x) for x, y in vl) / max(n_val, 1)
    if v < best:
        best = v; torch.save(net.state_dict(), a.out)
    print(f"epoch {ep+1:3d}  val loss {v:.5f}" + ("  *" if v == best else ""))
print("best val loss", best, "->", a.out)
