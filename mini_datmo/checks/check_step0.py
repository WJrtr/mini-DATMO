"""步驟 0 驗證：位姿轉換對不對。

量化：第 k 幀點轉到世界座標後，和第 k-5 幀點的最近鄰距離中位數。位姿正確時應該很小（< 0.1 m 等級）。
對照：不轉換（單位矩陣）時的數字。
另存兩張圖：50 幀疊圖、行車軌跡。
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kitti_io import load_dataset  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "step0")
os.makedirs(OUT, exist_ok=True)


def to_world(pts, T):
    return pts[:, :3] @ T[:3, :3].T + T[:3, 3]


def nn_median(ds, k, j, identity=False):
    a, b = ds.scan(k), ds.scan(j)
    a = a[np.linalg.norm(a[:, :2], axis=1) < 30]
    b = b[np.linalg.norm(b[:, :2], axis=1) < 30]
    Ta = np.eye(4) if identity else ds.T_w_velo[k]
    Tb = np.eye(4) if identity else ds.T_w_velo[j]
    d, _ = cKDTree(to_world(b, Tb)).query(to_world(a, Ta))
    return np.median(d)


for name, k0 in [("seq08", 1000), ("0005", 50), ("0060", 40)]:
    ds = load_dataset(name)
    t = ds.T_w_velo[:, :3, 3]
    step = np.linalg.norm(np.diff(t, axis=0), axis=1)
    print(f"[{name}] frames={ds.n}  path length={step.sum():.1f} m  mean speed={step.mean()/0.1:.1f} m/s")
    print(f"   NN median k={k0} vs k-5: with pose {nn_median(ds, k0, k0-5):.3f} m | identity {nn_median(ds, k0, k0-5, True):.3f} m")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].plot(t[:, 0], t[:, 1], "-")
    ax[0].set_aspect("equal"); ax[0].set_title(f"{name} trajectory (T_w_velo)")
    for k in range(max(0, k0 - 25), min(ds.n, k0 + 25), 2):
        p = ds.scan(k)
        p = p[(np.linalg.norm(p[:, :2], axis=1) < 30) & (p[:, 2] > -1.3)][::4]
        w = to_world(p, ds.T_w_velo[k])
        ax[1].scatter(w[:, 0], w[:, 1], s=0.05, c="k")
    ax[1].set_aspect("equal"); ax[1].set_title(f"{name}: ~25 scans overlaid around k={k0}")
    fig.savefig(os.path.join(OUT, f"{name}_step0.png"), dpi=110)
    plt.close(fig)
print("saved to", OUT)
