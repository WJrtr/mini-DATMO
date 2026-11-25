"""步驟 0 驗證：位姿轉換對不對。

量化：第 k 幀和第 k-5 幀的非地面點（ground.ground_mask，感測器 30 m 內）各自轉到世界座標後，
第 k 幀每一點到第 k-5 幀最近點的距離，取中位數。位姿正確時應該很小（< 0.1 m 等級）。
對照：
  - identity：不轉換，兩幀都用各自的感測器座標。
  - P only（只有 SemanticKITTI）：把 poses.txt 的相機位姿 P 直接套在 LiDAR 點上，不做 Tr 的共軛。
另存兩張圖：約 25 幀疊圖、行車軌跡。

用法：python checks/check_step0.py > ../results/pose_check.txt
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ground import ground_mask  # noqa: E402
from kitti_io import _to4x4, load_dataset  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "step0")
os.makedirs(OUT, exist_ok=True)
MAX_RANGE = 30.0


def to_world(pts, T):
    return pts[:, :3] @ T[:3, :3].T + T[:3, 3]


def nonground(pts):
    pts = pts[np.linalg.norm(pts[:, :2], axis=1) < MAX_RANGE]
    return pts[~ground_mask(pts)]


def nn_median(ds, k, j, poses):
    """poses：(n, 4, 4)，第 k 幀點 -> 世界座標。"""
    a, b = nonground(ds.scan(k)), nonground(ds.scan(j))
    d, _ = cKDTree(to_world(b, poses[j])).query(to_world(a, poses[k]))
    return np.median(d)


def camera_poses(ds):
    """SemanticKITTI 的 poses.txt（相機位姿 P），不做共軛。KITTI raw 沒有這個檔案，回傳 None。"""
    path = os.path.join(os.path.dirname(ds.scan_paths[0]), "..", "poses.txt")
    if not os.path.exists(path):
        return None
    return np.stack([_to4x4(p) for p in np.loadtxt(path).reshape(-1, 3, 4)])


print(f"非地面點（ground_mask）、感測器 {MAX_RANGE:.0f} m 內；第 k 幀對第 k-5 幀，最近點距離的中位數（m）")
for name, k0 in [("seq08", 1000), ("0005", 50), ("0060", 40)]:
    try:
        ds = load_dataset(name)
    except (FileNotFoundError, OSError) as err:  # KITTI raw 沒有下載時略過
        print(f"[{name}] 略過：{err}")
        continue
    t = ds.T_w_velo[:, :3, 3]
    step = np.linalg.norm(np.diff(t, axis=0), axis=1)
    moved = np.linalg.norm(t[k0] - t[k0 - 5])
    eye = np.stack([np.eye(4)] * ds.n)
    row = f"identity {nn_median(ds, k0, k0 - 5, eye):.3f} | T_w_velo {nn_median(ds, k0, k0 - 5, ds.T_w_velo):.3f}"
    P = camera_poses(ds)
    if P is not None:
        row += f" | P only {nn_median(ds, k0, k0 - 5, P):.3f}"
    print(f"[{name}] frames={ds.n}  path length={step.sum():.1f} m  mean speed={step.mean()/0.1:.1f} m/s")
    print(f"   k={k0} vs k-5（自車移動 {moved:.2f} m）: {row}")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].plot(t[:, 0], t[:, 1], "-")
    ax[0].set_aspect("equal"); ax[0].set_title(f"{name} trajectory (T_w_velo)")
    for k in range(max(0, k0 - 25), min(ds.n, k0 + 25), 2):
        w = to_world(nonground(ds.scan(k))[::4], ds.T_w_velo[k])
        ax[1].scatter(w[:, 0], w[:, 1], s=0.05, c="k")
    ax[1].set_aspect("equal"); ax[1].set_title(f"{name}: ~25 scans overlaid around k={k0}")
    fig.savefig(os.path.join(OUT, f"{name}_step0.png"), dpi=110)
    plt.close(fig)
print("saved to", OUT)
