"""步驟 ⑨：建圖應用。把一段序列的非地面點疊成世界座標地圖，比較：
  raw（全部點）/ ours（去掉我們標成移動的點）/ lmnet（去掉 LMNet 標成移動的點）/ gt（去掉真值移動點）

數字（voxel 0.2 m，每個 voxel 依裡面的點看真值）：
  - ghost voxels：只由真值移動點組成、卻還留在地圖裡的 voxel 數 → 鬼影
  - static kept：只由真值靜止點組成的 voxel，留下來的比例 → 誤刪越少越接近 100%
（思路同 DynamicMap Benchmark 的「動態點刪得掉、靜態點保得住」；那邊的正式指標定義未確認。）

用法：python mapping.py --pred out/seq08_g5/predictions --frames 3900:4071
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from consistency import transform
from ground import ego_mask, ground_mask
from kitti_io import load_dataset, read_label_moving

VOX = 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--out", default="out/mapping")
    args = ap.parse_args()
    ds = load_dataset("seq08")
    a, b = map(int, args.frames.split(":"))
    P, G, O, LM = [], [], [], []
    for k in range(a, b, args.every):
        p = ds.scan(k)
        keep = ~ground_mask(p) & ~ego_mask(p) & (np.linalg.norm(p[:, :2], axis=1) < 40)
        P.append(transform(p[keep, :3], ds.T_w_velo[k]))
        G.append(ds.labels(k)[keep])
        O.append(read_label_moving(os.path.join(args.pred, f"{k:06d}.label"))[keep])
        LM.append(ds.lmnet(k)[keep])
    P, G, O, LM = map(np.concatenate, (P, G, O, LM))
    v = np.floor(P / VOX).astype(np.int64)
    _, inv = np.unique(v, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    nv = inv.max() + 1
    n_all = np.bincount(inv, minlength=nv)
    n_gt = np.bincount(inv, weights=G, minlength=nv)
    dyn_vox = n_gt == n_all          # 只由移動點組成
    sta_vox = n_gt == 0              # 只由靜止點組成
    res = {}
    removed_sets = dict(raw=np.zeros_like(G), ours=O, lmnet=LM, gt=G)
    for name, rm in removed_sets.items():
        kept = np.bincount(inv, weights=~rm, minlength=nv) > 0
        res[name] = dict(ghost_voxels=int(np.sum(kept & dyn_vox)), dyn_voxels=int(dyn_vox.sum()),
                         static_kept=float(np.sum(kept & sta_vox) / max(sta_vox.sum(), 1)),
                         points_removed=int(rm.sum()))
        print(f"{name:6s} ghost voxels {res[name]['ghost_voxels']:7d} / {res[name]['dyn_voxels']}   "
              f"static kept {100*res[name]['static_kept']:.2f}%   removed pts {res[name]['points_removed']}")
    os.makedirs(args.out, exist_ok=True)
    tag = f"seq08_{a}-{b}"
    with open(os.path.join(args.out, f"{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    fig, axes = plt.subplots(1, 4, figsize=(24, 7), sharex=True, sharey=True)
    for ax, (name, rm) in zip(axes, removed_sets.items()):
        kept = ~rm
        sub = np.arange(len(P))[kept][::3]
        ax.scatter(P[sub, 0], P[sub, 1], s=0.05, c="0.35")
        ghost = kept & G  # 留在地圖裡的真值移動點 = 鬼影
        ax.scatter(P[ghost, 0], P[ghost, 1], s=0.2, c="red")
        ax.set_title(f"{name}: ghost voxels {res[name]['ghost_voxels']}, static kept {100*res[name]['static_kept']:.1f}%")
        ax.set_aspect("equal")
    fig.suptitle(f"SemanticKITTI seq 08 frames {a}-{b}: accumulated map (red = ground-truth moving points left in map)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"{tag}.png"), dpi=110)
    print("saved", os.path.join(args.out, f"{tag}.png"))


if __name__ == "__main__":
    main()
