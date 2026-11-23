"""mini-DATMO 主程式：跑標註器、輸出 .label、評估。

例：
  python run.py --data seq08 --frames 0:4071 --out out/seq08_g5
  python run.py --data seq08 --frames 0:500 --pose-noise 5 --seed 0
  python run.py --data seq08 --frames 0:500 --no-ego-motion
"""
import argparse
import json
import os
import time

import numpy as np

from evaluate import Counter
from kitti_io import load_dataset, write_label
from labeler import Labeler


def noisy_poses(T, level, seed=0):
    """7-2：每幀獨立加高斯雜訊。1 單位 = (0.1 m, 0.1 m, 1°)，沿用 LMNet Fig. 4 的單位。
    雜訊加在感測器座標系（右乘）：T_noisy = T · ΔT。"""
    rng = np.random.default_rng(seed)
    out = T.copy()
    for k in range(len(T)):
        dx, dy = rng.normal(0, 0.1 * level, 2)
        dyaw = np.deg2rad(rng.normal(0, 1.0 * level))
        D = np.eye(4)
        D[:2, :2] = [[np.cos(dyaw), -np.sin(dyaw)], [np.sin(dyaw), np.cos(dyaw)]]
        D[:2, 3] = [dx, dy]
        out[k] = T[k] @ D
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="seq08")
    ap.add_argument("--frames", default=None, help="a:b")
    ap.add_argument("--gap", default="5", help="例如 5 或 5,10,20")
    ap.add_argument("--merge", default="or", choices=["or", "and"])
    ap.add_argument("--horizon", type=int, default=0,
                    help=">0 時改用長時間可見性：和 ±5, ±15, ... ±horizon 幀比較（類似整段序列的地圖清理）")
    ap.add_argument("--min-votes", type=int, default=1)
    ap.add_argument("--ratio-thr", type=float, default=0.15)
    ap.add_argument("--kw", default="", help='其他 Labeler 參數，例如 "refine=0.5,disp_factor=0.5"')
    ap.add_argument("--pose-noise", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ego-motion", action="store_true")
    ap.add_argument("--out", default=None, help="輸出 .label 的資料夾")
    ap.add_argument("--json", default=None, help="把評估結果寫成 json")
    args = ap.parse_args()

    ds = load_dataset(args.data)
    a, b = (0, ds.n) if args.frames is None else map(int, args.frames.split(":"))
    poses = ds.T_w_velo
    if args.no_ego_motion:
        poses = np.repeat(np.eye(4)[None], ds.n, axis=0)
    elif args.pose_noise > 0:
        poses = noisy_poses(ds.T_w_velo, args.pose_noise, args.seed)
    gap = [int(g) for g in args.gap.split(",")]
    gap = gap[0] if len(gap) == 1 else gap
    if args.horizon > 0:
        gap = list(range(5, args.horizon + 1, 10))

    t0 = time.time()
    L = Labeler(ds, poses=poses, gap=gap, merge=args.merge, min_votes=args.min_votes,
                ratio_thr=args.ratio_thr, **eval("dict(" + args.kw + ")"))
    out = L.run(range(a, b))
    dt = time.time() - t0

    res = {}
    if ds.label_paths is not None:
        c = {n: Counter() for n in ["consistency_only", "labeler", "lmnet"]}
        c40 = {n: Counter() for n in c}
        for k, o in out.items():
            gt = ds.labels(k)
            near = np.linalg.norm(ds.scan(k)[:, :2], axis=1) < 40
            preds = dict(consistency_only=o["cand"], labeler=o["label"], lmnet=ds.lmnet(k))
            for n, p in preds.items():
                if p is None:
                    continue
                c[n].add(p, gt)
                c40[n].add(p, gt, near)
        print(f"[{args.data} {a}:{b}] gap={args.gap} noise={args.pose_noise} no_ego={args.no_ego_motion}  {dt/(b-a):.2f} s/frame")
        for n in c:
            if c[n].tp + c[n].fp + c[n].fn == 0:
                continue  # 這個序列沒有 LMNet 預測
            print(f"  {n:17s} all: {c[n]}   | <40m: {c40[n]}")
            res[n] = dict(all=c[n].result(), within40=c40[n].result())
    res["tracks"] = dict(total=len(L.track_info), moving=int(sum(t["moving"] for t in L.track_info)))
    res["args"] = vars(args)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)
    if args.out:
        for k, o in out.items():
            write_label(os.path.join(args.out, "predictions", f"{k:06d}.label"), o["label"])
        print("labels ->", args.out)


if __name__ == "__main__":
    main()
