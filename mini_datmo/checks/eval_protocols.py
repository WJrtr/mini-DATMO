"""兩種評估方式的比較：所有點都算，或是忽略真值為 unlabeled(0)、outlier(1) 的點。

SemanticKITTI-MOS 的官方設定（semantic-kitti-mos.yaml）把 0、1 對應到 unlabeled 並設為 ignore，
所以落在這些點上的預測不算 FP。LMNet 在序列 08 的官方預測：所有點都算是 58.4%，忽略 0/1 是 59.9%（論文 Table I 的數字）。

用法：python checks/eval_protocols.py --data seq08 --pred out/seq08_refine,out/seq08_ego20
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kitti_io import MOVING_MAX, MOVING_MIN, load_dataset, read_label_moving  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="seq08")
    ap.add_argument("--pred", default="", help="輸出資料夾，逗號分開")
    ap.add_argument("--frames", default=None)
    args = ap.parse_args()
    ds = load_dataset(args.data)
    a, b = (0, ds.n) if args.frames is None else map(int, args.frames.split(":"))
    dirs = [d for d in args.pred.split(",") if d]
    names = dirs + (["LMNet"] if ds.pred_paths else [])
    C = {n: np.zeros((2, 3), np.int64) for n in names}
    ign_n = tot = 0
    for k in range(a, b):
        sem = np.fromfile(ds.label_paths[k], dtype=np.uint32) & 0xFFFF
        gt = (sem >= MOVING_MIN) & (sem <= MOVING_MAX)
        ign = sem <= 1
        ign_n += int(ign.sum())
        tot += len(sem)
        for n in names:
            p = ds.lmnet(k) if n == "LMNet" else read_label_moving(os.path.join(n, "predictions", f"{k:06d}.label"))
            C[n][0] += [np.sum(p & gt), np.sum(p & ~gt), np.sum(~p & gt)]
            C[n][1] += [np.sum(p & gt), np.sum(p & ~gt & ~ign), np.sum(~p & gt)]
    print(f"== {args.data} {a}:{b}，unlabeled/outlier 占所有點 {100*ign_n/tot:.2f}%")
    for n, c in C.items():
        for j, lab in enumerate(["所有點", "忽略 0/1"]):
            tp, fp, fn = c[j]
            print(f"  {n:28s} {lab:7s} P {100*tp/max(tp+fp,1):5.1f}  R {100*tp/max(tp+fn,1):5.1f}  "
                  f"IoU {100*tp/max(tp+fp+fn,1):5.1f}   (TP {tp:,} FP {fp:,} FN {fn:,})")


if __name__ == "__main__":
    main()
