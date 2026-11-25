"""物體層級的長時間一致性（「消失檢查」）的可行性。結論：不可行（2026-11-25）。

想法：整段可見期間都停著的物體，短時間的參考幀看不出它是移動物體；唯一的證據是隔很久之後（或之前）
那個位置是空的。所以對所有非地面點分群，每個群算「被長時間參考幀看穿的點」占多少，比例高的群整個標成移動。

結果（序列 08 的開發範圍與 0:200）：
  - 不要求回波（--det-kw "need_return=False"）：誤判大多是停著的車。這些「看穿」來自參考幀沒有回波的格子。
  - 要求回波（預設）：±16、32、64 秒的參考幀一票都沒有，所有門檻下都標不到任何點。
這支程式也是發現「沒有回波被當成看穿」的地方，見 consistency.py 的 need_return。

用法：python checks/try_vanish.py --data seq08 --frames 1500:1800,3000:3300 --every 5 [--det-kw "need_return=False"]
"""
import argparse
import os
import sys
from collections import Counter as Cnt

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cluster import cluster  # noqa: E402
from consistency import ConsistencyDetector  # noqa: E402
from eval_by_speed import BINS, inst_centers  # noqa: E402
from ground import ego_mask, ground_mask  # noqa: E402
from kitti_io import load_dataset  # noqa: E402

ALL = [40, 80, 160, 320, 640]
SETS = [("±16、32、64 秒，1 票", [160, 320, 640], 1),
        ("±16、32、64 秒，2 票", [160, 320, 640], 2),
        ("±4 到 64 秒，2 票", [40, 80, 160, 320, 640], 2),
        ("±4 到 64 秒，3 票", [40, 80, 160, 320, 640], 3)]
THRS = [0.3, 0.5, 0.7, 0.9]
NAMES = {0: "unlabeled", 1: "outlier", 10: "car", 11: "bicycle", 13: "bus", 15: "motorcycle", 18: "truck",
         20: "other-vehicle", 30: "person", 31: "bicyclist", 32: "motorcyclist", 40: "road", 44: "parking",
         48: "sidewalk", 49: "other-ground", 50: "building", 51: "fence", 52: "other-structure",
         60: "lane-marking", 70: "vegetation", 71: "trunk", 72: "terrain", 80: "pole", 81: "traffic-sign",
         99: "other-object"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="seq08")
    ap.add_argument("--frames", default="1500:1800,3000:3300")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--max-extent", type=float, default=8.0)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--det-kw", default="", help='ConsistencyDetector 參數，例如 "need_return=True"')
    ap.add_argument("--pred", default=None, help="現有標籤的資料夾；有的話另外報告「聯集」之後的結果")
    args = ap.parse_args()
    ds = load_dataset(args.data)
    det = ConsistencyDetector(ds, cache_size=80, **eval("dict(" + args.det_kw + ")"))
    cache = {}

    def centers(k):
        if k not in cache:
            if len(cache) > 30:
                cache.pop(min(cache))
            cache[k] = inst_centers(ds, k)[3]
        return cache[k]

    gt_bin = np.zeros(4, np.int64)
    R = {(n, t): dict(tp=np.zeros(4, np.int64), fp=0, cls=Cnt()) for n, _, _ in SETS for t in THRS}
    frames = [k for w in args.frames.split(",") for k in range(*map(int, w.split(":")), args.every)]
    for k in frames:
        pts = ds.scan(k)
        sem, inst, mov, cen = inst_centers(ds, k)
        g = ground_mask(pts) | ego_mask(pts)
        sel = ~g & (np.linalg.norm(pts[:, :2], axis=1) < det.max_range)
        xyz = pts[sel, :3]
        votes = {}
        for gp in ALL:
            v = np.zeros(sel.sum(), np.int64)
            for j in (k - gp, k + gp):
                if 0 <= j < ds.n:
                    v += det._against(k, j, xyz)
            votes[gp] = v
        # 每個點的速度分組（3 = unknown）
        pbin = np.full(len(pts), 3)
        for key in set(zip(sem[mov].tolist(), inst[mov].tolist())):
            m = (sem == key[0]) & (inst == key[1])
            v = None
            if key in cen:
                ends = [centers(j).get(key) for j in (k - 5, k + 5) if 0 <= j < ds.n]
                ends = [e for e in ends if e is not None]
                if len(ends) == 2:
                    v = np.linalg.norm(ends[1] - ends[0])
                elif len(ends) == 1:
                    v = np.linalg.norm(ends[0] - cen[key]) / 0.5
            if v is not None:
                pbin[m] = next(i for i, (_, lo, hi) in enumerate(BINS) if lo <= v < hi)
        gt_bin += np.bincount(pbin[mov], minlength=4)
        cid, cl = cluster(pts[:, :3], ~g, eps=args.eps)
        cl = [q for q in cl if q["extent"] < args.max_extent]
        for name, gaps, mv in SETS:
            c = np.zeros(len(pts), bool)
            c[sel] = sum(votes[gp] for gp in gaps) >= mv
            for q in cl:
                ratio = c[q["idx"]].mean()
                for t in THRS:
                    if ratio > t:
                        idx = q["idx"]
                        mm = mov[idx]
                        r = R[(name, t)]
                        r["tp"] += np.bincount(pbin[idx][mm], minlength=4)
                        r["fp"] += int((~mm).sum())
                        r["cls"].update(sem[idx][~mm].tolist())
    G = gt_bin.sum()
    print(f"== {args.data} {args.frames} 每 {args.every} 幀；群上限 {args.max_extent} m，eps {args.eps}，{args.det_kw}；"
          f"真值移動點 {G:,}（stopped {gt_bin[0]:,} / slow {gt_bin[1]:,} / moving {gt_bin[2]:,}）")
    print(f"  {'參考幀':22s} 門檻     P      R   | R: stopped   slow  moving |     FP  FP 類別")
    for name, _, _ in SETS:
        for t in THRS:
            r = R[(name, t)]
            tp = r["tp"].sum()
            tot = max(sum(r["cls"].values()), 1)
            cls = "  ".join(f"{NAMES.get(s, s)} {100*n/tot:.0f}%" for s, n in r["cls"].most_common(4))
            rb = "  ".join(f"{100*r['tp'][i]/max(gt_bin[i],1):6.1f}" for i in range(3))
            print(f"  {name:22s} {t:.1f}  {100*tp/max(tp+r['fp'],1):6.1f} {100*tp/max(G,1):6.1f} | {rb} | {r['fp']:7,}  {cls}")


if __name__ == "__main__":
    main()
