"""比較兩組標註輸出，找出 IoU 差距來自哪裡。

用法：python checks/compare_runs.py --data seq08 --a out/seq08_refine --b out/seq08_path [--frames a:b]
輸出：
  1. 兩組的 P / R / IoU，以及 LMNet（序列 08 才有）
  2. 依估計速度分組的召回率
  3. 誤判（FP）依真值語意類別分組
  4. B 相對 A 多出來 / 少掉的 TP、FP，依類別與速度分組
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from consistency import transform  # noqa: E402
from kitti_io import MOVING_MAX, MOVING_MIN, load_dataset, read_label_moving  # noqa: E402

NAMES = {0: "unlabeled", 1: "outlier", 10: "car", 11: "bicycle", 13: "bus", 15: "motorcycle", 16: "on-rails",
         18: "truck", 20: "other-vehicle", 30: "person", 31: "bicyclist", 32: "motorcyclist", 40: "road",
         44: "parking", 48: "sidewalk", 49: "other-ground", 50: "building", 51: "fence", 52: "other-structure",
         60: "lane-marking", 70: "vegetation", 71: "trunk", 72: "terrain", 80: "pole", 81: "traffic-sign",
         99: "other-object", 252: "moving-car", 253: "moving-bicyclist", 254: "moving-person",
         255: "moving-motorcyclist", 256: "moving-on-rails", 257: "moving-bus", 258: "moving-truck",
         259: "moving-other-vehicle"}
BINS = [("stopped<0.5", 0, 0.5), ("slow0.5-2", 0.5, 2.0), ("moving>2", 2.0, 1e9)]


def speed_bin(v):
    if v is None:
        return "unknown"
    return next(n for n, lo, hi in BINS if lo <= v < hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="seq08")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--frames", default=None)
    args = ap.parse_args()
    ds = load_dataset(args.data)
    a0, b0 = (0, ds.n) if args.frames is None else map(int, args.frames.split(":"))

    cen_cache = {}

    def centers(k):
        if k not in cen_cache:
            L = np.fromfile(ds.label_paths[k], dtype=np.uint32)
            sem, inst = L & 0xFFFF, L >> 16
            mov = (sem >= MOVING_MIN) & (sem <= MOVING_MAX)
            p = ds.scan(k)
            d = {}
            for s, i in set(zip(sem[mov].tolist(), inst[mov].tolist())):
                m = (sem == s) & (inst == i)
                if m.sum() >= 10:
                    d[(s, i)] = transform(p[m, :3].mean(0)[None], ds.T_w_velo[k])[0][:2]
            cen_cache[k] = d
            if len(cen_cache) > 40:
                cen_cache.pop(min(cen_cache))
        return cen_cache[k]

    tot = defaultdict(lambda: np.zeros(3, np.int64))  # name -> tp, fp, fn
    rec = defaultdict(lambda: defaultdict(lambda: np.zeros(2, np.int64)))  # name -> bin -> (gt, hit)
    fpcls = defaultdict(lambda: defaultdict(int))
    delta = defaultdict(lambda: defaultdict(int))  # kind -> key -> count
    for k in range(a0, b0):
        L = np.fromfile(ds.label_paths[k], dtype=np.uint32)
        sem, inst = L & 0xFFFF, L >> 16
        gt = (sem >= MOVING_MIN) & (sem <= MOVING_MAX)
        preds = {"A": read_label_moving(os.path.join(args.a, "predictions", f"{k:06d}.label")),
                 "B": read_label_moving(os.path.join(args.b, "predictions", f"{k:06d}.label"))}
        if ds.pred_paths:
            preds["LMNet"] = ds.lmnet(k)
        # 每個移動物體的當下速度
        cen = centers(k)
        pt_bin = np.full(len(gt), "", dtype=object)
        for key in set(zip(sem[gt].tolist(), inst[gt].tolist())):
            m = (sem == key[0]) & (inst == key[1])
            v = None
            if key in cen:
                ends = [centers(j).get(key) for j in (k - 5, k + 5) if 0 <= j < ds.n]
                ends = [e for e in ends if e is not None]
                if len(ends) == 2:
                    v = np.linalg.norm(ends[1] - ends[0]) / 1.0
                elif len(ends) == 1:
                    v = np.linalg.norm(ends[0] - cen[key]) / 0.5
            pt_bin[m] = speed_bin(v)
        for name, p in preds.items():
            tot[name] += [np.sum(p & gt), np.sum(p & ~gt), np.sum(~p & gt)]
            for b in set(pt_bin[gt]):
                m = gt & (pt_bin == b)
                rec[name][b] += [m.sum(), np.sum(p & m)]
            for s, c in zip(*np.unique(sem[p & ~gt], return_counts=True)):
                fpcls[name][NAMES.get(int(s), str(s))] += int(c)
        pa, pb = preds["A"], preds["B"]
        for kind, m in [("TP+ (B 多抓到)", pb & ~pa & gt), ("TP- (B 漏掉)", pa & ~pb & gt),
                        ("FP+ (B 多誤判)", pb & ~pa & ~gt), ("FP- (B 少誤判)", pa & ~pb & ~gt)]:
            if not m.any():
                continue
            if "TP" in kind:
                for b, c in zip(*np.unique(pt_bin[m].astype(str), return_counts=True)):
                    delta[kind][b] += int(c)
            else:
                for s, c in zip(*np.unique(sem[m], return_counts=True)):
                    delta[kind][NAMES.get(int(s), str(s))] += int(c)

    print(f"== {args.data} {a0}:{b0}   A = {args.a}   B = {args.b}")
    for name, (tp, fp, fn) in tot.items():
        print(f"  {name:6s} P {100*tp/max(tp+fp,1):5.1f}  R {100*tp/max(tp+fn,1):5.1f}  IoU {100*tp/max(tp+fp+fn,1):5.1f}"
              f"   (TP {tp:,}  FP {fp:,}  FN {fn:,})")
    print("\n== 依速度分組的召回率")
    bins = [b for b, _, _ in BINS] + ["unknown"]
    for name in tot:
        row = "  ".join(f"{b} {100*rec[name][b][1]/max(rec[name][b][0],1):5.1f}" for b in bins if rec[name][b][0])
        print(f"  {name:6s} {row}")
    print("  真值點數：" + "  ".join(f"{b} {rec['A'][b][0]:,}" for b in bins if rec['A'][b][0]))
    print("\n== 誤判（FP）的真值類別，前 6 名")
    for name in tot:
        f = sorted(fpcls[name].items(), key=lambda x: -x[1])[:6]
        s = sum(fpcls[name].values())
        print(f"  {name:6s} " + "  ".join(f"{c} {100*n/max(s,1):.0f}%" for c, n in f))
    print("\n== B 相對 A 的變化")
    for kind in ["TP+ (B 多抓到)", "TP- (B 漏掉)", "FP+ (B 多誤判)", "FP- (B 少誤判)"]:
        d = delta[kind]
        s = sum(d.values())
        top = sorted(d.items(), key=lambda x: -x[1])[:5]
        print(f"  {kind}: {s:,} 點  " + "  ".join(f"{c} {n:,}" for c, n in top))


if __name__ == "__main__":
    main()
