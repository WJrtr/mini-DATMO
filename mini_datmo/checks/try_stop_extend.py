"""停留延伸（labeler.py 第 4 趟）的診斷：加上延伸之後多了多少 TP / FP，每個外框各自的結果。

用法：python checks/try_stop_extend.py [--data seq08] [--frames 1500:1800,3000:3300] [--kw "stop_drift=1.0"] [--rows 40]
輸出：
  1. 不加延伸 / 加延伸的 P、R、IoU
  2. 只靠延伸標到的點：TP、FP，以及 FP 的真值類別
  3. 每個延伸一列：track、方向、尾端漂移、標了幾幀、TP、FP、FP 最多的類別
"""
import argparse
import os
import sys
from collections import Counter as Cnt

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import Counter  # noqa: E402
from kitti_io import MOVING_MAX, MOVING_MIN, load_dataset  # noqa: E402
from labeler import Labeler  # noqa: E402

NAMES = {0: "unlabeled", 1: "outlier", 10: "car", 11: "bicycle", 13: "bus", 15: "motorcycle", 18: "truck",
         20: "other-vehicle", 30: "person", 31: "bicyclist", 32: "motorcyclist", 40: "road", 44: "parking",
         48: "sidewalk", 49: "other-ground", 50: "building", 51: "fence", 52: "other-structure",
         60: "lane-marking", 70: "vegetation", 71: "trunk", 72: "terrain", 80: "pole", 81: "traffic-sign",
         99: "other-object"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="seq08")
    ap.add_argument("--frames", default="1500:1800,3000:3300")
    ap.add_argument("--kw", default="")
    ap.add_argument("--rows", type=int, default=40)
    args = ap.parse_args()
    kw = eval("dict(" + args.kw + ")")
    ds = load_dataset(args.data)
    base, full = Counter(), Counter()
    ext_tp = ext_fp = 0
    fp_cls = Cnt()
    rows = []
    for w in args.frames.split(","):
        a, b = map(int, w.split(":"))
        L = Labeler(ds, stop_extend=True, **kw)
        L.stop_debug = True
        out = L.run(range(a, b))
        sem = {}
        for k, o in out.items():
            s = np.fromfile(ds.label_paths[k], dtype=np.uint32) & 0xFFFF
            sem[k] = s
            gt = (s >= MOVING_MIN) & (s <= MOVING_MAX)
            ext = o.get("ext", np.zeros(len(gt), bool))
            base.add(o["label"] & ~ext, gt)
            full.add(o["label"], gt)
            ext_tp += int(np.sum(ext & gt))
            ext_fp += int(np.sum(ext & ~gt))
            fp_cls.update(s[ext & ~gt].tolist())
        ti = {t["id"]: t for t in L.track_info}
        for e in L.ext_info:
            t = ti[e["track"]]
            info = (f"track n={t['n']} 位移 {t['disp']:.1f} 直線度 {t['straight']:.2f} 群大小 {t['extent']:.1f} "
                    f"比例 {t['ratio']:.2f}｜物體大小 {e['size']:.1f}")
            tp = fp = nf = 0
            c = Cnt()
            for k, n, r, ok, idx in e["frames"]:
                if not ok:
                    continue
                nf += 1
                s = sem[k][idx]
                g = (s >= MOVING_MIN) & (s <= MOVING_MAX)
                tp += int(g.sum())
                fp += int((~g).sum())
                c.update(s[~g].tolist())
            top = NAMES.get(c.most_common(1)[0][0], "?") if c else ""
            rows.append((tp + fp, f"  {w:10s} track {e['track']:5d} {'往後' if e['dir'] == 1 else '往前'} 從 {e['k0']:4d}  "
                                  f"漂移 {e['drift']:.2f} m  {'採用' if e['valid'] else '不採用'}  檢查 {len(e['frames']):4d} 幀  "
                                  f"標 {nf:4d} 幀  TP {tp:6d}  FP {fp:6d}  {top:12s} {info}"))
    print(f"== {args.data} {args.frames}  kw={kw}")
    print(f"  不加延伸  {base}")
    print(f"  加延伸    {full}")
    print(f"  只靠延伸標到的點：TP {ext_tp:,}  FP {ext_fp:,}（各外框互相重疊的點只算一次）")
    tot = max(sum(fp_cls.values()), 1)
    print("  延伸的 FP 類別：" + "  ".join(f"{NAMES.get(s, s)} {100*n/tot:.0f}%" for s, n in fp_cls.most_common(6)))
    print(f"== 外框共 {len(rows)} 個（每個延伸的 TP / FP 含原本就標到的點；不採用的延伸不會標進結果）")
    for _, r in sorted(rows, key=lambda x: -x[0])[:args.rows]:
        print(r)


if __name__ == "__main__":
    main()
