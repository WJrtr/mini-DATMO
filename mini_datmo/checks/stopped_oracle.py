"""暫停物體的上限：真值移動點裡「當下暫停」的那些，屬於「同一個物體在別的幀已經被標成移動」的有多少？

如果把「移動」沿著同一個物體往前後傳，最多能拿回這些點。用真值的 (類別, instance) 當物體身分，
所以這是上限（實際上要靠追蹤把暫停的群接回同一個物體）。

「已經被抓到的幀」：這個物體在該幀被標成移動的點 >= max(10, 20%)。
輸出：暫停的點依「離最近一個被抓到的幀有多遠」分組，以及中間這個物體是否一直看得到。

用法：python checks/stopped_oracle.py --pred out/seq08/predictions [--data seq08] [--frames a:b]
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_by_speed import BINS, inst_centers
from kitti_io import load_dataset, read_label_moving

NAMES = {252: "car", 253: "bicyclist", 254: "person", 255: "motorcyclist", 256: "on-rails",
         257: "bus", 258: "truck", 259: "other-vehicle"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--data", default="seq08")
    ap.add_argument("--frames", default=None)
    args = ap.parse_args()
    ds = load_dataset(args.data)
    a, b = (0, ds.n) if args.frames is None else map(int, args.frames.split(":"))
    cache = {}

    def centers(k):
        if k not in cache:
            if len(cache) > 30:
                cache.pop(min(cache))
            cache[k] = inst_centers(ds, k)[3]
        return cache[k]

    rec = defaultdict(dict)  # key -> {k: (n, tp, bin)}
    rng = defaultdict(dict)  # key -> {k: (range, speed)}
    for k in range(a, b):
        sem, inst, mov, cen = inst_centers(ds, k)
        cache[k] = cen
        p = read_label_moving(os.path.join(args.pred, f"{k:06d}.label"))
        for key in set(zip(sem[mov].tolist(), inst[mov].tolist())):
            m = (sem == key[0]) & (inst == key[1])
            v = None
            if key in cen:
                ends = [(j, centers(j).get(key)) for j in (k - 5, k + 5) if 0 <= j < ds.n]
                ends = [(j, c) for j, c in ends if c is not None]
                if len(ends) == 2:
                    v = np.linalg.norm(ends[1][1] - ends[0][1]) / 1.0
                elif len(ends) == 1:
                    v = np.linalg.norm(ends[0][1] - cen[key]) / 0.5
            bn = "unknown" if v is None else next(nm for nm, lo, hi in BINS if lo <= v < hi)
            rec[key][k] = (int(m.sum()), int(p[m].sum()), bn)
            rng[key][k] = (float(np.linalg.norm(cen[key] - ds.T_w_velo[k][:2, 3])) if key in cen else -1.0,
                           -1.0 if v is None else float(v))

    total_mov = sum(n for d in rec.values() for n, _, _ in d.values())
    stopped = sum(n for d in rec.values() for n, _, bn in d.values() if bn.startswith("stopped"))
    hit = sum(tp for d in rec.values() for n, tp, bn in d.values() if bn.startswith("stopped"))
    print(f"[{args.data} {a}:{b}] 真值移動點 {total_mov}，其中暫停 {stopped}（{100*stopped/total_mov:.1f}%），已抓到 {100*hit/max(stopped,1):.1f}%")

    edges = [0, 10, 50, 100, 200, 500, 10 ** 9]
    by_dist = defaultdict(int)       # 離最近被抓到的幀的距離
    by_dist_cont = defaultdict(int)  # 而且中間每一幀這個物體都看得到（允許缺 5 幀以內）
    never = 0
    by_class = defaultdict(lambda: [0, 0])  # 類別 -> [暫停的點, 其中物體曾被抓到]
    for key, d in rec.items():
        ks = np.array(sorted(d))
        det = np.array([d[k][1] >= max(10, 0.2 * d[k][0]) for k in ks])
        dk = ks[det]
        for k in ks:
            n, tp, bn = d[k]
            if not bn.startswith("stopped"):
                continue
            by_class[key[0]][0] += n
            if len(dk) == 0:
                never += n
                continue
            by_class[key[0]][1] += n
            j = dk[np.argmin(np.abs(dk - k))]
            dist = abs(int(j) - k)
            e = next(x for x in edges[1:] if dist <= x)
            by_dist[e] += n
            lo, hi = min(j, k), max(j, k)
            seen = ks[(ks >= lo) & (ks <= hi)]
            if len(seen) == 0 or np.max(np.diff(seen), initial=1) <= 5:
                by_dist_cont[e] += n
    print(f"暫停的點裡，物體從來沒被抓到：{100*never/max(stopped,1):.1f}%")
    print("離最近一個被抓到的幀（幀數）   占暫停的點   其中物體中間一直看得到")
    cum = cumc = 0
    for e in edges[1:]:
        cum += by_dist[e]
        cumc += by_dist_cont[e]
        lab = f"<= {e}" if e < 10 ** 9 else "更遠"
        print(f"  {lab:10s} 累計 {100*cum/max(stopped,1):5.1f}%   累計 {100*cumc/max(stopped,1):5.1f}%")
    print("依類別：暫停的點、其中物體曾被抓到的比例")
    for s, (n, h) in sorted(by_class.items(), key=lambda x: -x[1][0]):
        print(f"  {NAMES.get(s, s):14s} {n:9d}  {100*h/max(n,1):5.1f}%")
    top_instances(rec, rng)


def top_instances(rec, rng, top=15):
    print("stopped 點最多的物體：類別/instance、出現的幀、各速度分組的幀數、最高速度、暫停時的距離中位數、暫停的點、被抓到的幀數")
    rows = []
    for key, d in rec.items():
        st = sum(n for n, _, bn in d.values() if bn.startswith("stopped"))
        if st == 0:
            continue
        ks = sorted(d)
        nb = [sum(1 for k in ks if d[k][2].startswith(x)) for x in ("stopped", "slow", "moving", "unknown")]
        det = sum(1 for k in ks if d[k][1] >= max(10, 0.2 * d[k][0]))
        vmax = max(rng[key][k][1] for k in ks)
        rs = [rng[key][k][0] for k in ks if d[k][2].startswith("stopped") and rng[key][k][0] >= 0]
        rows.append((st, key, ks[0], ks[-1], nb, vmax, float(np.median(rs)) if rs else -1, det))
    for st, key, k0, k1, nb, vmax, r, det in sorted(rows, reverse=True)[:top]:
        print(f"  {NAMES.get(key[0], key[0]):8s} {key[1]:5d}  幀 {k0:4d}-{k1:4d}  停/慢/快/未知 {nb}  vmax {vmax:5.1f} m/s  距離 {r:5.1f} m  暫停點 {st:7d}  抓到 {det} 幀")


if __name__ == "__main__":
    main()
