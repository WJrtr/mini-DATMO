"""依物體「當下速度」分組的召回率，以及整體精確率。

當下速度：用真值 (類別, instance) 找出同一物體在 k-5、k+5 幀的點，世界座標中心的位移 / 1 秒。
只有一邊看得到就用那一邊（/ 0.5 秒）。點太少（< 10）的不估速度，歸到 unknown。
分組：stopped < 0.5 m/s、slow 0.5–2、moving > 2。

用法：python eval_by_speed.py --pred out/seq08_g5/predictions [--frames 0:4071] [--json ...]
"""
import argparse
import json
import os

import numpy as np

from consistency import transform
from kitti_io import MOVING_MAX, MOVING_MIN, load_dataset, read_label_moving

BINS = [("stopped(<0.5)", 0, 0.5), ("slow(0.5-2)", 0.5, 2.0), ("moving(>2)", 2.0, 1e9)]


def inst_centers(ds, k):
    L = np.fromfile(ds.label_paths[k], dtype=np.uint32)
    sem, inst = L & 0xFFFF, L >> 16
    mov = (sem >= MOVING_MIN) & (sem <= MOVING_MAX)
    p = ds.scan(k)
    out = {}
    for s, i in set(zip(sem[mov].tolist(), inst[mov].tolist())):
        m = (sem == s) & (inst == i)
        if m.sum() >= 10:
            out[(s, i)] = transform(p[m, :3].mean(0)[None], ds.T_w_velo[k])[0][:2]
    return sem, inst, mov, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--data", default="seq08")
    ap.add_argument("--frames", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    ds = load_dataset(args.data)
    a, b = (0, ds.n) if args.frames is None else map(int, args.frames.split(":"))
    names = ["ours", "lmnet"] if ds.pred_paths else ["ours"]
    rec = {n: {bn: [0, 0] for bn, _, _ in BINS + [("unknown", 0, 0)]} for n in names}
    fp = {n: 0 for n in names}
    tp_all = {n: 0 for n in names}
    cache = {}

    def centers(k):
        if k not in cache:
            if len(cache) > 30:
                cache.pop(min(cache))
            cache[k] = inst_centers(ds, k)[3]
        return cache[k]

    for k in range(a, b):
        sem, inst, mov, cen = inst_centers(ds, k)
        cache[k] = cen
        preds = dict(ours=read_label_moving(os.path.join(args.pred, f"{k:06d}.label")))
        if ds.pred_paths:
            preds["lmnet"] = ds.lmnet(k)
        for n, p in preds.items():
            fp[n] += int(np.sum(p & ~mov))
            tp_all[n] += int(np.sum(p & mov))
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
            for n, p in preds.items():
                rec[n][bn][0] += int(m.sum())
                rec[n][bn][1] += int(np.sum(p[m]))
    res = {}
    for n in names:
        print(f"== {n}: precision {100*tp_all[n]/max(tp_all[n]+fp[n],1):.1f}%")
        res[n] = {"precision": tp_all[n] / max(tp_all[n] + fp[n], 1)}
        for bn, (tot, hit) in rec[n].items():
            print(f"   {bn:14s} GT pts {tot:9d}  recall {100*hit/max(tot,1):5.1f}%")
            res[n][bn] = dict(gt=tot, recall=hit / max(tot, 1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
