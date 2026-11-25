"""候選階段的召回率：參考幀的時間範圍從 ±0.5 秒拉長到整段序列，候選點能多抓到多少。

對應 AutoMOS Table I 上半部（只做地圖清理、不追蹤）的那幾列：這一步的召回率決定了整個流程的召回率上限。
輸出兩種召回率：
  點：真值移動點裡，被標成候選的比例。
  物體：把同一個真值物體的點合起來看，候選比例 ≥ 5% 的物體，它的點全部算抓到（對應「群內候選比例 ≥ 5% 才送追蹤」）。

用法：python checks/candidate_recall.py --data seq08 --every 10
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from consistency import ConsistencyDetector  # noqa: E402
from eval_by_speed import BINS, inst_centers  # noqa: E402
from ground import ego_mask, ground_mask  # noqa: E402
from kitti_io import load_dataset  # noqa: E402

SETS = [("±0.5 秒（第一版）", [5], 1),
        ("±0.5、1、2 秒", [5, 10, 20], 1),
        ("到 ±8 秒", [5, 10, 20, 40, 80], 1),
        ("到 ±8 秒，至少 3 票（預設）", [5, 10, 20, 40, 80], 3),
        ("到 ±64 秒", [5, 10, 20, 40, 80, 160, 320, 640], 1),
        ("到 ±64 秒，至少 2 票", [5, 10, 20, 40, 80, 160, 320, 640], 2),
        ("到 ±64 秒，至少 3 票", [5, 10, 20, 40, 80, 160, 320, 640], 3)]
ALL = [5, 10, 20, 40, 80, 160, 320, 640]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="seq08")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--ego-radius", type=float, default=2.0)
    ap.add_argument("--det-kw", default="", help='ConsistencyDetector 參數；"need_return=False" 是 2026-11-25 之前的做法')
    args = ap.parse_args()
    ds = load_dataset(args.data)
    det = ConsistencyDetector(ds, cache_size=60, **eval("dict(" + args.det_kw + ")"))
    cache = {}

    def centers(k):
        if k not in cache:
            if len(cache) > 30:
                cache.pop(min(cache))
            cache[k] = inst_centers(ds, k)[3]
        return cache[k]

    gt_bin = np.zeros(4, np.int64)
    S = {n: dict(tp=np.zeros(4, np.int64), obj=np.zeros(4, np.int64), fp=0) for n, _, _ in SETS}
    for k in range(0, ds.n, args.every):
        pts = ds.scan(k)
        sem, inst, mov, cen = inst_centers(ds, k)
        g = ground_mask(pts) | ego_mask(pts, args.ego_radius)
        sel = ~g & (np.linalg.norm(pts[:, :2], axis=1) < det.max_range)
        xyz = pts[sel, :3]
        votes = {}
        for gp in ALL:
            v = np.zeros(sel.sum(), np.int64)
            for j in (k - gp, k + gp):
                if 0 <= j < ds.n:
                    v += det._against(k, j, xyz)
            votes[gp] = v
        keys = set(zip(sem[mov].tolist(), inst[mov].tolist()))
        masks = {}
        for key in keys:
            m = (sem == key[0]) & (inst == key[1])
            v, b = None, 3
            if key in cen:
                ends = [centers(j).get(key) for j in (k - 5, k + 5) if 0 <= j < ds.n]
                ends = [e for e in ends if e is not None]
                if len(ends) == 2:
                    v = np.linalg.norm(ends[1] - ends[0])
                elif len(ends) == 1:
                    v = np.linalg.norm(ends[0] - cen[key]) / 0.5
            if v is not None:
                b = next(i for i, (_, lo, hi) in enumerate(BINS) if lo <= v < hi)
            masks[key] = (m, b)
            gt_bin[b] += m.sum()
        for name, gaps, mv in SETS:
            c = np.zeros(len(pts), bool)
            c[sel] = sum(votes[gp] for gp in gaps) >= mv
            S[name]["fp"] += int((c & ~mov).sum())
            for key, (m, b) in masks.items():
                n = int((c & m).sum())
                S[name]["tp"][b] += n
                if n >= 0.05 * max(int((m & sel).sum()), 1) and n > 0:
                    S[name]["obj"][b] += m.sum()
    G = gt_bin.sum()
    print(f"== {args.data}，每 {args.every} 幀取一幀，{args.det_kw or '預設（need_return=True）'}；真值移動點 {G:,}（stopped {gt_bin[0]:,} / slow {gt_bin[1]:,} / moving {gt_bin[2]:,}）")
    print(f"  {'參考幀範圍':22s} {'點 P':>6s} {'點 R':>6s} | 點 R: stopped  slow  moving | {'物體 R':>6s} | 物體 R: stopped  slow  moving")
    for name, _, _ in SETS:
        s = S[name]
        tp = s["tp"].sum()
        r = lambda a: "  ".join(f"{100*a[i]/max(gt_bin[i],1):6.1f}" for i in range(3))  # noqa: E731
        print(f"  {name:22s} {100*tp/max(tp+s['fp'],1):6.1f} {100*tp/G:6.1f} | {r(s['tp'])} | {100*s['obj'].sum()/G:6.1f} | {r(s['obj'])}")


if __name__ == "__main__":
    main()
