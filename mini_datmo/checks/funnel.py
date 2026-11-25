"""逐階段漏斗分析：真值移動點在流程的哪一步被丟掉，以及換判斷規則能拿回多少。

標註器只跑一次（第 1、2 趟最花時間），每個群記下「真值移動點數（依速度分組）」和「非移動點數」。
第 3 趟的判斷規則只用到每條 track 的統計量，所以換規則不必重跑，直接用記下來的點數重算 P / R / IoU。

用法：python checks/funnel.py --data seq08 [--frames 1500:1800,3000:3300] [--kw "max_extent_track=8"] [--json out.json]
輸出：
  1. 漏斗：每個速度分組的真值移動點，在哪一步被丟掉（占該組真值點的百分比）
  2. 規則比較：淨位移、累積軌跡長度、放寬直線度，以及三個上限（oracle）
  3. 誤判（FP）來源：整條 track 都沒有真值移動點（純誤判軌跡）或混在移動物體的群裡，以及真值類別
漏斗和規則比較只看第 1–3 趟（候選、分群、追蹤、判斷）。第 4 趟（停留延伸）另外列一行：它多拿回多少點、多了多少誤判。
"""
import argparse
import json
import os
import sys
from collections import Counter as Cnt

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_by_speed import BINS, inst_centers  # noqa: E402
from kitti_io import load_dataset, write_label  # noqa: E402
from labeler import Labeler  # noqa: E402

BN = [b for b, _, _ in BINS] + ["unknown"]
NAMES = {0: "unlabeled", 1: "outlier", 10: "car", 11: "bicycle", 13: "bus", 15: "motorcycle", 18: "truck",
         20: "other-vehicle", 30: "person", 31: "bicyclist", 32: "motorcyclist", 40: "road", 44: "parking",
         48: "sidewalk", 49: "other-ground", 50: "building", 51: "fence", 52: "other-structure",
         60: "lane-marking", 70: "vegetation", 71: "trunk", 72: "terrain", 80: "pole", 81: "traffic-sign",
         99: "other-object"}
Z = lambda: np.zeros(4, np.int64)  # noqa: E731


def bc(x):
    return np.bincount(x, minlength=4).astype(np.int64)


def decide(t, mode="net", min_straight=0.6, min_len=5, ratio_thr=0.15, min_disp=0.5):
    """回傳 (是否移動, 第一個沒通過的條件)。條件順序和 labeler.py 第 3 趟相同。"""
    if not (t["confirmed"] and t["n"] >= min_len):
        return False, "track 太短或未確認"
    if t["straight"] < min_straight:
        return False, "直線度不足"
    travel = t["disp"] if mode == "net" else t["path"]
    if not travel > max(t["extent"], min_disp):
        return False, "位移不足"
    if not t["ratio"] > ratio_thr:
        return False, "候選比例不足"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="seq08")
    ap.add_argument("--frames", default=None, help="a:b，可用逗號分開多段")
    ap.add_argument("--kw", default="", help='Labeler 參數，例如 "max_extent_track=8"；空字串 = 預設值')
    ap.add_argument("--json", default=None)
    ap.add_argument("--out", default=None, help="同時把標籤寫到 <out>/predictions")
    args = ap.parse_args()
    kw = eval("dict(" + args.kw + ")")
    ds = load_dataset(args.data)
    wins = [(0, ds.n)] if args.frames is None else [tuple(map(int, w.split(":"))) for w in args.frames.split(",")]

    gtot = Z()
    pre = {s: Z() for s in ["地面或自車", "超過 40 m", "沒有分到群", "群超過追蹤上限", "群內候選比例 < 5%"]}
    tracks = []      # 每條 track：統計量 + gt / ngt（整群）+ rgt / rngt（refine 之後）+ 類別
    untracked = dict(gt=Z(), ngt=0, rgt=Z(), rngt=0)   # 有送進追蹤、但 track 只有 1 幀
    allcl = dict(gt_major=Z(), ngt_major=0)            # 上限：所有群裡，真值移動點過半的群
    keptcl = dict(gt_major=Z(), ngt_major=0)           # 上限：送進追蹤的群裡，真值移動點過半的群（refine 之後）
    actual = np.zeros(3, np.int64)                     # 第 1–3 趟的 TP / FP / FN，用來核對
    ext_tp, ext_fp, ext_cls = Z(), 0, Cnt()            # 只靠第 4 趟（停留延伸）標到的點
    cache = {}

    def centers(k):
        if k not in cache:
            if len(cache) > 30:
                cache.pop(min(cache))
            cache[k] = inst_centers(ds, k)[3]
        return cache[k]

    for a, b in wins:
        lab = Labeler(ds, **kw)
        max_range = lab.cluster_kw.get("max_range", 40.0)
        recs = {}
        movs = {}
        bins, sems = {}, {}

        def hook(k, pts, g, cand, cid, cl):
            sem, inst, mov, cen = inst_centers(ds, k)
            cache[k] = cen
            binid = np.full(len(sem), 3, np.int64)
            for key in set(zip(sem[mov].tolist(), inst[mov].tolist())):
                v = None
                if key in cen:
                    ends = [centers(j).get(key) for j in (k - 5, k + 5) if 0 <= j < ds.n]
                    ends = [e for e in ends if e is not None]
                    if len(ends) == 2:
                        v = np.linalg.norm(ends[1] - ends[0]) / 1.0
                    elif len(ends) == 1:
                        v = np.linalg.norm(ends[0] - cen[key]) / 0.5
                if v is not None:
                    binid[(sem == key[0]) & (inst == key[1])] = next(i for i, (_, lo, hi) in enumerate(BINS) if lo <= v < hi)
            movs[k] = mov
            bins[k] = binid[mov].astype(np.int8)
            sems[k] = sem.astype(np.uint16)
            gtot[:] += bc(binid[mov])
            far = ~g & (np.linalg.norm(pts[:, :2], axis=1) >= max_range)
            pre["地面或自車"] += bc(binid[mov & g])
            pre["超過 40 m"] += bc(binid[mov & far])
            pre["沒有分到群"] += bc(binid[mov & ~g & ~far & (cid < 0)])
            for q in cl:
                idx = q["idx"]
                m, bi = mov[idx], binid[idx]
                gt, ngt = bc(bi[m]), int((~m).sum())
                if gt.sum() > ngt:
                    allcl["gt_major"] += gt
                    allcl["ngt_major"] += ngt
                if q["extent"] >= lab.max_extent_track:
                    pre["群超過追蹤上限"] += gt
                    continue
                if q["ratio"] < lab.track_ratio:
                    pre["群內候選比例 < 5%"] += gt
                    continue
                keep = np.ones(len(idx), bool)
                c = cand[idx]
                if lab.refine > 0 and c.any():
                    xyz = pts[idx, :3]
                    keep = cKDTree(xyz[c]).query(xyz)[0] < lab.refine
                rgt, rngt = bc(bi[m & keep]), int((~m & keep).sum())
                if rgt.sum() > rngt:
                    keptcl["gt_major"] += rgt
                    keptcl["ngt_major"] += rngt
                recs[(k, q["id"])] = dict(gt=gt, ngt=ngt, rgt=rgt, rngt=rngt, cls=Cnt(sem[idx][~m & keep].tolist()))

        out = lab.run(range(a, b), hook=hook)
        for k, o in out.items():
            p, gt = o["label"], movs[k]
            if args.out:
                write_label(os.path.join(args.out, "predictions", f"{k:06d}.label"), p)
            if "ext" in o:
                e = o["ext"]
                ext_tp += bc(bins[k][e[gt]].astype(np.int64))
                ext_fp += int((e & ~gt).sum())
                ext_cls.update(sems[k][e & ~gt].tolist())
                p = p & ~e
            actual += [np.sum(p & gt), np.sum(p & ~gt), np.sum(~p & gt)]
        seen = set()
        for t in lab.track_info:
            r = dict(gt=Z(), ngt=0, rgt=Z(), rngt=0, cls=Cnt())
            for key in t["history"]:
                q = recs[key]
                seen.add(key)
                for f in ("gt", "ngt", "rgt", "rngt"):
                    r[f] = r[f] + q[f]
                r["cls"].update(q["cls"])
            r.update({f: t[f] for f in ("n", "confirmed", "disp", "path", "straight", "extent", "ratio", "moving")})
            tracks.append(r)
        for key, q in recs.items():
            if key not in seen:
                for f in ("gt", "ngt", "rgt", "rngt"):
                    untracked[f] = untracked[f] + q[f]
        print(f"  {a}:{b} 完成，{len(lab.track_info)} 條 track", flush=True)

    G = gtot.sum()
    pct = lambda v, d: 100.0 * v / max(d, 1)  # noqa: E731

    def evaluate(pick):
        """pick(t) → 這條 track 是否標成移動。回傳 TP（依速度）、FP。"""
        tp, fp = Z(), 0
        for t in tracks:
            if pick(t):
                tp, fp = tp + t["rgt"], fp + t["rngt"]
        return tp, fp

    def line(name, tp, fp):
        s = tp.sum()
        return (f"  {name:34s} P {pct(s, s+fp):5.1f}  R {pct(s, G):5.1f}  IoU {pct(s, G+fp):5.1f}   "
                + "  ".join(f"{pct(tp[i], gtot[i]):5.1f}" for i in range(3)))

    print(f"\n== {args.data} {wins}  kw={kw}")
    tp0, fp0 = evaluate(lambda t: t["moving"])
    print(f"  核對：第 1–3 趟實際輸出 TP {actual[0]:,} FP {actual[1]:,} FN {actual[2]:,}；"
          f"由 track 統計重算 TP {tp0.sum():,} FP {fp0:,}")
    ftp, ffp = actual[0] + ext_tp.sum(), actual[1] + ext_fp
    print(f"  第 4 趟（停留延伸）另外拿回 TP {ext_tp.sum():,}（占各組真值點 "
          + " / ".join(f"{n} {pct(ext_tp[i], gtot[i]):.1f}%" for i, n in enumerate(BN[:3]))
          + f"），多出 FP {ext_fp:,}（"
          + "  ".join(f"{NAMES.get(c, str(c))} {pct(n, ext_fp):.0f}%" for c, n in ext_cls.most_common(4)) + "）")
    print(f"  最後輸出：P {pct(ftp, ftp + ffp):.1f}  R {pct(ftp, G):.1f}  IoU {pct(ftp, G + ffp):.1f}")

    # 1. 漏斗
    result = dict(data=args.data, frames=wins, kw=args.kw, gt_points={n: int(v) for n, v in zip(BN, gtot)})
    result["stop_extend"] = dict(tp=[int(x) for x in ext_tp], fp=int(ext_fp))
    for mode in ("net", "path"):
        stages = dict(pre)
        stages["track 只有 1 幀"] = untracked["gt"]
        for s in ["track 太短或未確認", "直線度不足", "位移不足", "候選比例不足", "refine 去掉", "抓到（TP）"]:
            stages[s] = Z()
        for t in tracks:
            ok, why = decide(t, mode)
            if ok:
                stages["抓到（TP）"] = stages["抓到（TP）"] + t["rgt"]
                stages["refine 去掉"] = stages["refine 去掉"] + t["gt"] - t["rgt"]
            else:
                stages[why] = stages[why] + t["gt"]
        print(f"\n== 漏斗（判斷規則：{'淨位移' if mode == 'net' else '累積軌跡長度'}）：各速度分組的真值移動點在哪一步被丟掉，單位 %")
        print(f"  {'階段':22s}" + "".join(f"{n:>16s}" for n in BN[:3]) + f"{'全部':>10s}")
        for s, v in stages.items():
            print(f"  {s:22s}" + "".join(f"{pct(v[i], gtot[i]):16.1f}" for i in range(3)) + f"{pct(v.sum(), G):10.1f}")
        print(f"  {'真值點數':22s}" + "".join(f"{gtot[i]:16,d}" for i in range(3)) + f"{G:10,d}")
        chk = sum(v.sum() for v in stages.values())
        if chk != G:
            print(f"  （各階段加總 {chk:,} ≠ 真值 {G:,}）")
        result[f"funnel_{mode}"] = {s: [int(x) for x in v] for s, v in stages.items()}

    # 2. 規則比較
    print("\n== 規則比較（同一次執行的群和 track，只換第 3 趟的判斷）；最後三欄是 stopped / slow / moving 的召回率")
    rules = [
        ("淨位移，直線度 ≥ 0.6（目前）", lambda t: decide(t, "net")[0]),
        ("累積長度，直線度 ≥ 0.6", lambda t: decide(t, "path")[0]),
        ("累積長度，直線度 ≥ 0.3", lambda t: decide(t, "path", 0.3)[0]),
        ("累積長度，不看直線度", lambda t: decide(t, "path", 0.0)[0]),
        ("淨位移，直線度 ≥ 0.3", lambda t: decide(t, "net", 0.3)[0]),
        ("上限：track 判斷全對", lambda t: t["rgt"].sum() > t["rngt"]),
    ]
    result["rules"] = {}
    for name, pick in rules:
        tp, fp = evaluate(pick)
        print(line(name, tp, fp))
        result["rules"][name] = dict(tp=[int(x) for x in tp], fp=int(fp))
    for name, d in [("上限：送進追蹤的群判斷全對", keptcl), ("上限：所有群判斷全對（不看候選）", allcl)]:
        print(line(name, d["gt_major"], d["ngt_major"]))
        result["rules"][name] = dict(tp=[int(x) for x in d["gt_major"]], fp=int(d["ngt_major"]))

    # 3. FP 來源
    result["fp"] = {}
    for mode in ("net", "path"):
        mv = [t for t in tracks if decide(t, mode)[0]]
        pure = [t for t in mv if t["gt"].sum() == 0]
        fp_pure = sum(t["rngt"] for t in pure)
        fp_all = sum(t["rngt"] for t in mv)
        cls_pure, cls_mix = Cnt(), Cnt()
        for t in mv:
            (cls_pure if t["gt"].sum() == 0 else cls_mix).update(t["cls"])
        top = lambda c: "  ".join(f"{NAMES.get(s, str(s))} {pct(n, fp_all):.0f}%" for s, n in c.most_common(5))  # noqa: E731
        print(f"\n== 誤判來源（{'淨位移' if mode == 'net' else '累積軌跡長度'}）：FP 共 {fp_all:,} 點，判成移動的 track {len(mv)} 條")
        print(f"  整條 track 沒有任何真值移動點：{len(pure)} 條，占 FP {pct(fp_pure, fp_all):.0f}%   {top(cls_pure)}")
        print(f"  track 含真值移動點（群裡混入其他點）：占 FP {pct(fp_all - fp_pure, fp_all):.0f}%   {top(cls_mix)}")
        result["fp"][mode] = dict(total=int(fp_all), pure_tracks=len(pure), pure=int(fp_pure),
                                  pure_cls={NAMES.get(s, str(s)): n for s, n in cls_pure.most_common(8)},
                                  mixed_cls={NAMES.get(s, str(s)): n for s, n in cls_mix.most_common(8)})

    if args.json:
        result["tracks"] = [dict(n=t["n"], confirmed=bool(t["confirmed"]), disp=t["disp"], path=t["path"],
                                 straight=t["straight"], extent=t["extent"], ratio=t["ratio"],
                                 gt=[int(x) for x in t["gt"]], ngt=int(t["ngt"]),
                                 rgt=[int(x) for x in t["rgt"]], rngt=int(t["rngt"])) for t in tracks]
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
