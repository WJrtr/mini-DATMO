"""多間隔 ⑤b + 分群投票（群內候選比例 > thr → 整群標成移動）。"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from consistency import ConsistencyDetector
from cluster import cluster
from evaluate import Counter
from ground import ground_mask
from kitti_io import load_dataset

ds = load_dataset("seq08")
frames = range(0, 200, 4)
gaps_sets = {"g5": [5], "g5,10,20": [5, 10, 20]}
dets = {g: ConsistencyDetector(ds, gap=g, merge="or") for g in [5, 10, 20]}
shared = {}
for d in dets.values(): d._cache = shared
res = {}
t0 = time.time()
for k in frames:
    pts = ds.scan(k); gt = ds.labels(k); g = ground_mask(pts)
    cand = {g_: dets[g_].candidates(k, pts[:, :3], ~g) for g_ in [5, 10, 20]}
    cid, cl = cluster(pts[:, :3], ~g)
    for name, gs in gaps_sets.items():
        c = np.logical_or.reduce([cand[x] for x in gs])
        res.setdefault((name, "point"), Counter()).add(c, gt)
        ratio = np.zeros(len(cl))
        for i, q in enumerate(cl):
            ratio[i] = c[q["idx"]].mean()
        for thr in [0.1, 0.2, 0.3, 0.5]:
            lab = np.zeros(len(pts), bool)
            for i, q in enumerate(cl):
                if ratio[i] > thr: lab[q["idx"]] = True
            res.setdefault((name, f"cluster>{thr}"), Counter()).add(lab, gt)
print(f"{(time.time()-t0)/len(frames):.2f} s/frame")
for k, v in res.items(): print(k, v)
