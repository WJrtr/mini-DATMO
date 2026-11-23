"""序列 08 一小段：去地面誤殺率 + ⑤b-only（OR / AND）vs. LMNet。
用法：python checks/check_consistency.py 0 200 [step]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from consistency import ConsistencyDetector  # noqa: E402
from evaluate import Counter  # noqa: E402
from ground import ground_mask  # noqa: E402
from kitti_io import load_dataset  # noqa: E402

a, b = int(sys.argv[1]), int(sys.argv[2])
step = int(sys.argv[3]) if len(sys.argv) > 3 else 1
ds = load_dataset("seq08")
det_or = ConsistencyDetector(ds, merge="or")
det_and = ConsistencyDetector(ds, merge="and")
det_and._cache = det_or._cache  # 共用距離圖快取

c = {k: Counter() for k in ["or", "and", "lmnet"]}
c40 = {k: Counter() for k in ["or", "and", "lmnet"]}
gt_moving = gt_moving_ground = gt_moving_far = 0
t0 = time.time()
for k in range(a, b, step):
    pts = ds.scan(k)
    gt = ds.labels(k)
    g = ground_mask(pts)
    near = np.linalg.norm(pts[:, :2], axis=1) < 40
    gt_moving += gt.sum(); gt_moving_ground += (gt & g).sum(); gt_moving_far += (gt & ~near).sum()
    p_or = det_or.candidates(k, pts[:, :3], ~g)
    p_and = det_and.candidates(k, pts[:, :3], ~g)
    lm = ds.lmnet(k)
    for name, p in [("or", p_or), ("and", p_and), ("lmnet", lm)]:
        c[name].add(p, gt)
        c40[name].add(p, gt, near)
dt = (time.time() - t0) / len(range(a, b, step))
print(f"frames {a}:{b}:{step}  {dt:.2f} s/frame")
print(f"GT moving points: {gt_moving}  被當成地面: {100*gt_moving_ground/max(gt_moving,1):.1f}%  在 40 m 外: {100*gt_moving_far/max(gt_moving,1):.1f}%")
for name in ["or", "and", "lmnet"]:
    print(f"  {name:6s} all: {c[name]}   |  <40m: {c40[name]}")
