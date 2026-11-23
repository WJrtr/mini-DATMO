"""診斷 ⑤b 漏判：漏掉的移動點，是沒有資訊（r_ref = inf）、被擋住（r > r_ref），還是差一點（δ 內）？
另外用 GT instance id 估每個移動物體的世界座標速度。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from polar import lookup, range_image
from consistency import transform
from ground import ground_mask
from kitti_io import load_dataset

ds = load_dataset(sys.argv[1] if len(sys.argv) > 1 else "seq08")
a, b = 0, 200
lab = lambda k: np.fromfile(ds.label_paths[k], dtype=np.uint32)
stats = dict(inf=0, behind=0, within=0, hit=0, total=0)
per_inst = {}
for k in range(a, b, 4):
    pts = ds.scan(k); L = lab(k); sem = L & 0xFFFF; inst = L >> 16
    mov = (sem >= 251) & (sem <= 259); g = ground_mask(pts)
    sel = mov & ~g
    best = np.full(sel.sum(), np.inf)  # 兩個方向中 r - r_ref 最小的
    anyinfo = np.zeros(sel.sum(), bool)
    for j in (k - 5, k + 5):
        if not (0 <= j < ds.n): continue
        img = range_image(ds.scan(j)[:, :3], neighborhood=0)
        r, rr = lookup(img, transform(pts[sel, :3], np.linalg.inv(ds.T_w_velo[j]) @ ds.T_w_velo[k]))
        anyinfo |= np.isfinite(rr)
        best = np.minimum(best, r - rr)
    stats["total"] += sel.sum()
    stats["inf"] += (~anyinfo).sum()
    stats["hit"] += (best < -0.5).sum()
    stats["within"] += ((best >= -0.5) & (best <= 0.5)).sum()
    stats["behind"] += (anyinfo & (best > 0.5)).sum()
    w = transform(pts[:, :3], ds.T_w_velo[k])
    for i in np.unique(inst[sel]):
        m = sel & (inst == i)
        per_inst.setdefault((int(sem[m][0]), int(i)), []).append((k, w[m].mean(0), m.sum(), np.linalg.norm(pts[m, :2], axis=1).mean()))
print({k: (v, f"{100*v/max(stats['total'],1):.1f}%") for k, v in stats.items()})
print("instances (class, id): frames, mean speed m/s, mean pts, mean range")
for key, v in sorted(per_inst.items(), key=lambda kv: -sum(x[2] for x in kv[1]))[:15]:
    ks = np.array([x[0] for x in v]); c = np.array([x[1] for x in v])
    sp = np.linalg.norm(c[-1, :2] - c[0, :2]) / max((ks[-1] - ks[0]) * 0.1, 1e-6) if len(v) > 1 else float("nan")
    print(key, len(v), f"{sp:.1f}", int(np.mean([x[2] for x in v])), f"{np.mean([x[3] for x in v]):.1f} m")
