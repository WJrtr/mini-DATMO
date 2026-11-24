"""用真值 instance 追蹤 Labeler.run 三個階段的數值（序列 08 前 200 幀，refine=0.5）。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from consistency import transform
from kitti_io import load_dataset
from labeler import Labeler

ds = load_dataset("seq08")
L = Labeler(ds, refine=0.5)
a, b = (int(x) for x in sys.argv[1].split(":")) if len(sys.argv) > 1 else (0, 200)
frames = range(a, b)
out = L.run(frames)
# (frame, cluster_id) -> track
key2t = {}
for t in L.track_info:
    for h in t["history"]:
        key2t[h] = t

def trace(sem_id, inst_id, show):
    print(f"\n===== GT instance ({sem_id},{inst_id}) =====")
    tracks_seen = {}
    rows = []
    for k in frames:
        lab = np.fromfile(ds.label_paths[k], dtype=np.uint32)
        m = ((lab & 0xFFFF) == sem_id) & ((lab >> 16) == inst_id)
        if m.sum() == 0:
            continue
        pts = ds.scan(k)
        cand = out[k]["cand"]
        w = transform(pts[m, :3].mean(0)[None], ds.T_w_velo[k])[0]
        # 找包含最多這個物體點的群（只看有被追蹤的群）
        best, bestn = None, 0
        for cid, q in L.per_frame[k]["clusters"].items():
            n = np.isin(q["idx"], np.where(m)[0]).sum()
            if n > bestn:
                best, bestn = (cid, q), n
        t = key2t.get((k, best[0])) if best else None
        if t:
            tracks_seen[t["id"]] = t
        rows.append((k, int(m.sum()), cand[m].mean(), w[:2], best, bestn, t, out[k]["label"][m].mean()))
    for r in rows:
        k, n, cr, w, best, bestn, t, lr = r
        if k not in show:
            continue
        cl = (f"群 {best[0]}: {len(best[1]['idx'])} 點, extent {best[1]['extent']:.2f} m, 候選比例 {best[1]['ratio']:.2f}"
              if best else "沒有進入追蹤的群（候選比例 < 0.05 或 extent ≥ 8 m）")
        tr = f"track {t['id']}" if t else "無 track"
        print(f" k={k:3d} | 真值 {n:4d} 點, 世界中心 ({w[0]:6.1f},{w[1]:6.1f}) | ⑤b 候選 {100*cr:5.1f}% | {cl} | {tr} | 標成移動 {100*lr:5.1f}%")
    for tid, t in tracks_seen.items():
        print(f"  track {tid}: 長度 {t['n']} 幀, 淨位移 {t['disp']:.2f} m, 直線度 {t['straight']:.2f}, 外框中位數 {t['extent']:.2f} m, 平均候選比例 {t['ratio']:.2f} → moving={t['moving']}")
    tot = sum(r[1] for r in rows); lab = sum(r[1] * r[7] for r in rows)
    print(f"  整段：真值點 {tot}，被標成移動 {100*lab/tot:.1f}%")

if len(sys.argv) > 2:
    sid, iid = map(int, sys.argv[2].split(","))
    trace(sid, iid, show=set(range(a, b, int(sys.argv[3]) if len(sys.argv) > 3 else 10)))
else:
    trace(252, 4, show={10, 20, 30, 40, 50, 60, 70, 78})
    trace(252, 3, show={0, 20, 40, 60, 80, 100, 120})
