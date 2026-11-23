"""步驟 ②：分群。

非地面、max_range 內的點 → voxel 降採樣（只用來加速 DBSCAN）→ x-y 平面 DBSCAN（eps = 1 m，對應 IJRR 2007 §5.1）
→ 把群編號傳回所有原始點。
回傳 cluster_id (N,)：-1 = 不屬於任何群（地面、太遠、雜訊、被過濾掉的群）。
"""
import numpy as np
from sklearn.cluster import DBSCAN


def cluster(pts_xyz, nonground, max_range=40.0, voxel=0.15, eps=1.0, min_samples=3,
            min_points=5, max_extent=20.0):
    n = len(pts_xyz)
    cid = np.full(n, -1, dtype=np.int64)
    sel = np.where(nonground & (np.linalg.norm(pts_xyz[:, :2], axis=1) < max_range))[0]
    if len(sel) == 0:
        return cid, []
    v = np.floor(pts_xyz[sel] / voxel).astype(np.int64)
    _, rep, inv = np.unique(v, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    lab_v = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts_xyz[sel[rep], :2])
    lab = lab_v[inv]
    clusters = []
    new_id = 0
    for c in np.unique(lab):
        if c < 0:
            continue
        idx = sel[lab == c]
        p = pts_xyz[idx]
        ext = p[:, :2].max(0) - p[:, :2].min(0)
        if len(idx) < min_points or ext.max() > max_extent:
            continue
        cid[idx] = new_id
        clusters.append(dict(id=new_id, idx=idx, center=p.mean(0),
                             bbox_min=p.min(0), bbox_max=p.max(0), extent=float(ext.max())))
        new_id += 1
    return cid, clusters
