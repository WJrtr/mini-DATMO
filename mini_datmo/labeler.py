"""步驟 ⑤a⑥：完整標註器（離線，三趟）。

第 1 趟（逐幀）：去地面 → ⑤b 候選 → 分群 → 每群的候選比例、世界座標中心、外框大小。
第 2 趟（整段序列）：只把「候選比例 > track_ratio」的群拿去追蹤。
第 3 趟（逐 track）：用整條軌跡判斷移動（離線，所以移動前後的幀也一起標）：
    淨位移 > max(外框最長邊中位數 × disp_factor, min_disp)  且  平均候選比例 > ratio_thr
    → 這條 track 所有幀的群，裡面的點都標成移動。
其他點都是靜止。
"""
import numpy as np

from cluster import cluster
from consistency import ConsistencyDetector, transform
from ground import ego_mask, ground_mask
from tracker import Tracker


class Labeler:
    def __init__(self, ds, poses=None, gap=5, merge="or", track_ratio=0.05, ratio_thr=0.15,
                 disp_factor=1.0, min_disp=0.5, max_extent_track=8.0, min_len=5, min_straight=0.6,
                 cluster_kw=None, refine=0.0, ratio_stat="mean", **det_kw):
        self.ratio_stat = ratio_stat  # "mean" 或 "p90"：停下來的片段沒有候選點時，用 p90 不會被平均稀釋
        self.cluster_kw = cluster_kw or {}
        self.refine = refine  # >0：移動群裡只標「離候選點 refine 公尺內」的點
        self.min_len = min_len
        self.min_straight = min_straight
        self.ds = ds
        self.T = ds.T_w_velo if poses is None else poses
        self.det = ConsistencyDetector(ds, gap=gap, merge=merge, poses=self.T, **det_kw)
        self.track_ratio = track_ratio
        self.ratio_thr = ratio_thr
        self.disp_factor = disp_factor
        self.min_disp = min_disp
        self.max_extent_track = max_extent_track

    def run(self, frames, step_dt=0.1):
        frames = list(frames)
        per_frame = {}
        tracker = Tracker(dt=step_dt)
        for k in frames:
            pts = self.ds.scan(k)
            g = ground_mask(pts) | ego_mask(pts)  # 地面和自己的車身都不處理，一律靜止
            cand = self.det.candidates(k, pts[:, :3], ~g)
            _, cl = cluster(pts[:, :3], ~g, **self.cluster_kw)
            keep = []
            for q in cl:
                q["ratio"] = float(cand[q["idx"]].mean())
                if q["ratio"] >= self.track_ratio and q["extent"] < self.max_extent_track:
                    q["world"] = transform(q["center"][None], self.T[k])[0]
                    keep.append(q)
            per_frame[k] = dict(n=len(pts), cand=cand, clusters={q["id"]: q for q in keep},
                                xyz=pts[:, :3] if self.refine > 0 else None)
            centers = np.array([q["world"][:2] for q in keep]).reshape(-1, 2)
            tracker.step(k, centers, [q["id"] for q in keep])
        # 第 3 趟
        moving_keys = set()
        self.track_info = []
        for t in tracker.all_tracks():
            if len(t.history) < 2:
                continue
            qs = [per_frame[k]["clusters"][cid] for k, cid in t.history]
            w = np.array([q["world"][:2] for q in qs])
            disp = float(np.linalg.norm(w[-1] - w[0]))
            path = float(np.sum(np.linalg.norm(np.diff(w, axis=0), axis=1)))
            straight = disp / max(path, 1e-6)  # 錯誤配對會讓軌跡跳來跳去，直線度很低
            ext = float(np.median([q["extent"] for q in qs]))
            rs = [q["ratio"] for q in qs]
            ratio = float(np.mean(rs) if self.ratio_stat == "mean" else np.percentile(rs, 90))
            moving = (t.confirmed and len(qs) >= self.min_len and straight >= self.min_straight
                      and disp > max(ext * self.disp_factor, self.min_disp) and ratio > self.ratio_thr)
            self.track_info.append(dict(id=t.id, n=len(qs), disp=disp, straight=straight, extent=ext,
                                        ratio=ratio, moving=moving))
            self.track_info[-1]["history"] = t.history
            if moving:
                moving_keys.update(t.history)
        self.per_frame = per_frame
        out = {}
        for k in frames:
            lab = np.zeros(per_frame[k]["n"], dtype=bool)
            for cid, q in per_frame[k]["clusters"].items():
                if (k, cid) in moving_keys:
                    idx = q["idx"]
                    if self.refine > 0:
                        c = per_frame[k]["cand"][idx]
                        if c.any():
                            from scipy.spatial import cKDTree
                            xyz = per_frame[k]["xyz"][idx]
                            d, _ = cKDTree(xyz[c]).query(xyz)
                            idx = idx[d < self.refine]
                    lab[idx] = True
            out[k] = dict(label=lab, cand=per_frame[k]["cand"])
        return out
