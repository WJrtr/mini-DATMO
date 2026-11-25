"""步驟 ⑤a⑥：完整標註器（離線，四趟）。

第 1 趟（逐幀）：去地面 → ⑤b 候選 → 只對候選點附近的非地面點分群 → 每群的候選比例、世界座標中心、外框大小。
第 2 趟（整段序列）：只把「候選比例 > track_ratio」的群拿去追蹤。
第 3 趟（逐 track）：用整條軌跡判斷移動（離線，所以移動前後的幀也一起標）：
    淨位移 > max(外框最長邊中位數 × disp_factor, min_disp)  且  平均候選比例 > ratio_thr
    → 這條 track 所有幀的群，裡面離候選點 refine 公尺內的點都標成移動。
第 4 趟（停留延伸，stop_extend）：移動 track 的兩端如果物體慢下來了，用所有非地面點繼續跟著它，
    把停著的那段也標成移動。說明見 _extend_stops。
其他點都是靜止。

預設值：參考幀 k±{5,10,20,40,80}、至少 3 票、參考幀那個方向要有回波才算看穿（need_return）、
只對候選點 0.5 m 內的點分群、群上限 20 m、自車遮罩 2.0 m、refine 0.5 m、停留延伸開啟。
SHORT_HORIZON 是第一版的設定（只看前後 0.5 秒、對所有非地面點分群），用來重現消融表的基準列：
    Labeler(ds, **SHORT_HORIZON)
"""
import numpy as np

from cluster import cluster
from consistency import ConsistencyDetector, transform
from ground import ego_mask, ground_mask
from tracker import Tracker


SHORT_HORIZON = dict(gap=5, min_votes=1, ego_radius=2.8, cand_grow=0.0, max_extent_track=8.0,
                     need_return=False, stop_extend=False)


class Labeler:
    def __init__(self, ds, poses=None, gap=(5, 10, 20, 40, 80), min_votes=3, merge="or", track_ratio=0.05,
                 ratio_thr=0.15, disp_factor=1.0, min_disp=0.5, max_extent_track=20.0, min_len=5, min_straight=0.6,
                 cluster_kw=None, refine=0.5, ratio_stat="mean", disp_mode="net", ego_radius=2.0, cand_grow=0.5,
                 stop_extend=True, stop_win=10, stop_drift=2.0, stop_max_size=8.0, stop_margin=0.2,
                 stop_eps=0.5, stop_min_pts=5, stop_miss=20, stop_disp_factor=1.5, stop_min_disp=2.0, **det_kw):
        # 第 4 趟（停留延伸）的參數，說明見 _extend_stops
        self.stop_extend, self.stop_win, self.stop_drift = stop_extend, stop_win, stop_drift
        self.stop_max_size, self.stop_margin, self.stop_eps = stop_max_size, stop_margin, stop_eps
        self.stop_min_pts, self.stop_miss, self.stop_disp_factor = stop_min_pts, stop_miss, stop_disp_factor
        self.stop_min_disp = stop_min_disp
        self.stop_debug = False  # True：ext_info 的每一幀多存框內點的編號（checks/try_stop_extend.py 用）
        self.cand_grow = cand_grow  # >0：只對「離候選點 cand_grow 公尺內」的非地面點分群（AutoMOS 的做法：對動態候選分群）
        self.disp_mode = disp_mode
        self.ego_radius = ego_radius
        self.ratio_stat = ratio_stat  # "mean" 或 "p90"：停下來的片段沒有候選點時，用 p90 不會被平均稀釋
        self.cluster_kw = cluster_kw or {}
        self.refine = refine  # >0：移動群裡只標「離候選點 refine 公尺內」的點
        self.min_len = min_len
        self.min_straight = min_straight
        self.ds = ds
        self.T = ds.T_w_velo if poses is None else poses
        self.det = ConsistencyDetector(ds, gap=gap, merge=merge, poses=self.T, min_votes=min_votes, **det_kw)
        self.track_ratio = track_ratio
        self.ratio_thr = ratio_thr
        self.disp_factor = disp_factor
        self.min_disp = min_disp
        self.max_extent_track = max_extent_track

    def run(self, frames, step_dt=0.1, hook=None):
        """hook(k, pts, g, cand, cid, clusters)：每幀第 1 趟結束時呼叫，診斷用（checks/funnel.py）。"""
        frames = list(frames)
        per_frame = {}
        tracker = Tracker(dt=step_dt)
        for k in frames:
            pts = self.ds.scan(k)
            g = ground_mask(pts) | ego_mask(pts, self.ego_radius)  # 地面和自己的車身都不處理，一律靜止
            cand = self.det.candidates(k, pts[:, :3], ~g)
            sel = ~g
            if self.cand_grow > 0:
                sel = np.zeros(len(pts), bool)
                if cand.any():
                    from scipy.spatial import cKDTree
                    ng = np.where(~g)[0]
                    d, _ = cKDTree(pts[cand, :2]).query(pts[ng, :2], distance_upper_bound=self.cand_grow)
                    sel[ng[np.isfinite(d)]] = True
            cid, cl = cluster(pts[:, :3], sel, **self.cluster_kw)
            keep = []
            for q in cl:
                q["ratio"] = float(cand[q["idx"]].mean())
                if q["ratio"] >= self.track_ratio and q["extent"] < self.max_extent_track:
                    q["world"] = transform(q["center"][None], self.T[k])[0]
                    if self.refine > 0 or self.stop_extend:
                        q["xyz"] = pts[q["idx"], :3].copy()  # 只留群內的點，整幀都存會佔好幾 GB
                    keep.append(q)
            if hook is not None:
                hook(k, pts, g, cand, cid, cl)
            per_frame[k] = dict(n=len(pts), cand=cand, clusters={q["id"]: q for q in keep})
            if self.stop_extend:
                per_frame[k]["ng"] = np.packbits(~g)  # 第 4 趟要用非地面遮罩，壓成 bit 存
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
            # "net"：淨位移（頭尾直線距離）；"path"：累積軌跡長度（AutoMOS 的規則）
            travel = disp if self.disp_mode == "net" else path
            moving = (t.confirmed and len(qs) >= self.min_len and straight >= self.min_straight
                      and travel > max(ext * self.disp_factor, self.min_disp) and ratio > self.ratio_thr)
            self.track_info.append(dict(id=t.id, n=len(qs), confirmed=t.confirmed, disp=disp, path=path,
                                        straight=straight, extent=ext, ratio=ratio, moving=moving))
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
                            xyz = q["xyz"]
                            d, _ = cKDTree(xyz[c]).query(xyz)
                            idx = idx[d < self.refine]
                    lab[idx] = True
            out[k] = dict(label=lab, cand=per_frame[k]["cand"])
        if self.stop_extend:
            extra = self._extend_stops(frames, per_frame, [t for t in self.track_info if t["moving"]])
            for k, m in extra.items():
                out[k]["ext"] = m & ~out[k]["label"]  # 只靠停留延伸標到的點（診斷用）
                out[k]["label"] = out[k]["label"] | m
        return out

    def _extend_stops(self, frames, per_frame, moving_tracks):
        """第 4 趟：停留延伸。

        物體停下來之後，參考幀看不穿它，候選點會消失，track 就斷了；SemanticKITTI 的真值卻把
        「序列中動過的物體」整段都標成移動。所以從移動 track 的兩端，用「所有非地面點」繼續跟著這個物體：
          1. 移動 track 的尾端（或開頭）stop_win 幀，群中心的移動 <= stop_drift 公尺（物體慢下來了）。
             用端點那一幀的群點（世界座標）的外框當起點。
          2. 往後（開頭則往前）逐幀：外框（放大 stop_margin）裡的非地面點當種子，對附近的非地面點
             分群（DBSCAN，eps = stop_eps），取種子最多的那一群當成這個物體。
             群的最長邊 <= stop_max_size（沒有和牆、樹籬連在一起）→ 整群標成移動，外框更新成這一群的外框。
          3. 連續 stop_miss 幀不成立（物體開走、被擋住、離開 40 m、和別的東西連在一起）就停止。
          4. 最後檢查：track 的淨位移 >= max(stop_disp_factor × 物體大小, stop_min_disp) 才採用
             （物體大小 = 延伸期間群最長邊的中位數）。
             第 3 趟比的是「候選點附近那一小群」的大小；靜止物體上的誤判 track，候選點只是在物體表面上移動，
             位移不會超過整個物體的大小，這裡用整個物體的大小再檢查一次。stop_min_disp 擋掉樹幹、
             桿子這類很細的靜止物體（候選點在樹冠範圍內移動 1 m 左右，就超過樹幹的寬度）。
        回傳 {幀: bool 遮罩}。self.ext_info 記錄每個延伸：是否採用（valid）、物體大小（size），
        以及每一幀的 (幀, 種子點數, 群大小, 是否成立, 點編號)。
        """
        W, mg = self.stop_win, self.stop_margin
        exts = []
        for t in moving_tracks:
            h = t["history"]
            if len(h) < W:
                continue
            for d in (1, -1):
                part = h[-W:] if d == 1 else h[:W]
                k_end, c_end = part[-1] if d == 1 else part[0]
                cen = np.array([per_frame[k]["clusters"][c]["world"][:2] for k, c in part])
                drift = float(np.linalg.norm(cen - (cen[-1] if d == 1 else cen[0]), axis=1).max())
                if drift > self.stop_drift:
                    continue
                pw = transform(per_frame[k_end]["clusters"][c_end]["xyz"], self.T[k_end])
                exts.append(dict(track=t["id"], dir=d, k0=k_end, lo=pw.min(0), hi=pw.max(0), drift=drift,
                                 disp=t["disp"], miss=0, frames=[]))
        self.ext_info = exts
        max_range = self.det.max_range
        S = self.stop_max_size
        for d in (1, -1):
            pend = sorted([e for e in exts if e["dir"] == d], key=lambda e: d * e["k0"])
            active, i = [], 0
            for k in (frames if d == 1 else frames[::-1]):
                while i < len(pend) and d * pend[i]["k0"] <= d * k:
                    active.append(pend[i])
                    i += 1
                if not active:
                    continue
                pts = self.ds.scan(k)
                ng = np.unpackbits(per_frame[k]["ng"])[:len(pts)].astype(bool)
                ids = np.where(ng & (np.linalg.norm(pts[:, :2], axis=1) < max_range))[0]
                w = transform(pts[ids, :3], self.T[k])
                ego = self.T[k][:2, 3]
                still = []
                for e in active:
                    lo, hi = e["lo"] - mg, e["hi"] + mg
                    ok, sel = False, None
                    if np.linalg.norm((lo[:2] + hi[:2]) / 2 - ego) < max_range:
                        # 附近的點：水平方向外框再加 stop_max_size，高度只多看 0.5 m（不把頭上的樹冠算進來）
                        near = np.where(np.all((w[:, :2] >= lo[:2] - S) & (w[:, :2] <= hi[:2] + S), axis=1)
                                        & (w[:, 2] >= lo[2] - 0.5) & (w[:, 2] <= hi[2] + 0.5))[0]
                        wn = w[near]
                        seed = np.all((wn >= lo) & (wn <= hi), axis=1)
                        n_seed, size = int(seed.sum()), -1.0
                        if n_seed >= self.stop_min_pts:
                            cid, cl = cluster(wn, np.ones(len(wn), bool), max_range=1e9, eps=self.stop_eps,
                                              max_extent=1e9)
                            c = cid[seed]
                            c = c[c >= 0]
                            if len(c):
                                q = cl[int(np.bincount(c).argmax())]
                                size = q["extent"]
                                if size <= S:
                                    ok = True
                                    e["lo"], e["hi"] = q["bbox_min"], q["bbox_max"]
                                    sel = ids[near[q["idx"]]]
                        e["frames"].append((k, n_seed, size, ok, sel))
                    e["miss"] = 0 if ok else e["miss"] + 1
                    if e["miss"] <= self.stop_miss:
                        still.append(e)
                active = still
        extra = {}
        for e in exts:
            sizes = [f[2] for f in e["frames"] if f[3]]
            e["size"] = float(np.median(sizes)) if sizes else -1.0
            e["valid"] = bool(sizes) and e["disp"] >= max(self.stop_disp_factor * e["size"], self.stop_min_disp)
            if e["valid"]:
                for k, _, _, ok, sel in e["frames"]:
                    if ok:
                        if k not in extra:
                            extra[k] = np.zeros(per_frame[k]["n"], dtype=bool)
                        extra[k][sel] = True
            if not self.stop_debug:
                e["frames"] = [f[:4] for f in e["frames"]]  # 點編號只在診斷時留著
        return extra
