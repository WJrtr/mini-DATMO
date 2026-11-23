"""步驟 ⑤b：雙向極座標一致性偵測（IJRR 2007 §7.1 的離線版）。

對第 k 幀的每個非地面點，和參考幀 j ∈ {k-gap, k+gap} 比較：
  1. 參考幀用「所有點」（含地面）建距離圖：每條光束第一次打到東西的距離 r_ref。
  2. 把第 k 幀的點轉到參考幀的感測器座標：inv(T_w_velo[j]) · T_w_velo[k]。
  3. r < r_ref - δ(r) → 這個點落在參考幀看得穿的空間裡 → 候選移動。
     δ(r) = delta_abs + delta_rel · r（遠處光束較寬、位姿誤差被放大）。
雙向合併：merge="or" 任一方向是候選就算（召回率高）；"and" 兩個方向都要（精確率高）。
"""
import numpy as np

from polar import lookup, range_image


def transform(pts_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    return pts_xyz @ T[:3, :3].T + T[:3, 3]


class ConsistencyDetector:
    def __init__(self, ds, gap=5, delta_abs=0.5, delta_rel=0.02, max_range=40.0, merge="or",
                 poses=None, min_votes=1, cache_size=400):
        self.ds = ds
        self.min_votes = min_votes
        self.cache_size = cache_size
        self.gap = gap
        self.delta_abs = delta_abs
        self.delta_rel = delta_rel
        self.max_range = max_range
        self.merge = merge
        self.T = ds.T_w_velo if poses is None else poses  # 7-2：可以換成加了雜訊的位姿
        self._cache = {}

    def _ref_image(self, j):
        if j not in self._cache:
            if len(self._cache) > self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[j] = range_image(self.ds.scan(j)[:, :3])
        return self._cache[j]

    def _against(self, k, j, xyz_k):
        T_j_k = np.linalg.inv(self.T[j]) @ self.T[k]
        r, r_ref = lookup(self._ref_image(j), transform(xyz_k, T_j_k))
        return r < r_ref - (self.delta_abs + self.delta_rel * r)

    def candidates(self, k, xyz_k, nonground):
        """回傳 (N,) bool：候選移動點。只對非地面、max_range 內的點判斷。"""
        sel = nonground & (np.linalg.norm(xyz_k[:, :2], axis=1) < self.max_range)
        out = np.zeros(len(xyz_k), dtype=bool)
        gaps = self.gap if isinstance(self.gap, (list, tuple)) else [self.gap]
        if self.merge == "and" and len(gaps) == 1:
            refs = [j for j in (k - gaps[0], k + gaps[0]) if 0 <= j < self.ds.n]
        else:  # 多個間隔：一律 OR（離線才能用未來的幀）
            refs = [j for gp in gaps for j in (k - gp, k + gp) if 0 <= j < self.ds.n]
        votes = [self._against(k, j, xyz_k[sel]) for j in refs]
        if not votes:
            return out
        if self.merge == "and" and len(votes) == 2:
            res = votes[0] & votes[1]
        else:  # 至少 min_votes 個參考幀「看穿」這個位置
            res = np.sum(votes, axis=0) >= min(self.min_votes, len(votes))
        out[sel] = res
        return out
