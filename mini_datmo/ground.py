"""步驟 ①：去地面。

做法：把點放進 x-y 格子（預設 1 m），每格取最低點 z_min。
一個點是地面，要同時滿足：
  - z - z_min < height_thr（離格子最低點很近）
  - z_min < ground_max_z（這一格的最低點夠低，真的看到了地面；整格都是車身時不會成立）
Velodyne 離地約 1.73 m（KITTI），所以地面大約在 z = -1.73。
"""
import numpy as np


def ego_mask(pts: np.ndarray, radius=2.0) -> np.ndarray:
    """自己車身的點（感測器水平距離 < radius）。這些點跟著車子移動，在世界座標裡會被誤認為移動物體。"""
    return np.linalg.norm(pts[:, :2], axis=1) < radius


def ground_mask(pts: np.ndarray, cell=1.0, height_thr=0.25, ground_max_z=-1.3) -> np.ndarray:
    xy = np.floor(pts[:, :2] / cell).astype(np.int64)
    # 把 2D 格子編號壓成 1D key
    key = (xy[:, 0] + 100000) * 200003 + (xy[:, 1] + 100000)
    uniq, inv = np.unique(key, return_inverse=True)
    zmin = np.full(len(uniq), np.inf)
    np.minimum.at(zmin, inv, pts[:, 2])
    zc = zmin[inv]
    return (pts[:, 2] - zc < height_thr) & (zc < ground_max_z)
