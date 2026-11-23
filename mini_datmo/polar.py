"""極座標距離圖（球座標：方位角 × 仰角）。

range_image(pts) 把一幀的點（在該幀自己的感測器座標）放進 (仰角, 方位角) 格子，
每格記下最近的距離，也就是那條光束「第一次打到東西」的距離。
這個距離之前的空間，在那一刻是「看得穿」的（IJRR 2007 §7.1）。
"""
import numpy as np

AZ_RES = np.deg2rad(0.5)
EL_RES = np.deg2rad(1.0)
EL_MIN, EL_MAX = np.deg2rad(-26.0), np.deg2rad(4.0)
N_AZ = int(round(2 * np.pi / AZ_RES))
N_EL = int(round((EL_MAX - EL_MIN) / EL_RES))


def spherical(xyz: np.ndarray):
    r = np.linalg.norm(xyz, axis=1)
    az = np.arctan2(xyz[:, 1], xyz[:, 0])
    el = np.arcsin(np.clip(xyz[:, 2] / np.maximum(r, 1e-6), -1, 1))
    return r, az, el


def cell_index(az, el):
    ia = np.floor((az + np.pi) / AZ_RES).astype(np.int64) % N_AZ
    ie = np.floor((el - EL_MIN) / EL_RES).astype(np.int64)
    valid = (ie >= 0) & (ie < N_EL)
    return ie, ia, valid


def range_image(xyz: np.ndarray, neighborhood=1) -> np.ndarray:
    """回傳 (N_EL, N_AZ) 的最近距離。沒有回波的格子 = inf（沒有資訊）。
    neighborhood=1 時，再對每格取相鄰 ±1 格的最小值，讓比較偏保守（降低邊界雜訊）。"""
    r, az, el = spherical(xyz)
    ie, ia, valid = cell_index(az, el)
    img = np.full((N_EL, N_AZ), np.inf)
    np.minimum.at(img, (ie[valid], ia[valid]), r[valid])
    if neighborhood > 0:
        out = img.copy()
        for de in range(-neighborhood, neighborhood + 1):
            for da in range(-neighborhood, neighborhood + 1):
                if de == 0 and da == 0:
                    continue
                sh = np.roll(img, da, axis=1)  # 方位角是環狀的
                if de > 0:
                    sh = np.vstack([np.full((de, N_AZ), np.inf), sh[:-de]])
                elif de < 0:
                    sh = np.vstack([sh[-de:], np.full((-de, N_AZ), np.inf)])
                np.minimum(out, sh, out=out)
        img = out
    return img


def lookup(img: np.ndarray, xyz: np.ndarray):
    """回傳每個點的 (r, r_ref)。r_ref = 參考距離圖在該方向的最近距離；格子外或沒資訊 = inf。"""
    r, az, el = spherical(xyz)
    ie, ia, valid = cell_index(az, el)
    r_ref = np.full(len(r), np.inf)
    r_ref[valid] = img[ie[valid], ia[valid]]
    return r, r_ref
