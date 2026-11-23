"""步驟 0：讀資料。

兩種格式輸出同一種介面：
    ds = load_dataset("seq08" | "0005" | "0060")
    ds.n                 幀數
    ds.scan(k)           (N, 4) float32：x, y, z, intensity（Velodyne 座標）
    ds.T_w_velo[k]       4x4，Velodyne 座標 -> 世界座標
    ds.labels(k)         (N,) bool，True = 移動（沒有真值時回傳 None）
    ds.tracklets         KITTI raw 才有：list of dict（Velodyne 座標的外框）
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEQ08_DIR = os.path.join(ROOT, "datasets", "lmnet_toy_seq08")
RAW_DIR = os.path.join(ROOT, "datasets", "kitti_raw", "2011_09_26")
SK_DIR = os.path.join(ROOT, "datasets", "semantickitti")

MOVING_MIN, MOVING_MAX = 251, 259  # SemanticKITTI-MOS：251–259 = 移動


def read_bin(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def read_label_moving(path: str) -> np.ndarray:
    sem = np.fromfile(path, dtype=np.uint32) & 0xFFFF
    return (sem >= MOVING_MIN) & (sem <= MOVING_MAX)


def write_label(path: str, moving: np.ndarray) -> None:
    """輸出 SemanticKITTI 格式：9 = 靜止，251 = 移動。"""
    out = np.where(moving, 251, 9).astype(np.uint32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.tofile(path)


def _to4x4(m34: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :4] = m34.reshape(3, 4)
    return T


@dataclass
class Dataset:
    name: str
    n: int
    T_w_velo: np.ndarray  # (n, 4, 4)
    scan_paths: list
    label_paths: list | None = None
    pred_paths: list | None = None  # LMNet 附帶的預測（序列 08）
    tracklets: list = field(default_factory=list)

    def scan(self, k: int) -> np.ndarray:
        return read_bin(self.scan_paths[k])

    def labels(self, k: int):
        if self.label_paths is None:
            return None
        return read_label_moving(self.label_paths[k])

    def lmnet(self, k: int):
        if self.pred_paths is None:
            return None
        return read_label_moving(self.pred_paths[k])


# ---------------------------------------------------------------- SemanticKITTI
def load_semantickitti(seq_dir: str, pred_dir: str | None = None, name="seq08") -> Dataset:
    calib = {}
    with open(os.path.join(seq_dir, "calib.txt")) as f:
        for line in f:
            if ":" in line:
                key, val = line.split(":", 1)
                calib[key.strip()] = np.array([float(x) for x in val.split()])
    Tr = _to4x4(calib["Tr"])  # Velodyne -> 左相機
    P = np.loadtxt(os.path.join(seq_dir, "poses.txt")).reshape(-1, 3, 4)
    # poses.txt：左相機第 k 幀 -> 左相機第 0 幀。轉成 Velodyne 座標：
    # T_w_velo[k] = inv(Tr) · P[k] · Tr
    Tr_inv = np.linalg.inv(Tr)
    T = np.stack([Tr_inv @ _to4x4(p) @ Tr for p in P])
    n = len(T)
    scans = [os.path.join(seq_dir, "velodyne", f"{k:06d}.bin") for k in range(n)]
    labels = [os.path.join(seq_dir, "labels", f"{k:06d}.label") for k in range(n)]
    preds = None
    if pred_dir and os.path.isdir(pred_dir):
        preds = [os.path.join(pred_dir, f"{k:06d}.label") for k in range(n)]
    return Dataset(name, n, T, scans, labels, preds)


# ---------------------------------------------------------------- KITTI raw
def _read_calib_rt(path: str) -> np.ndarray:
    d = {}
    with open(path) as f:
        for line in f:
            if ":" in line:
                key, val = line.split(":", 1)
                try:
                    d[key.strip()] = np.array([float(x) for x in val.split()])
                except ValueError:
                    pass
    T = np.eye(4)
    T[:3, :3] = d["R"].reshape(3, 3)
    T[:3, 3] = d["T"]
    return T


def _rot(roll, pitch, yaw) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def oxts_to_poses(oxts: np.ndarray) -> np.ndarray:
    """KITTI devkit 的 Mercator 投影（scale 取第一幀緯度），回傳 T_w_imu，原點設在第一幀。"""
    er = 6378137.0
    scale = np.cos(oxts[0, 0] * np.pi / 180.0)
    poses = []
    for lat, lon, alt, roll, pitch, yaw in oxts[:, :6]:
        tx = scale * lon * np.pi * er / 180.0
        ty = scale * er * np.log(np.tan((90.0 + lat) * np.pi / 360.0))
        T = np.eye(4)
        T[:3, :3] = _rot(roll, pitch, yaw)
        T[:3, 3] = [tx, ty, alt]
        poses.append(T)
    poses = np.stack(poses)
    origin = np.linalg.inv(poses[0])
    return np.stack([origin @ T for T in poses])


def read_tracklets(path: str) -> list:
    """回傳 list of dict：type, h, w, l, first_frame, poses (M, 6: tx ty tz rx ry rz)。Velodyne 座標，tz 是外框底部。"""
    root = ET.parse(path).getroot()
    out = []
    for item in root.find("tracklets").findall("item"):
        poses = []
        for p in item.find("poses").findall("item"):
            poses.append([float(p.find(t).text) for t in ("tx", "ty", "tz", "rx", "ry", "rz")])
        out.append(dict(
            type=item.find("objectType").text,
            h=float(item.find("h").text), w=float(item.find("w").text), l=float(item.find("l").text),
            first_frame=int(item.find("first_frame").text),
            poses=np.array(poses),
        ))
    return out


def load_raw(drive: str) -> Dataset:
    ddir = os.path.join(RAW_DIR, f"2011_09_26_drive_{drive}_sync")
    vdir = os.path.join(ddir, "velodyne_points", "data")
    n = len([f for f in os.listdir(vdir) if f.endswith(".bin")])
    oxts = np.stack([np.loadtxt(os.path.join(ddir, "oxts", "data", f"{k:010d}.txt")) for k in range(n)])
    T_w_imu = oxts_to_poses(oxts)
    T_velo_imu = _read_calib_rt(os.path.join(RAW_DIR, "calib_imu_to_velo.txt"))
    T_imu_velo = np.linalg.inv(T_velo_imu)
    T = np.stack([Ti @ T_imu_velo for Ti in T_w_imu])
    scans = [os.path.join(vdir, f"{k:010d}.bin") for k in range(n)]
    trk_path = os.path.join(ddir, "tracklet_labels.xml")
    trk = read_tracklets(trk_path) if os.path.exists(trk_path) else []
    return Dataset(drive, n, T, scans, tracklets=trk)


def load_dataset(name: str) -> Dataset:
    if name == "seq08":
        return load_semantickitti(
            os.path.join(SEQ08_DIR, "sequences", "08"),
            os.path.join(SEQ08_DIR, "predictions_salsanext_residual_1_valid", "sequences", "08", "predictions"),
        )
    if name.startswith("seq"):  # 其他 SemanticKITTI 序列：datasets/semantickitti/dataset/sequences/NN
        return load_semantickitti(os.path.join(SK_DIR, "dataset", "sequences", name[3:]), None, name=name)
    return load_raw(name)
