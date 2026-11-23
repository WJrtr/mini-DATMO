"""步驟 ④：追蹤。等速 Kalman + 馬氏距離 gating（χ² 2 自由度 99% = 9.21）+ 匈牙利配對。

量測只有群中心的 [x, y]（世界座標），因為雷射量不到速度（IJRR 2007 §6.2）。
track 管理：連續配到 3 次 → 確認；待確認連續 2 次沒配到 → 刪除；確認的 track 沒配到時只 predict，保留 n_old 幀（AutoMOS n_old = 5）。
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

GATE = 9.21


class Track:
    _next = 0

    def __init__(self, z, k, dt, sigma_a, r):
        self.id = Track._next
        Track._next += 1
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([0.5 ** 2, 0.5 ** 2, 10.0 ** 2, 10.0 ** 2])
        self.hits = 1
        self.miss = 0
        self.confirmed = False
        self.history = []  # (frame, cluster_key)

    def predict(self, F, Q):
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q


class Tracker:
    def __init__(self, dt=0.1, sigma_a=2.0, r=0.3, n_confirm=3, n_tentative_miss=2, n_old=5):
        self.dt = dt
        self.F = np.eye(4)
        self.F[0, 2] = self.F[1, 3] = dt
        q = sigma_a ** 2
        G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        self.Q = G @ G.T * q
        self.H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        self.R = np.eye(2) * r ** 2
        self.n_confirm, self.n_tentative_miss, self.n_old = n_confirm, n_tentative_miss, n_old
        self.tracks = []
        self.finished = []

    def step(self, k, centers, keys):
        """centers: (M, 2) 世界座標；keys: 每個量測的識別（例如 (frame, cluster_id)）。"""
        for t in self.tracks:
            t.predict(self.F, self.Q)
        T, M = len(self.tracks), len(centers)
        cost = np.full((T, M), 1e6)
        for i, t in enumerate(self.tracks):
            S = self.H @ t.P @ self.H.T + self.R
            Si = np.linalg.inv(S)
            y = centers - (self.H @ t.x)[None, :]
            d2 = np.einsum("mi,ij,mj->m", y, Si, y)
            cost[i] = np.where(d2 <= GATE, d2, 1e6)
        matched_t, matched_m = set(), set()
        if T and M:
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] >= 1e6:
                    continue
                t = self.tracks[i]
                S = self.H @ t.P @ self.H.T + self.R
                K = t.P @ self.H.T @ np.linalg.inv(S)
                t.x = t.x + K @ (centers[j] - self.H @ t.x)
                t.P = (np.eye(4) - K @ self.H) @ t.P
                t.hits += 1
                t.miss = 0
                if t.hits >= self.n_confirm:
                    t.confirmed = True
                t.history.append((k, keys[j]))
                matched_t.add(i)
                matched_m.add(j)
        alive = []
        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.miss += 1
            limit = self.n_old if t.confirmed else self.n_tentative_miss
            if t.miss > limit:
                self.finished.append(t)
            else:
                alive.append(t)
        for j in range(M):
            if j not in matched_m:
                t = Track(centers[j], k, self.dt, None, None)
                t.history.append((k, keys[j]))
                alive.append(t)
        self.tracks = alive

    def all_tracks(self):
        return self.finished + self.tracks
