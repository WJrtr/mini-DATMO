"""產生簡報用的圖：每一步都用序列 08 的實際資料處理結果。

輸出到 docs/report_figures/*.png。主要示範幀 K = 45（一台約 9.6 m/s 的車正從前方駛近，
旁邊還有兩台停在路口、但被人工標成「移動」的車）。
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from consistency import transform  # noqa: E402
from ground import ego_mask, ground_mask  # noqa: E402
from kitti_io import load_dataset  # noqa: E402
from labeler import Labeler  # noqa: E402
from polar import EL_MIN, EL_RES, N_AZ, N_EL, range_image  # noqa: E402

OUT = os.path.join(HERE, "..", "..", "docs", "report_figures")
os.makedirs(OUT, exist_ok=True)

for f in [r"C:\Windows\Fonts\msjh.ttc", r"C:\Windows\Fonts\msjhbd.ttc"]:
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": ["Microsoft JhengHei", "DejaVu Sans"], "axes.unicode_minus": False,
    "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15, "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF", "savefig.facecolor": "#FFFFFF", "axes.edgecolor": "#8A8F98",
    "axes.titleweight": "normal", "axes.titlecolor": "#14213D", "text.color": "#14213D",
})
INK, GREY, LIGHT = "#14213D", "#9AA0A8", "#D9D6CF"
ORANGE, BLUE, RED = "#E07A2E", "#2F6FB0", "#9B2226"

ds = load_dataset("seq08")
K = 45


def bev(ax, lim=35, x0=0, y0=0):
    ax.set_xlim(x0 - lim, x0 + lim)
    ax.set_ylim(y0 - lim * 0.62, y0 + lim * 0.62)
    ax.set_aspect("equal")
    ax.set_xlabel("x（m，前方）")
    ax.set_ylabel("y（m，左方）")


def ego(ax):
    ax.add_patch(plt.Rectangle((-2.2, -0.9), 4.4, 1.8, color=INK, zorder=5))
    ax.annotate("", xy=(4.5, 0), xytext=(2.2, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2), zorder=5)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


pts = ds.scan(K)
xyz = pts[:, :3]
gt = ds.labels(K)
g = ground_mask(pts)
e = ego_mask(pts)

# ---------------------------------------------------------------- 01 原始點雲
fig, ax = plt.subplots(figsize=(15, 9.5))
sel = np.random.default_rng(0).permutation(len(pts))[::2]
sc = ax.scatter(xyz[sel, 0], xyz[sel, 1], c=np.clip(xyz[sel, 2], -2, 1.5), s=0.6, cmap="viridis")
cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
cb.set_label("高度 z（m）")
ego(ax)
bev(ax)
save(fig, "01_raw_bev.png")

# ---------------------------------------------------------------- 02 位姿對齊
fig, axes = plt.subplots(1, 2, figsize=(18, 8.2))
ks = range(K - 20, K + 21, 4)
for ax, use_pose, title in [(axes[0], False, "(a) 未做座標轉換"),
                            (axes[1], True, "(b) 以 T_w_velo 轉到世界座標")]:
    for k in ks:
        p = ds.scan(k)
        p = p[(~ground_mask(p)) & (np.linalg.norm(p[:, :2], axis=1) < 40)][::3]
        w = transform(p[:, :3], ds.T_w_velo[k]) if use_pose else p[:, :3]
        ax.scatter(w[:, 0], w[:, 1], s=0.15, c=INK, alpha=0.35)
    c = ds.T_w_velo[K][:3, 3] if use_pose else np.zeros(3)
    bev(ax, 30, c[0], c[1])
    ax.set_title(title)
save(fig, "02_pose_alignment.png")

# ---------------------------------------------------------------- 03 去地面與車身
fig, axes = plt.subplots(1, 2, figsize=(18, 7.6), gridspec_kw=dict(width_ratios=[1.45, 1]))
ax = axes[0]
ax.scatter(xyz[g, 0], xyz[g, 1], s=0.3, c=LIGHT, label=f"地面 {g.sum():,} 點")
ng = ~g & ~e
ax.scatter(xyz[ng, 0], xyz[ng, 1], s=0.5, c=INK, label=f"非地面 {ng.sum():,} 點")
ax.scatter(xyz[e, 0], xyz[e, 1], s=6, c=RED, label=f"自己的車身 {e.sum()} 點")
ax.add_patch(plt.Circle((0, 0), 2.0, fill=False, ls="--", color=RED, lw=1.5))
bev(ax, 30)
ax.legend(loc="upper left", markerscale=12, fontsize=14, framealpha=0.9)
ax.set_title("(a) 鳥瞰")
ax = axes[1]
side = np.abs(xyz[:, 1]) < 2.0
s1 = side & g
s2 = side & ~g
ax.scatter(xyz[s1, 0], xyz[s1, 2], s=1, c=GREY, label="地面")
ax.scatter(xyz[s2, 0], xyz[s2, 2], s=1, c=INK, label="非地面")
ax.axhline(-1.73, color=BLUE, ls=":", lw=1.5)
ax.text(1, -1.62, "地面約在 z = −1.73 m（感測器離地 1.73 m）", color=BLUE, fontsize=13)
ax.set_xlim(0, 40)
ax.set_ylim(-2.4, 2.0)
ax.set_xlabel("x（m）")
ax.set_ylabel("z（m）")
ax.set_title("(b) 側視（|y| < 2 m）")
ax.legend(loc="upper right", markerscale=8, fontsize=13)
save(fig, "03_ground_removal.png")

# ---------------------------------------------------------------- 04 參考幀距離圖
img = range_image(ds.scan(K - 5)[:, :3], neighborhood=0)
fig, ax = plt.subplots(figsize=(18, 4.6))
show = np.where(np.isfinite(img), np.clip(img, 0, 40), np.nan)
im = ax.imshow(show, origin="lower", aspect="auto", cmap="magma_r",
               extent=[-180, 180, np.degrees(EL_MIN), np.degrees(EL_MIN + N_EL * EL_RES)])
cb = fig.colorbar(im, ax=ax, pad=0.01)
cb.set_label("r_ref（m）")
ax.set_xlabel("方位角（°），0° = 正前方")
ax.set_ylabel("仰角（°）")
save(fig, "04_range_image.png")

# ---------------------------------------------------------------- Labeler（前 200 幀）
L = Labeler(ds)
snap = {}


def keep_clusters(k, pts_, g_, cand_, cid_, cl_):
    if k == K:
        snap["cl"] = cl_


out = L.run(range(0, 200), hook=keep_clusters)
cand = out[K]["cand"]
lab = out[K]["label"]
lm = ds.lmnet(K)

# ---------------------------------------------------------------- 05 候選點 vs 真值
fig, axes = plt.subplots(1, 2, figsize=(18, 7.4))
for ax, mask, color, title in [(axes[0], cand, ORANGE, f"(a) 一致性候選點（{cand.sum():,}）"),
                               (axes[1], gt, BLUE, f"(b) 人工標註的移動點（{gt.sum():,}）")]:
    ax.scatter(xyz[ng, 0], xyz[ng, 1], s=0.4, c=LIGHT)
    ax.scatter(xyz[mask, 0], xyz[mask, 1], s=2.5, c=color)
    ego(ax)
    bev(ax, 30)
    ax.set_title(title)
save(fig, "05_candidates_vs_gt.png")

# ---------------------------------------------------------------- 06 分群
cl = snap["cl"]  # 只對離候選點 0.5 m 內的非地面點分群
fig, ax = plt.subplots(figsize=(15, 8.6))
ax.scatter(xyz[g, 0], xyz[g, 1], s=0.2, c="#ECEAE4")
ax.scatter(xyz[ng, 0], xyz[ng, 1], s=0.4, c=LIGHT)
pal = ["#2F6FB0", "#6A4C93", "#2A9D8F", "#8D6E63", "#5C677D", "#B56576", "#3D5A80", "#7B9E3E"]
for i, q in enumerate(cl):
    r = cand[q["idx"]].mean()
    p = xyz[q["idx"]]
    sent = r >= L.track_ratio and q["extent"] < L.max_extent_track
    col = ORANGE if sent else pal[i % len(pal)]
    ax.scatter(p[:, 0], p[:, 1], s=1.6, color=col)
    if sent and abs(q["center"][0]) < 28 and abs(q["center"][1]) < 17:
        ax.add_patch(plt.Rectangle(q["bbox_min"][:2], *(q["bbox_max"][:2] - q["bbox_min"][:2]),
                                   fill=False, ec=INK, lw=1.2))
        ax.text(q["bbox_max"][0] + 0.4, q["bbox_max"][1], f"{r:.2f}", fontsize=13, color=INK)
ego(ax)
bev(ax, 30)
save(fig, "06_clusters.png")

# ---------------------------------------------------------------- 07 追蹤軌跡（世界座標）
fig, ax = plt.subplots(figsize=(15, 9))
for k in range(0, 200, 10):
    p = ds.scan(k)
    p = p[(~ground_mask(p)) & (np.linalg.norm(p[:, :2], axis=1) < 35)][::6]
    w = transform(p[:, :3], ds.T_w_velo[k])
    ax.scatter(w[:, 0], w[:, 1], s=0.1, c=LIGHT)
traj = ds.T_w_velo[:200, :3, 3]
ax.plot(traj[:, 0], traj[:, 1], color=BLUE, lw=3, label="自己的車")
best = None
for t in L.track_info:
    if t["n"] < 5:
        continue
    w = np.array([L.per_frame[k]["clusters"][c]["world"][:2] for k, c in t["history"]])
    if t["moving"]:
        ax.plot(w[:, 0], w[:, 1], color=ORANGE, lw=2.5)
        if best is None or t["disp"] > best[0]["disp"]:
            best = (t, w)
    else:
        ax.plot(w[:, 0], w[:, 1], color=GREY, lw=1)
ax.plot([], [], color=ORANGE, lw=2.5, label="判為移動的 track")
ax.plot([], [], color=GREY, lw=1, label="判為靜止的 track")
if best:
    t, w = best
    ax.annotate(f"track {t['id']}：{t['n']} 幀，淨位移 {t['disp']:.1f} m，直線度 {t['straight']:.2f}",
                xy=w[len(w) // 2], xytext=(w[len(w) // 2][0] - 5, w[len(w) // 2][1] + 14), fontsize=14,
                arrowprops=dict(arrowstyle="->", color=INK), color=INK)
ax.set_aspect("equal")
ax.set_xlabel("世界座標 x（m）")
ax.set_ylabel("世界座標 y（m）")
ax.legend(loc="lower left", fontsize=14)
save(fig, "07_tracks_world.png")

# ---------------------------------------------------------------- 08 最終標籤 vs 真值 vs LMNet
fig, axes = plt.subplots(1, 2, figsize=(18, 7.4))
for ax, pred, name in [(axes[0], lab, "(a) 本專案"), (axes[1], lm, "(b) LMNet")]:
    tp, fp, fn = pred & gt, pred & ~gt, ~pred & gt
    ax.scatter(xyz[ng, 0], xyz[ng, 1], s=0.4, c=LIGHT)
    ax.scatter(xyz[tp, 0], xyz[tp, 1], s=2.5, c=ORANGE, label=f"TP 正確 {tp.sum():,}")
    ax.scatter(xyz[fp, 0], xyz[fp, 1], s=2.5, c=RED, label=f"FP 誤判 {fp.sum():,}")
    ax.scatter(xyz[fn, 0], xyz[fn, 1], s=2.5, c=BLUE, label=f"FN 漏判 {fn.sum():,}")
    ego(ax)
    bev(ax, 30)
    ax.legend(loc="upper left", markerscale=6, fontsize=14, framealpha=0.9)
    ax.set_title(name)
save(fig, "08_final_vs_lmnet.png")

# ---------------------------------------------------------------- 09 慢速車：中心抖動
ds2 = ds
L2 = Labeler(ds2)
out2 = L2.run(range(3150, 3330))
key2t = {h: t for t in L2.track_info for h in t["history"]}
gtc, trk = [], {}
for k in range(3150, 3330):
    labk = np.fromfile(ds2.label_paths[k], dtype=np.uint32)
    m = ((labk & 0xFFFF) == 252) & ((labk >> 16) == 20)
    if m.sum() < 20:
        continue
    p = ds2.scan(k)
    gtc.append((k, transform(p[m, :3].mean(0)[None], ds2.T_w_velo[k])[0][:2]))
    bestq, bn = None, 0
    for cid, q in L2.per_frame[k]["clusters"].items():
        n = np.isin(q["idx"], np.where(m)[0]).sum()
        if n > bn:
            bestq, bn = (cid, q), n
    if bestq and (k, bestq[0]) in key2t:
        t = key2t[(k, bestq[0])]
        trk.setdefault(t["id"], (t, []))[1].append((k, bestq[1]["world"][:2]))
fig, ax = plt.subplots(figsize=(15, 7.5))
gk = np.array([x[0] for x in gtc])
gw = np.array([x[1] for x in gtc])
d0 = gw[0]
ax.plot(gk, np.linalg.norm(gw - d0, axis=1), color=BLUE, lw=3, label="真值物體中心離起點的距離")
top = sorted(trk.items(), key=lambda x: -len(x[1][1]))[:3]  # 和這台車重疊最多幀的三條 track
for (tid, (t, pts_)), col in zip(sorted(top, key=lambda x: x[1][1][0][0]), [ORANGE, RED, "#6A4C93"]):
    kk = np.array([x[0] for x in pts_])
    ww = np.array([x[1] for x in pts_])
    verdict = "判為移動" if t["moving"] else "判為靜止"
    ax.plot(kk, np.linalg.norm(ww - d0, axis=1), "o-", color=col, lw=2,
            label=f"track {tid}：淨位移 {t['disp']:.2f} m，直線度 {t['straight']:.2f} → {verdict}")
ax.axhline(4.0, color=GREY, ls="--")
ax.text(gk[0] + 1, 4.15, "要判為移動：淨位移 > 外框約 4 m，且直線度 ≥ 0.6", color=INK, fontsize=14)
ax.set_xlabel("幀")
ax.set_ylabel("離起點的距離（m）")
ax.legend(loc="upper left", fontsize=13)
save(fig, "09_slow_car.png")

# ---------------------------------------------------------------- 10 結果長條圖
fig, ax = plt.subplots(figsize=(15, 6.8))
names = ["本專案\n序列 07", "本專案\n序列 08", "AutoMOS\n只用可見性", "LMNet\n官方預測（08）", "AutoMOS\n完整方法"]
RES = os.path.join(HERE, "..", "..", "results")


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


vals = [100 * load("seq07_full.json")["labeler"]["all"]["iou"], 100 * load("seq08_full.json")["labeler"]["all"]["iou"],
        36.6, 100 * load("seq08_full.json")["lmnet"]["all"]["iou"], 74.2]  # 36.6、74.2：AutoMOS Table I
cols = [ORANGE, ORANGE, GREY, BLUE, INK]
bars = ax.bar(names, vals, color=cols, width=0.62)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", fontsize=18, fontweight="bold", color=INK)
ax.set_ylim(0, 85)
ax.set_ylabel("moving IoU（%）")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
save(fig, "10_iou_comparison.png")

# ---------------------------------------------------------------- 11 依速度分組
fig, ax = plt.subplots(figsize=(15, 6.8))
sp = load("seq08_by_speed.json")
bins = ["stopped(<0.5)", "slow(0.5-2)", "moving(>2)"]
n_gt = sum(sp["ours"][b]["gt"] for b in bins + ["unknown"])
share = [100 * sp["ours"][b]["gt"] / n_gt for b in bins]
groups = [f"暫停\n< 0.5 m/s（{share[0]:.0f}%）", f"慢速\n0.5–2 m/s（{share[1]:.0f}%）", f"快速\n> 2 m/s（{share[2]:.0f}%）"]
ours = [100 * sp["ours"][b]["recall"] for b in bins]
lmn = [100 * sp["lmnet"][b]["recall"] for b in bins]
x = np.arange(3)
b1 = ax.bar(x - 0.2, ours, 0.38, color=ORANGE, label="本專案")
b2 = ax.bar(x + 0.2, lmn, 0.38, color=BLUE, label="LMNet")
for bs, vs in [(b1, ours), (b2, lmn)]:
    for b, v in zip(bs, vs):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", fontsize=17, fontweight="bold", color=INK)
ax.set_xticks(x, groups)
ax.set_ylim(0, 100)
ax.set_ylabel("召回率（%）")
ax.legend(fontsize=15)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
save(fig, "11_recall_by_speed.png")

print("K =", K, "| cand", int(cand.sum()), "| label", int(lab.sum()), "| gt", int(gt.sum()), "| lmnet", int(lm.sum()))
print("clusters at K:", len(cl), "| sent:", sum(1 for q in cl if cand[q["idx"]].mean() >= L.track_ratio and q["extent"] < L.max_extent_track))
for nm, pr in [("ours", lab), ("lmnet", lm)]:
    print(nm, "TP", int((pr & gt).sum()), "FP", int((pr & ~gt).sum()), "FN", int((~pr & gt).sum()))
if best:
    t = best[0]
    print("best track", t["id"], "n", t["n"], "disp %.1f straight %.2f extent %.1f ratio %.2f" % (t["disp"], t["straight"], t["extent"], t["ratio"]),
          "| contains K:", any(k == K for k, _ in t["history"]))
for tid, (t, pts_) in top:
    print("slow car track", tid, "overlap", len(pts_), "n", t["n"], "disp %.2f straight %.2f moving %s" % (t["disp"], t["straight"], t["moving"]))
