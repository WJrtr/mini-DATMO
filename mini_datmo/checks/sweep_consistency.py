"""⑤b 參數掃描（序列 08 一小段，只看 OR）。"""
import os, sys, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import polar
from consistency import ConsistencyDetector
from evaluate import Counter
from ground import ground_mask
from kitti_io import load_dataset

ds = load_dataset("seq08")
frames = range(0, 200, 4)
data = [(k, ds.scan(k), ds.labels(k)) for k in frames]

def run(el_res_deg, nb_el, nb_az, delta_abs, delta_rel, merge="or", gap=5):
    polar.EL_RES = np.deg2rad(el_res_deg)
    polar.N_EL = int(round((polar.EL_MAX - polar.EL_MIN) / polar.EL_RES))
    orig = polar.range_image
    def ri(xyz, neighborhood=1):
        img = orig(xyz, neighborhood=0)
        out = img.copy()
        for de in range(-nb_el, nb_el + 1):
            for da in range(-nb_az, nb_az + 1):
                sh = np.roll(img, da, axis=1)
                if de > 0: sh = np.vstack([np.full((de, polar.N_AZ), np.inf), sh[:-de]])
                elif de < 0: sh = np.vstack([sh[-de:], np.full((-de, polar.N_AZ), np.inf)])
                np.minimum(out, sh, out=out)
        return out
    import consistency
    consistency.range_image = ri
    det = ConsistencyDetector(ds, gap=gap, delta_abs=delta_abs, delta_rel=delta_rel, merge=merge)
    c = Counter()
    for k, pts, gt in data:
        g = ground_mask(pts)
        c.add(det.candidates(k, pts[:, :3], ~g), gt)
    consistency.range_image = orig
    return c

for args in [(1.0,1,1,0.5,0.02),(1.0,0,1,0.5,0.02),(1.0,0,0,0.5,0.02),(0.5,0,1,0.5,0.02),(0.4,0,1,0.3,0.01),(0.4,1,1,0.3,0.01),(0.4,0,1,0.5,0.02)]:
    print(args, run(*args), flush=True)
print("gap 3", run(0.4,0,1,0.3,0.01,gap=3)); print("gap 10", run(0.4,0,1,0.3,0.01,gap=10))
