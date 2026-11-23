import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import Counter
from kitti_io import load_dataset
from labeler import Labeler
a, b = int(sys.argv[1]), int(sys.argv[2])
kw = eval("dict(" + (sys.argv[3] if len(sys.argv) > 3 else "") + ")")
ds = load_dataset("seq08")
t0 = time.time()
L = Labeler(ds, **kw)
out = L.run(range(a, b))
print(f"{(time.time()-t0)/(b-a):.2f} s/frame", kw)
c = {n: Counter() for n in ["cand", "labeler", "lmnet"]}
for k, o in out.items():
    gt = ds.labels(k)
    c["cand"].add(o["cand"], gt); c["labeler"].add(o["label"], gt); c["lmnet"].add(ds.lmnet(k), gt)
for n, v in c.items(): print(f"  {n:8s} {v}")
tr = [t for t in L.track_info if t["n"] >= 3]
print("tracks:", len(tr), "moving:", sum(t["moving"] for t in tr))
for t in sorted(tr, key=lambda t: -t["ratio"])[:12]:
    print("   ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in t.items()})
