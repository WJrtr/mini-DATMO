"""開發集：序列 08 的 1500:1800 和 3000:3300 兩段，回報標註器的 IoU/P/R。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import Counter
from kitti_io import load_dataset
from labeler import Labeler
kw = eval("dict(" + (sys.argv[1] if len(sys.argv) > 1 else "") + ")")
ds = load_dataset("seq08")
c, cl = Counter(), Counter()
for a, b in [(1500, 1800), (3000, 3300)]:
    out = Labeler(ds, **kw).run(range(a, b))
    for k, o in out.items():
        gt = ds.labels(k); c.add(o["label"], gt); cl.add(ds.lmnet(k), gt)
print(f"{str(kw):70s} ours {c} | lmnet {cl}", flush=True)
