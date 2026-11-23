"""步驟 ⑧：逐點 moving IoU / precision / recall（LMNet 式 4：IoU = TP / (TP + FP + FN)）。
所有幀的點加總起來再算。"""
import numpy as np


class Counter:
    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def add(self, pred: np.ndarray, gt: np.ndarray, mask=None):
        if mask is not None:
            pred, gt = pred[mask], gt[mask]
        self.tp += int(np.sum(pred & gt))
        self.fp += int(np.sum(pred & ~gt))
        self.fn += int(np.sum(~pred & gt))

    def result(self):
        tp, fp, fn = self.tp, self.fp, self.fn
        iou = tp / max(tp + fp + fn, 1)
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        return dict(iou=iou, precision=p, recall=r, tp=tp, fp=fp, fn=fn)

    def __str__(self):
        d = self.result()
        return f"IoU {100*d['iou']:5.1f}%  P {100*d['precision']:5.1f}%  R {100*d['recall']:5.1f}%"
