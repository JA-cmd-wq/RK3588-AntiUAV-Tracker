"""Numerically equivalent sparse YOLOv8 DFL decode for the existing model.

The classification outputs are already scores. Filter anchors before DFL
softmax instead of evaluating the 64 regression channels for all 8400 anchors.
Output validation, coordinate arithmetic, branch order, NMS, clipping and
candidate format retain anti_uav_demo/yolo.py semantics. No model change.
"""
from __future__ import annotations
import numpy as np
from yolo import nchw, nms


def postprocess(outputs, transform, conf=.25, iou=.45):
    if outputs is None or len(outputs)!=9:
        raise ValueError('Expected the validated 9-output drone YOLOv8 model')
    boxes,scores=[],[]
    bins=np.arange(16,dtype=np.float32).reshape(1,16,1)
    for branch,stride in enumerate((8,16,32)):
        side=640//stride
        regression=nchw(outputs[branch*3],64,side).reshape(4,16,side*side)
        cls=nchw(outputs[branch*3+1],1,side).reshape(-1)
        nchw(outputs[branch*3+2],1,side)  # validated but unused, as in v1
        indices=np.flatnonzero(cls>=conf)
        if not indices.size:
            continue
        logits=regression[:,:,indices]
        logits=logits-logits.max(axis=1,keepdims=True)
        probabilities=np.exp(logits)
        probabilities/=probabilities.sum(axis=1,keepdims=True)
        distances=(probabilities*bins).sum(axis=1)
        grid=np.stack((indices%side,indices//side))
        positions=np.concatenate((grid+.5-distances[:2],grid+.5+distances[2:]),axis=0)*stride
        boxes.append(positions.T)
        scores.append(cls[indices])
    if not scores:
        return []
    boxes=np.concatenate(boxes).astype(np.float32)
    scores=np.concatenate(scores)
    keep=nms(boxes,scores,iou)[:100]
    boxes,scores=boxes[keep],scores[keep]
    scale,left,top,width,height=transform
    boxes[:,(0,2)]=((boxes[:,(0,2)]-left)/scale).clip(0,width)
    boxes[:,(1,3)]=((boxes[:,(1,3)]-top)/scale).clip(0,height)
    return [{'bbox':[float(x1),float(y1),float(x2-x1),float(y2-y1)],'score':float(s)}
            for (x1,y1,x2,y2),s in zip(boxes,scores) if x2>x1 and y2>y1]
