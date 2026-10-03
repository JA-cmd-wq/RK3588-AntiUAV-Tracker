"""Single-class 9-output YOLOv8 RKNN decode (per stride: 64-channel DFL box
branch, class score, score_sum). No model conversion or training is performed.
"""
import time
import cv2
import numpy as np


def preprocess(bgr):
    height, width = bgr.shape[:2]
    scale = min(640 / height, 640 / width)
    resized = (round(width * scale), round(height * scale))
    dw, dh = (640 - resized[0]) / 2, (640 - resized[1]) / 2
    left, right = round(dw - .1), round(dw + .1)
    top, bottom = round(dh - .1), round(dh + .1)
    image = cv2.resize(bgr, resized, interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[None]), (scale, left, top, width, height)


def nchw(output, channels, grid):
    a = np.asarray(output)
    if a.shape == (1, grid, grid, channels):
        a = a.transpose(0, 3, 1, 2)
    if a.shape != (1, channels, grid, grid) or a.dtype.kind != 'f' or not np.isfinite(a).all():
        raise ValueError(f'Unexpected YOLO output shape/dtype {a.shape}/{a.dtype}')
    return a.astype(np.float32, copy=False)


def nms(boxes, scores, threshold):
    areas = np.prod(np.maximum(0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    order, keep = scores.argsort()[::-1], []
    while order.size:
        first, rest = order[0], order[1:]
        keep.append(first)
        overlap = np.maximum(0, np.minimum(boxes[first, 2:], boxes[rest, 2:]) -
                             np.maximum(boxes[first, :2], boxes[rest, :2]) + .00001)
        intersection = np.prod(overlap, axis=1)
        iou = intersection / np.maximum(areas[first] + areas[rest] - intersection, 1e-12)
        order = rest[iou <= threshold]
    return np.asarray(keep, dtype=np.int64)


def postprocess(outputs, transform, conf=.25, iou=.45):
    if outputs is None or len(outputs) != 9:
        raise ValueError('Expected the validated 9-output drone YOLOv8 model')
    boxes, scores = [], []
    for branch, stride in enumerate((8, 16, 32)):
        side = 640 // stride
        logits = nchw(outputs[branch * 3], 64, side).reshape(1, 4, 16, side, side)
        logits = logits - logits.max(axis=2, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=2, keepdims=True)
        distances = (probabilities * np.arange(16, dtype=np.float32).reshape(1, 1, -1, 1, 1)).sum(axis=2)
        col, row = np.meshgrid(np.arange(side), np.arange(side))
        grid = np.stack((col, row)).reshape(1, 2, side, side)
        positions = np.concatenate((grid + .5 - distances[:, :2], grid + .5 + distances[:, 2:]), axis=1) * stride
        boxes.append(positions.transpose(0, 2, 3, 1).reshape(-1, 4))
        scores.append(nchw(outputs[branch * 3 + 1], 1, side).reshape(-1))
        nchw(outputs[branch * 3 + 2], 1, side)  # score_sum is intentionally unused
    boxes, scores = np.concatenate(boxes).astype(np.float32), np.concatenate(scores)
    keep = scores >= conf
    boxes, scores = boxes[keep], scores[keep]
    if not len(scores):
        return []
    keep = nms(boxes, scores, iou)[:100]
    boxes, scores = boxes[keep], scores[keep]
    scale, left, top, width, height = transform
    boxes[:, (0, 2)] = ((boxes[:, (0, 2)] - left) / scale).clip(0, width)
    boxes[:, (1, 3)] = ((boxes[:, (1, 3)] - top) / scale).clip(0, height)
    return [{'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)], 'score': float(s)}
            for (x1, y1, x2, y2), s in zip(boxes, scores) if x2 > x1 and y2 > y1]


class Detector:
    def __init__(self, model, core='0', conf=.25, iou=.45):
        from rknnlite.api import RKNNLite
        self.session = RKNNLite()
        self.conf, self.iou = conf, iou
        try:
            if self.session.load_rknn(str(model)) != 0:
                raise RuntimeError('YOLO load_rknn failed')
            if self.session.init_runtime(core_mask=getattr(RKNNLite, 'NPU_CORE_' + core)) != 0:
                raise RuntimeError('YOLO init_runtime failed')
        except BaseException:
            self.close()
            raise

    def detect(self, bgr):
        t = time.perf_counter()
        image, transform = preprocess(bgr)
        t1 = time.perf_counter()
        outputs = self.session.inference(inputs=[image], data_format=['nhwc'])
        t2 = time.perf_counter()
        candidates = postprocess(outputs, transform, self.conf, self.iou)
        t3 = time.perf_counter()
        return candidates, {'yolo_pre_ms': (t1-t)*1000, 'yolo_inference_ms': (t2-t1)*1000,
                            'yolo_post_ms': (t3-t2)*1000, 'yolo_ms': (t3-t)*1000}

    def close(self):
        self.session.release()
