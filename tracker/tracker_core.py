"""FEAR-XS video tracking without the upstream training dependencies.

The crop, clamp, grid and optional smoothing formulas follow
PinataFarms/FEARTracker. Model wrappers consume RGB 0..255 float32 NCHW
and perform ImageNet normalization inside the exported graph.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import cv2
import numpy as np


def ensure_bbox_boundaries(bbox, image_shape):
    """Reproduce upstream ordering, including clipping x/y before x2/y2."""
    x, y, w, h = bbox
    x = min(max(0, x), image_shape[1])
    y = min(max(0, y), image_shape[0])
    x2 = min(max(0, x + w), image_shape[1])
    y2 = min(max(0, y + h), image_shape[0])
    return np.asarray([x, y, x2 - x, y2 - y], dtype=np.int32)


def clamp_bbox(bbox, image_shape, min_side=3):
    x, y, w, h = ensure_bbox_boundaries(bbox, image_shape)
    if w < min_side:
        w = min_side
        x -= max(0, x + w - image_shape[1])
    if h < min_side:
        h = min_side
        y -= max(0, y + h - image_shape[0])
    return np.asarray([x, y, w, h])


def extended_crop(image, bbox, crop_size, offset, padding_value=None):
    """Return (resized RGB crop, bbox in crop coordinates, original context).

    FEAR extends width and height independently; the context is generally
    rectangular and is resized directly to a square, using bilinear OpenCV
    interpolation, as in the official Albumentations Resize transform.
    """
    if padding_value is None:
        padding_value = np.mean(image, axis=(0, 1))
    x, y, w, h = bbox
    context = np.asarray(
        [x - w * offset, y - h * offset,
         w * (1.0 + offset + offset), h * (1.0 + offset + offset)],
        dtype=np.int32,
    )
    left, top = max(-int(context[0]), 0), max(-int(context[1]), 0)
    right = max(int(context[0] + context[2]) - image.shape[1], 0)
    bottom = max(int(context[1] + context[3]) - image.shape[0], 0)
    crop = image[
        context[1] + top:context[1] + context[3] - bottom,
        context[0] + left:context[0] + context[2] - right,
    ]
    padded = cv2.copyMakeBorder(
        crop, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=tuple(float(v) for v in padding_value),
    )
    if padded.size == 0:
        raise ValueError("The tracking crop is empty; check the initial bbox.")
    padded_bbox = ensure_bbox_boundaries(
        [bbox[0] - context[0], bbox[1] - context[1], bbox[2], bbox[3]],
        padded.shape[:2],
    )
    scale = np.asarray(
        [crop_size / padded.shape[1], crop_size / padded.shape[0]] * 2,
        dtype=np.float64,
    )
    crop_bbox = padded_bbox * scale
    resized = cv2.resize(padded, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    return resized, crop_bbox, context


def raw_nchw(image):
    return np.ascontiguousarray(image[:, :, :3].transpose(2, 0, 1)[None], dtype=np.float32)


def sigmoid(values):
    """Stable float32 sigmoid without overflow in background logits."""
    values = np.asarray(values, dtype=np.float32)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def nchw_output(output, channels, side, name):
    """Validate outputs, allowing explicitly recognizable NHWC results."""
    output = np.asarray(output)
    expected = (1, channels, side, side)
    if output.shape == (1, side, side, channels):
        output = output.transpose(0, 3, 1, 2)
    if output.shape != expected:
        raise ValueError(f"{name} shape {output.shape}; expected {expected}")
    return np.ascontiguousarray(output, dtype=np.float32)


class FEARTracker:
    """Single-template tracker. All image arguments are uint8 RGB arrays."""

    def __init__(self, runtime, smooth=False, template_size=128, instance_size=256,
                 score_size=16, total_stride=16, template_bbox_offset=0.2,
                 search_context=2.0, penalty_k=0.062, window_influence=0.38,
                 lr=0.765):
        self.runtime = runtime
        self.smooth = smooth
        self.template_size = template_size
        self.instance_size = instance_size
        self.score_size = score_size
        self.template_bbox_offset = template_bbox_offset
        self.search_context = search_context
        self.penalty_k = penalty_k
        self.window_influence = window_influence
        self.lr = lr
        positions = (np.arange(score_size) - score_size // 2) * total_stride + instance_size // 2
        # Upstream make_grid subtracts np.floor(), producing float64 grids.
        # Keep this precision: float32 decoding changes rounding at boundaries.
        self.grid_x, self.grid_y = [v.astype(np.float64) for v in np.meshgrid(positions, positions)]
        self.window = np.outer(np.hanning(score_size), np.hanning(score_size))
        self.bbox = None
        self.template_features = None
        self.template_inference_ms = None
        self.initialization_ms = None
        self.last_search_input = None

    def initialize(self, image, bbox):
        begin = time.perf_counter()
        self.bbox = clamp_bbox(bbox, image.shape)
        self.mean_color = np.mean(image, axis=(0, 1))
        crop, _, _ = extended_crop(
            image, self.bbox, self.template_size, self.template_bbox_offset,
        )
        inference_begin = time.perf_counter()
        features = self.runtime.template(raw_nchw(crop))
        self.template_inference_ms = (time.perf_counter() - inference_begin) * 1000
        self.template_features = nchw_output(features, 256, 8, "template_features")
        self.initialization_ms = (time.perf_counter() - begin) * 1000
        return self.bbox.copy()

    def prepare_search(self, image, bbox=None):
        """Prepare raw NCHW search input and update crop mapping/prev_size.

        Passing bbox permits teacher-forced calibration from a ground-truth
        box without changing the tracker state. The template remains fixed.
        """
        if self.template_features is None:
            raise RuntimeError("Call initialize() before prepare_search().")
        crop, crop_bbox, context = extended_crop(
            image, self.bbox if bbox is None else bbox,
            self.instance_size, self.search_context, self.mean_color,
        )
        self.mapping = context
        self.prev_size = crop_bbox[2:]
        self.last_search_input = raw_nchw(crop)
        return self.last_search_input

    def decode(self, bbox_map, cls_logits):
        regression = nchw_output(bbox_map, 4, self.score_size, "bbox")[0]
        cls_score = sigmoid(nchw_output(cls_logits, 1, self.score_size, "cls")[0, 0])
        left = self.grid_x - regression[0]
        top = self.grid_y - regression[1]
        right = self.grid_x + regression[2]
        bottom = self.grid_y + regression[3]
        penalty = None
        score = cls_score
        if self.smooth:
            def squared_size(width, height):
                pad = (width + height) * 0.5
                return np.sqrt((width + pad) * (height + pad))

            width, height = right - left, bottom - top
            scale = squared_size(width, height) / squared_size(*self.prev_size)
            scale = np.maximum(scale, 1.0 / scale)
            ratio = (self.prev_size[0] / self.prev_size[1]) / (width / height)
            ratio = np.maximum(ratio, 1.0 / ratio)
            penalty = np.exp(-(ratio * scale - 1) * self.penalty_k)
            score = (penalty * cls_score * (1 - self.window_influence)
                     + self.window * self.window_influence)
        row, column = np.unravel_index(np.argmax(score), score.shape)
        pred_bbox = np.asarray([
            left[row, column], top[row, column],
            right[row, column] - left[row, column],
            bottom[row, column] - top[row, column],
        ])
        if self.smooth:
            # Preserve upstream _smooth_size exactly, including its second
            # multiplication by lr; changing this changes tracking behavior.
            learning_rate = float(penalty[row, column] * cls_score[row, column] * self.lr)
            pred_size = pred_bbox[2:] * learning_rate
            prev_size = self.prev_size * (1 - learning_rate)
            pred_bbox[2:] = prev_size + learning_rate * (pred_size + prev_size)
        return pred_bbox, float(cls_score[row, column])

    def update(self, image):
        begin = time.perf_counter()
        search = self.prepare_search(image)
        preprocessing_ms = (time.perf_counter() - begin) * 1000
        inference_begin = time.perf_counter()
        bbox_map, cls_logits = self.runtime.search(search, self.template_features)
        inference_ms = (time.perf_counter() - inference_begin) * 1000
        post_begin = time.perf_counter()
        pred_bbox, score = self.decode(bbox_map, cls_logits)
        x_scale = self.mapping[2] / self.instance_size
        y_scale = self.mapping[3] / self.instance_size
        pred_bbox = np.asarray([
            round(float(pred_bbox[0] * x_scale + self.mapping[0])),
            round(float(pred_bbox[1] * y_scale + self.mapping[1])),
            max(3, round(float(pred_bbox[2] * x_scale))),
            max(3, round(float(pred_bbox[3] * y_scale))),
        ], dtype=np.int32)
        self.bbox = clamp_bbox(pred_bbox, image.shape)
        postprocessing_ms = (time.perf_counter() - post_begin) * 1000
        return {
            "bbox": self.bbox.copy(), "score": score,
            "preprocessing_ms": preprocessing_ms,
            "search_inference_ms": inference_ms,
            "postprocessing_ms": postprocessing_ms,
            "tracking_ms": (time.perf_counter() - begin) * 1000,
        }


class ONNXRuntime:
    def __init__(self, template_path, search_path, threads=1):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.template_session = ort.InferenceSession(
            str(template_path), options, providers=["CPUExecutionProvider"],
        )
        self.search_session = ort.InferenceSession(
            str(search_path), options, providers=["CPUExecutionProvider"],
        )
        self.template_input = self.template_session.get_inputs()[0].name
        self.search_inputs = [item.name for item in self.search_session.get_inputs()]
        self.description = {"backend": "onnx", "threads": threads,
                            "template": str(template_path), "search": str(search_path)}

    def template(self, image):
        return self.template_session.run(None, {self.template_input: image})[0]

    def search(self, image, template_features):
        return self.search_session.run(
            None, dict(zip(self.search_inputs, [image, template_features])),
        )[:2]

    def close(self):
        pass


class RKNNRuntime:
    def __init__(self, template_path, search_path, core_mask="auto", bbox_logits=False):
        from rknnlite.api import RKNNLite
        masks = {
            "auto": "NPU_CORE_AUTO", "0": "NPU_CORE_0", "1": "NPU_CORE_1",
            "2": "NPU_CORE_2", "01": "NPU_CORE_0_1", "012": "NPU_CORE_0_1_2",
        }
        self.template_session = RKNNLite()
        self.search_session = RKNNLite()
        self.bbox_logits = bbox_logits
        try:
            for session, path in [(self.template_session, template_path), (self.search_session, search_path)]:
                result = session.load_rknn(str(path))
                if result != 0:
                    raise RuntimeError(f"load_rknn({path}) returned {result}")
                result = session.init_runtime(core_mask=getattr(RKNNLite, masks[core_mask]))
                if result != 0:
                    raise RuntimeError(f"init_runtime({path}) returned {result}")
        except BaseException:
            self.close()
            raise
        self.description = {"backend": "rknn", "core_mask": core_mask,
                            "template": str(template_path), "search": str(search_path),
                            "bbox_output": "log_distances_restored_with_cpu_exp" if bbox_logits else "distances"}

    def template(self, image):
        outputs = self.template_session.inference(inputs=[image], data_format=["nchw"])
        if outputs is None:
            raise RuntimeError("Template RKNN inference returned no output.")
        return outputs[0]

    def search(self, image, template_features):
        outputs = self.search_session.inference(
            inputs=[image, np.ascontiguousarray(template_features, dtype=np.float32)],
            data_format=["nchw", "nchw"],
        )
        if outputs is None:
            raise RuntimeError("Search RKNN inference returned no output.")
        if self.bbox_logits:
            # The exported NPU graph ends before upstream torch.exp(). Keep
            # the pretrained regression function by restoring it on the CPU.
            outputs[0] = np.exp(np.asarray(outputs[0], dtype=np.float32))
        return outputs[:2]

    def close(self):
        self.template_session.release()
        self.search_session.release()


def load_pytorch_runtime(module_path, weights_path, source_dir=None, variant="original"):
    """Load the exporter's create_runtime factory without importing training packages here."""
    module_path = Path(module_path).resolve()
    spec = importlib.util.spec_from_file_location("fear_export_runtime", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load export module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime = module.create_runtime(
        weights_path=weights_path, source_dir=source_dir, variant=variant,
    )
    runtime.description = {"backend": "pytorch", "variant": variant,
                           "weights": str(weights_path), "module": str(module_path)}
    return runtime
