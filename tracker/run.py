#!/usr/bin/env python3
"""YOLO INT8 + existing FEAR FP16 + constant-velocity Kalman, board only."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tracker_core import FEARTracker, RKNNRuntime, clamp_bbox
from yolo import Detector

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get('ANTI_UAV_MODELS', REPO_ROOT / 'models'))

TIMINGS = ('decode_ms', 'rgb_ms', 'yolo_ms', 'yolo_pre_ms', 'yolo_inference_ms',
           'yolo_post_ms', 'fear_ms', 'fear_pre_ms', 'fear_inference_ms',
           'fear_post_ms', 'fear_init_ms', 'kalman_ms', 'draw_ms', 'encode_ms', 'frame_ms')


def center(box):
    return np.asarray(box[:2], dtype=float) + np.asarray(box[2:], dtype=float) / 2


def iou(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    intersection = np.prod(np.maximum(0, np.minimum(a[:2]+a[2:], b[:2]+b[2:]) - np.maximum(a[:2], b[:2])))
    return float(intersection / max(np.prod(a[2:])+np.prod(b[2:])-intersection, 1e-9))


class BoxKalman:
    """8D [cx, cy, w, h, vx, vy, vw, vh], dt=one source frame.

    Internal state is never image-clamped, so out-of-view motion persists.
    """
    def __init__(self, box):
        self.x = np.r_[center(box), box[2:], np.zeros(4)].astype(float)
        self.P = np.diag([25., 25., 16., 16., 100., 100., 16., 16.])
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)
        self.Q = np.diag([1., 1., .2, .2, 4., 4., .1, .1])
        self.R = np.diag([4., 4., 16., 16.])

    def box(self):
        size = np.maximum(3., self.x[2:4])
        return np.r_[self.x[:2]-size/2, size]

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.box()

    def update(self, box):
        z = np.r_[center(box), box[2:]]
        residual = z - self.H @ self.x
        gain = np.linalg.solve(self.H @ self.P @ self.H.T + self.R, self.H @ self.P).T
        self.x += gain @ residual
        # Joseph form retains symmetric positive covariance across long videos.
        m = np.eye(8) - gain @ self.H
        self.P = m @ self.P @ m.T + gain @ self.R @ gain.T
        return self.box()


class Fusion:
    def __init__(self, detector, tracker, args):
        self.detector, self.tracker, self.args = detector, tracker, args
        self.state, self.kalman = 'SEARCH', None
        self.misses = self.detector_misses = 0
        self.last_yolo = -args.yolo_interval
        self.last_detection_box = None
        self.last_detection_frame = None
        self.ever_locked = False
        self.losses = self.recoveries = self.coast_recoveries = self.initial_locks = 0

    def step(self, frame, index):
        a = self.args
        row = {name: 0. for name in TIMINGS}
        row.update(frame=index, state_before=self.state, state=self.state, source='NONE', reason='',
                   fear_score=None, fear_bbox=None, bbox=None, prediction=None,
                   yolo_called=0, fear_called=0, yolo_conf=None, yolo_distance_px=None,
                   yolo_candidates=[], yolo_size_rejected=0, event='', misses=0, detector_misses=0)
        rgb = None

        def get_rgb():
            nonlocal rgb
            if rgb is None:
                t = time.perf_counter()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                row['rgb_ms'] += (time.perf_counter()-t)*1000
            return rgb

        def initialize(box):
            image = get_rgb()
            t = time.perf_counter()
            self.tracker.initialize(image, box)
            elapsed = (time.perf_counter()-t)*1000
            row['fear_init_ms'] += elapsed
            row['fear_ms'] += elapsed

        if self.state == 'LOST':
            self.state = 'SEARCH'
        prediction, fear_box = None, None
        if self.state == 'TRACK':
            t = time.perf_counter()
            prediction = self.kalman.predict()
            row['kalman_ms'] += (time.perf_counter()-t)*1000
            row['prediction'] = prediction.tolist()
            image = get_rgb()
            self.tracker.bbox = clamp_bbox(prediction, image.shape)
            t = time.perf_counter()
            result = self.tracker.update(image)
            row['fear_ms'] += (time.perf_counter()-t)*1000
            row.update(fear_called=1, fear_score=result['score'], fear_bbox=result['bbox'].tolist(),
                       fear_pre_ms=result['preprocessing_ms'], fear_inference_ms=result['search_inference_ms'],
                       fear_post_ms=result['postprocessing_ms'])
            if result['score'] >= a.fear_threshold:
                fear_box = result['bbox']
            # FEAR.update mutates bbox even on low confidence. Never retain that low-confidence crop.
            self.tracker.bbox = clamp_bbox(prediction, image.shape)

        needs_yolo = self.state == 'SEARCH' or fear_box is None or index-self.last_yolo >= a.yolo_interval
        selected = None
        if needs_yolo:
            candidates, timings = self.detector.detect(frame)
            row.update(timings)
            row.update(yolo_called=1, yolo_candidates=candidates)
            self.last_yolo = index
            # A short-lived size prior survives LOST. Reject drastic size changes
            # before association, so a tiny background false positive does not
            # replace the known drone just because detection missed several frames.
            eligible = candidates
            if self.last_detection_box is not None and index-self.last_detection_frame <= a.size_memory_frames:
                previous_area = float(np.prod(self.last_detection_box[2:]))
                eligible = [c for c in candidates if 1/a.size_ratio_limit <=
                            np.prod(c['bbox'][2:])/max(previous_area, 1.) <= a.size_ratio_limit]
                row['yolo_size_rejected'] = len(candidates)-len(eligible)
            if eligible:
                if prediction is None:
                    selected = max(eligible, key=lambda c: c['score'])
                else:
                    selected = min(eligible, key=lambda c: np.linalg.norm(center(c['bbox'])-center(prediction)))
                    distance = float(np.linalg.norm(center(selected['bbox'])-center(prediction)))
                    row['yolo_distance_px'] = distance
                    gate = max(a.association_gate_px, a.association_gate_scale * np.linalg.norm(prediction[2:]))
                    gate *= 1 + min(self.misses, a.max_misses) / a.max_misses
                    if distance > gate:
                        selected = None
            # Only a confident detection elsewhere counts against FEAR. An empty or weak YOLO
            # result is no evidence: the detector may not see the target at this size or angle.
            if selected is not None:
                self.detector_misses = 0
            elif any(c['score'] >= a.disagree_conf for c in eligible):
                self.detector_misses += 1

        if selected is not None:
            box = np.asarray(selected['bbox'])
            row.update(source='YOLO', bbox=box.tolist(), yolo_conf=selected['score'])
            if self.state == 'SEARCH':
                initialize(box)
                t = time.perf_counter()
                self.kalman = BoxKalman(box)
                row['kalman_ms'] += (time.perf_counter()-t)*1000
                if self.ever_locked:
                    self.recoveries += 1
                    row['event'] = 'REACQUIRED'
                else:
                    self.initial_locks += 1
                    row['event'] = 'INITIAL_LOCK'
                self.ever_locked = True
                row['reason'] = 'global_detection'
            else:
                if self.misses or fear_box is None:
                    self.coast_recoveries += 1
                    row['event'] = 'COAST_RECOVERY'
                    initialize(box)
                else:
                    row['event'] = 'YOLO_CORRECTION'
                    # FEAR's box scale follows its template; refresh it when the target has
                    # clearly grown or shrunk since the template was taken.
                    ratio = float(np.prod(box[2:]))/max(float(np.prod(fear_box[2:])), 1.)
                    if not 1/a.template_refresh_ratio <= ratio <= a.template_refresh_ratio:
                        initialize(box)
                        row['event'] = 'TEMPLATE_REFRESH'
                t = time.perf_counter()
                self.kalman.update(box)
                row['kalman_ms'] += (time.perf_counter()-t)*1000
                row['reason'] = 'low_score_recovery' if fear_box is None else 'periodic_correction'
            self.tracker.bbox = clamp_bbox(box, frame.shape)
            self.last_detection_box = box.copy()
            self.last_detection_frame = index
            self.state = 'TRACK'
            self.misses = 0
        elif self.state == 'TRACK':
            if fear_box is not None and self.detector_misses < a.detector_miss_limit:
                t = time.perf_counter()
                self.kalman.update(fear_box)
                row['kalman_ms'] += (time.perf_counter()-t)*1000
                row.update(source='FEAR', bbox=fear_box.tolist(), reason='high_fear_score')
                self.tracker.bbox = fear_box.copy()
                self.misses = 0
            else:
                self.misses += 1
                row['reason'] = 'low_fear_score' if fear_box is None else 'detector_disagrees'
                if self.misses <= a.coast_frames:
                    row.update(source='KALMAN', bbox=clamp_bbox(prediction, frame.shape).tolist())
                if self.misses >= a.max_misses or (fear_box is not None and self.detector_misses >= a.detector_miss_limit):
                    self.state = 'LOST'
                    self.losses += 1
                    row.update(source='NONE', bbox=None, event='LOST')
                    self.kalman = None
        else:
            row['reason'] = 'no_detection'
        row.update(state=self.state, misses=self.misses, detector_misses=self.detector_misses)
        return row


def annotate(frame, row, fps):
    colors = {'YOLO': (0, 210, 255), 'FEAR': (70, 240, 60), 'KALMAN': (255, 120, 30), 'NONE': (180, 180, 180)}
    color = colors[row['source']]
    box = row['bbox']
    if box is not None:
        x, y, w, h = map(round, box)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, row['source'], (x, max(95, y-8)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
    overlay = frame[:85, :min(760, frame.shape[1])]
    overlay[:] = (overlay.astype(np.uint16) // 3).astype(np.uint8)
    cv2.putText(frame, f"{row['state']} / {row['source']} | FPS {fps:.1f} | frame {row['frame']}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2)
    score = '-' if row['fear_score'] is None else f"{row['fear_score']:.3f}"
    cv2.putText(frame, f"FEAR {score} | miss {row['misses']} | YOLO=yellow FEAR=green KALMAN=blue",
                (12, 57), cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1)
    return frame


def read_gt(path):
    if path is None:
        return None
    value = json.loads(Path(path).read_text())
    # UCAS visible.json; never use infrared coordinates for RGB.
    return value


def distribution(values):
    return {'mean': float(np.mean(values)), 'p95': float(np.percentile(values, 95))} if values else None


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def run(a):
    if not 0 <= a.fear_threshold <= 1 or not 0 <= a.yolo_conf <= 1:
        raise ValueError('Confidence thresholds must be in [0,1]')
    if not 1 <= a.coast_frames < a.max_misses or a.yolo_interval < 1 or a.detector_miss_limit < 1:
        raise ValueError('Require 1 <= coast-frames < max-misses and positive YOLO intervals/limits')
    if a.yolo_core == a.fear_core:
        raise ValueError('YOLO and FEAR must use different NPU cores')
    if 'int8' in str(a.template_model).lower() or 'int8' in str(a.search_model).lower():
        raise ValueError('FEAR INT8 is disallowed; use the existing FP16 models')
    cv2.setNumThreads(a.cv_threads)
    detector = Detector(a.yolo_model, a.yolo_core, a.yolo_conf)
    runtime = None
    cap = writer = None
    rows = []
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        runtime = RKNNRuntime(a.template_model, a.search_model, core_mask=a.fear_core, bbox_logits=True)
        tracker = FEARTracker(runtime, smooth=False)
        fusion = Fusion(detector, tracker, a)
        cap = cv2.VideoCapture(str(a.video))
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open {a.video}')
        width, height = int(cap.get(3)), int(cap.get(4))
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(source_fps) or source_fps <= 0:
            source_fps = 25.
        if not a.no_video:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*a.codec), source_fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError('Cannot open output video writer')
        truth = read_gt(a.gt)
        rolling = deque(maxlen=30)
        begin = time.perf_counter()
        index = 0
        while a.max_frames == 0 or index < a.max_frames:
            start = time.perf_counter()
            ok, frame = cap.read()
            decode_ms = (time.perf_counter()-start)*1000
            if not ok:
                break
            row = fusion.step(frame, index)
            row['decode_ms'] = decode_ms
            row.update(gt_exists=None, gt_valid=None, gt_iou=None, gt_center_error_px=None)
            if truth is not None and index < len(truth['gt_rect']):
                exists = bool(truth.get('exist', [1]*len(truth['gt_rect']))[index])
                row['gt_exists'] = int(exists)
                target = truth['gt_rect'][index]
                row['gt_valid'] = int(exists and len(target) == 4 and target[2] > 0 and target[3] > 0)
                if row['gt_valid']:
                    row['gt_iou'] = iou(row['bbox'], target) if row['bbox'] is not None else 0.
                    row['gt_center_error_px'] = float(np.linalg.norm(center(row['bbox'])-center(target))) if row['bbox'] is not None else None
            if writer is not None:
                t = time.perf_counter()
                display_fps = len(rolling) / sum(rolling) if rolling else 0.
                annotate(frame, row, display_fps)
                row['draw_ms'] = (time.perf_counter()-t)*1000
                t = time.perf_counter()
                writer.write(frame)
                row['encode_ms'] = (time.perf_counter()-t)*1000
            row['frame_ms'] = (time.perf_counter()-start)*1000
            rolling.append(row['frame_ms']/1000)
            rows.append(row)
            index += 1
            if index % 100 == 0:
                print(json.dumps({'frames': index, 'fps': index/(time.perf_counter()-begin),
                                  'state': row['state'], 'losses': fusion.losses, 'recoveries': fusion.recoveries}), flush=True)
        t = time.perf_counter()
        if writer is not None:
            writer.release()
            writer = None
        encode_flush_ms = (time.perf_counter()-t)*1000
        cap.release()
        cap = None
        seconds = time.perf_counter()-begin
        if not rows:
            raise RuntimeError('No frames decoded')
        csv_path = out.with_suffix('.csv')
        fields = [k for k in rows[0] if k not in ('bbox', 'prediction', 'fear_bbox')]
        fields += [f'{prefix}_{axis}' for prefix in ('box', 'prediction', 'fear_box') for axis in ('x', 'y', 'w', 'h')]
        with csv_path.open('w', newline='') as stream:
            csv_writer = csv.DictWriter(stream, fieldnames=fields)
            csv_writer.writeheader()
            for r in rows:
                serial = {k: v for k, v in r.items() if k in fields}
                serial['yolo_candidates'] = json.dumps(serial['yolo_candidates'], separators=(',', ':'))
                for key, prefix in (('bbox', 'box'), ('prediction', 'prediction'), ('fear_bbox', 'fear_box')):
                    for j, axis in enumerate(('x', 'y', 'w', 'h')):
                        serial[f'{prefix}_{axis}'] = r[key][j] if r[key] is not None else None
                csv_writer.writerow(serial)
        means = {name: float(np.mean([r[name] for r in rows])) for name in TIMINGS}
        means['encode_ms_including_flush'] = means['encode_ms'] + encode_flush_ms/len(rows)
        yolo_rows = [r for r in rows if r['yolo_called']]
        fear_rows = [r for r in rows if r['fear_called']]
        valid = [r for r in rows if r['gt_valid'] == 1]
        summary = {'frames': len(rows), 'source_fps': source_fps, 'width': width, 'height': height,
                   'end_to_end_fps': len(rows)/seconds, 'end_to_end_seconds': seconds,
                   'average_ms_per_frame': means, 'encode_flush_ms': encode_flush_ms,
                   'average_ms_per_yolo_call': distribution([r['yolo_ms'] for r in yolo_rows]),
                   'average_ms_per_fear_search': distribution([r['fear_ms']-r['fear_init_ms'] for r in fear_rows]),
                   'yolo_calls': len(yolo_rows), 'fear_search_calls': len(fear_rows),
                   'source_counts': dict(Counter(r['source'] for r in rows)),
                   'state_counts': dict(Counter(r['state'] for r in rows)),
                   'initial_locks': fusion.initial_locks, 'lost_transitions': fusion.losses,
                   'search_reacquisitions': fusion.recoveries, 'coast_recoveries': fusion.coast_recoveries,
                   'gt_visible_frames': len(valid),
                   'gt_mean_iou': float(np.mean([r['gt_iou'] for r in valid])) if valid else None,
                   'gt_iou_ge_0_5': sum(r['gt_iou'] >= .5 for r in valid)/len(valid) if valid else None,
                   'boxed_gt_iou_zero_frames': sum(r['bbox'] is not None and r['gt_iou'] == 0 for r in valid),
                   'config': vars(a),
                   'model_sha256': {str(p): sha256(p) for p in (a.yolo_model, a.template_model, a.search_model)},
                   'timing_note': 'perf_counter synchronous wall timings, not NPU kernel-only. FPS includes every frame, initial acquisition/template initialization, decode, RGB, drawing, OpenCV video encoding and encoder flush; excludes model loading, CSV/JSON writing. YOLO/FEAR means are both per-all-frames and per-call. FEAR inference includes the original CPU Exp restoration. RGB is separate. No GT is used by Fusion.'}
        out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        detector.close()
        if runtime is not None:
            runtime.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--gt', help='UCAS visible.json, evaluation only')
    p.add_argument('--yolo-model', default=str(MODEL_DIR / 'yolo' / 'drone_yolov8n_int8.rknn'))
    p.add_argument('--template-model', default=str(MODEL_DIR / 'fear' / 'template_fp16.rknn'))
    p.add_argument('--search-model', default=str(MODEL_DIR / 'fear' / 'search_fp16.rknn'))
    p.add_argument('--yolo-core', choices=('0', '1', '2'), default='0')
    p.add_argument('--fear-core', choices=('0', '1', '2'), default='1')
    p.add_argument('--fear-threshold', type=float, default=.70)
    p.add_argument('--yolo-conf', type=float, default=.25)
    p.add_argument('--yolo-interval', type=int, default=15)
    p.add_argument('--coast-frames', type=int, default=5)
    p.add_argument('--max-misses', type=int, default=15)
    p.add_argument('--detector-miss-limit', type=int, default=3,
                   help='Consecutive confident YOLO detections elsewhere force global search even when FEAR stays confident')
    p.add_argument('--disagree-conf', type=float, default=.5,
                   help='YOLO score needed for a detection elsewhere to count against FEAR')
    p.add_argument('--template-refresh-ratio', type=float, default=1.5,
                   help='Refresh the FEAR template when YOLO and FEAR box areas differ by more than this')
    p.add_argument('--association-gate-px', type=float, default=120.)
    p.add_argument('--association-gate-scale', type=float, default=2.)
    p.add_argument('--size-ratio-limit', type=float, default=3., help='Accepted area ratio is [1/limit, limit] against last YOLO box')
    p.add_argument('--size-memory-frames', type=int, default=120, help='Keep size prior across LOST; expires to allow changed target scale')
    p.add_argument('--cv-threads', type=int, default=2)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--codec', default='mp4v')
    p.add_argument('--no-video', action='store_true', help='Ablation: decode+inference, no annotation/encoding')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
