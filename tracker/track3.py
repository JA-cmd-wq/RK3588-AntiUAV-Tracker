#!/usr/bin/env python3
"""Three tracking modes on RK3588 (MPP decode + RGA crops + RKNN NPU), same harness:

  yolo_kalman       YOLO every frame (NPU cores 0+2, async) + 8-D constant-velocity Kalman
  fear_kalman       FEAR (NPU core 1) + Kalman, one manual init box, no detector
  yolo_fear_kalman  v2 behaviour: YOLO finds / re-checks, FEAR tracks every frame, Kalman

Timed run = decode thread -> track thread (-> optional hw-encode thread).  The
annotated video is rendered afterwards from frames.csv by render.py.

Without --single this script drives the repeated FPS runs itself (separate
processes): N no-encode runs (first one writes frames.csv) + M with-encode runs.
"""
from __future__ import annotations
import argparse, collections, csv, json, os, queue, statistics, subprocess, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get('ANTI_UAV_MODELS', HERE.parent / 'models'))
sys.dont_write_bytecode = True
sys.path[:0] = [str(HERE)]
import numpy as np

SOURCE_FPS_DEFAULT = 30.0


# ----------------------------------------------------------------------------- args
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', required=True, choices=('yolo_kalman', 'fear_kalman', 'yolo_fear_kalman'))
    p.add_argument('--video', required=True)
    p.add_argument('--out', required=True, help='output dir (frames.csv, summary.json, runs/)')
    p.add_argument('--init', default='', help='JSON {"frame":[x,y,w,h]} manual init box(es)')
    p.add_argument('--gt', default='', help='UCAS visible.json or groundtruth.csv (evaluation only)')
    p.add_argument('--no-encode', action='store_true', help='single run without hw encode')
    p.add_argument('--single', action='store_true', help='one timed run in this process (internal)')
    p.add_argument('--run-index', type=int, default=0)
    p.add_argument('--repeat-noenc', type=int, default=3)
    p.add_argument('--repeat-enc', type=int, default=2)
    p.add_argument('--source-fps', type=float, default=0., help='0 = ffprobe')
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--yolo-model', default=str(MODEL_DIR / 'yolo' / 'drone_yolov8n_int8.rknn'))
    p.add_argument('--template-model', default=str(MODEL_DIR / 'fear' / 'template_fp16.rknn'))
    p.add_argument('--search-model', default=str(MODEL_DIR / 'fear' / 'search_fp16.rknn'))
    p.add_argument('--fear-core', default='1')
    # v2 defaults, identical for every mode and video
    p.add_argument('--fear-threshold', type=float, default=.70)
    p.add_argument('--yolo-conf', type=float, default=.25)
    p.add_argument('--yolo-interval', type=int, default=15)
    p.add_argument('--coast-frames', type=int, default=5)
    p.add_argument('--max-misses', type=int, default=15)
    p.add_argument('--detector-miss-limit', type=int, default=3)
    p.add_argument('--association-gate-px', type=float, default=120.)
    p.add_argument('--association-gate-scale', type=float, default=2.)
    p.add_argument('--size-ratio-limit', type=float, default=3.)
    p.add_argument('--size-memory-frames', type=int, default=120)
    p.add_argument('--yolo-suspect-checks', type=int, default=2)
    p.add_argument('--disagree-conf', type=float, default=0.5)
    p.add_argument('--template-refresh-ratio', type=float, default=1.5)
    p.add_argument('--yolo-queue', type=int, default=0, help='0 = mode default (v2: 2, yolo_kalman: 16)')
    p.add_argument('--history-frames', type=int, default=8)
    p.add_argument('--max-yolo-age', type=int, default=4)
    p.add_argument('--decode-queue', type=int, default=8)
    p.add_argument('--encode-queue', type=int, default=4)
    p.add_argument('--bitrate', type=int, default=6000000)
    for stage, cpu in (('decode', 5), ('track', 6), ('yolo', 7), ('encode', 4)):
        p.add_argument('--' + stage + '-cpu', type=int, default=cpu)
    p.add_argument('--yolo-second-cpu', type=int, default=5)
    return p


# ----------------------------------------------------------------------------- GT
def load_gt(path):
    path = Path(path)
    if path.suffix == '.json':
        j = json.loads(path.read_text())
        return [(bool(e), r) for e, r in zip(j['exist'], j['gt_rect'])]
    out = []
    for r in csv.DictReader(path.open()):
        out.append((bool(int(r['exist'])), [float(r[k]) for k in 'xywh']))
    return out


def probe_fps(video):
    try:
        s = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                     'stream=r_frame_rate', '-of', 'csv=p=0', str(video)], text=True).strip()
        a, b = s.split('/')
        return float(a) / float(b)
    except Exception:
        return SOURCE_FPS_DEFAULT


# ----------------------------------------------------------------------------- steppers
def blank_row(idx):
    return dict(frame=idx, state='LOST', bbox=None, score=None, source='NONE', event='', reason='',
                fear_score=None, yolo_ms=0., yolo_npu_ms=None, fear_ms=0., fear_npu_ms=None, kalman_ms=0.,
                yolo_called=0, yolo_ncand=0, yolo_conf=None, misses=0, detector_misses=0,
                fear_calls=0, fear_init_ms=0.)


class YoloKalman:
    """YOLO on every frame, v2-style gating + size memory, Kalman coast <= coast_frames, then LOST."""
    armed = True

    def __init__(self, detector, args, mods):
        self.det, self.a, self.m = detector, args, mods
        self.kal = None; self.misses = 0; self.ever = False
        self.last_box = None; self.last_frame = -10**6
        self.generation = 0
        self.state_for_decode = 'SEARCH'

    def lookahead(self, index):
        return True

    def _size_ok(self, cands, index):
        if self.last_box is not None and index - self.last_frame <= self.a.size_memory_frames:
            area = float(np.prod(self.last_box[2:]))
            r = self.a.size_ratio_limit
            return [c for c in cands if 1 / r <= np.prod(c['bbox'][2:]) / max(area, 1.) <= r]
        return cands

    def step(self, frame, index):
        a, m = self.a, self.m
        row = blank_row(index)
        self.det.submit(frame, index, self.generation, priority=True)
        res = self.det.take(index, self.generation, wait=True)[0]
        cands = res['candidates']
        row.update(yolo_ms=res['yolo_ms'], yolo_npu_ms=res['yolo_npu_ms'], yolo_called=1, yolo_ncand=len(cands),
                   yolo_conf=max((c['score'] for c in cands), default=None))
        elig = self._size_ok(cands, index)
        H, W = frame.height, frame.width
        if self.kal is None:
            if elig:
                c = max(elig, key=lambda c: c['score'])
                box = np.asarray(c['bbox'], float)
                t = time.perf_counter(); self.kal = m['BoxKalman'](box); row['kalman_ms'] = (time.perf_counter() - t) * 1000
                row.update(state='TRACK', source='YOLO', bbox=box.tolist(), score=c['score'],
                           event='REACQUIRED' if self.ever else 'INITIAL_LOCK', reason='global_detection')
                self.ever = True; self.misses = 0; self.last_box = box; self.last_frame = index
            else:
                row.update(reason='no_detection')
            return row
        t = time.perf_counter(); pred = self.kal.predict(); row['kalman_ms'] = (time.perf_counter() - t) * 1000
        cand = None
        if elig:
            cand = min(elig, key=lambda c: np.linalg.norm(m['center'](c['bbox']) - m['center'](pred)))
            dist = float(np.linalg.norm(m['center'](cand['bbox']) - m['center'](pred)))
            gate = max(a.association_gate_px, a.association_gate_scale * np.linalg.norm(pred[2:]))
            gate *= 1 + min(self.misses, a.max_misses) / a.max_misses
            if dist > gate:
                cand = None
        if cand is not None:
            box = np.asarray(cand['bbox'], float)
            t = time.perf_counter(); self.kal.update(box); row['kalman_ms'] += (time.perf_counter() - t) * 1000
            row.update(state='TRACK', source='YOLO', bbox=box.tolist(), score=cand['score'], reason='detection_match')
            self.misses = 0; self.last_box = box; self.last_frame = index
        else:
            self.misses += 1
            row['misses'] = self.misses
            if self.misses <= a.coast_frames:
                row.update(state='COAST', source='KALMAN', bbox=m['clamp_bbox'](pred, (H, W)).tolist(), score=0.,
                           reason='no_detection_match')
            else:
                self.kal = None; self.misses = 0
                row.update(state='LOST', event='LOST', reason='coast_exhausted')
        row['misses'] = self.misses
        return row


class FearKalman:
    """FEAR + Kalman from manual box(es); TRACK if FEAR score >= thr else COAST <= N frames then LOST (permanent)."""
    armed = True

    def __init__(self, tracker, args, mods, inits):
        self.tr, self.a, self.m, self.inits = tracker, args, mods, inits
        self.kal = None; self.misses = 0

    def lookahead(self, index):
        return False

    def step(self, frame, index):
        a, m = self.a, self.m
        row = blank_row(index)
        H, W = frame.height, frame.width
        if index in self.inits:
            box = self.inits[index]
            ms, _, _ = self.tr.initialize(frame, box)
            self.kal = m['BoxKalman'](box); self.misses = 0
            row.update(state='TRACK', source='MANUAL', bbox=list(map(float, box)), score=1., event='MANUAL_INIT',
                       fear_init_ms=ms, fear_ms=ms, reason='manual_init')
        elif self.kal is not None:
            t = time.perf_counter(); pred = self.kal.predict(); row['kalman_ms'] = (time.perf_counter() - t) * 1000
            box, score, tm = self.tr.update(frame, pred)
            row.update(fear_score=float(score), fear_ms=tm['fear_ms'], fear_npu_ms=tm.get('fear_npu_ms'), fear_calls=1)
            if score >= a.fear_threshold:
                t = time.perf_counter(); self.kal.update(box); row['kalman_ms'] += (time.perf_counter() - t) * 1000
                self.misses = 0
                row.update(state='TRACK', source='FEAR', bbox=[float(v) for v in box], score=float(score), reason='high_fear_score')
            else:
                self.misses += 1
                if self.misses <= a.coast_frames:
                    row.update(state='COAST', source='KALMAN', bbox=m['clamp_bbox'](pred, (H, W)).tolist(), score=float(score),
                               reason='low_fear_score')
                else:
                    self.kal = None
                    row.update(state='LOST', event='LOST', reason='coast_exhausted')
            row['misses'] = self.misses
        return row


class V2Fusion:
    """Unmodified v2 Fusion; optional manual init box that replaces the first YOLO lock."""

    def __init__(self, fusion, args, mods, inits):
        self.f, self.a, self.m, self.inits = fusion, args, mods, inits
        self.armed = not inits  # lookahead YOLO only after manual init is processed
        self.init_done = not inits

    def lookahead(self, index):
        f = self.f
        return self.armed and (index % self.a.yolo_interval == 0 or f.state != 'TRACK')

    def step(self, frame, index):
        f, m = self.f, self.m
        if index in self.inits and not self.init_done:
            box = np.asarray(self.inits[index], float)
            row = blank_row(index)
            ms, _, _ = f.tracker.initialize(frame, box)
            f.kalman = m['BoxKalman'](box); f.state = 'TRACK'; f.ever_locked = True; f.generation += 1
            f.history.clear(); f.history[index] = (frame, box.copy()); f.measurements[index] = box
            f.misses = f.detector_misses = 0
            f.last_detection_box = None
            row.update(state='TRACK', source='MANUAL', bbox=box.tolist(), score=1., event='MANUAL_INIT',
                       fear_init_ms=ms, fear_ms=ms, reason='manual_init')
            self.init_done = True; self.armed = True
            return row
        r = f.step(frame, index)
        row = blank_row(index)
        for k in ('source', 'event', 'reason', 'fear_score', 'yolo_ms', 'yolo_npu_ms', 'fear_ms', 'fear_npu_ms',
                  'kalman_ms', 'yolo_called', 'yolo_conf', 'misses', 'detector_misses', 'fear_calls', 'fear_init_ms'):
            if r.get(k) is not None or k in ('fear_score', 'yolo_npu_ms', 'fear_npu_ms', 'yolo_conf'):
                row[k] = r.get(k)
        row['yolo_ncand'] = len(r['yolo_candidates'])
        row['bbox'] = r['bbox']
        row['score'] = r['score'] if r['bbox'] is not None else None
        row['v2_state'] = r['state']
        if r['bbox'] is None:
            row['state'] = 'LOST'
        elif r['source'] == 'KALMAN':
            row['state'] = 'COAST'
        else:
            row['state'] = 'TRACK'
        return row


# ----------------------------------------------------------------------------- one timed run
def run_single(a):
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('OMP_NUM_THREADS', '1')
    import cv2
    cv2.setNumThreads(1)
    import pipeline as P
    from run import BoxKalman, center, iou
    from tracker_core import clamp_bbox
    from infer import NativeFear, NativeYolo, validate_cores
    hw = P.hw
    mods = dict(BoxKalman=BoxKalman, center=center, clamp_bbox=clamp_bbox)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs = out / 'runs'; runs.mkdir(exist_ok=True)
    src_fps = a.source_fps or probe_fps(a.video)
    inits = {int(k): v for k, v in json.loads(a.init).items()} if a.init else {}
    if a.mode == 'fear_kalman' and not inits:
        raise SystemExit('fear_kalman needs --init')
    if not a.yolo_queue:
        a.yolo_queue = 16 if a.mode == 'yolo_kalman' else 2
    gt = load_gt(a.gt) if a.gt else None
    use_yolo = a.mode in ('yolo_kalman', 'yolo_fear_kalman')
    use_fear = a.mode in ('fear_kalman', 'yolo_fear_kalman')
    if use_yolo and use_fear:
        validate_cores(a.fear_core, '02')

    fear = yolos = None
    detector = None
    errors = queue.Queue()
    yolos = []
    try:
        if use_fear:
            fear = NativeFear(a.template_model, a.search_model, core=a.fear_core, query_perf=True)
        if use_yolo:
            yolos.append(NativeYolo(a.yolo_model, core='0', query_perf=True))
            yolos.append(NativeYolo(a.yolo_model, core='2', query_perf=True))
            detector = P.AsyncDetector(yolos, a, errors)
        tracker = P.RGATracker(fear) if use_fear else None
        if a.mode == 'yolo_kalman':
            stepper = YoloKalman(detector, a, mods)
        elif a.mode == 'fear_kalman':
            stepper = FearKalman(tracker, a, mods, inits)
        else:
            stepper = V2Fusion(P.Fusion(detector, tracker, a), a, mods, inits)

        inputs = queue.Queue(a.decode_queue); outputs = queue.Queue(a.encode_queue)
        rows = []; stop = threading.Event()
        decoder = hw.Decoder(a.video); encoder_box = [None]; flush = [0.]
        enc_path = runs / f'timing_encode_{a.run_index}.mp4'
        t_end = {}

        def put(q, item):
            while not stop.is_set():
                try:
                    q.put(item, timeout=.2); return
                except queue.Full:
                    pass

        def decode_worker():
            try:
                P.pin_cpu(a.decode_cpu)
                i = 0
                while not stop.is_set() and (a.max_frames == 0 or i < a.max_frames):
                    t = time.perf_counter(); frame = decoder.read(); ms = (time.perf_counter() - t) * 1000
                    if frame is None:
                        break
                    if detector is not None and stepper.lookahead(i):
                        gen = stepper.f.generation if isinstance(stepper, V2Fusion) else stepper.generation
                        detector.submit(frame, i, gen)
                    put(inputs, (frame, i, ms)); i += 1
                put(inputs, None)
            except BaseException as e:
                errors.put(e); stop.set()

        def track_worker():
            try:
                P.pin_cpu(a.track_cpu)
                while not stop.is_set():
                    try:
                        item = inputs.get(timeout=.2)
                    except queue.Empty:
                        continue
                    if item is None:
                        break
                    frame, idx, dms = item
                    t = time.perf_counter(); row = stepper.step(frame, idx)
                    row['track_ms'] = (time.perf_counter() - t) * 1000; row['decode_ms'] = dms
                    row['t_done_ms'] = (time.perf_counter() - start) * 1000
                    if gt is not None and idx < len(gt):
                        ex, g = gt[idx]
                        valid = ex and len(g) == 4 and g[2] > 0 and g[3] > 0
                        row['gt_exists'] = int(ex); row['gt_valid'] = int(valid)
                        if valid:
                            row['gt'] = g
                            row['iou'] = iou(row['bbox'], g) if row['bbox'] is not None else 0.
                            row['center_err'] = (float(np.linalg.norm(center(row['bbox']) - center(g)))
                                                 if row['bbox'] is not None else None)
                    rows.append(row)
                    put(outputs, (frame, row))
                t_end['track'] = time.perf_counter() - start
                put(outputs, None)
            except BaseException as e:
                errors.put(e); stop.set()

        def encode_worker():
            try:
                P.pin_cpu(a.encode_cpu)
                roll = collections.deque(maxlen=30); last = time.perf_counter()
                colors = {'TRACK': (40, 240, 80), 'COAST': (255, 200, 0)}
                while not stop.is_set():
                    try:
                        item = outputs.get(timeout=.2)
                    except queue.Empty:
                        continue
                    if item is None:
                        break
                    frame, row = item
                    if not a.no_encode:
                        if encoder_box[0] is None:
                            encoder_box[0] = hw.Encoder(enc_path, frame.width, frame.height, src_fps, a.bitrate)
                        d = frame.clone_for_draw(); img = d.rgb()
                        if row['bbox'] is not None:
                            x, y, w, h = map(round, row['bbox'])
                            cv2.rectangle(img, (x, y), (x + w, y + h), colors.get(row['state'], (255, 255, 255)), 2)
                        fps = len(roll) / sum(roll) if roll else 0.
                        cv2.putText(img, f"{row['state']} FPS {fps:.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
                        encoder_box[0].write(d)
                    now = time.perf_counter(); roll.append(now - last); last = now
                if encoder_box[0] is not None:
                    flush[0] = encoder_box[0].close(); encoder_box[0] = None
            except BaseException as e:
                errors.put(e); stop.set()

        threads = [threading.Thread(target=decode_worker, daemon=True), threading.Thread(target=track_worker, daemon=True)]
        if not a.no_encode:
            threads.append(threading.Thread(target=encode_worker, daemon=True))
        else:  # drain outputs
            def drain():
                while not stop.is_set():
                    try:
                        item = outputs.get(timeout=.2)
                    except queue.Empty:
                        continue
                    if item is None:
                        break
            threads.append(threading.Thread(target=drain, daemon=True))
        start = time.perf_counter()
        for t in threads: t.start()
        for t in threads:
            while t.is_alive():
                t.join(.2)
                if not errors.empty():
                    raise errors.get()
        seconds = time.perf_counter() - start
        if not errors.empty():
            raise errors.get()
        if not rows:
            raise RuntimeError('no frames processed')
        calls = list(detector.calls) if detector is not None else []
        if detector is not None:
            detector.close()
    finally:
        try:
            stop.set()
        except Exception:
            pass
        for y in yolos: y.close()
        if fear is not None: fear.close()
        try:
            decoder.close()
        except Exception:
            pass

    return rows, calls, seconds, t_end.get('track', seconds), src_fps, inits, gt, flush[0], enc_path


# ----------------------------------------------------------------------------- outputs
FIELDS = ['frame', 'state', 'x', 'y', 'w', 'h', 'score', 'source', 'event', 'reason', 'fear_score', 'yolo_ms', 'yolo_npu_ms',
          'fear_ms', 'fear_npu_ms', 'kalman_ms', 'track_ms', 'decode_ms', 't_done_ms', 'yolo_called', 'yolo_ncand', 'yolo_conf',
          'misses', 'detector_misses', 'fear_calls', 'fear_init_ms', 'v2_state',
          'gt_exists', 'gt_valid', 'gt_x', 'gt_y', 'gt_w', 'gt_h', 'iou', 'center_err']


def write_csv(rows, path):
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator='\n'); w.writeheader()
        for r in rows:
            s = {k: r.get(k) for k in FIELDS}
            for k, v in zip('xywh', r['bbox'] or [None] * 4): s[k] = None if v is None else round(float(v), 2)
            for k, v in zip(('gt_x', 'gt_y', 'gt_w', 'gt_h'), r.get('gt') or [None] * 4): s[k] = v
            for k in ('score', 'fear_score', 'yolo_ms', 'yolo_npu_ms', 'fear_ms', 'fear_npu_ms', 'kalman_ms', 'track_ms',
                      'decode_ms', 't_done_ms', 'yolo_conf', 'iou', 'center_err', 'fear_init_ms'):
                if s[k] is not None: s[k] = round(float(s[k]), 4)
            w.writerow(s)


def gt_metrics(rows):
    v = [r for r in rows if r.get('gt_valid')]
    if not v:
        return {}
    ious = np.array([r['iou'] for r in v])
    errs = [r['center_err'] for r in v]
    curve = [float((ious >= t).mean()) for t in np.round(np.arange(0, 1.0001, .05), 2)]
    return dict(gt_valid_frames=len(v), gt_total_frames=len(rows), mean_iou=float(ious.mean()),
                success_at_0_5=float((ious >= .5).mean()),
                precision_at_20px=float(np.mean([(e is not None and e <= 20) for e in errs])),
                success_auc=float(np.mean(curve)),
                success_curve=dict(thresholds=[round(float(t), 2) for t in np.arange(0, 1.0001, .05)], success=curve))


def fps_stats(rows, seconds, track_seconds):
    t = np.array([r['t_done_ms'] for r in rows]) / 1000
    win = 30
    inst = [(win) / (t[i] - t[i - win]) for i in range(win, len(t))] if len(t) > win else []
    return dict(fps_wall=len(rows) / seconds, fps_track_loop=len(rows) / track_seconds,
                fps_rolling30_median=float(np.median(inst)) if inst else None,
                fps_rolling30_mean=float(np.mean(inst)) if inst else None)


def med(x):
    x = [float(v) for v in x if v is not None]
    return float(np.median(x)) if x else None


def single_main(a):
    rows, calls, seconds, track_s, src_fps, inits, gt, flush_ms, enc_path = run_single(a)
    out = Path(a.out)
    st = collections.Counter(r['state'] for r in rows)
    n = len(rows)
    fear_rows = [r for r in rows if r['fear_calls'] == 1 and not r['fear_init_ms'] and r['fear_ms']]
    # v2 replays several FEAR updates in one row after a delayed YOLO correction; keep single-update rows only
    summ = dict(mode=a.mode, video=str(a.video), frames=n, run_index=a.run_index, encode=not a.no_encode,
                source_fps=src_fps, init=inits or None,
                track=st['TRACK'], coast=st['COAST'], lost=st['LOST'], track_pct=100 * st['TRACK'] / n,
                coast_pct=100 * st['COAST'] / n, lost_pct=100 * st['LOST'] / n,
                seconds=seconds, track_loop_seconds=track_s, encoder_flush_ms=flush_ms,
                **fps_stats(rows, seconds, track_s),
                yolo_calls=len(calls), yolo_ms_median=med([c['yolo_ms'] for c in calls]),
                yolo_npu_ms_median=med([c['yolo_npu_ms'] for c in calls]),
                yolo_pre_ms_median=med([c['yolo_pre_ms'] for c in calls]),
                yolo_post_ms_median=med([c['yolo_post_ms'] for c in calls]),
                yolo_calls_by_worker=(None if not calls else dict(collections.Counter(c['worker'] for c in calls))),
                fear_ms_median=med([r['fear_ms'] for r in fear_rows]),
                fear_npu_ms_median=med([r['fear_npu_ms'] for r in fear_rows]),
                fear_updates=sum(r['fear_calls'] for r in rows),
                kalman_ms_median=med([r['kalman_ms'] for r in rows if r['kalman_ms']]),
                npu_cores=dict(yolo=[0, 2] if a.mode != 'fear_kalman' else None,
                               fear=[1] if a.mode != 'yolo_kalman' else None),
                events=dict(collections.Counter(r['event'] for r in rows if r['event'])),
                models=dict(yolo=a.yolo_model if a.mode != 'fear_kalman' else None,
                            fear=[a.template_model, a.search_model] if a.mode != 'yolo_kalman' else None),
                params={k: getattr(a, k) for k in ('fear_threshold', 'yolo_conf', 'yolo_interval', 'coast_frames', 'max_misses',
                                                  'detector_miss_limit', 'association_gate_px', 'association_gate_scale',
                                                  'size_ratio_limit', 'size_memory_frames', 'yolo_suspect_checks', 'disagree_conf', 'template_refresh_ratio', 'yolo_queue')},
                note='')
    summ.update(gt_metrics(rows) if gt is not None else {})
    runs = out / 'runs'; runs.mkdir(parents=True, exist_ok=True)
    tag = ('noenc' if a.no_encode else 'enc') + f'_{a.run_index}'
    (runs / f'summary_{tag}.json').write_text(json.dumps(summ, indent=1))
    if a.no_encode and a.run_index == 0:
        write_csv(rows, out / 'frames.csv')
    if enc_path.exists():
        enc_path.unlink()
    print(json.dumps({k: summ[k] for k in ('mode', 'encode', 'frames', 'fps_wall', 'track_pct', 'coast_pct', 'lost_pct',
                                           'yolo_ms_median', 'fear_ms_median')}), flush=True)


def driver(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['OPENBLAS_NUM_THREADS'] = '1'; env['OMP_NUM_THREADS'] = '1'; env['PYTHONDONTWRITEBYTECODE'] = '1'
    base = [sys.executable, str(Path(__file__).resolve()), '--single', '--mode', a.mode, '--video', a.video, '--out', a.out]
    for opt in ('init', 'gt'):
        if getattr(a, opt): base += ['--' + opt, getattr(a, opt)]
    base += ['--yolo-model', a.yolo_model, '--template-model', a.template_model, '--search-model', a.search_model]
    if a.max_frames: base += ['--max-frames', str(a.max_frames)]
    if a.source_fps: base += ['--source-fps', str(a.source_fps)]
    plan = [(True, i) for i in range(a.repeat_noenc)] + [(False, i) for i in range(a.repeat_enc)]
    for noenc, i in plan:
        cmd = base + ['--run-index', str(i)] + (['--no-encode'] if noenc else [])
        r = subprocess.run(cmd, env=env)
        if r.returncode:
            print('run failed', cmd, r.returncode, flush=True)
    # aggregate
    rs = {}
    for f in sorted((out / 'runs').glob('summary_*.json')):
        rs[f.stem[len('summary_'):]] = json.loads(f.read_text())
    ne = [rs[k] for k in sorted(rs) if k.startswith('noenc')]
    en = [rs[k] for k in sorted(rs) if k.startswith('enc')]
    if not ne:
        raise SystemExit('no successful no-encode run')
    s = dict(ne[0])
    s.pop('run_index', None); s.pop('encode', None)
    s['fps_noenc_runs'] = [r['fps_wall'] for r in ne]
    s['fps_enc_runs'] = [r['fps_wall'] for r in en]
    s['fps_noenc_median'] = float(np.median(s['fps_noenc_runs']))
    s['fps_enc_median'] = float(np.median(s['fps_enc_runs'])) if en else None
    s['fps_noenc_rolling30_median_of_run0'] = ne[0]['fps_rolling30_median']
    s['fps_noenc_rolling30_mean_of_run0'] = ne[0]['fps_rolling30_mean']
    s['fps_enc_rolling30_median'] = float(np.median([r['fps_rolling30_median'] for r in en])) if en else None
    s['fps_enc_rolling30_mean'] = float(np.mean([r['fps_rolling30_mean'] for r in en])) if en else None
    s['track_pct_runs'] = [r['track_pct'] for r in ne]
    s['yolo_ms_median'] = med([r['yolo_ms_median'] for r in ne])
    s['yolo_npu_ms_median'] = med([r['yolo_npu_ms_median'] for r in ne])
    s['fear_ms_median'] = med([r['fear_ms_median'] for r in ne])
    s['fear_npu_ms_median'] = med([r['fear_npu_ms_median'] for r in ne])
    s['summary_of_frames_csv'] = 'noenc run 0 (frames.csv and track/coast/lost/GT metrics come from this run)'
    (out / 'summary.json').write_text(json.dumps(s, indent=1))
    print('SUMMARY', a.mode, json.dumps({k: s[k] for k in ('fps_noenc_median', 'fps_enc_median', 'track_pct')}), flush=True)


if __name__ == '__main__':
    args = build_parser().parse_args()
    if args.single:
        single_main(args)
    else:
        driver(args)
