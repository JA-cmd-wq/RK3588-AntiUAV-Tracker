# Tracker pipeline: MPP/RGA asynchronous YOLO + FEAR + Kalman

Runs on RK3588 with the YOLOv8n INT8 detector and the FEAR template/search FP16 models. No training, conversion or FEAR INT8 is involved at run time. `tracker_core.py` grid decoding and CPU Exp restoration are retained. GT is used only for evaluation.

```bash
bash tracker/build_native.sh     # libnative_io.so  (GStreamer MPP + RGA)
bash tracker/build_infer.sh      # libanti_uav_infer.so  (RKNN C API wrapper; set RKNN_INCLUDE / RKNN_LIB if needed)
bash tracker/run_board.sh \
  --video test_videos/rice_field.mp4 \
  --output out/rice_field.mp4
```

Models default to `models/fear/{template,search}_fp16.rknn` and `models/yolo/drone_yolov8n_int8.rknn`; override with `--template-model`, `--search-model`, `--yolo-model` or the `ANTI_UAV_MODELS` environment variable. `run_board.sh` uses `python3` by default (`PYTHON=...` to change; `EXTRA_PYTHONPATH=...` to add an extra site-packages directory containing `rknnlite`).

Board dependencies: GStreamer 1.24 with `mppvideodec`/`mpph264enc`, librga 2.1.0, DMA heap access, librknnrt 2.3.2 and C headers, C++ compiler, a Python environment with NumPy, OpenCV and RKNN Toolkit Lite2. `run_board.sh` limits BLAS threads. `RKNN_INCLUDE` / `RKNN_LIB` override the RKNN header and runtime library locations at build time.

The decoder handles MP4 input and demands NV12 DMA-BUF output. It errors rather than falling back to CPU decode/resize/colorspace conversion. RGA converts decoded NV12 to RGB DMA storage for FEAR's exact rectangular context crop. YOLO uses RGA to convert and letterbox directly from NV12. RGA also copies an independent drawing frame and converts it back to NV12 for MPP H.264 encoding. Small RGB network tensors are copied into NumPy and RKNN `inputs_set`; this is **partial DMA zero copy**, not end-to-end input zero copy. Production `cv2.cvtColor` and `cv2.resize` calls raise errors. Offline dataset preparation and visual audits may use CPU image operations and are excluded from timed runs.

Decode, sequential FEAR/Kalman tracking and drawing/encoding each have a worker; two asynchronous YOLO workers own separate contexts by default (five application workers total). Bounded queues preserve every source frame; excess pending YOLO jobs coalesce. TRACK never waits for a detector result. SEARCH waits for the current frame's detection because it has no track yet. Generation IDs discard results from a previous lock. After two unmatched detector checks, per-frame asynchronous detection starts before LOST, so a high FEAR score cannot hide a missed recovery window. Delayed detections associate against the prediction on their **source frame**, then move to current time using FEAR measurement displacement when available, falling back to the Kalman prediction displacement. Low-score recovery initializes the template on that source image and replays intervening FEAR frames serially; a low-score replay cannot claim recovery. FEAR is never run across frames in parallel. Drawing uses a cloned frame so template/replay source pixels stay untouched.

FEAR uses core 1 (mask 2). Two independent YOLO contexts use core 0 (mask 1) and core 2 (mask 4), so all three NPU cores participate. `--yolo-workers 1` selects the measured single-detector reference. A single YOLO context using the combined core 0+2 mask 5 returns `rknn_set_core_mask=-1` on this runtime; the two-context solution avoids that unsupported mask. RKNN initialization does not use AUTO. Each worker has its own context, and an inflight set prevents duplicate jobs. Results arriving out of order cannot supersede an already accepted newer source frame. CPU workers default to the board's A76 CPUs 4–7; `-1` disables a stage's affinity. Native inference releases the Python GIL and removes full-image float32 NCHW packing. YOLO postprocessing filters anchors by confidence before DFL; numerical equivalence was verified against v1 on real cached outputs.

Default display has one green final target box, `cls=0 score=...`, state and rolling processing FPS. `--debug` colors the final box by its source (YOLO yellow, FEAR green, Kalman blue) and adds source and frame index. Kalman-only boxes have score 0, because no detector/tracker measurement supports them.

| New parameter | Default | Meaning |
|---|---:|---|
| `--source-fps` | 20 | Output playback FPS |
| `--decode-queue` | 8 | Decoded frame queue capacity |
| `--encode-queue` | 4 | Tracked frame queue capacity |
| `--yolo-workers` | 2 | Independent YOLO contexts on cores 0 and 2; use 1 for the reference |
| `--yolo-second-cpu` | 5 | Second detector worker Linux CPU affinity |
| `--yolo-queue` | 2 | Pending detector queue capacity |
| `--yolo-suspect-checks` | 2 | Run asynchronous YOLO each frame after repeated detector disagreement |
| `--max-yolo-age` | 4 | Maximum source-frame age of accepted detection |
| `--history-frames` | 8 | Frame/prediction history for association and serial recovery replay |
| `--bitrate` | 6000000 | MPP H.264 target bits per second |
| `--decode-cpu` | 5 | Decoder consumer Linux CPU affinity |
| `--track-cpu` | 6 | Sequential FEAR/Kalman CPU affinity |
| `--yolo-cpu` | 7 | Detector CPU affinity |
| `--encode-cpu` | 4 | Drawing/encoder CPU affinity |
| `--debug` | off | Source-colored box plus state/FPS labels |
| `--no-encode` | off | Full decode/inference benchmark without drawing or encoding |

The v1 fusion parameters remain available: `--fear-threshold .7`, `--yolo-conf .25`, `--yolo-interval 15`, `--coast-frames 5`, `--max-misses 15`, `--detector-miss-limit 3`, association gate 120 pixels / 2 box diagonals, box-area ratio gate 3 and size memory 120 frames. `--no-video` aliases `--no-encode`. The inherited `--codec`/`--cv-threads` options do not select v2 behavior: encoding is always MPP H.264 and OpenCV uses one thread.

Each output stem writes `.csv`, `.summary.json`, `.yolo_calls.json` and, unless encoding is disabled, `.mp4`. CSV includes frame state, final/predicted/raw FEAR boxes, score, accepted YOLO source frame/age/candidates, events, GT IoU and Kalman prediction error, plus stage timings. The calls JSON logs every actual detector execution against its source frame; skipped/coalesced jobs are not detector misses.

Timings are wall times within concurrent workers and cannot be summed to infer FPS. `decode_ms` is appsink read/wait, not the MPP device kernel duration; upstream GStreamer decode runs concurrently. `format_ms` is the lazy full-frame RGA NV12→RGB conversion. YOLO's direct letterbox includes its own conversion. `fear_pre_ms` excludes the separately recorded full-frame conversion. `fear_api_ms` includes input transfer and runtime overhead; `fear_npu_ms` is RKNN PERF_RUN duration, not a silicon-only claim. RGA RGB→NV12 is `encode_convert_ms`; `encode_ms` is appsrc submission/backpressure, not hardware kernel time. EOS flush is included in total FPS and separately reported. FPS excludes model load and CSV/JSON serialization, includes initial acquisition, every source frame and encoder EOS. `--no-encode` skips drawing as well as encoding.

Concurrency and state-machine tests run on the board without NPU inference or video IO (they use fake detector/tracker objects):

```bash
bash tracker/run_board.sh --help
python3 tracker/test_async.py
```

Measured speed and per-clip tracking results are in the [README](../README.md).

RGA robustness: MPP can supply physical pages above 4 GiB. NV12 source operations explicitly select RGA3, while application RGB scratch buffers use DMA32 so RGA2 remains safe. Exact one-pixel-origin rectangles use the legacy hardware blit API to avoid an im2d checker limitation. Extreme resize ratios use multiple RGA stages, preserving the context mapping; oversized search contexts are centered and capped at the hardware 8192-pixel extent. There is no CPU fallback.

`monitor_npu.py` samples driver per-core load and frequency separately during benchmarks. Summary `worker_cpu_seconds`/`worker_cpu_utilization_percent` use each application worker’s thread CPU clock, excluding waiting; the YOLO value sums two workers. GStreamer and RKNN internal threads are not included in these application-worker CPU values. NPU utilization and frame throughput are different metrics: repeated detector jobs are not inserted solely to inflate load.
