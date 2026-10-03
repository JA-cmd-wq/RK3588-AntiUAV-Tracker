# Fusion state machine: YOLO + FEAR + Kalman

Single-target RGB anti-UAV tracking built from a single-class YOLOv8n INT8 detector and the FEAR-XS FP16 template/search models, tied together by an 8D constant-velocity Kalman filter. No training or model conversion is involved at run time. `tracker/tracker_core.py` implements crop, CPU Exp restoration, grid decode and clamping.

`tracker/run.py` is the reference (single-threaded, OpenCV video IO) implementation of the state machine and the source of the shared helpers (`BoxKalman`, `iou`, CLI parser). The optimized MPP/RGA pipeline in `tracker/pipeline.py` reuses the same fusion rules with asynchronous detection; see [TRACKER_PIPELINE.md](TRACKER_PIPELINE.md).

```bash
python3 tracker/run.py --video input.mp4 --output out/result.mp4
```

`--gt` (optional, evaluation only) takes a UCAS-style `visible.json` with `gt_rect`/`exist` arrays. The fusion state machine never sees annotations or a manually supplied initial box. Output MP4 preserves the source resolution, frame count and playback FPS; overlay FPS is measured processing speed over the previous 30 complete frames. Each run writes an adjacent CSV and summary JSON.

## State machine and defaults

| Parameter | Default | Purpose |
|---|---:|---|
| `--fear-threshold` | 0.70 | Accept raw FEAR sigmoid classification score |
| `--yolo-conf` | 0.25 | YOLO confidence filter |
| `--yolo-interval` | 15 frames | Full-frame periodic correction |
| `--coast-frames` | 5 | Maximum displayed Kalman-only frames |
| `--max-misses` | 15 | Low-confidence frames without recovery before LOST |
| `--detector-miss-limit` | 3 | Consecutive YOLO checks that confidently see the target elsewhere override persistent high FEAR scores |
| `--disagree-conf` | 0.5 | YOLO score a detection elsewhere needs before it counts against FEAR |
| `--template-refresh-ratio` | 1.5 | Refresh the FEAR template on a periodic YOLO match whose box area differs from FEAR's by more than this |
| `--association-gate-px` | 120 | Minimum association radius, pixels |
| `--association-gate-scale` | 2 | Radius in predicted box diagonal lengths |
| `--size-ratio-limit` | 3 | Accepted box-area ratio against previous YOLO measurement: 1/3–3 |
| `--size-memory-frames` | 120 | Size prior retained across LOST, then expires |
| `--yolo-core` | 0 | RKNNLite mask 1 |
| `--fear-core` | 1 | RKNNLite mask 2 for both FP16 branches |

SEARCH runs YOLO every frame and initializes FEAR plus Kalman from the highest-confidence detection. TRACK first predicts a box with an 8D constant-velocity Kalman filter (one source-frame time step), then centers the original FEAR search crop on that prediction. High-score FEAR updates the filter. Low-score FEAR never updates it and is discarded; YOLO runs immediately and the closest detection to the **pre-update prediction** is associated. The radius expands during misses. A match updates Kalman and low-score recovery refreshes the FEAR template. Periodic matches correct position, and also refresh the template when the YOLO and FEAR box areas differ by more than 1.5×: FEAR's box scale follows its template, so a target that has grown or shrunk since initialization would otherwise be boxed at the old size. Five missing frames may display blue Kalman boxes; later waiting frames have no box. Fifteen uncorrected low-score frames enter LOST; the next frame returns to global SEARCH. Three detector checks that each see a confident detection (score ≥ 0.5) elsewhere, none inside the gate, also force global search despite high FEAR scores. A check with no detection, or only weak ones, is not counted: the detector may simply not see the target at that size or angle (stars on a night sky are typical weak hits), and letting it veto a confident FEAR track made the combined mode worse than FEAR alone.

State and box source are separate CSV fields. Yellow=YOLO, green=FEAR, blue=Kalman. Initial acquisition is not counted as recovery. `search_reacquisitions` counts new locks after LOST; `coast_recoveries` includes low-score YOLO recovery on the same frame as the initial low score, even before a Kalman-only frame was displayed. `event`, `reason` and all YOLO candidates are retained for auditing.

The Kalman internal state is not image-clamped; crop and drawing boxes are clamped. It tracks center, width/height and their velocities. A box-area gate against the last YOLO measurement excludes drastic size changes before nearest-candidate association; it remains active during short global searches and expires after 120 frames to allow later scale changes. Gradual scale changes remain possible because each accepted YOLO measurement updates this prior. It is a single-target tracker, with spatial/size association rather than an appearance identity model. FEAR scores are not calibrated presence probabilities: historical validation had score 0.982 on a zero-IoU prediction. These are starting parameters; the size gate was added after the first smoke/clip runs exposed background false acquisitions. Ground truth was used to audit the results, not in inference. This is an engineering demo, not a held-out benchmark or fitted presence model.

## Measurement

`decode_ms` measures `VideoCapture.read`; RGB conversion is separate. YOLO contains letterbox, synchronous RKNN inference and DFL/NMS. FEAR contains the existing crop, synchronous runtime (including CPU Exp), and decode; template initialization is separately identified. Kalman prediction/correction, drawing, and `VideoWriter.write` have separate timers. Encoder closure/flush is included in total FPS and amortized into `encode_ms_including_flush`. Model loading and final CSV/JSON serialization are excluded. Inference timers include memory transfers and Python/runtime overhead, not only NPU kernels.

Summary JSON reports all-frame averages, actual-call averages and call counts because YOLO executes intermittently. Use `--no-video` on the same full input for the decode+inference ablation; it skips both drawing and encoding. Do not compare different resolutions or clips as a controlled benchmark.

CSV includes zero-based frame index, state, source, final XYWH box, predicted and raw FEAR boxes, raw FEAR score, detector score/distance/candidates, miss counters/events, all timings, and optional GT presence/validity/IoU/center error. A visible annotation with nonpositive width/height is invalid and excluded from accuracy aggregates.

## Data and verification notes

The rules above were regression-checked with `--gt` on two visible-light RGB clips of the UCAS Anti-UAV300 test set ([ucas-vg/Anti-UAV](https://github.com/ucas-vg/Anti-UAV), `visible.json` XYWH annotations). Anti-UAV410/600 infrared clips are not used. With the earlier "any unmatched check counts" rule the exit/re-entry clip had IoU ≥ 0.5 on 83.3% of frames and zero IoU on 8.9%; with the current rules 91.1% and 2.7% (older YOLO model, which suits that clip). That dataset is not redistributed here. Zero IoU is evidence of a wrong location, not proof of an identity switch; visual review and target identities are needed to claim another object was followed.
