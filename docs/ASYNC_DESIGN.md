# v2 asynchronous fusion design notes

These notes describe implementation invariants, not measured v2 results. They
were checked against `tracker/run.py`, `tracker/tracker_core.py`, and the v1 reference
run on 2026-10-02. Ground truth must remain evaluation-only.

## Sequential tracking with asynchronous detection

Use one decode producer, one tracking owner, one draw/encode consumer, and one
YOLO worker. Only the tracking owner may mutate FEAR, Kalman, state, counters,
or template features. FEAR frame i+1 must start after frame i has completed.
The detector has a separate RKNN context on core 0, or a supported 0+2 mask;
both FEAR contexts use core 1. Reject intersecting masks rather than comparing
their CLI strings. Core 0+2 is the bit mask 5 if RKNNLite exposes no named
`NPU_CORE_0_2` constant; check API acceptance on the board.

Packets retain source frame ID, source PTS, input dimensions, immutable pixel
buffer/DMA ownership, and timing fields. Detector jobs/results retain source
frame ID, generation ID, submit/start/finish monotonic times, candidates,
preprocess transform, and all detector timings. A generation increments after
LOST/new lock so an earlier track cannot update a new target.

Bound decode and encode queues (for example 8/8). Block producers when full;
never drop decoded or tracked frames during an offline accuracy benchmark.
`appsink drop=false`, `sync=false`, and encoder/appsrc `block=true` preserve the
original frame count without pacing to 20 FPS. Keep every output frame at its
original source PTS and report processing FPS separately from playback FPS.
Holding a GstSample reference is essential: its DMA fd is invalid for use after
the buffer has returned to the MPP pool. Frames in the YOLO queue/history need
their own references. Do not reuse an output DMA buffer while the encoder still
owns it. Shutdown/error cancellation must unblock all queue producers.

Only the YOLO request queue may coalesce pending redundant requests, and every
coalesced job should be explicitly counted. One running job plus one pending
job avoids arbitrarily stale detection backlog. Search/low-score requests have
priority over periodic requests. A continuously busy search worker can accept
the latest decoded frame without replacing the job already in inference.
Tracking must consume completed results without waiting for YOLO in TRACK.

## Safe association of delayed detections

A detector box belongs to its source frame s, not the current frame i. Never
call `initialize(frame_i, box_s)`: moving targets/camera motion would produce a
background template. Never perform `kalman.update(box_s)` directly at i.

Prefer decode lookahead: submit periodic jobs when their immutable decoded
frame becomes available, before the tracking consumer reaches it. This can
hide detector latency without a stale update. Schedule from source frame IDs,
not wall time, and expose source/apply frame and lag in CSV. Search requests
need an explicit policy because preserving all frames and acquiring on exactly
the first detected frame cannot be guaranteed by a completely nonblocking
search when YOLO takes longer than the tracking stage.

For an on-time result (s == i), preserve v1 association and update logic. For a
delayed result in TRACK:

1. Require s <= i, same generation, source history retained, and lag within
   `--max-yolo-lag-frames` (suggest 3 initially). Process each result once in
   increasing source order; log discarded future, stale, and obsolete results.
2. Select the nearest eligible candidate to the prediction saved at frame s,
   applying the size memory and spatial gate at s. Association with prediction
   at i would make a correct moving target look too far away.
3. If all intervening tracking steps are confident, transport the candidate
   using their displacement, or preferably use an out-of-sequence Kalman
   measurement update: update the saved predicted x/P at s, replay saved
   prediction/accepted FEAR measurements s+1..i, then install the resulting
   current x/P. Preserve source measurement covariance and gate at i as well.
   Do not rewrite already emitted CSV/video rows.
4. A transported correction should not replace a confident current FEAR box
   with an unadjusted old box. Log original and propagated boxes separately.
   Retain the existing template on periodic corrections as v1 did.
5. If intervening steps were low confidence, FEAR displacements cannot safely
   transport the candidate. The conservative option rejects that delayed
   correction and requests a new detection. A recovery path can initialize
   FEAR on retained source frame s and then run its retained frames s+1..i
   serially in the tracking thread. These extra calls are counted as replay
   work, not parallel FEAR. Accept the replay only after confidence/association
   validation. Replaying becomes unreliable/costly as lag grows, hence a bound.

For SEARCH/reacquisition, initialize the template on the retained source frame
s. Either retain source→current frames for serial FEAR catch-up, or allow the
SEARCH state to await a fresh result before committing that source frame. This
wait overlaps decode and encode but must be reported as acquisition wait. It
does not block per-frame tracking in TRACK. Do not backfill ground truth or
retroactively report that earlier SEARCH frames were tracked.

Negative YOLO results also require correct chronology. Count one detector miss
per accepted completed job, not once for each tracking frame that sees the same
empty result. A stale negative should not veto a new confident target or new
generation. Preserve v1's detector miss override only if the result is current
enough and still pertains to the track being tested. Otherwise async throughput
can silently create extra LOST transitions or leave high-score drift unchecked.

Deterministic detector injection tests should cover result arrival at source,
lag 1..max, stale lag, generation change, empty results, out-of-order results,
two candidates, and position/size gate failures. A moving synthetic target
tests that initialization uses source pixels and that delayed boxes are
propagated. Verify high-confidence FEAR never initializes a template from a
YOLO box at a different time. Use no-model fake tracker/detector tests on the
board; they should test chronology/state semantics, not duplicate math.

## RGA preprocessing while retaining FEAR semantics

The v1 crop path uses a uint8 RGB source. FEAR template uses 128x128 with offset
0.2; search uses 256x256 with offset 2.0. Context is rectangular:
`[x-w*offset, y-h*offset, w*(1+2*offset), h*(1+2*offset)]` converted to int32
with truncation toward zero. Width and height are extended independently. The
rectangle is padded with initialization-frame mean RGB and warped directly to
a square using bilinear resize. It is not a square context or a letterbox.
Preserve the original integer context, bbox clipping order, padded bbox scale,
and original coordinate mapping even if RGA source alignment is expanded.

MPP should produce NV12 DMA buffers. RGA converts NV12→RGB/BGR and performs
all resize/crop/pad pixel work. No `cv2.cvtColor`, GStreamer `videoconvert`,
FFmpeg `swscale`, CPU NV12 arithmetic, or hidden OpenCV capture remains in the
benchmark path. Verify and report that decode really uses MPP and the actual
caps/format/allocator. Loading RKNN input from a small CPU array is allowed;
uint8 RGB→float32 NCHW is input packing, not a color conversion or resize.
CPU Exp remains unchanged on the small regression output.

NV12 chroma crop coordinates/dimensions must be even. For exact odd-coordinate
FEAR geometry, either RGA-convert the full frame once to an RGB DMA buffer then
crop that, or RGA-convert an even-expanded source ROI to RGB and perform the
exact odd RGB crop with RGA. Full-frame RGA conversion is the simpler baseline;
its measured cost should be reported separately, and no CPU whole-frame copy
is necessary for FEAR or YOLO. A draw thread can map the RGA RGB/BGR buffer to
draw and RGA-convert to encoder NV12, with DMA cache sync around CPU writes.
On the board the existing `/usr/include/rga` and librga 2.1.0 support these
operations; fail on an RGA error rather than silently reverting to CPU resize.

YOLO keeps the 640x640 letterbox with original scale, exact rounded left/top
and dimensions, fill RGB (114,114,114), and original postprocess. RGA may have
maximum scale factor/alignment restrictions: use an intermediate RGA buffer
for an exceptionally small FEAR template/search context; count both passes.
Keep NV12 stride and vertical stride from GstVideoMeta or caps, not width and
height assumptions. RGB_888 width stride must satisfy the installed RGA
driver; align DMA stride while exposing only logical image columns to RKNN.
Different jobs need separate destinations or a mutex around a shared scratch
pool. RGA hardware submissions can contend even with independent NPU cores.

The exact full-image initial mean is CPU arithmetic over a RGA RGB result and
adds acquisition-only overhead. Approximating it with a downscaled image could
change border padding and tracking, so do not silently substitute. Avoid its
second unnecessary computation (v1 initialize sets mean_color then
extended_crop recomputes the same mean because padding_value is omitted).
Steady search needs no whole-frame CPU mean. The largest v1 FEAR CPU cost is
full-frame BGR→RGB plus border/crop/resize and array input packing, not CPU Exp.

Check stream colorimetry/range. RGA's default BT.601 limited conversion is not
always equivalent to an input tagged BT.709/full. Set an explicit supported
conversion mode from caps and record it. Compare representative sky, greenery,
and near-edge crops with v1 software-decoded RGB on the board. RGA bilinear
sampling can differ from OpenCV; do not claim bit parity without measuring it.
Run the same complete 1000-frame annotated sequences and evaluate IoU/events
before deciding whether interpolation differences are acceptable.

## Timings, accuracy, and audit evidence

Each stage records busy/service time separately from queue wait. Overlapping
stage milliseconds do not add to a total frame time. Aggregate FPS is processed
frame count / (first decode start→all stage joins and final encoder EOS/flush).
Exclude model loading and writing final CSV/JSON consistently with v1. Include
initial detection, template initialization, all replay work, pending jobs at
shutdown, and all final encoded frames. No-video mode still decodes and tracks
all frames with identical scheduling inputs. Do not compare a latest-only
frame dropping run to the full-frame v1 baseline.

Record decode, NV12→RGB, YOLO preprocess/inference/postprocess, FEAR preprocess,
FEAR search NPU runtime call, CPU Exp/postprocess, template initialization,
Kalman, draw, RGB→NV12, encode submission, encoder wait/flush, and queue waits.
An encoder submit call can return before hardware work: report frame/byte
completion and EOS drain, not submit timing alone. RGA sync calls and RKNNLite
runtime calls are host wall timings rather than kernel-only hardware time.
Report both per-job/per-FEAR-call timings and amortized per-all-frame timings.

CSV rows need frame/source PTS, state/source/box/score/prediction, consumed
YOLO source/apply frame and age, job ID, original/propagated candidate, reason,
event, misses, detector misses, per-stage times, and queue wait. Detector job
log captures jobs that complete after their source CSV row has been emitted;
attach detector cost to the apply row for accounting or keep a separate jobs
CSV, clearly documenting it. Do not mutate rows from the YOLO thread.

Baseline acceptance thresholds are sky mean IoU >=0.793 and IoU>=0.5 rate
>=97.40%, complex mean IoU >=0.598 and success >=82.75%, without additional
visible misses or confirmed wrong-target locks. Compare raw summaries rather
than rounded report figures. Loss/recovery counts alone can be misleading:
fewer false losses are an improvement; fewer recoveries due to more time lost
are not. Report visible missed frames, false positive boxed frames, recovery
latency at reentry, per-frame IoU, and events. Original complex GT anomaly at
434..442 remains documented; use the same valid-frame mask as v1.

For air-to-air, detector miss rate must be defined on annotated visible target
frames. Periodic fusion detector calls produce only a sampled miss rate; a
separate exhaustive YOLO evaluation on every annotated frame is needed for an
honest full-video miss rate and it is not included in fusion FPS. Compare the
same IoU/confidence criterion across clips. Camera motion prediction error is
pre-update Kalman center error against GT (and normalized by target diagonal)
on visible frames; assess p50/p95 and the probability that the target remains
inside the FEAR search context. Image-plane constant velocity has no ego-motion
compensation, so sudden tilt/shake may invalidate it even with high throughput.
