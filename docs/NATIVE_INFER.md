# Native inference wrapper

Build **on RK3588** with `bash tracker/build_infer.sh` (set `RKNN_INCLUDE`/`RKNN_LIB`
if the RKNN header/runtime are not in the system paths); g++, Python and NumPy are required. The library is written to a temporary
name and atomically installed so rebuilding cannot truncate a running mapping.
This wrapper does not retrain or convert models. Its standard inference
contexts use `rknn_init` flags 0.

`NativeFear(template, search, core='1')` accepts contiguous uint8 RGB images:
`template(rgb128)` returns a timings dict and stores the template internally;
`search(rgb256)` returns `(bbox_distances, cls_logits, timings)` as float32 NCHW
arrays compatible with the original `FEARTracker.decode`. Regression Exp
remains NumPy on CPU. `.feature` exposes float32 NCHW template features for
parity checks, but search uses the internal FP16 features, transposed exactly
once into the queried NHWC search-input layout. No FP32 whole-image NCHW
transpose occurs in Python.

`NativeYolo(model, core='0')` accepts contiguous `rgb640` and returns nine
float32 NCHW outputs plus timings for the existing YOLO postprocessing. Core
0 and FEAR core 1 are disjoint. The wrapper recognizes mask `02` as bit value
5, but the installed RKNN 2.3.2 runtime/this model rejected it with
`rknn_set_core_mask returned -1`; **use YOLO core 0**. Do not claim core 2 is
used in the demonstrated configuration. `validate_cores()` rejects overlapping
bit masks; the pipeline should invoke it before opening sessions.

Image `rknn_input` remains uint8 NHWC with `pass_through=0`; the runtime performs
its required internal conversion to model FP16. This is **not DMA input zero
copy**. Native RKNN outputs are requested with `want_float=0`, then converted
using ARM NEON FP16 conversion / INT8 dequantization and copied into NumPy
arrays. Native output conversion is therefore explicitly measured. The
unvalidated pass-through input experiment was removed from delivered source.

Timings, in milliseconds:

- `inputs_ms`: synchronous `rknn_inputs_set`, including runtime conversion.
- `run_get_ms`: `rknn_run` plus native `rknn_outputs_get` wall time.
- `outputs_ms`: CPU native-output conversion/copy.
- `npu_ms`: `RKNN_QUERY_PERF_RUN.run_duration / 1000`, or None if unavailable.
  The SDK defines this as real inference time, not strictly an isolated NPU
  kernel time if a model has CPU/GPU fallback. Existing FEAR profiling showed
  all compute operators on NPU with CPU I/O operators only; keep that scope
  clear when reporting it. No performance-collection flag is enabled to obtain
  the normal-run value, which was available on this board.
- `cpu_exp_ms`: original CPU bbox restoration for FEAR, zero for template/YOLO.
- `total_ms`: complete Python call including CPU Exp, ctypes and output copy.
- Additional detailed keys expose release/query/native-call timing.

`profiling=True` enables a **separate diagnostic context** with the collection
flag; it must not be used for final pipeline FPS. Timings mark whether that
flag is active. The full JSON descriptions preserve queried input/output
shape, dtype, quantization and layout, SDK/driver and core mask.

The (unshipped) probe script constructed identical v1 tensors using software decode/resize
solely for validation, then runs both implementations on the board. This is
not the hardware video pipeline. The recorded 20-repeat probe
`native_infer_probe.json` (not included in this repository) showed exact FEAR template/bbox/cls equality and all
YOLO outputs within 1e-5, with the final detection box/confidence identical.
FEAR C API call mean was 7.10 ms versus RKNNLite including Python packing
10.24 ms; YOLO 11.16 versus 12.33 ms on core 0. These are short same-session
microbenchmarks, not v2 end-to-end FPS or a substitute for full-sequence IoU.
Concurrent load, clock frequency and one-core versus all-core execution change
absolute timings; final pipeline CSV/summary are authoritative.
