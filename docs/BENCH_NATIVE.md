# Native RKNN latency and device profiling

`bench_native.c` uses the board's RKNN C API and runtime. No package installation is needed.

Compile on RK3588, from `fear_to_rknn/bench_native/`. `RKNN_INCLUDE` is the directory containing `rknn_api.h` (shipped with rknn-toolkit2 under `rknpu2/runtime/Linux/librknn_api/include`):

```sh
mkdir -p native_bench
gcc -O2 -std=c11 -Wall -Wextra -Werror -I"$RKNN_INCLUDE" \
  bench_native.c /usr/lib/librknnrt.so -lm -o native_bench/bench_native
```

The raw input files `template_template.bin`, `search_search.bin` and `search_template_features.bin` are written by `fear_to_rknn/export_models.py` into its output directory (they are not committed). Provide a raw binary for every model input, in input index order. Each binary must be contiguous, little-endian float32, in NCHW order, with exactly `input.n_elems * 4` bytes. The tool explicitly sets `RKNN_TENSOR_FLOAT32` and `pass_through=0`, allowing RKNN to convert to the model's internal dtype and layout. Its default input layout is NCHW. **For these FEAR models use `--input-layout nhwc`: RKNN 2.3.2 reports normalize errors with NCHW inputs even when `rknn_inputs_set` returns success.** This option transposes the NCHW raw files to NHWC once, before timing, and passes `RKNN_TENSOR_NHWC` to the runtime. Both image and feature inputs are transposed. The JSON records the actual runtime layout and excludes the one-time transposition from model timing. Use exactly the same preprocessing contract as the reference export; a model configured with mean/std preprocessing will apply that configuration.

On a machine with existing NumPy, an NCHW input can be saved with `np.ascontiguousarray(input, dtype='<f4').tofile('input.f32')`.

```sh
mkdir -p results
# Single-input template model
./native_bench/bench_native models/template_fp16.rknn models/template_template.bin \
  --prefix results/template_fp16 --core 0,1,2 --input-layout nhwc \
  --warmup 100 --iterations 500 --dump-prefix results/template_fp16

# Two-input search model: confirm each model input index in the output JSON
./native_bench/bench_native models/search_fp16.rknn models/search_search.bin models/search_template_features.bin \
  --prefix results/search_fp16 --core 0,1,2 --input-layout nhwc \
  --warmup 100 --iterations 500 --dump-prefix results/search_fp16
```

Create the output directory first. Select `--core auto`, `0`, `1`, `2`, `0,1`, or `0,1,2` as needed. Default warmup/repeats are 100/500 **for each pass**.

The program creates two fresh synchronous contexts:

1. `plain`: `rknn_init` flags = 0. This pass measures ordinary native API wall latency and queries `RKNN_QUERY_PERF_RUN` after every `rknn_outputs_get`.
2. `profile`: flags = `RKNN_FLAG_COLLECT_PERF_MASK` (`0x8`). This pass repeats the measurements and copies `RKNN_QUERY_PERF_DETAIL` after the final `rknn_outputs_get`, before releasing outputs or destroying the context.

The board's actual header documents `rknn_perf_run.run_duration` in **microseconds**, so the files report milliseconds. It also explicitly warns that collection mode reduces frame rate. Use the plain pass for normal latency; **do not calculate final tracker FPS from the profiling pass wall time**.

Outputs:

- `PREFIX.json`: model I/O attributes, runtime and driver versions, flags and core mask, mean/median/p95/min/max/stddev for each latency stage, and device-token rows extracted from the detailed report.
- `PREFIX.csv`: each sample from both passes. The `wall_ms` sum includes `inputs_set`, `run + outputs_get` with native outputs, and `outputs_release`; it excludes both performance-query calls. The complete pass elapsed time, which includes querying, is separately recorded as `elapsed_with_queries_ms`.
- `PREFIX.perf.txt`: the exact runtime performance report. Review each operator's device field for CPU/GPU fallback. The JSON parses RKNN 2.3.2's `ID OpType DataType Target ... Time(us)` rows into device counts, per-device profiled time sums, and `cpu_compute_operator_rows` that exclude InputOperator/OutputOperator boundary processing. It also keeps conservative CPU/GPU token lists. If the report has no recognizable device rows, `device_assignment_not_reported` means device placement is unverified, not that all operators ran on NPU. Per-operator sums are from **one profiling inference**, not the normal 500-repeat mean.
- Optional `DUMP_PREFIX.output_N.f32`: float32 dequantized outputs from one additional **untimed** inference using the plain context. Reshape according to the corresponding output's model layout and dimensions in JSON; use these for PyTorch/ONNX comparisons.

`PERF_RUN` is documented as real inference time. If fallback exists, do not label its total as exclusively NPU time. Keep the raw per-operator report as evidence and separately identify device-specific costs. An input layout conversion or RKNN runtime overhead should also not automatically be called a model CPU fallback.

This tool isolates native model execution. Full tracking FPS must be measured separately with actual completed frame counts, preprocessing, state update, decoding, drawing, and output encoding included or excluded explicitly. Record model SHA256, NPU frequency/governor, temperature, core mask, and whether another process was using the NPU. Do not derive FPS from a per-frame `Infer time` label printed by other demos.

## Reproduce the final four-model comparison

Run only when other NPU work is idle. The wrapper uses directory/binary arguments and records environment and hashes without changing clocks or stopping processes:

```sh
bash run_native_benchmarks.sh ./models ./validation ./native_bench/bench_native
```

It requires `template_fp16.rknn`, `search_fp16.rknn`, `template_int8.rknn`, `search_int8.rknn`, and the three raw input files shown above in MODEL_DIR. It also accepts development names containing `_npu_` when the final model names are absent. It runs sequentially, with 100 warmups and 500 measurements in each plain/profiling context, writing `final_native_template_fp16`, `final_native_search_fp16`, `final_native_template_int8`, and `final_native_search_int8` prefixes. The inputs are random smoke tensors for repeatable model speed measurement, not real-crop tracking accuracy evidence.

The final search model outputs bbox **logits**: apply `exp` to its first output for comparison to the original model's positive bbox distances. Keep that exact CPU decoder outside the RKNN model. A report with no CPU compute operators permits describing PERF_RUN as model NPU inference time; still list InputOperator/OutputOperator boundary rows separately. Model normalization is now encoded in RKNN configuration; supply raw RGB float32 pixels according to the export contract, not pre-normalized pixels.

Note: the INT8 models were an experiment and are not shipped in this repository, so `run_native_benchmarks.sh` (which expects all four models) needs INT8 models you convert yourself; for the shipped FP16 pair, run the two `bench_native` commands above directly.
