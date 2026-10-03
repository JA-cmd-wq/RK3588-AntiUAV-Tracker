#!/usr/bin/env bash
# Sequential FEAR model timing; run only when other NPU work is idle.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    printf 'Usage: %s MODEL_DIR RESULTS_DIR BENCH_BINARY\n' "$0" >&2
    printf 'Inputs must be MODEL_DIR/template_template.bin, search_search.bin, search_template_features.bin (NCHW float32).\n' >&2
    exit 2
fi

model_dir=$1
results_dir=$2
bench_binary=$3

if [[ ! -x "$bench_binary" ]]; then
    printf 'Benchmark binary is missing or not executable: %s\n' "$bench_binary" >&2
    exit 2
fi
model_file() {
    local branch=$1 dtype=$2
    if [[ -s "$model_dir/${branch}_${dtype}.rknn" ]]; then
        printf '%s/%s_%s.rknn\n' "$model_dir" "$branch" "$dtype"
    elif [[ -s "$model_dir/${branch}_npu_${dtype}.rknn" ]]; then
        printf '%s/%s_npu_%s.rknn\n' "$model_dir" "$branch" "$dtype"
    else
        printf 'Required model is missing: %s/%s_%s.rknn\n' "$model_dir" "$branch" "$dtype" >&2
        return 2
    fi
}
template_fp16=$(model_file template fp16)
search_fp16=$(model_file search fp16)
template_int8=$(model_file template int8)
search_int8=$(model_file search int8)
for file in template_template.bin search_search.bin search_template_features.bin; do
    if [[ ! -s "$model_dir/$file" ]]; then
        printf 'Required model/input is missing or empty: %s\n' "$model_dir/$file" >&2
        exit 2
    fi
done
mkdir -p "$results_dir"

capture_environment() {
    date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
    uname -a
    printf 'Load average: '; cat /proc/loadavg 2>/dev/null || true
    for npu_path in /sys/class/devfreq/*npu*; do
        [[ -d "$npu_path" ]] || continue
        for field in governor cur_freq available_frequencies; do
            printf '%s/%s: ' "$npu_path" "$field"
            cat "$npu_path/$field" 2>/dev/null || true
        done
    done
    for thermal_path in /sys/class/thermal/thermal_zone*; do
        [[ -d "$thermal_path" ]] || continue
        printf '%s type=' "$thermal_path"
        cat "$thermal_path/type" 2>/dev/null || true
        printf '%s temp_millidegrees=' "$thermal_path"
        cat "$thermal_path/temp" 2>/dev/null || true
    done
    for debug_file in /sys/kernel/debug/rknpu/load /sys/kernel/debug/rknpu/version; do
        if [[ -r "$debug_file" ]]; then
            cat "$debug_file"
        elif command -v sudo >/dev/null 2>&1 && sudo -n test -r "$debug_file" 2>/dev/null; then
            sudo -n cat "$debug_file"
        fi
    done
    # Read-only owner inventory; never stop processes or change the governor.
    if command -v fuser >/dev/null 2>&1; then
        fuser -v /dev/dri/by-path/platform-fdab0000.npu-render 2>&1 || true
    fi
    ps -eo pid,comm,pcpu,pmem 2>/dev/null || true
}

capture_environment > "$results_dir/final_native_environment_before.txt" 2>&1
sha256sum "$template_fp16" "$search_fp16" "$template_int8" "$search_int8" \
    "$model_dir"/template_template.bin "$model_dir"/search_search.bin \
    "$model_dir"/search_template_features.bin > "$results_dir/final_native_sha256.txt"

for dtype in fp16 int8; do
    prefix="$results_dir/final_native_template_$dtype"
    printf 'Running template %s (100 warmup, 500 repeats per context)\n' "$dtype"
    "$bench_binary" "$(model_file template "$dtype")" "$model_dir/template_template.bin" \
        --prefix "$prefix" --dump-prefix "$prefix" \
        --input-layout nhwc --core 0,1,2 --warmup 100 --iterations 500 \
        > "$prefix.log" 2>&1
    capture_environment > "$prefix.environment.txt" 2>&1

    prefix="$results_dir/final_native_search_$dtype"
    printf 'Running search %s (100 warmup, 500 repeats per context)\n' "$dtype"
    "$bench_binary" "$(model_file search "$dtype")" "$model_dir/search_search.bin" \
        "$model_dir/search_template_features.bin" \
        --prefix "$prefix" --dump-prefix "$prefix" \
        --input-layout nhwc --core 0,1,2 --warmup 100 --iterations 500 \
        > "$prefix.log" 2>&1
    capture_environment > "$prefix.environment.txt" 2>&1
done
capture_environment > "$results_dir/final_native_environment_after.txt" 2>&1

printf 'Saved native JSON/CSV/per-operator reports and untimed outputs in %s\n' "$results_dir"
printf 'Check cpu_compute_operator_rows before calling PERF_RUN exclusively NPU time.\n'
printf 'InputOperator/OutputOperator CPU rows are boundary handling, recorded separately.\n'
printf 'Do not use profile wall to report final tracking FPS.\n'
