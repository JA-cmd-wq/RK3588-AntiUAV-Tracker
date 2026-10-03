/* RKNN 2.x synchronous native benchmark. No dependencies beyond librknnrt/libm.
 * Inputs: contiguous little-endian NCHW float32, in model input index order.
 * Separate unprofiled and COLLECT_PERF contexts: profiling wall is diagnostic.
 */
#define _POSIX_C_SOURCE 200809L
#include "rknn_api.h"
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_INPUTS 16

typedef struct {
    const char *model, *input[MAX_INPUTS], *prefix, *dump_prefix;
    int n_inputs, warmup, iterations;
    int nhwc_inputs;
    rknn_core_mask core;
    const char *core_name;
} options;

typedef struct {
    double input_ms, run_get_ms, release_ms, wall_ms, query_run_ms;
    int query_ret;
} sample;

typedef struct {
    uint32_t flags;
    rknn_sdk_version version;
    rknn_input_output_num io;
    rknn_tensor_attr *input_attrs, *output_attrs;
    sample *samples;
    char *detail;
    size_t detail_len;
    int detail_ret, valid_queries;
    double elapsed_ms;
} result;

static double now_ms(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t) != 0) {
        perror("clock_gettime"); exit(1);
    }
    return (double)t.tv_sec * 1000.0 + (double)t.tv_nsec / 1000000.0;
}

static void fail_api(const char *api, int ret) {
    fprintf(stderr, "%s failed: %d\n", api, ret); exit(1);
}

static void *checked_calloc(size_t n, size_t bytes) {
    void *p = calloc(n, bytes);
    if (!p) { perror("calloc"); exit(1); }
    return p;
}

static void *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    if (fseek(f, 0, SEEK_END) != 0) { perror(path); exit(1); }
    long len = ftell(f);
    if (len <= 0 || (uint64_t)len > UINT32_MAX) {
        fprintf(stderr, "Invalid/oversized file: %s (%ld bytes)\n", path, len);
        exit(1);
    }
    rewind(f);
    void *data = checked_calloc((size_t)len, 1);
    if (fread(data, 1, (size_t)len, f) != (size_t)len) {
        fprintf(stderr, "Short read: %s\n", path); exit(1);
    }
    fclose(f); *size = (size_t)len;
    return data;
}

static char *filename(const char *prefix, const char *suffix) {
    size_t n = strlen(prefix) + strlen(suffix) + 1;
    char *p = checked_calloc(n, 1);
    snprintf(p, n, "%s%s", prefix, suffix);
    return p;
}

static FILE *open_output(const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); exit(1); }
    return f;
}

static int positive_number(const char *s, int allow_zero) {
    char *end;
    errno = 0;
    long n = strtol(s, &end, 10);
    if (errno || !*s || *end || n < (allow_zero ? 0 : 1) || n > 1000000) {
        fprintf(stderr, "Invalid iteration count: %s\n", s); exit(2);
    }
    return (int)n;
}

static void usage(const char *name) {
    fprintf(stderr,
        "Usage: %s MODEL.rknn INPUT0.f32 [INPUT1.f32 ...] [options]\n"
        "  --prefix PATH      Output PATH.json/.csv/.perf.txt (default bench)\n"
        "  --warmup N         Warmup per context (default 100)\n"
        "  --iterations N     Measured repeats per context (default 500)\n"
        "  --core MASK        auto, 0, 1, 2, 0,1, or 0,1,2 (default auto)\n"
        "  --input-layout L   nchw (default) or nhwc; raw files remain NCHW\n"
        "  --dump-prefix PATH Untimed float32 outputs PATH.output_N.f32\n"
        "Input files must match each input n_elems * 4 bytes exactly.\n", name);
}

static options parse(int argc, char **argv) {
    options o = {0};
    o.prefix = "bench"; o.warmup = 100; o.iterations = 500;
    o.core = RKNN_NPU_CORE_AUTO; o.core_name = "auto";
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (!strcmp(a, "--help")) { usage(argv[0]); exit(0); }
        if (!strncmp(a, "--", 2)) {
            if (i + 1 == argc) { usage(argv[0]); exit(2); }
            const char *value = argv[++i];
            if (!strcmp(a, "--prefix")) o.prefix = value;
            else if (!strcmp(a, "--dump-prefix")) o.dump_prefix = value;
            else if (!strcmp(a, "--warmup")) o.warmup = positive_number(value, 1);
            else if (!strcmp(a, "--iterations")) o.iterations = positive_number(value, 0);
            else if (!strcmp(a, "--input-layout")) {
                if (!strcmp(value, "nchw")) o.nhwc_inputs = 0;
                else if (!strcmp(value, "nhwc")) o.nhwc_inputs = 1;
                else { fprintf(stderr, "Invalid input layout: %s\n", value); exit(2); }
            }
            else if (!strcmp(a, "--core")) {
                o.core_name = value;
                if (!strcmp(value, "auto")) o.core = RKNN_NPU_CORE_AUTO;
                else if (!strcmp(value, "0")) o.core = RKNN_NPU_CORE_0;
                else if (!strcmp(value, "1")) o.core = RKNN_NPU_CORE_1;
                else if (!strcmp(value, "2")) o.core = RKNN_NPU_CORE_2;
                else if (!strcmp(value, "0,1")) o.core = RKNN_NPU_CORE_0_1;
                else if (!strcmp(value, "0,1,2")) o.core = RKNN_NPU_CORE_0_1_2;
                else { fprintf(stderr, "Invalid core mask: %s\n", value); exit(2); }
            } else { fprintf(stderr, "Unknown option: %s\n", a); exit(2); }
        } else if (!o.model) o.model = a;
        else if (o.n_inputs < MAX_INPUTS) o.input[o.n_inputs++] = a;
        else { fprintf(stderr, "Too many inputs\n"); exit(2); }
    }
    if (!o.model || !o.n_inputs) { usage(argv[0]); exit(2); }
    return o;
}

/* PERF_DETAIL is runtime-owned. Copy while outputs are live; do not free it. */
static void collect_detail(rknn_context ctx, result *r) {
    rknn_perf_detail p = {0};
    r->detail_ret = rknn_query(ctx, RKNN_QUERY_PERF_DETAIL, &p, sizeof(p));
    if (r->detail_ret || !p.perf_data || !p.data_len) return;
    if (p.data_len > 16 * 1024 * 1024) {
        fprintf(stderr, "Unreasonable PERF_DETAIL length\n"); exit(1);
    }
    r->detail_len = (size_t)p.data_len;
    r->detail = checked_calloc(r->detail_len + 1, 1);
    memcpy(r->detail, p.perf_data, r->detail_len);
    while (r->detail_len && r->detail[r->detail_len - 1] == '\0') --r->detail_len;
}

static sample infer(rknn_context ctx, rknn_input *inputs, rknn_input_output_num io,
                    rknn_output *outputs, int measure, result *detail_result) {
    sample s = {0}; s.query_ret = -1; s.query_run_ms = NAN;
    double t0 = now_ms();
    int ret = rknn_inputs_set(ctx, io.n_input, inputs);
    if (ret) fail_api("rknn_inputs_set", ret);
    double t1 = now_ms();
    ret = rknn_run(ctx, NULL);
    if (ret) fail_api("rknn_run", ret);
    ret = rknn_outputs_get(ctx, io.n_output, outputs, NULL);
    if (ret) fail_api("rknn_outputs_get", ret);
    double t2 = now_ms();
    /* Both queries are explicitly documented as valid AFTER outputs_get. */
    if (measure) {
        rknn_perf_run p = {0};
        s.query_ret = rknn_query(ctx, RKNN_QUERY_PERF_RUN, &p, sizeof(p));
        if (!s.query_ret && p.run_duration > 0)
            s.query_run_ms = (double)p.run_duration / 1000.0;
    }
    if (detail_result) collect_detail(ctx, detail_result);
    double t3 = now_ms();
    ret = rknn_outputs_release(ctx, io.n_output, outputs);
    if (ret) fail_api("rknn_outputs_release", ret);
    double t4 = now_ms();
    s.input_ms = t1 - t0; s.run_get_ms = t2 - t1;
    s.release_ms = t4 - t3;
    /* Deliberately exclude PERF_RUN/PERF_DETAIL querying from API wall. */
    s.wall_ms = s.input_ms + s.run_get_ms + s.release_ms;
    return s;
}

static void dump_outputs(rknn_context ctx, const options *o,
                         rknn_input *inputs, const result *r) {
    if (!o->dump_prefix) return;
    rknn_output *outputs = checked_calloc(r->io.n_output, sizeof(*outputs));
    for (uint32_t i = 0; i < r->io.n_output; ++i) {
        outputs[i].index = i; outputs[i].want_float = 1;
    }
    int ret = rknn_inputs_set(ctx, r->io.n_input, inputs);
    if (ret) fail_api("dump inputs_set", ret);
    ret = rknn_run(ctx, NULL);
    if (ret) fail_api("dump run", ret);
    ret = rknn_outputs_get(ctx, r->io.n_output, outputs, NULL);
    if (ret) fail_api("dump outputs_get", ret);
    for (uint32_t i = 0; i < r->io.n_output; ++i) {
        char suffix[64]; snprintf(suffix, sizeof(suffix), ".output_%u.f32", i);
        char *path = filename(o->dump_prefix, suffix);
        FILE *f = open_output(path);
        size_t expected = (size_t)r->output_attrs[i].n_elems * sizeof(float);
        if (outputs[i].size != expected) {
            fprintf(stderr, "Float output %u has %u bytes; expected %zu\n",
                    i, outputs[i].size, expected); exit(1);
        }
        if (fwrite(outputs[i].buf, 1, expected, f) != expected) {
            perror(path); exit(1);
        }
        fclose(f); free(path);
    }
    ret = rknn_outputs_release(ctx, r->io.n_output, outputs);
    if (ret) fail_api("dump outputs_release", ret);
    free(outputs);
}

static void transpose_input_to_nhwc(rknn_input *input, const rknn_tensor_attr *attr) {
    if (attr->n_dims != 4 || (attr->fmt != RKNN_TENSOR_NHWC && attr->fmt != RKNN_TENSOR_NCHW)) {
        fprintf(stderr, "--input-layout nhwc requires 4D NHWC/NCHW model inputs\n"); exit(1);
    }
    size_t n = attr->dims[0];
    size_t c = attr->dims[attr->fmt == RKNN_TENSOR_NHWC ? 3 : 1];
    size_t h = attr->dims[attr->fmt == RKNN_TENSOR_NHWC ? 1 : 2];
    size_t w = attr->dims[attr->fmt == RKNN_TENSOR_NHWC ? 2 : 3];
    float *src = input->buf;
    float *dst = checked_calloc(attr->n_elems, sizeof(*dst));
    for (size_t b = 0; b < n; ++b)
        for (size_t y = 0; y < h; ++y)
            for (size_t x = 0; x < w; ++x)
                for (size_t ch = 0; ch < c; ++ch)
                    dst[((b * h + y) * w + x) * c + ch] = src[((b * c + ch) * h + y) * w + x];
    free(input->buf); input->buf = dst; input->fmt = RKNN_TENSOR_NHWC;
}

static result benchmark(const options *o, uint32_t flags, void *model, size_t model_size) {
    result r = {0}; r.flags = flags; r.detail_ret = -1;
    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, model, (uint32_t)model_size, flags, NULL);
    if (ret) fail_api("rknn_init", ret);
    ret = rknn_set_core_mask(ctx, o->core);
    if (ret) fail_api("rknn_set_core_mask", ret);
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &r.version, sizeof(r.version));
    if (ret) fail_api("SDK_VERSION", ret);
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &r.io, sizeof(r.io));
    if (ret) fail_api("IN_OUT_NUM", ret);
    if (r.io.n_input != (uint32_t)o->n_inputs || !r.io.n_output) {
        fprintf(stderr, "Model expects %u inputs/%u outputs; got %d files\n",
                r.io.n_input, r.io.n_output, o->n_inputs); exit(1);
    }
    r.input_attrs = checked_calloc(r.io.n_input, sizeof(*r.input_attrs));
    r.output_attrs = checked_calloc(r.io.n_output, sizeof(*r.output_attrs));
    rknn_input *inputs = checked_calloc(r.io.n_input, sizeof(*inputs));
    rknn_output *outputs = checked_calloc(r.io.n_output, sizeof(*outputs));
    for (uint32_t i = 0; i < r.io.n_input; ++i) {
        r.input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &r.input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret) fail_api("INPUT_ATTR", ret);
        fprintf(stderr, "Input %u: name=%.*s model_fmt=%d model_type=%d dims=[",
                i, RKNN_MAX_NAME_LEN, r.input_attrs[i].name,
                r.input_attrs[i].fmt, r.input_attrs[i].type);
        for (uint32_t d = 0; d < r.input_attrs[i].n_dims; ++d)
            fprintf(stderr, "%s%u", d ? "," : "", r.input_attrs[i].dims[d]);
        fprintf(stderr, "] n_elems=%u raw=NCHW supplied=%s/FLOAT32/pass_through=0\n",
                r.input_attrs[i].n_elems, o->nhwc_inputs ? "NHWC" : "NCHW");
        size_t bytes;
        inputs[i].buf = read_file(o->input[i], &bytes);
        if (bytes != (size_t)r.input_attrs[i].n_elems * sizeof(float)) {
            fprintf(stderr, "Input %u size mismatch: %zu bytes != %u * 4\n",
                    i, bytes, r.input_attrs[i].n_elems); exit(1);
        }
        inputs[i].index = i; inputs[i].size = (uint32_t)bytes;
        inputs[i].type = RKNN_TENSOR_FLOAT32; inputs[i].fmt = RKNN_TENSOR_NCHW;
        inputs[i].pass_through = 0;
        if (o->nhwc_inputs) transpose_input_to_nhwc(&inputs[i], &r.input_attrs[i]);
    }
    for (uint32_t i = 0; i < r.io.n_output; ++i) {
        r.output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &r.output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret) fail_api("OUTPUT_ATTR", ret);
        outputs[i].index = i; outputs[i].want_float = 0;
    }
    fprintf(stderr, "Pass %s: warmup=%d repeats=%d core=%s\n",
            flags ? "collect_perf" : "plain", o->warmup, o->iterations, o->core_name);
    for (int i = 0; i < o->warmup; ++i)
        infer(ctx, inputs, r.io, outputs, 0, NULL);
    r.samples = checked_calloc((size_t)o->iterations, sizeof(*r.samples));
    double begin = now_ms();
    for (int i = 0; i < o->iterations; ++i) {
        result *detail_target = flags && i == o->iterations - 1 ? &r : NULL;
        r.samples[i] = infer(ctx, inputs, r.io, outputs, 1, detail_target);
        if (isfinite(r.samples[i].query_run_ms)) ++r.valid_queries;
    }
    r.elapsed_ms = now_ms() - begin;
    if (!flags) dump_outputs(ctx, o, inputs, &r); /* outside all timed samples */
    for (uint32_t i = 0; i < r.io.n_input; ++i) free(inputs[i].buf);
    free(inputs); free(outputs);
    ret = rknn_destroy(ctx);
    if (ret) fail_api("rknn_destroy", ret);
    return r;
}

static void json_string(FILE *f, const char *s, size_t n) {
    fputc('"', f);
    for (size_t i = 0; i < n; ++i) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') { fputc('\\', f); fputc(c, f); }
        else if (c < 32) fprintf(f, "\\u%04x", c);
        else fputc(c, f);
    }
    fputc('"', f);
}

static void str(FILE *f, const char *s) { json_string(f, s, strlen(s)); }
static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double value(const sample *s, int field) {
    switch (field) {
        case 0: return s->wall_ms;
        case 1: return s->query_run_ms;
        case 2: return s->input_ms;
        case 3: return s->run_get_ms;
        default: return s->release_ms;
    }
}

static double percentile(const double *v, int n, double p) {
    double index = (n - 1) * p;
    int lo = (int)index, hi = lo + 1 < n ? lo + 1 : lo;
    return v[lo] + (v[hi] - v[lo]) * (index - lo);
}

static void stats(FILE *f, const result *r, int repeats, int field) {
    double *v = checked_calloc((size_t)repeats, sizeof(*v));
    double sum = 0.0;
    int n = 0;
    for (int i = 0; i < repeats; ++i) {
        double x = value(&r->samples[i], field);
        if (isfinite(x)) { v[n++] = x; sum += x; }
    }
    if (!n) { fputs("null", f); free(v); return; }
    qsort(v, (size_t)n, sizeof(*v), cmp_double);
    double mean = sum / n, variance = 0.0;
    for (int i = 0; i < n; ++i) variance += (v[i] - mean) * (v[i] - mean);
    fprintf(f, "{\"count\":%d,\"mean\":%.6f,\"median\":%.6f,\"p95\":%.6f,"
               "\"min\":%.6f,\"max\":%.6f,\"stddev\":%.6f}",
            n, mean, percentile(v, n, 0.5), percentile(v, n, 0.95),
            v[0], v[n - 1], sqrt(variance / n));
    free(v);
}

static void attrs(FILE *f, const rknn_tensor_attr *a, uint32_t n) {
    fputc('[', f);
    for (uint32_t i = 0; i < n; ++i) {
        if (i) fputc(',', f);
        fprintf(f, "{\"index\":%u,\"name\":", a[i].index);
        json_string(f, a[i].name, strnlen(a[i].name, sizeof(a[i].name)));
        fprintf(f, ",\"dims\":[");
        for (uint32_t d = 0; d < a[i].n_dims; ++d)
            fprintf(f, "%s%u", d ? "," : "", a[i].dims[d]);
        fprintf(f, "],\"n_elems\":%u,\"model_fmt\":%d,\"model_type\":%d,"
                   "\"qnt_type\":%d,\"zero_point\":%d,\"scale\":%.9g}",
                a[i].n_elems, a[i].fmt, a[i].type, a[i].qnt_type, a[i].zp, a[i].scale);
    }
    fputc(']', f);
}

static int has_word(const char *s, size_t n, const char *word) {
    size_t w = strlen(word);
    for (size_t i = 0; i + w <= n; ++i) {
        if ((i && (isalnum((unsigned char)s[i - 1]) || s[i - 1] == '_')) ||
            (i + w < n && (isalnum((unsigned char)s[i + w]) || s[i + w] == '_'))) continue;
        size_t j = 0;
        while (j < w && toupper((unsigned char)s[i + j]) == word[j]) ++j;
        if (j == w) return 1;
    }
    return 0;
}

/* Keep the full row for audit: table format can vary across runtime versions. */
static int device_rows(FILE *f, const result *r, const char *word) {
    int count = 0;
    fputc('[', f);
    if (r->detail) {
        size_t start = 0;
        while (start < r->detail_len) {
            size_t end = start;
            while (end < r->detail_len && r->detail[end] != '\n') ++end;
            if (has_word(r->detail + start, end - start, word)) {
                if (count++) fputc(',', f);
                json_string(f, r->detail + start, end - start);
            }
            start = end + 1;
        }
    }
    fputc(']', f);
    return count;
}

/* Parse only actual layer rows, never the summary/header's NPU/CPU labels. */
static int operator_rows(FILE *f, const result *r, const char *target, int compute_only,
                         int64_t *time_us) {
    int count = 0;
    *time_us = 0;
    fputc('[', f);
    if (r->detail) {
        size_t start = 0;
        while (start < r->detail_len) {
            size_t end = start;
            while (end < r->detail_len && r->detail[end] != '\n') ++end;
            char *line = checked_calloc(end - start + 1, 1);
            memcpy(line, r->detail + start, end - start);
            unsigned id;
            char op[128], dtype[64], device[16], in_shape[512], out_shape[512], cycles[128];
            int64_t duration;
            int fields = sscanf(line, "%u %127s %63s %15s %511s %511s %127s %" SCNd64,
                                &id, op, dtype, device, in_shape, out_shape, cycles, &duration);
            if (fields == 8 && !strcmp(device, target) &&
                (!compute_only || (strcmp(op, "InputOperator") && strcmp(op, "OutputOperator")))) {
                if (count++) fputc(',', f);
                json_string(f, line, end - start);
                *time_us += duration;
            }
            free(line); start = end + 1;
        }
    }
    fputc(']', f);
    return count;
}

static void pass_json(FILE *f, const result *r, const options *o) {
    fprintf(f, "{\"flags\":%u,\"profiling_enabled\":%s,\"api_version\":",
            r->flags, r->flags ? "true" : "false");
    str(f, r->version.api_version);
    fputs(",\"driver_version\":", f); str(f, r->version.drv_version);
    fprintf(f, ",\"measured_repeats\":%d,\"perf_run_valid_queries\":%d,"
               "\"elapsed_with_queries_ms\":%.6f,\"wall_ms\":",
            o->iterations, r->valid_queries, r->elapsed_ms);
    stats(f, r, o->iterations, 0);
    fputs(",\"perf_run_ms\":", f); stats(f, r, o->iterations, 1);
    fputs(",\"input_set_ms\":", f); stats(f, r, o->iterations, 2);
    fputs(",\"run_and_get_ms\":", f); stats(f, r, o->iterations, 3);
    fputs(",\"output_release_ms\":", f); stats(f, r, o->iterations, 4);
    fputc('}', f);
}

static void write_results(const options *o, const result *plain, const result *profile) {
    char *json_path = filename(o->prefix, ".json");
    char *csv_path = filename(o->prefix, ".csv");
    char *detail_path = filename(o->prefix, ".perf.txt");
    FILE *f = open_output(json_path);
    fputs("{\"schema_version\":1,\"model\":", f); str(f, o->model);
    fprintf(f, ",\"warmup_per_pass\":%d,\"core_mask\":%u,\"core_name\":",
            o->warmup, (unsigned)o->core);
    str(f, o->core_name);
    fprintf(f, ",\"raw_input_contract\":\"contiguous NCHW float32\","
          "\"runtime_input_layout\":\"%s\",\"input_pass_through\":0,"
          "\"input_transpose_timed\":false,"
          "\"wall_scope\":\"inputs_set + run + outputs_get(native) + outputs_release; excludes queries\","
          "\"perf_run_scope\":\"RKNN_QUERY_PERF_RUN real inference time; NPU-only only if device report confirms it\","
          "\"profiling_wall_is_final_fps\":false,\"inputs\":", o->nhwc_inputs ? "NHWC" : "NCHW");
    attrs(f, plain->input_attrs, plain->io.n_input);
    fputs(",\"outputs\":", f); attrs(f, plain->output_attrs, plain->io.n_output);
    fputs(",\"plain\":", f); pass_json(f, plain, o);
    fputs(",\"profile\":", f); pass_json(f, profile, o);
    fprintf(f, ",\"perf_detail_query_ret\":%d,\"perf_detail_file\":", profile->detail_ret);
    str(f, detail_path);
    fputs(",\"cpu_device_token_lines\":", f); int cpu = device_rows(f, profile, "CPU");
    fputs(",\"gpu_device_token_lines\":", f); int gpu = device_rows(f, profile, "GPU");
    fputs(",\"npu_device_token_lines\":", f); int npu = device_rows(f, profile, "NPU");
    int64_t cpu_us, gpu_us, npu_us, cpu_compute_us;
    fputs(",\"cpu_operator_rows\":", f); int cpu_ops = operator_rows(f, profile, "CPU", 0, &cpu_us);
    fputs(",\"gpu_operator_rows\":", f); int gpu_ops = operator_rows(f, profile, "GPU", 0, &gpu_us);
    fputs(",\"npu_operator_rows\":", f); int npu_ops = operator_rows(f, profile, "NPU", 0, &npu_us);
    fputs(",\"cpu_compute_operator_rows\":", f);
    int cpu_compute_ops = operator_rows(f, profile, "CPU", 1, &cpu_compute_us);
    fprintf(f, ",\"profile_operator_counts\":{\"CPU\":%d,\"GPU\":%d,\"NPU\":%d,\"CPU_compute\":%d},"
               "\"profile_operator_time_us\":{\"CPU\":%" PRId64 ",\"GPU\":%" PRId64
               ",\"NPU\":%" PRId64 ",\"CPU_compute\":%" PRId64 "},",
            cpu_ops, gpu_ops, npu_ops, cpu_compute_ops, cpu_us, gpu_us, npu_us, cpu_compute_us);
    fprintf(f, "\"device_token_line_counts\":{\"CPU\":%d,\"GPU\":%d,\"NPU\":%d},"
               "\"fallback_scan\":", cpu, gpu, npu);
    str(f, !profile->detail ? "unavailable" : cpu_compute_ops || gpu_ops ? "cpu_or_gpu_compute_operator_detected" :
           npu_ops ? "all_reported_compute_operators_on_npu" : "device_assignment_not_reported");
    fputs("}\n", f); if (fclose(f)) { perror(json_path); exit(1); }

    f = open_output(csv_path);
    fputs("pass,iteration,input_set_ms,run_and_get_ms,output_release_ms,wall_ms,perf_run_ms,perf_query_ret\n", f);
    for (int pass = 0; pass < 2; ++pass) {
        const result *r = pass ? profile : plain;
        for (int i = 0; i < o->iterations; ++i) {
            const sample *s = &r->samples[i];
            fprintf(f, "%s,%d,%.6f,%.6f,%.6f,%.6f,", pass ? "profile" : "plain", i,
                    s->input_ms, s->run_get_ms, s->release_ms, s->wall_ms);
            if (isfinite(s->query_run_ms)) fprintf(f, "%.6f", s->query_run_ms);
            fprintf(f, ",%d\n", s->query_ret);
        }
    }
    if (fclose(f)) { perror(csv_path); exit(1); }
    f = open_output(detail_path);
    if (profile->detail) {
        if (fwrite(profile->detail, 1, profile->detail_len, f) != profile->detail_len) {
            perror(detail_path); exit(1);
        }
    } else fprintf(f, "PERF_DETAIL unavailable: query returned %d\n", profile->detail_ret);
    if (fclose(f)) { perror(detail_path); exit(1); }
    fprintf(stderr, "Saved %s, %s, %s\nCPU/GPU token lines=%d/%d (inspect raw rows)\n",
            json_path, csv_path, detail_path, cpu, gpu);
    free(json_path); free(csv_path); free(detail_path);
}

static void free_result(result *r) {
    free(r->input_attrs); free(r->output_attrs); free(r->samples); free(r->detail);
}

int main(int argc, char **argv) {
    options o = parse(argc, argv);
    const uint16_t endian = 1;
    if (*(const uint8_t *)&endian != 1 || sizeof(float) != 4) {
        fprintf(stderr, "Requires little-endian IEEE float32 host\n"); return 1;
    }
    size_t model_size;
    void *model = read_file(o.model, &model_size);
    result plain = benchmark(&o, 0, model, model_size);
    result profile = benchmark(&o, RKNN_FLAG_COLLECT_PERF_MASK, model, model_size);
    write_results(&o, &plain, &profile);
    free_result(&plain); free_result(&profile); free(model);
    return 0;
}
