# FEAR-XS (RKNN, FP16)

Two RKNN graphs for RK3588 (`target_platform=rk3588`, RKNN Toolkit2 2.3.2, runtime 2.3.2), converted from the official FEAR-XS-NoEmbs checkpoint of [PinataFarms/FEARTracker](https://github.com/PinataFarms/FEARTracker) (MIT). Weights are unchanged; the only graph modifications are listed in [docs/FEAR_RKNN_CHANGES.md](../../docs/FEAR_RKNN_CHANGES.md).

| File | Role | Runs | Size |
|---|---|---|---:|
| `template_fp16.rknn` | Encodes the target template crop into a feature map | once per (re)initialisation | 1.95 MB |
| `search_fp16.rknn` | Correlates a search crop with the template feature; produces box regression and classification maps | every frame | 3.43 MB |

## Inputs and outputs

| Graph | Tensor | Shape | Type / range | Notes |
|---|---|---|---|---|
| template | `template` (in) | `1×3×128×128` (RKNN reports NHWC `1×128×128×3`) | RGB, 0–255 | crop = bbox extended by 0.2 on each side, resized to 128×128 |
| template | `template_features` (out) | `1×256×8×8` | float | cache it; feed it to every `search` call |
| search | `search` (in) | `1×3×256×256` (NHWC `1×256×256×3`) | RGB, 0–255 | crop = predicted bbox extended by 2.0 on each side (5× the box), resized to 256×256 |
| search | `template_features` (in) | `1×256×8×8` (NHWC `1×8×8×256`) | float | **not** normalised |
| search | bbox (out) | `1×4×16×16` | float logits | **apply `exp` on the CPU** to get the l/t/r/b distances (see change #1) |
| search | `cls` (out) | `1×1×16×16` | float logits | apply sigmoid in post-processing |

All shapes are static.

## Pre-processing configured inside the models

ImageNet normalisation is part of the RKNN input configuration, so the host sends raw 0–255 RGB:

| Input | mean | std |
|---|---|---|
| `template` / `search` image | `[123.675, 116.28, 103.53]` | `[58.395, 57.12, 57.375]` |
| search `template_features` | 0 (×256 channels) | 1 (×256 channels) |

Decoding (grid, window/penalty smoothing, clamping) follows the upstream tracker and is implemented in [`tracker/tracker_core.py`](../../tracker/tracker_core.py).

## Measured on RK3588 (FP16, RKNN 2.3.2, NPU at 1 GHz)

NPU time (`RKNN_QUERY_PERF_RUN`, 500 runs after 100 warm-up): template 0.64 ms, search 2.45 ms. All 47 / 76 compute operators run on the NPU. Details and accuracy in [docs/FEAR_RKNN_RESULTS.md](../../docs/FEAR_RKNN_RESULTS.md).

## Provenance and license

- Source: PinataFarms/FEARTracker @ `0a3bd039918909c79c1b7e55a4bfb7807520abde`, checkpoint `FEAR-XS-NoEmbs.ckpt` (SHA256 `8efe7dfd3498e385f332fd655f360a848d723f5fd77c2d433b2c084029616be5`).
- Converted from the official FEAR-XS checkpoint (FEARTracker, MIT, see `fear_to_rknn/upstream/FEARTracker.LICENSE`). Released here under the repository license (AGPL-3.0).
- Integrity: `shasum -a 256 -c SHA256SUMS`.
