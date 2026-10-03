# Drone detector: YOLOv8n INT8 (RKNN)

| | |
|---|---|
| File | `drone_yolov8n_int8.rknn` (3.9 MB) |
| Architecture | YOLOv8n, single class (`drone`), anchor-free head with DFL (reg_max = 16); the compiled graph uses ReLU activations |
| Target | RK3588 NPU, RKNN Toolkit2 / runtime 2.3.2 |
| Precision | INT8 (asymmetric affine quantisation) |
| Input | `1×640×640×3` NHWC, RGB, uint8 0–255 (letterbox, fill value 114); the /255 scaling is folded into the model's input quantisation |
| Outputs | 9 tensors, 3 per stride (8, 16, 32) |
| Post-processing | CPU: DFL decode, confidence filter (default 0.25), NMS (IoU 0.45) in [`tracker/yolo_post.py`](../../tracker/yolo_post.py) |

## Output layout

For each stride `s ∈ {8, 16, 32}` (grid `640/s` = 80, 40, 20) the model returns, in this order:

| Index | Shape (NCHW) | Meaning |
|---|---|---|
| `3k + 0` | `1×64×G×G` | box distance logits: 4 sides × 16 DFL bins |
| `3k + 1` | `1×1×G×G` | class score (sigmoid already applied in the graph) |
| `3k + 2` | `1×1×G×G` | score sum (ReduceSum output); unused by the decoder |

(`k` = 0, 1, 2 for strides 8, 16, 32.) Outputs are INT8 in the RKNN runtime and are dequantised by the host wrapper (`tracker/native_infer.cpp`). The runtime may report the tensors as NCHW or NHWC; `tracker/infer.py` handles both.

Output layout verified on the RK3588 runtime (9 outputs, shapes as above).

## Provenance

- Training data: visible-light (RGB) drone images with emphasis on small and distant targets; single class `drone`.
- Strong on drones a few to a few dozen pixels across. Weak on large, close-up drones filling a big part of the frame (they are under-represented in the training set); the combined mode leans on FEAR there.
- Trained with Ultralytics YOLOv8; released under the repository license (AGPL-3.0).
- Integrity: `shasum -a 256 -c SHA256SUMS`.
