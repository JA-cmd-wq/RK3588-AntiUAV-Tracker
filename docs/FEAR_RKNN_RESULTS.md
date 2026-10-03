# FEAR-XS → RKNN：RK3588 实测记录

日期：2026-10-02。在 RK3588 上完成双分支转换、板端连续跟踪与测量。仓库发布的是 FP16 模型；INT8 仅作为实验版本测过，精度损失明显，不随仓库发布，也没有继续优化、训练或微调。改动清单见 [FEAR_RKNN_CHANGES.md](FEAR_RKNN_CHANGES.md)。

## 模型边界

- 模板：`template_fp16.rknn`，RGB `[1,3,128,128]` → `[1,256,8,8]`，初始化（及重建模板）时运行。
- 搜索：`search_fp16.rknn`，RGB `[1,3,256,256]` + 固定模板特征 `[1,256,8,8]` → bbox logits `[1,4,16,16]` 与 cls logits `[1,1,16,16]`，逐帧运行。
- 图像是 RGB 0–255；ImageNet 归一化配置在 RKNN 输入（mean `[123.675,116.28,103.53]`，std `[58.395,57.12,57.375]`）。模板特征输入 mean=0/std=1。输入形状全部固定。
- 搜索模型输出的是 bbox **logits**；运行时在 CPU 用 float32 `np.exp` 还原距离，再执行官方的裁剪映射与解码（`tracker/tracker_core.py`）。
- 官方 FEAR-XS-NoEmbs checkpoint，严格加载全部参数。保留 FBNet / ReLU / 相关性 MatMul / 回归头的原始权重和激活；没有近似激活替换，没有 Dynamic Template Update。

## 环境与测量口径

RK3588，Ubuntu 24.04，aarch64，8 GB 内存；RKNN Toolkit2 / Lite2 / librknnrt 2.3.2，NPU driver 0.9.8。转换环境：Python 3.11、PyTorch 2.2.0、ONNX 1.16.1、ONNX Runtime 1.26.0、NumPy 1.26.4、OpenCV 4.11.0。

三 NPU 核（mask=7），NPU governor=performance、频率 1 GHz，测量开始时约 41–42 °C，测量窗口内无其他 NPU 使用者，三核初始负载 0%。没有改 governor。

原生测量每模型预热 100 次、正式 500 次，独立创建普通 flags=0 与 profiling flags=8 的 context。表中 NPU 值来自普通 context 的 `RKNN_QUERY_PERF_RUN`，profiling 单独用于验证设备归属；flags=8 的墙钟不用于报 FPS。原生 wall 包括 `inputs_set`、`run`、`outputs_get`、`release`；输入文件 NCHW→NHWC 的一次转换在计时之外。PERF_RUN 是 SDK 报告的实际推理时长，可能包含 runtime 调度，不能等同于理论算力推算。

| 分支 | 精度 | NPU 均值 ms | 中位数 ms | p95 ms | 原生 API wall 均值 ms |
|---|---|---:|---:|---:|---:|
| 模板 | FP16 | 0.6372 | 0.6240 | 0.6620 | 1.0900 |
| 搜索 | FP16 | 2.4530 | 2.4325 | 2.5506 | 4.3130 |
| 模板 | INT8（未发布） | 0.4272 | 0.4190 | 0.4600 | 0.9625 |
| 搜索 | INT8（未发布） | 1.5652 | 1.5370 | 1.8102 | 3.7890 |

最终模板 47 个、搜索 76 个计算算子全部报告在 NPU。CPU 只剩 InputOperator / OutputOperator 边界（模板 2 行、搜索 4 行）；模型内部 CPU/GPU 计算算子均为 0。复现方法见 [BENCH_NATIVE.md](BENCH_NATIVE.md)。

## 算子检查、改写与拒绝的方案

原始搜索 ONNX 包含 67 Conv、38 Relu、2 MatMul、5 Reshape、1 Transpose、1 Exp 等。Toolkit 将相关性 MatMul lower 为 exMatMul；固定 Reshape、Transpose 布局变换和卷积均在 NPU。build 日志没有报告 unsupported，但原生设备报告发现最终 **Exp → CPU**，空闲 profiling 单次 17 μs。这说明 build 成功不等于全 NPU。

| 修改 | 原因 | 精度/硬件验证 | 决定 |
|---|---|---|---|
| `Exp(x) = 1/Sigmoid(-x)-1` | 尝试等价的 NPU 表达 | 20 个真实 crop 的 FP32 ONNX bbox 最大差 0.000397；硬件上 ConvSigmoid/Add 在 NPU，但倒数 lower 成 **FLOAT Div CPU**（样本 130 μs），未加速 | 拒绝，保留 `rewrite_exp.py` 与 parity 数据 |
| bbox logits 提前作为搜索输出；NumPy Exp 放进 CPU decode | 消除 RKNN 图内 Exp fallback，保持 learned head 原值 | 20 个真实 crop 的 FP32 boundary parity：bbox 最大差 7.63e-06，分类/模板逐位一致；最终硬件 CPU 计算算子为 0 | 采用 |
| `(x/255-mean)/std` 从 ONNX 移到 RKNN mean/std 输入配置 | 避免额外归一化节点，保持数学语义 | mean `[123.675,116.28,103.53]`，std `[58.395,57.12,57.375]`；模板特征输入 mean=0/std=1；同输入 FP32 parity 通过 | 采用 |
| Python 3.7 `collections.Mapping` → `collections.abc.Mapping` 兼容别名 | 上游 mobile-vision 在 Python 3.11 import 报错 | 仅 Python 类型别名，无张量/算子改动，strict checkpoint 成功 | 采用 |

改前/改后空闲窗口的公平比较（同一份 raw 输入、三核、100/500 次）：

| FP16 分支 | 改前 runtime 均值 ms | 改后均值 ms | 降低 | 改前→改后原生 wall ms |
|---|---:|---:|---:|---|
| 模板 | 0.7182 | 0.6372 | 11.28% | 1.1730 → 1.0900 |
| 搜索 | 2.7474 | 2.4530 | 10.72% | 4.5976 → 4.3130 |

改前搜索含 CPU Exp，所以改前列称 runtime 时间。归一化节点移出后，模板 NPU 算子数 49→47、搜索 78→76，搜索的 CPU 计算算子 1→0。早期在转换任务占用 CPU 时跑出的测速、以及第一轮用原生 NCHW 输入 normalize 失败的记录没有用于正式结果：RKNN 2.3.2 在 NCHW 输入下即使 `rknn_inputs_set` 返回 0 也会报 normalize error，改为显式 NHWC 后输出才与参考一致。

## 精度：PyTorch / ONNX / RKNN 同输入

INT8 校准使用官方视频中均匀抽取的 100 组真实目标模板、搜索 crop 与原 PyTorch feature，未使用随机 feature。两段无人机序列未参与校准。独立数值验证为 40 组原始 JPEG crop，每序列 20 个等距帧：固定首帧模板，搜索 crop 使用前一帧 GT（teacher forcing），避免漂移改变输入而使比较失真。

以下是实际 RKNN 模板→搜索完整链路对 PyTorch 的结果，bbox 已恢复 Exp；最大差的单位是模型 crop 回归距离或 raw cls logit，不是原视频像素框误差。

| 输出 | FP16 最低/平均余弦 | FP16 最大绝对差 | INT8 最低/平均余弦 | INT8 最大绝对差 |
|---|---|---|---|---|
| 模板特征 | 0.999991 / 0.999991 | 0.02693 | 0.989013 / 0.990128 | 0.84213 |
| bbox 距离 | 0.999967 / 0.999992 | 1.73824 | 0.874448 / 0.944255 | 73.95969 |
| cls logits | 0.999991 / 0.999997 | 0.22478 | 0.974487 / 0.991981 | 11.48122 |

分类峰值位置与 PyTorch 一致：FP16 39/40（97.5%），INT8 27/40（67.5%）。INT8 bbox 最大 relative L2 为 0.67837。校准只来自官方人像视频，泛化到无人机目标时校准覆盖不足可能是原因之一，但这只是推测，没有证实，也没有继续校准或微调以掩盖结果。推荐并发布 FP16。

原始 PyTorch→ONNX 余弦均接近 1；完整两段视频上 ONNX 与 PyTorch 的整数框逐帧一致。官方裁剪/解码另用 100 组含越界输入验证，两种 smoothing 下：crop/context/最终整数框误差 0，非平滑解码误差 0，平滑浮点最大差 1.42e-14。

## 完整无人机视频跟踪

使用官方 [VOT2018](https://data.votchallenge.net/vot2018/main/description.json) 的 `drone_across`（147 帧）与 `drone_flip`（113 帧），1280×720，metadata 30 FPS；原始 archives 用官方 SHA1 与 ZIP CRC 校验。画面为机载 FPV 视角追踪另一飞行目标。各后端在同一生成 MP4 上从相同的第一帧框连续跟踪，无重置，不提供中间帧 GT。

GT polygon 取 min/max AABB，continuous IoU 无 +1，排除初始化帧。这是本次的连续单目标 AABB 测量，不是官方 VOT EAO/reset 协议。

| 序列 | 后端 | 对 GT 平均 IoU | IoU≥0.5 帧占比 | 中心误差≤20 px | 对 PyTorch 平均框 IoU |
|---|---|---:|---:|---:|---:|
| drone_across | PyTorch / ONNX | 0.74049 | 100% | 100% | 1.00000 |
| drone_across | FP16 | 0.73941 | 100% | 100% | 0.98948 |
| drone_across | INT8 | 0.72348 | 99.32% | 100% | 0.86950 |
| drone_flip | PyTorch / ONNX | 0.01563 | 2.68% | 3.57% | 1.00000 |
| drone_flip | FP16 | 0.01563 | 2.68% | 3.57% | 0.91640 |
| drone_flip | INT8 | 0.01572 | 1.79% | 3.57% | 0.64900 |

`drone_across` 上 FP16 全程跟住，相对 PyTorch 的原视频框最大 xywh 分量差 2 px；INT8 最大 15 px。`drone_flip` 中原始 PyTorch 模型本身就丢失目标并在背景上漂移，FP16 的 GT 指标与原模型相同；这段视频不能称为成功跟踪，也不能把原模型的失败归因于 RKNN 转换。INT8 的漂移轨迹相对 PyTorch 差异更大。

速度来自板上完整视频的单次完整运行，不是理论 NPU FPS。每次模板 1 次、搜索 warmup 5 次；“计算 FPS”包括 crop / inference / Exp / decode；“整体 FPS”再包含读取、RGB 转换、绘框、编码及 flush，排除加载、ROI、初始化与首帧输出。这是 Python + RKNN Lite 加 OpenCV 读写的路径，不是 `tracker/` 里的 MPP/RGA 流水线。

| 视频 | 精度 | 搜索 Lite 调用均值 ms | 跟踪计算 FPS | 含读写视频整体 FPS |
|---|---|---|---:|---:|
| drone_across | FP16 | 7.465 | 115.53 | 37.45 |
| drone_across | INT8 | 6.580 | 128.86 | 38.89 |
| drone_flip | FP16 | 7.151 | 120.74 | 38.83 |
| drone_flip | INT8 | 7.017 | 122.33 | 39.12 |

Lite 调用时间与原生 NPU 时长不同：它包含 Python / layout / 输入输出传输与 CPU Exp。原生测速使用随机 smoke 固定输入，视频测速使用真实逐帧输入，两种口径不混用。

## 局限

- 只在两段 VOT2018 无人机序列上做了连续跟踪验证，没有做 UAV123 / VisDrone 全数据集评测。
- 后续若要改善 INT8 泛化或困难翻转序列，需要另外评估更有代表性的校准/跟踪数据、模板更新或微调；本次停在已验证的 FP16 部署上，没有修改模型或训练。
