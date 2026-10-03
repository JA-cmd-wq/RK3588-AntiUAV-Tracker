# FEAR-XS → RKNN：相对原版的改动清单

基线：PinataFarms/FEARTracker（commit `0a3bd039918909c79c1b7e55a4bfb7807520abde`），官方 checkpoint `FEAR-XS-NoEmbs.ckpt`（SHA256 `8efe7dfd3498e385f332fd655f360a848d723f5fd77c2d433b2c084029616be5`），无 Dynamic Template Update。目标：RK3588，RKNN Toolkit2 / Runtime 2.3.2，NPU driver 0.9.8。

**权重一字未改**（`strict=True` 载入全部参数），**没有替换任何激活、没有重训、没有微调**。FEAR-XS 的主干与头部本来就只用 ReLU（导出的 ONNX 中无 SiLU/Swish/GELU）。出厂模型只有 FP16；INT8 版本精度损失明显，未随仓库发布。

## 1. 改动表

| # | 改动 | 原版 | RKNN 版 | 原因 | 精度影响 |
|---|---|---|---|---|---|
| 1 | bbox 的 `Exp` 移出计算图 | 回归头末端 `x = torch.exp(adjust * bbox_pred + bias)`，Exp 在图内 | 搜索模型直接输出 bbox **logits**（`[1,4,16,16]`），CPU 上用 `np.exp`（float32）还原距离后再按官方方式解码 | 图内唯一的 `Exp` 在 RKNN 上落到 CPU（空闲 profiling 单次约 17 μs），NPU 图被拆开；build 日志不报错，只有板端 per-op 设备报告才能看到 | 与原图逐样本对比（20 组真实 crop，FP32 ONNX）：bbox 最大绝对差 **7.63e-06**，cls 与模板特征最大绝对差 0（逐位一致），bbox 余弦 ≥ 0.9999999999999993。数据：`evidence/fear_npu_boundary_parity.json` |
| 2 | ImageNet 归一化移入 RKNN 输入配置 | 预处理（albumentations `Normalize`，mean `[0.485,0.456,0.406]`、std `[0.229,0.224,0.225]`，作用于 0–1 图像）在 CPU 上做；导出 ONNX 时若放进图内会多出 `Div/Sub/Div` 节点 | 图内不含归一化；RKNN `config(mean_values=[[123.675,116.28,103.53]], std_values=[[58.395,57.12,57.375]])`，板端直接送 RGB uint8（0–255）。搜索模型第二个输入（模板特征 256 通道）配 mean=0、std=1，即不归一化 | 避免额外的归一化算子，也省掉 CPU 上的逐像素浮点运算 | 数学上精确（`123.675 = 255×0.485` 等）。与改动 1 同一份对比数据覆盖（对比的 NPU 边界图 = 去掉归一化 + 去掉 Exp，输入由外部做 `(x/255-mean)/std`），模板特征最大差 0 |
| 3 | 固定形状，拆成 template / search 两张图 | PyTorch 动态图；`get_features` 初始化时算一次模板特征并缓存，逐帧 `track(search, template_features)` | **template**：RGB `[1,3,128,128]` → `[1,256,8,8]`，初始化/重建模板时运行；**search**：RGB `[1,3,256,256]` + 模板特征 `[1,256,8,8]` → bbox logits `[1,4,16,16]`、cls logits `[1,1,16,16]`，每帧运行。两张图输入输出形状全部固定（opset 12） | RKNN 需要静态形状；两张图对应原版 `get_features` / `connect_model`，调用关系不变 | 无（同一计算，仅静态化）。PyTorch→ONNX：模板最大绝对差 4.4e-06，搜索 bbox 3.3e-04、cls 1.6e-05（PyTorch 与 ONNX Runtime 同输入对比，bbox 为 Exp 之后的距离）；`evidence/fear_export_report.json` |
| 4 | Python 3.11 `collections.Mapping` 别名 | 上游 mobile-vision 面向 Python 3.7，`import` 时引用 `collections.Mapping` 等 | `export_models.py` 在 import 之前补 `collections.Mapping / MutableMapping / Sequence / Iterable` → `collections.abc` 的别名 | 仅让上游代码能在 Python 3.11 里 import | 无（纯 import 兼容，不触及张量与算子） |

## 2. 试过并拒绝的方案

| 方案 | 做法 | 结果 | 结论 |
|---|---|---|---|
| `Exp(x) = 1 / Sigmoid(-x) − 1`（`rewrite_exp.py`） | 把 Exp 改写成 `Neg → Sigmoid → Reciprocal → Sub`，试图让 Exp 在 NPU 上等价执行 | 数值上等价（20 组真实 crop，bbox 最大绝对差 3.97e-04，`evidence/fear_rejected_sigmoid_exp_parity.json`）；Sigmoid/Add 在 NPU，但 `Reciprocal` 被 lower 成 **FLOAT Div 落到 CPU**（样本 130 μs），比原先 Exp 的 17 μs 更慢 | 拒绝；脚本保留作为“已尝试”的记录，最终采用改动 1 |

## 3. 没有改动的部分

- 权重：`weights_changed = false`，`learned_activation_changed = false`（见 `evidence/fear_npu_boundary_parity.json`）。
- 激活函数：FBNet-C 主干、相关性模块、box/cls 塔全部原样 ReLU；没有近似激活替换。
- 网络结构：相关性 `MatMul`、`SepConv`、回归头原样；不做 Dynamic Template Update。
- 精度：出厂只有 FP16，不量化。INT8 实验版（校准集为官方人像测试视频，泛化到无人机时精度明显下降：分类峰位置与 PyTorch 一致 27/40，对比 FP16 的 39/40）未发布。

## 4. 结果

下表数据出自 RKNN 原生 C API，每模型预热 100 次、统计 500 次，三核，`RKNN_QUERY_PERF_RUN` 取自普通（flags=0）context；随机 smoke 输入，仅用于测模型速度。详见 `FEAR_RKNN_RESULTS.md` 与 `BENCH_NATIVE.md`。

**算子归属**：最终 template 47 个、search 76 个计算算子全部在 NPU；CPU 上只剩 `InputOperator / OutputOperator` 边界行（template 2 行、search 4 行），模型内部 CPU/GPU 计算算子为 0。

原版导出图（`evidence/fear_export_report.json`）的算子统计，对照上面的改动：

| 图 | 原始 ONNX 算子（数量） | 改动后 NPU 计算算子 |
|---|---|---|
| template | Conv 47、Relu 30、Add 12、Div 2、Sub 1、Constant 1 | 47（49 → 47：去掉归一化节点） |
| search | Conv 67、Relu 38、Add 13、Reshape 5、MatMul 2、Concat 2、Mul 2、Transpose 1、Exp 1、Div 2、Sub 1、Constant 7 | 76（78 → 76；Exp 移出，CPU 计算算子 1 → 0） |

速度（FP16，空闲窗口，同一份输入、三核、100/500 次，改动前后公平对比）：

| 分支 | 改前 NPU 均值 ms | 改后 NPU 均值 ms | 降低 | 原生 API wall（改前 → 改后）ms |
|---|---:|---:|---:|---|
| template | 0.7182 | **0.6372** | 11.28% | 1.1730 → 1.0900 |
| search | 2.7474 | **2.4530** | 10.72% | 4.5976 → 4.3130 |

“改前”的 search 含 CPU 上的 Exp，所以该列是 runtime 时间，不是纯 NPU 时间。原生 API wall 额外包含 `inputs_set / run / outputs_get / release`。改后 CPU 端 `np.exp` 对 `[1,4,16,16]` 张量的耗时可忽略。

FP16 整体精度（与本次改动无关，是 FP16 数值本身的误差；PyTorch 与实际 RKNN 模板→搜索完整链路，40 组真实 crop，bbox 已 Exp 还原）：

| 输出 | 最低 / 平均余弦 | 最大绝对差 |
|---|---|---|
| 模板特征 | 0.999991 / 0.999991 | 0.02693 |
| bbox 距离 | 0.999967 / 0.999992 | 1.73824 |
| cls logits | 0.999991 / 0.999997 | 0.22478 |

分类峰值位置与 PyTorch 一致：39/40（97.5%）。bbox 最大差的单位是模型 crop 回归距离，不是原视频像素。

## 5. 复现

在装有 RKNN Toolkit2 2.3.2 的 Linux（x86 或 RK3588）上，`fear_to_rknn/` 目录内：

```bash
pip install -r requirements_convert.txt
export PYTHONPATH="$PWD/upstream${PYTHONPATH:+:$PYTHONPATH}"

# 1) 导出固定形状 ONNX（含归一化与 Exp 的 FP32 参考图），checkpoint 需自行从上游获取
python export_models.py --weights /path/to/FEAR-XS-NoEmbs.ckpt --source-dir upstream --output models

# 2) 用真实视频生成 manifest 与（可选）校准集；--tracks 是逐帧 bbox 的 JSON：{"frames":[{"bbox":[x,y,w,h]}, ...]}
python prepare_calibration.py --video clip.mp4 --tracks tracks.json \
  --weights /path/to/FEAR-XS-NoEmbs.ckpt

# 3) 改写为 NPU 边界图：去掉归一化子图，bbox 输出改为 Exp 之前的 logits，并与参考图对比
python prepare_npu_onnx.py

# 4) 转 RKNN（FP16）
python convert_rknn.py --precision fp16 --normalize-input \
  --template-stem template_npu --search-stem search_npu
```

产物在 `models/`：`template_npu_fp16.rknn`、`search_npu_fp16.rknn`（即仓库中的 `template_fp16.rknn` / `search_fp16.rknn`，仅文件名不同）。`rewrite_exp.py` 是被拒绝的方案，仅供对照。FEAR 的 ONNX 文件（FP32 约 3–5 MB 每个）不入库，按上面的步骤即可重新生成。
