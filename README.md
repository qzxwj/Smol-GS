# Smol-GS

按随仓库提供的 [Smol-GS 论文](paper/arxiv.tex) 独立实现的训练、渲染和压缩流程。当前实现边界见 [FIDELITY.md](docs/FIDELITY.md)。公开参考 CSV 位于 `official_snapshot/results/`，不代表本实现的实测结果。

## 目录

| 路径 | 用途 |
|---|---|
| `train.py` | 五阶段训练与当前格式检查点续训 |
| `render.py` | 检查点或独立压缩包的渲染与评估 |
| `compress.py` | 完成训练后的编码、独立解码 |
| `metrics.py` | 对保存的 PNG 计算 PSNR、SSIM、LPIPS |
| `configs/` | 基础、原生 MLP 和流程测试配置 |
| `scene/`、`gaussian_renderer/` | 模型、数据读取和渲染 |
| `smolgs/`、`utils/` | 压缩、评估及共享工具 |
| `scripts/` | 安装、环境检查、批量训练和编解码验证 |
| `tests/` | 当前实现的回归测试 |
| `submodules/` | 运行所需第三方扩展源码 |

## 环境

Python 3.11、PyTorch 2.4.1、CUDA 12.4；完整依赖见 `environment.yml` 和 `requirements.txt`。已有环境可直接使用；新环境按以下方式安装：

```bash
conda env create -f environment.yml
conda activate smol-gs
SMOLGS_PYTHON=python bash scripts/build_extensions.sh
python scripts/check_environment.py
```

原生 MLP 配置还需要带神经网络支持的 tiny-cuda-nn：

```bash
SMOLGS_PYTHON=python bash scripts/install_tcnn.sh
```

安装脚本固定 tiny-cuda-nn 版本，可通过 `SMOLGS_TCNN_SOURCE` 指定本地源码，通过 `CUDA_HOME` 指定 CUDA 工具链。LPIPS 使用本地缓存的 VGG 权重。

## 配置

| 配置 | 用途 |
|---|---|
| `configs/base.json` | Torch MLP 基础配置，位置 MLP 学习率 0.008 |
| `configs/small.json` | 保留配置名；当前关闭 opacity 正则，与基础设置相同 |
| `configs/large.json` | 保留配置名；当前关闭 opacity 正则，与基础设置相同 |
| `configs/native.json` | 原生外观和位置 MLP，位置 MLP 学习率 0.001 |
| `configs/smoke.json` | 缩短为 50 步的五阶段流程检查 |
| `configs/explicit.json` | 显式几何配置；所有配置默认均使用显式尺度/旋转 |
| `configs/ecvq.json` | 显式几何＋尺度/旋转 ECVQ，支持独立编解码 |

所有配置使用论文的 `abs(tanh(raw))` opacity，默认坐标八叉树深度为 16。没有额外的 MLP 学习率暖启动。原生配置保留当前选择的 0.001 位置学习率，属于明确的论文差异。

当前训练代码已删除透明度正则项及其配置/命令行入口，图像重建仍正常优化透明度，
低透明度/不可见剪枝保持不变。旧配置中的该字段不再接受；旧格式检查点和压缩包不再兼容。
历史实验快照保留原行为，策略差异见 [FIDELITY.md](docs/FIDELITY.md)。

## 单场景训练

数据应为未畸变 COLMAP 格式，包含图像目录及 `sparse/0/` 中的相机、图像和点云文件。标准划分按文件名排序，每八张中一张用于测试。

```bash
python train.py -s ../data/tandt/train -i images -r 1 \
  -m outputs/train_native --gpu 0 --config configs/native.json \
  --eval-lpips --monitor-octree
```

Torch 基础配置使用 `--config configs/base.json`。正常训练不设初始点数或最终点数上限。阶段边界为 500、15000、20000、30000、35000 步。

所有配置统一使用显式几何：每点直接训练 `log_scales` 和原始 wxyz 四元数，
尺度为 `exp(log_scales)`，四元数在渲染前归一化。尺度按 SfM 邻近距离初始化，
旋转从 `(1,0,0,0)` 开始。固定学习率默认分别为 0.005、0.001，可用
`--explicit-scaling-lr`、`--explicit-rotation-lr` 调整。
网络预测尺度和旋转的路径、对应网络及学习率入口已删除。
实际优化器学习率配置记录在 `run.json` 的 `effective_learning_rates`。

坐标冻结时默认保留同格首点。实验选项 `--coordinate-selection opacity` 按全部训练视角的
可见 opacity 均值选择代表点；评分定义与严格对照方法见 [COORDINATE_SELECTION.md](docs/COORDINATE_SELECTION.md)。

默认只对特征执行标量量化和 Gaussian 码率优化；几何以 FP32 无损保存。
启用 ECVQ 后再压缩尺度和旋转。增密、剪枝、坐标冻结和外观训练保持五阶段调度。
已移除模式的配置和模型不再兼容，重新训练应使用新输出目录。

ECVQ 使用 `--config configs/ecvq.json`。原生网络可在
`--config configs/native.json` 后添加
`--geometry-quantization ecvq`。
默认在 20001 步初始化两个几何码本并开启 VQ，在 25001 步开启包含码率的分配，
其余阶段沿用 Smol-GS。完成训练后生成包含几何索引流的独立压缩包。
参考来源、参数和所有适配见 [ECVQ.md](docs/ECVQ.md)。

当前格式检查点可保留 Adam 与随机数状态续训：

```bash
python train.py -s ../data/tandt/train -i images -r 1 \
  -m outputs/train_native --gpu 0 \
  --resume outputs/train_native/checkpoint_latest.pt --eval-lpips
```

完成 35k 后生成 `model.smolgs.zip`。检查点和压缩包必须记录 `opacity_activation=abs_tanh`，只支持当前配置结构，不再提供旧公式或旧配置迁移分支。

## 渲染与指标

```bash
python render.py -m outputs/train_native/model.smolgs.zip \
  -s ../data/tandt/train -i images -r 1 \
  -o outputs/train_native/test/ours_35000 --save-images
python metrics.py -m outputs/train_native
```

`render.py` 输出浮点渲染的平均 PSNR、SSIM、LPIPS、FPS 和逐视角指标；`metrics.py` 评估保存的 8-bit PNG，结果可能有量化差异。LPIPS 统一使用 Inria 3DGS 的 VGG v0.1 口径，直接传入 RGB [0,1]。

## 编解码

```bash
python compress.py encode outputs/train_native/checkpoint_35000.pt \
  outputs/train_native/model.smolgs.zip --device cuda --codec cuda
python compress.py decode outputs/train_native/model.smolgs.zip \
  outputs/train_native/decoded.pt --device cuda
python scripts/verify_roundtrip.py \
  --checkpoint outputs/train_native/checkpoint_35000.pt \
  --archive outputs/train_native/model.smolgs.zip \
  --output outputs/train_native/roundtrip.json
```

压缩包包含坐标八叉树、属性算术流、网络、二值哈希表和元数据。Size 以整个 ZIP 的实际字节数除以 1,000,000 计算 MB。原生 MLP 始终需要 CUDA；Torch MLP 支持 CPU 参考编解码。

当前实现统一使用 `smolgs-checkpoint-v6` 和 `smolgs-archive-v10`。
显式几何未启用 ECVQ 时，
尺度的 log 参数和原始四元数以 FP32 无损保存，并与坐标八叉树顺序对齐。
此时总包大小包含未量化几何，不能当作已经实现 ECVQ 的压缩结果。
坐标编码统一使用独立网格，`coordinate_depth` 默认为 16，必须在 1–20 之间。
不提供旧检查点、旧压缩包或零深度坐标路径的兼容与迁移。

ECVQ 通过统一格式中的配置和几何存储元数据标识。几何码本先裁掉未使用项，
索引使用 RDO 的 16 位 rANS，整数 CDF 随包保存。安装其 CPU 扩展：

```bash
python -m pip install --no-build-isolation --no-deps ./submodules/rdo-entropy
```

当前环境已安装；完整扩展构建脚本也包含此项。原有 compress.py、render.py 和
scripts/verify_roundtrip.py 支持 ECVQ 包。解码模型直接查表恢复几何，只用于推理；
续训使用原始训练检查点。

## 批量运行与验证

```bash
python full_eval.py --dataset tandt --scene train --gpu 0 \
  --config configs/native.json --output-root outputs/native --dry-run
python full_eval.py --dataset tandt --scene train --gpu 0 \
  --config configs/native.json --output-root outputs/native
python scripts/collect_results.py --output-root outputs/native
python -m unittest discover -s tests -v
```

批量入口支持 MipNeRF360、Tanks and Temples、Deep Blending。MipNeRF360 使用室内 `images_2`、室外 `images_4`，其他数据集使用 `images`。缺少 CUDA 或本地 VGG 权重的测试会明确跳过；流程测试不能替代完整训练质量验证。

许可证与第三方来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
