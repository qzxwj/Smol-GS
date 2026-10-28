# Smol-GS

按随仓库提供的 [Smol-GS 论文](paper/arxiv.tex) 独立实现的训练、渲染和压缩流程。当前实现边界见 [FIDELITY.md](docs/FIDELITY.md)。公开参考 CSV 位于 `results/` 和 `official_snapshot/`，不代表本实现的实测结果。

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
| `configs/small.json` | 更强 opacity 正则，其他基础设置相同 |
| `configs/large.json` | 更弱 opacity 正则，其他基础设置相同 |
| `configs/native.json` | 原生外观和位置 MLP，位置 MLP 学习率 0.001 |
| `configs/smoke.json` | 缩短为 50 步的五阶段流程检查 |
| `configs/explicit.json` | 第一步实验：Torch 外观网络＋显式尺度/旋转，无几何量化 |

所有配置使用论文的 `abs(tanh(raw))` opacity，默认坐标八叉树深度为 16。没有额外的 MLP 学习率暖启动。原生配置保留当前选择的 0.001 位置学习率，属于明确的论文差异。

## 单场景训练

数据应为未畸变 COLMAP 格式，包含图像目录及 `sparse/0/` 中的相机、图像和点云文件。标准划分按文件名排序，每八张中一张用于测试。

```bash
python train.py -s ../data/tandt/train -i images -r 1 \
  -m outputs/train_native --gpu 0 --config configs/native.json \
  --eval-lpips --monitor-octree
```

Torch 基础配置使用 `--config configs/base.json`。正常训练不设初始点数或最终点数上限。阶段边界为 500、15000、20000、30000、35000 步。

显式几何实验可使用 `--config configs/explicit.json`，或在原生配置命令上加
`--geometry-mode explicit`。它直接训练 `exp(log_scales)` 和每点原始 wxyz 四元数，
渲染前归一化四元数，移除尺度和旋转 MLP。尺度沿用 SfM 邻近距离初始化，旋转从
`(1,0,0,0)` 开始。默认固定学习率分别为 0.005、0.001，可用
`--explicit-scaling-lr`、`--explicit-rotation-lr` 调整；原配置中的
`log_scales`、`scale_mlp`、`rotation_mlp` 学习率不用于该模式。
实际优化器学习率配置记录在 `run.json` 的 `effective_learning_rates`。

该实验只保留特征的量化和 Gaussian 码率项，不对尺度或旋转量化，不启用 VQ/ECVQ。
增密、剪枝、坐标冻结和外观训练继续按原阶段执行。首次训练需要新输出目录，
不能通过续训把 neural 检查点转为 explicit。

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

显式几何使用 `smolgs-checkpoint-v4-explicit` 和 `smolgs-archive-v7-explicit`，
尺度的 log 参数和原始四元数以 FP32 无损保存，并与坐标八叉树顺序对齐。
此时总包大小包含未量化几何，不能当作已经实现 ECVQ 的压缩结果。
默认 neural 模式继续使用原有 v3 检查点和 v6 压缩包。

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
