# Smol-GS 论文复现

实现对应用户提供的 **Smol-GS: Compact Representations for Abstract 3D Gaussian
Splatting** 论文，按本机 `HAC`、`HAC-plus`、`Scaffold-GS`、`LocoGS` 的工程习惯组织。

**这是按论文独立实现，不是作者公开源码。** 2026-10-21 检查
[官方仓库](https://github.com/AaltoML/Smol-GS)，README 仍为
`Codes to be released.`。官方结果快照保存在 `official_snapshot/`。
论文明确内容、推定细节和差异逐项列在 [docs/FIDELITY.md](docs/FIDELITY.md)。
不保证逐行代码、码流、速度或最终质量与作者实现相同。

已包含完整的 **训练 → 增密/剪枝 → 特征压缩 → 坐标压缩 → 实际编码 →
独立解码 → 渲染/PSNR/SSIM/LPIPS评估** 流程，没有将原始 checkpoint 当成压缩模型。

## 目录

```text
Smol-GS/
├── arguments/__init__.py       # 附录参数、学习率、3DGS风格命令行参数
├── scene/
│   ├── gaussian_model.py       # 单splat神经模型、六个MLP、ADC和Adam状态迁移
│   ├── dataset_readers.py      # COLMAP/Blender相机、图像、SfM
│   ├── cameras.py
│   └── colmap_loader.py        # 原始Inria COLMAP解析
├── gaussian_renderer/          # Inria/HAC CUDA光栅器和小场景参考光栅器
├── utils/                      # 几何、混合哈希、Huffman/算术编码、loss、checkpoint
├── submodules/                 # 四个CUDA扩展的完整源码
├── smolgs/                     # 自包含压缩容器及评估流程
├── configs/{small,base,large}.json
├── train.py                    # 五阶段训练
├── compress.py                 # 独立编码/解码
├── render.py                   # checkpoint或压缩包渲染/评估
├── metrics.py                  # 3DGS式离线PNG指标计算
├── run_shell_{mip360,tnt,db}.py # 与本机HAC相似的启动入口
├── full_eval.py                # 全部13场景
├── scripts/                    # 构建、检查、队列、结果汇总
├── tests/                      # 码流、优化器、五阶段、CUDA一致性验证
├── docs/                       # 复现差异、来源、验证结果
└── official_snapshot/          # 作者结果CSV，和本项目实测结果分开
```

## 安装

本机已经配置并验证了目录内的独立环境，直接使用即可：

```bash
cd /mnt/dataset/lsh/xwj/Smol-GS
source .env/bin/activate
python run_shell_mip360.py --scene garden --gpu 2 --dry-run
```

下列安装命令供重新创建环境时使用。

论文环境为 Python 3.11、PyTorch 2.4.1、CUDA 12.4。本机为 RTX 4090，
论文为 H200 141GB；正式训练的显存和时间不能按论文硬件直接推算。

```bash
cd /mnt/dataset/lsh/xwj/Smol-GS
conda env create -f environment.yml
conda activate smol-gs
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST=8.9  # 本机RTX4090；其他显卡按实际架构设置
bash scripts/build_extensions.sh
python scripts/check_environment.py
```

也可使用 Python 3.11 venv，先安装 PyTorch CUDA wheel，再安装依赖及编译扩展：

```bash
python3.11 -m venv .env
.env/bin/python -m pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu124
.env/bin/python -m pip install -r requirements.txt
# CUDA_HOME必须指向CUDA 12.x编译工具链；系统默认CUDA13不适合编译Torch2.4扩展。
CUDA_HOME=/usr/local/cuda-12.2 TORCH_CUDA_ARCH_LIST=8.9 \
  SMOLGS_PYTHON="$PWD/.env/bin/python" bash scripts/build_extensions.sh
```

上面的第二种方式使用本机已有12.2编译器与cu124运行库；若要求编译器也严格12.4，
使用conda方案。`scripts/build_extensions.sh`只向当前选择的Python环境安装。
LPIPS统一采用官方Inria 3DGS口径：VGG v0.1，RGB图像直接以`[0,1]`输入，
不额外做`2*x-1`，逐图计算后取均值。训练验证、独立渲染、离线PNG及诊断共用此实现。
使用本机已缓存的VGG16权重；没有缓存时需先下载预训练权重。
新结果记录`lpips_protocol=inria-3dgs-vgg-v0.1-direct-01`；离线PNG评测另写
`evaluation_protocol.json`。旧结果没有被重新标记或换算，需重新评测才能按新口径比较。

## 单场景正式训练

```bash
python train.py \
  -s /mnt/dataset/lsh/xwj/data/mipnerf360/garden \
  -i images_4 -r 1 \
  -m outputs/mipnerf360/garden/base \
  --variant base --eval --gpu 2 --eval-lpips
```

增密使用光栅器投影位置梯度前两个分量的范数，并按可见观测次数取平均；
阈值为2e−4。新运行的`run.json`记录`adc_gradient_space: screen_xy`。

默认35000步、f=8、R=16、λ_s=0.2、λ_o=3e−7、λ_q=5e−4，
学习率及网络结构与附录表格对应。`small/base/large`仅改变论文给出的λ_o。
`--config configs/base.json`也可加载完整配置。命令行覆盖配置的具体字段，例如：

```bash
python train.py -s /path/to/scene -m outputs/example \
  --feat_dim 8 --recursion 16 --lambda_o 3e-7 --lambda_q 5e-4 \
  --position_lr_init 0.0002 --position_lr_final 0.00001
```

支持`--source_path`、`--model_path`、`--white_background`、`--test_iterations`、
`--start_checkpoint`、`--densify_grad_threshold`等现有仓库常见参数名。
完整清单用`python train.py --help`。

| 阶段 | 步数 | 行为 |
|---|---:|---|
| warmup | 1–500 | SfM位置/尺度、随机抽象特征初始化后训练 |
| densification | 501–15000 | 增密与剪枝 |
| compaction | 15001–20000 | opacity L1正则与剪枝 |
| feature compression | 20001–30000 | f/s量化训练和Gaussian NLL |
| coordinate compression | 30001–35000 | 网格中心量化、重复叶去重、固定坐标微调 |

正式训练不设置点数上限。`--max-initial-points`、`--max-splats`、
`configs/smoke.json`仅供调试，会改变论文实验设置。

输出包含`config.json`、JSON格式的`cfg_args`、`run.json`、`training.jsonl`、
`checkpoint_latest.pt`、阶段评估和最终`model.smolgs.zip`。
checkpoint同时保存Adam动量、随机数状态和训练阶段；使用：

```bash
python train.py -s /path/to/scene -i images_4 -r 1 -m outputs/example \
  --resume outputs/example/checkpoint_latest.pt
```

若只运行前1000步验证，使用`--stop-after 1000`；它不会把五阶段压缩成1000步。
恢复时使用checkpoint的模型配置，数据路径、图像目录、分辨率和划分必须一致。

## 完整13场景和汇总

数据默认读取`/mnt/dataset/lsh/xwj/data`，无需复制或更改现有数据。
室外MipNeRF360使用images_4，室内使用images_2；T&T、DeepBlending使用images。
这些已缩小的图片统一-r 1，避免再次降采样。

```bash
# 先查看全部命令，检查数据是否齐全
python full_eval.py --gpu 2 --variant base --dry-run

# 依次训练并从真实压缩包评估，结束后汇总CSV/JSON
python full_eval.py --gpu 2 --variant base

# 只运行一个数据集/场景；其他入口同理
python run_shell_mip360.py --scene garden --gpu 2 --variant base
python run_shell_tnt.py --gpu 2 --variant small
python run_shell_db.py --gpu 2 --variant large

# 恢复队列中未完成的训练，已完成模型重新评估
python full_eval.py --gpu 2 --variant base --resume
python scripts/collect_results.py --output-root outputs
```

队列在`outputs/<dataset>/<scene>/<variant>/launcher.log`记录进度。
它按顺序执行，不自动占满多个GPU。缺失LPIPS权重时可显式加`--skip-lpips`；
此时LPIPS为null，不能作为完整论文指标。
结果汇总在`outputs/benchmark_results.{csv,json}`，平均值标注实际测量场景数。

## 压缩、独立解码及评估

完成训练自动生成`model.smolgs.zip`，也可独立编码：

```bash
python compress.py encode outputs/example/checkpoint_35000.pt outputs/example/model.smolgs.zip
python compress.py decode outputs/example/model.smolgs.zip outputs/example/decoded.pt

# 直接从压缩包独立解码并评估，解码不需要原始checkpoint
python render.py -m outputs/example/model.smolgs.zip \
  -s /path/to/scene -i images_4 -r 1 \
  -o outputs/example/test/ours_35000 --save-images

# 对保存的PNG重新计算指标（PNG量化会使数值略有差异）
python metrics.py -m outputs/example
```

压缩包包含occupancy-octree的Huffman码流、f/s的HAC Gaussian算术码流、
六个MLP、二值哈希参数及完整元数据。每个属性块核对整数符号SHA-256。
统计整个ZIP文件大小，报告十进制MB和二进制MiB，不漏掉网络或哈希开销。
CUDA码流使用匹配的CUDA网格和CDF实现；CPU参考码流使用`--codec cpu`，
可以在CPU上独立解码，但编码效率和速度不用于对比论文。

## 验证

```bash
python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=2 python scripts/check_environment.py --output outputs/validation/environment.json

# 真实Garden数据、缩短五阶段、低分辨率，只验证链路
CUDA_VISIBLE_DEVICES=2 python train.py \
  -s /mnt/dataset/lsh/xwj/data/mipnerf360/garden -i images_4 -r 160 \
  -m outputs/validation/garden_cuda_50 --config configs/smoke.json \
  --max-initial-points 2000 --max-splats 5000 --test-iterations 50 \
  --log-interval 5 --checkpoint-interval 25
python scripts/verify_roundtrip.py \
  --checkpoint outputs/validation/garden_cuda_50/checkpoint_50.pt \
  --archive outputs/validation/garden_cuda_50/model.smolgs.zip \
  --output outputs/validation/garden_cuda_50/roundtrip.json
```

测试覆盖Huffman/八叉树边界、条件Gaussian算术编码、尾概率稳定性、
增密后Adam状态、checkpoint恢复随机序列、五阶段梯度、压缩前后图像一致性、
HAC CUDA与参考哈希网格的输出及参数梯度。
实际已执行的验证及版本记录见[docs/VALIDATION.md](docs/VALIDATION.md)。

许可证与复制来源见[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)和`LICENSE.md`。
