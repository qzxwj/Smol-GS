# 与论文的对应关系及复现边界

实现依据是用户提供的 `smol-gs-2512.00850/arxiv.tex`，本目录的
`paper/arxiv.tex` 保存了该版本。2026-10-21 检查作者的
[项目主页](https://aaltoml.github.io/Smol-GS/)和
[官方仓库](https://github.com/AaltoML/Smol-GS)，仓库 README 仍为
“Codes to be released.”，仅公开了结果 CSV。
因此本项目是**按论文独立实现**，不能保证作者源码相同、码流相同，
也不能在未完成标准训练的情况下声称复现了论文指标。

## 已按正文和附录实现

| 论文内容 | 实现位置 | 配置 |
|---|---|---|
| 每个 splat 的独立 xyz、抽象特征 f、缩放控制器 s | `scene/gaussian_model.py` | f=8，s=3；没有 anchor/offset |
| Morton 二进制位置嵌入 | `utils/graphics_utils.py` | R=16，输入48维 |
| MLP_oct | `scene/gaussian_model.py` | 48→64→64→8，ReLU |
| MLP_o / c / r / s | 同上 | 20→128→128→1/3/4/3；Tanh/Sigmoid/None/None |
| 相机方向、距离、位置嵌入拼接 | `SmolGS.attributes` | 方向为 camera_center−xyz |
| opacity=abs(MLP_o)，s*=s·sigmoid(MLP_s) | 同上 | 四元数为 wxyz，归一化 |
| 原始 3DGS alpha compositing | `gaussian_renderer/` | 使用 HAC 提供的 Inria CUDA 光栅器 |
| 二值哈希网格 | `utils/hashgrid.py` | 3D:12层×4维、2¹³；XY/XZ/YZ各4层×4维、2¹⁵ |
| 分辨率表 | `arguments/__init__.py` | 3D和2D逐项抄录附录，合计96维 |
| MLP_h | `scene/gaussian_model.py` | 96→128→128→(2n_f+8) |
| 特征与缩放的 Gaussian prior、学习量化步长、NLL | 同上 | 每 splat 的 f 和 s 联合位数对 N 求平均 |
| SfM 初始化、增密、剪枝 | 同上 | 投影位置XY梯度阈值2e−4、opacity阈值0.005 |
| 五阶段训练 | `train.py` | 0–500、500–15000、15000–20000、20000–30000、30000–35000 |
| compaction阶段 opacity L1 | `utils/loss_utils.py` | λ_o·sum(abs(opacity))，仅第三阶段 |
| compression阶段 NLL | 同上 | λ_q=5e−4，仅第四、五阶段 |
| L1+SSIM | 同上 | λ_s=0.2，11×11 Gaussian SSIM窗口 |
| Adam和十组指数学习率 | `arguments/`、`SmolGS.update_learning_rates` | 附录学习率表，I=30000 |
| 坐标网格中心量化、重复叶剪枝 | `SmolGS.freeze_coordinates` | 30001步，R=16 |
| BFS occupancy octree + Huffman | `utils/octree.py` | 每内部节点一个occupancy byte |
| 属性算术编码 | `utils/encodings_cuda.py`、`smolgs/codec.py` | 默认使用原始HAC CUDA编码器 |
| 独立压缩模型 | `smolgs/codec.py` | ZIP包含坐标、f/s码流、网络、二值哈希和元数据 |
| 标准划分 | `scene/dataset_readers.py` | 图像名排序后，index%8==0为测试，其余训练 |
| 三个数据集13个场景 | `scripts/run_benchmark.py` | MipNeRF360:9、T&T:2、DeepBlending:2 |
| 三种模型 | `configs/{small,base,large}.json` | λ_o=4e−7 / 3e−7 / 1e−7 |

论文的 MLP_h 输出维度写成 `8+2n_f`，说明实现只能输出每 splat 一个 f 步长
和一个 s 步长；正文说“per-dimension step size”与表格不完全一致。本实现遵循表格。
三个 Linear 层解释为输入到隐藏、隐藏到隐藏、隐藏到输出。

## 未公开细节的明确选择

以下是需要验证的实现选择，不是从作者源码确认的事实。

1. **量化训练**：第四、五阶段用 uniform(−Δ/2, Δ/2) noise，推理和编码用
   round(f/Δ)·Δ；这是 HAC 的常用处理。步长为基准步长乘(1+tanh(adj))，
   f基准为1，s基准为0.001。Gaussian标准差使用softplus保持正值。
   s使用exp(log_s)的正值参数化；量化到零的控制器以1e−7作数值下界。
2. **NLL箱宽**：正文积分上下界写成 f_hat±Δ，但round(f/Δ)对应的互不重叠
   量化箱为 f_hat±Δ/2，HAC的实际码流也用±Δ/2。默认选择后者以保证率模型与
   实际算术编码一致。`--rate_bin_half_width 1` 可切换为正文书写的积分箱宽。
   位数使用log2；不把自然对数当成比特数。
3. **ADC周期与不可见判据**：采用邻近仓库惯例，每100步更新；连续1000步
   不可见的点可剪枝。增密分数为可见观测下投影位置梯度前两个分量的范数均值，
   即 `viewspace_points.grad[visible, :2].norm(dim=-1)`。这遵循本机 Inria 风格实现；
   作者未公开源码，不能视为已确认的作者实现细节。
   当前采用下述参考3DGS的尺度感知增密补充：小点同坐标克隆，大点移除父点并
   生成两个子点，按当前相机协方差采样，控制器s缩小为原来的1/1.6，特征f继承。
   **这与正文对所有点随机克隆的描述存在差别，是稳定性工程选择，不是已知作者实现。**
   固定场景尺度采用训练相机中心的1.1倍最大离心半径，独立于不断变化的点云包围盒。
   默认分裂阈值为0.01倍场景尺度，从3000步起剪去s最大分量超过0.1倍场景尺度的点。
   Smol-GS的s是最终轴尺度的上界，因此此项比直接检查单一视角的最终轴尺度更保守。
   最低保留一个有效、未超尺度的点；全体点几何无效时明确报错。
   不加入Scaffold-GS的offset层或HAC的体积正则。
4. **初始化**：s使用邻近3个SfM点的距离平方均值的平方根（与Inria的simple-knn一致）；
   f用N(0,0.01²)。学习率直接按论文表格，不额外乘场景尺度或加warmup延迟。
5. **包围盒**：位置嵌入的包围盒在ADC周期更新；20001步固定哈希归一化包围盒；
   30001步用全体坐标min/max固定八叉树包围盒，去重后坐标不再优化。
   哈希包围盒单独保存，避免第五阶段改变已学习的熵上下文。
   最大边界坐标夹到最后一个网格，退化轴保留1e−6范围。
   量化前若重复叶将删除超过50%的点，明确报错，保持原模型与优化器状态；
   50%是工程故障检查阈值，不是论文超参数，不裁剪包围盒或提高R来绕过检查。
6. **采样和剪枝统计**：每步均匀随机选择一个训练相机；可见点窗口累计opacity和梯度，
   未观察到的点只由不可见时间判据处理。Adam只迁移保留点的状态，新点动量清零。
7. **存储格式**：Huffman码本、ZIP容器布局、网络权重精度、哈希编码格式没有公开。
   此处网络为float32，二值哈希按位打包，再随ZIP无损压缩；没有用未压缩PLY
   或checkpoint冒充模型码流。网络、哈希、元数据和ZIP开销全部计入size_MB。
   因float32权重、码本、块大小及实现不同，不能保证论文大小和编解码时间。
8. **熵上下文的可移植性**：CPU参考码流用CPU预测上下文，CUDA码流用CUDA。
   CUDA码流使用HAC的浮点CDF，要求匹配的运行库/网格后端，解码每块均核对量化
   整数符号SHA-256；不静默接受不同后端导致的错误解码。禁用TF32。
9. **图像分辨率**：完整实验脚本采用原始3DGS惯例，MipNeRF360室外images_4、
   室内images_2，其他数据集原images；读取这些图片时-r 1避免重复缩小。
   这与本机已有压缩仓库的数据组织一致，本文没有逐场景明确写出分辨率。
10. **硬件和验证**：论文使用H200 141GB、Python3.11、Torch2.4.1、CUDA12.4。
    本机是共享RTX4090 24GB。缩短测试、低分辨率或限制点数不能作为论文质量复现证据。
    `configs/smoke.json`只验证流程；正式配置不缩短阶段、不限制点数。

## 与邻近仓库的取舍

沿用HAC、HAC++、Scaffold-GS、LocoGS的目录和命令行习惯，以及本地HAC的CUDA依赖。
但不把其他方法的`n_offsets`、`voxel_size`、`use_feat_bank`、SH progression、opacity reset、
mask loss等直接搬到Smol-GS：论文明确使用单splat显式坐标、8维抽象特征和六个MLP。
所以布局和工具链相似，方法参数仍服从本论文。具体参考文件见`docs/REFERENCE_REPOS.md`。

## 2026-10-21 投影位置梯度接口与验证

CUDA 和 Torch 参考渲染器均返回保留梯度的 `viewspace_points`。
训练在 loss.backward 后累计可见点的投影 XY 梯度范数，以观测次数归一化后
与阈值 2e-4 比较。Torch 参考渲染器按 W/2、H/2 缩放，以匹配 CUDA 光栅器
投影位置梯度的 NDC 约定。缺失投影梯度时明确报错。

按用户要求，旧实验产物和三维坐标梯度兼容分支均已删除。当前实现只保留
屏幕空间 XY 增密统计。后续Truck完整训练暴露了尺度增长和量化去重故障，
其诊断及修复见[经典仓库审计](REFERENCE_ADC_AUDIT.md)。
11 项测试通过，包括可见性/mask 索引、忽略投影梯度第三分量、缺失梯度报错、
CPU/CUDA 渲染反传到增密统计的连接，以及既有编码和优化器测试。
本次验证日志位于 `outputs/validation/screen_adc/tests.log`。

## 2026-10-21 尺度稳定性补充

新增的尺度感知增密和剪枝只作为可追踪的工程补充，不能据此声称作者采用了
同一方案，或已复现质量指标。新训练日志记录`3dgs-size-aware-v1`、固定场景尺度、
包围盒、控制器最大尺度及clone/split/异常剪枝数量。旧检查点仍可读取和诊断，
但缺少固定场景尺度的旧训练不能静默续跑为新策略。尚未启动修复后的正式Truck训练。

## 宽字母表属性的独立编码

当单个 splat 的属性跨通道符号范围超过算术编码器字母表限制时，编码器继续
沿属性通道拆分码流，保留量化值并将所有码流及元数据计入总大小。
此类分段码流由 smolgs-archive-v2 引入，各属性保留整数符号
SHA-256 校验。当前新导出使用下述 v3 上下文约定，解码器继续支持 v1/v2。
CPU/CUDA 宽字母表往返回归测试通过。

## 2026-10-22 编码与推理的上下文批次

Truck已完成35000步，但首次导出的v2包在一致性核验中失败：
检查点按整场景计算prior，编码按自适应码流块计算prior。CUDA GEMM对不同批次形状
产生微小浮点差异，点411790的一个特征刚好落在半整数舍入边界两侧，量化整数不一致。
算术码流SHA-256通过不等于其量化结果与检查点整场景推理一致。

新导出的`smolgs-archive-v3`明确记录`context_layout=full-scene-v1`：编码、解码
按完整场景预测一次prior，码流分块只切片上下文，不再随块大小重新计算MLP。
这与当前检查点渲染的计算批次一致。v1/v2解码仍使用原来的逐块约定以兼容旧CDF。
没有修改训练检查点、量化公式、属性和图像核验阈值；新回归测试模拟1 ULP的
批次差异，验证CPU/CUDA下改变码流块大小不会改变半整数附近的量化结果。
17项测试通过，日志位于`outputs/validation/codec_truck_sizeaware/tests.log`。

## 2026-10-22 整理阶段剪枝与位置网络活动

15k–20k归因实验表明，原有额外世界尺寸阈值删除了有效点。仅改用全部训练视角的
最大渲染轴尺度改善很小，移除整理阶段的额外尺寸条件使20k PSNR从19.944提升至24.791。
因此新增`adc_size_prune_until`限定这项工程规则的适用范围；兼容默认值20000重现旧配置，
本轮显式使用15000，保留其他剪枝条件和增密阶段的规则。

旧15k位置MLP第二个ReLU死亡，真实图像损失反传确认隐藏层和输出权重均无梯度，只有末层偏置更新。
新增`octree_init=kaiming`候选：隐藏层Kaiming normal、输出层Xavier normal、零偏置，
独立生成器seed由`octree_init_seed`记录。结构和学习率不变；旧配置默认`torch`，
旧检查点权重不会被改写。初始化是论文未公开细节的独立选择，不代表作者源码。
训练可通过`--monitor-octree --require-active-octree`记录活动，并在全点确认死亡时停止。
22项测试通过。此时仅完成1000步初始化对照，35000步稳定性和质量仍需实测。
详见`outputs/truck_octree_fix_20261022/diagnosis_report.md`及其后续日志。


### 同日后续诊断状态

Kaiming候选在7k的1515123个点中，只有5241个点的第二层仍有正激活，
相机均值附近组仅197/630987；因此该候选被拒绝继续，不能视为退化已修复。
正偏置、较大初始特征方差、位置组Adam eps1e-8的1000步对照也未建立稳定修复。
LeakyReLU(0.01)仅作为机制对照，明确偏离论文ReLU；未应用于当前完整参照。
当前完整参照从已验证的no_size/20k继续至35k，只用于量化剪枝修正的最终收益，
保留原ReLU及已知退化，权重仍为3e-7/5e-4。

另新增默认关闭的`octree_input_center`参数化对照：中心化二进制输入并补偿第一层
初始偏置，保持初始函数及MLP函数族相同，不增加参数、改变激活或学习率。
这改变了优化器的参数坐标，仍是未确认的独立实现选择；仅排队1000步验证，不是已验证修复。
25项测试通过，包括中心化的初始函数与折叠等价、检查点及压缩包往返。
最新状态见`outputs/truck_octree_fix_20261022/diagnosis_report.md`。


## 2026-10-22 LPIPS评测口径统一

根据用户要求，所有当前评测入口统一采用官方Inria 3DGS的VGG v0.1 LPIPS口径，
RGB图像以[0,1]直接输入，不执行2*x-1。共享包装器使用已安装lpips包的
normalize=False；数值与Inria lpipsPyTorch实测核对。训练损失不含LPIPS，训练目标不变。
训练验证、render.py、metrics.py、剪枝/位置网络诊断及协议审计均调用共享包装器。
新报告写明lpips_protocol；历史实验和source快照保留原数值与原代码，不做静默换算。
已启动的Python进程保留其加载的旧代码；要使用新口径需新进程重新评测。
浮点图像评测与保存PNG的评测仍有8位量化差异，最终PNG对照应使用metrics.py。
这是本次复现采用的统一口径；Smol-GS作者评测源码尚未发布。

本次验证：31项CPU测试中30项通过，1项CUDA专用测试跳过；包含独立Inria参考值、
共享渲染评测入口、真实metrics.py子进程及禁止混合口径均值的回归检查。
Truck完整32张PNG：LPIPS=0.1809527683，首尾两张与Inria原实现差异最大1.49e-8。
验证日志和逐图结果位于outputs/validation/lpips_inria/。
