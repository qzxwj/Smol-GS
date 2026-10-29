# 几何 ECVQ：第二步实现与 RDO 对照

当前实现训练、评估、完整检查点续训以及独立编解码。
使用 `--geometry-mode explicit --geometry-quantization ecvq`，
或 `--config configs/ecvq.json`。保持第一步行为则使用
`--geometry-mode explicit`，默认 `geometry_quantization=none`。

## 对照来源

来源是本地 USTC-IMCL/RDO-Gaussian，修订
`a7bb99b0374efbe3e2a23c5cf3e9109fbf53baef`。
参考实现的路径与行为对应关系列于下表；仓库不再保留冻结参考源码和动态执行参考源码的测试。
根目录 LICENSE.md 与该参考仓库许可证内容一致。

| 行为 | RDO 参考位置 | 当前实现 |
|---|---|---|
| 输入向量 | `scene/gaussian_model.py:vector_dict` | 物理尺度 exp(log_scales)、原始四元数 |
| 量化 | `vq/ecvq.py:ECVQ.quant` | 两个独立完整向量码本，维度 3 和 4 |
| 分配代价 | 同上 | 欧氏 L2 距离 + (-log2 p)/lambda，不使用平方距离 |
| 索引刷新 | `train.py` 的 `update_index`、渲染器索引缓存 | 训练按 `(iteration - 1) % 10 == 0` 重新分配，其余步复用索引；评估/导出重新分配 |
| 码本初始化 | `vq/kmeans.py:kmeans` | 相同抽样策略，十次 Lloyd 更新，保留空簇旧中心 |
| 概率初始化 | `vq/ecvq.py:kmeans_initialize` | 原参考的 L2-normalized counts，然后取 log；零值处理见下文 |
| 渲染梯度 | `gaussian_renderer/__init__.py:ste` | (x_hat - x).detach() + x，旋转随后归一化 |
| VQ 误差 | 同文件的 activate_vq 循环 | 选中码字与输入的 L2 范数和，不添加 VQ-VAE commitment 项 |
| 损失分母 | 同文件的 vq_dim 分支 | 保留 RDO 完整六组属性的 55N，维持几何项相对于图像损失的权重；N 为全场景点数，mask 不改变分母 |
| 几何码率 | 同上 | (bits_scale/lambda_scale + bits_rot/lambda_rot)/(55N) |
| 优化器 | `scene/gaussian_model.py:training_setup` | 独立 Adam，eps=1e-15；码本 2e-4、logits 2e-3 |
| VQ 调度器 | 同上 | MultiStepLR，milestones=[12000]，gamma=0.1，仅随 VQ 更新计数 |
| 死码字重激活 | `train.py`、`vq/ecvq.py:reactivate_codeword` | 全局迭代整除 1000 且尚未开启码率约束时执行；阈值 1e-5、噪声标准差 1e-4 |

当前训练损失为图像损失及特征码率项，再直接加几何 VQ 项和几何码率项；
不包含透明度正则。几何两项不会再乘 Smol-GS 的 lambda_rate。
特征不进入几何码本，旋转不在 ECVQ 输入前归一化。

RDO 六组量化向量的维数为 3+4+3+9+15+21=55。这里移除颜色/SH 的 VQ 项，
但保留 55N 分母，使几何两项相对于图像损失的系数不变；这等价于对原来的
7N 归一化结果整体乘以 7/55。码字分配中的 lambda 和距离代价保持不变。

## 阶段与参数来源

默认值取自上述本地参考的当前配置：20001 开始 VQ，25001 开始包含码率的分配；
尺度/旋转码本均为 8192，lambda 分别为 4096 和 256，点块大小为 16384。
这同时保留 Smol-GS 到 20000 的 compaction 和 30001 的坐标冻结。
K-means 在首次到达 VQ 阶段时对全体点执行一次；评估不会隐式初始化或消耗初始化随机数。
25001 之前也计算几何码率损失，区别仅是分配代价尚未加入码率。

本地参考的参数注释明确指出：原始 RDO 的 vq_start_iter=15001、
rate_constrain_iter=20001、vq_scale_lmbda=32768。这里采用的是本地当前参数与
Smol-GS 阶段的组合，不能称为原始论文的完全复现。

## 明确的必要适配

1. **维度与空集合。** 原参考在索引中使用 squeeze()，N=1 会变成标量。
   当前显式保留 [N,1] 索引，N=0 返回空向量与可微的零损失；K-means 不接受空输入。
2. **零频码字。** 原参考 log(0) 产生 -inf logits。使用其已有 1e-9 数值下限，
   对归一化计数取下限后再 log，避免单点、重复四元数或 K>N 导致不可恢复的 -inf 参数。
   不改变 K-means 的距离、中心更新或空簇保留规则。
3. **训练/评估一致。** 参考 Softmax.log_pmf 仅在 requires_grad 时截断到 log(1e-9)。
   当前训练与 no_grad 评估都使用这个相同下限；极低概率时，这与参考 no_grad 分支不同。
4. **尺度码字处理。** 与 RDO 一致，训练期间不投影尺度码本；导出时先把严格小于零的
   码字改为 1e-8，再重新分配索引并裁剪码本。零值和小于 1e-8 的正值保持原样。
   此处理在导出副本上执行，不修改训练模型；导出参考和实际编码使用同一规则。
5. **接口和依赖。** 原代码的 einops 变换换成对应的原生 Torch reshape/transpose，
   限定每类属性一个完整码本；argmin 搜索不保留自动微分图，但查表、码率和误差保留梯度。
6. **索引缓存。** 与 RDO 一致，训练按 `(iteration - 1) % 10 == 0` 刷新索引，其余步查缓存，
   但仍使用当前码本和 logits 计算码字、损失及梯度。评估和导出重新分配索引。
   缓存按模型保存进检查点，坐标排序/去重时同步重排；新点和首次使用的点补算索引。
   当前检查点必须包含缓存，不再为历史检查点补充缺失字段。
7. **训练循环。** 保留 Smol-GS 每步更新（包括最后一步）的行为。参考 RDO 跳过最终迭代的优化器更新；
   本步骤不把这项训练循环差异迁入。未引入其 GS/SH mask、几何规范化或 AbsGS。

## 保存、续训与编码边界

检查点统一为 `smolgs-checkpoint-v6`，ECVQ 模式记录：两个码本及 logits、初始化标志、
码率约束阶段标志、逐点索引缓存、VQ 更新数、主 Adam、VQ Adam、MultiStepLR 以及 Python/NumPy/Torch/CUDA RNG。
加载时验证格式、阶段和调度器更新数。续训不能用命令行偷偷切换量化模式、码本大小或调度参数。
VQ 启用前的检查点仍保存未初始化码本状态，在跨过起始迭代时才执行 K-means。

从该检查点可以直接渲染评估；独立压缩包统一使用下面的 v10 格式。

## 独立编解码（第三步）

导出按最终坐标八叉树顺序排列点，使用当前码本与已保存的码率约束阶段标志分配一次索引。
随后按 RDO `utils/encode_utils.py:shrink_codebook` 的 unique/searchsorted 规则，
裁掉未使用码字并重映射索引。裁剪后不会再运行 ECVQ 分配。

每个属性保存 FP32 码本、整数 CDF 和 rANS 索引流；不保存逐点原始尺度、四元数或训练优化器。
码本概率使用裁剪后的 logits 经 softmax，按参考 `_build_tables` 添加 overflow 符号，
调用参考 C++ pmf_to_quantized_cdf，精度为 16 位，再调用参考 RansEncoder。
整数 CDF 直接随包保存，解码不重算概率表，也不需要 logits 或训练场景。
特征继续使用原 Gaussian 流，坐标继续使用原八叉树，所有文件及 ZIP 开销计入 Size。

`submodules/rdo-entropy` 收录参考 imcl-compression 的 C++ rANS、CDF 和 ryg_rans 源码，
仅有一处 C++ 修正：flush 的输出缓冲区额外保留两个 uint32 给 64 位终止状态。
原参考按符号数分配缓冲，单个索引会发生堆越界；这已用单点编解码用例复现并修复。
此修正不改变正常输入的码流；当前测试检查 CDF 合法性与单点、多码字索引往返。
来源和修订见 `rdo_entropy_manifest.json`；Apache-2 和 ryg_rans 许可证保存在扩展目录。

解码查表恢复尺度和原始四元数，旋转随后归一化。物理尺度单独保存为运行时 buffer，
避免 exp(log(codeword)) 的额外舍入；不再运行 K-means、概率分配或 STE。
保存为 decoded.pt 后仍可独立加载渲染，禁止作为训练检查点续训。
历史检查点不再兼容；当前格式必须完整保存运行时 buffer 和索引缓存。

`smolgs.codec.export_reference` 独立重现实际导出决策，不读取编码后的索引流。
验证脚本以它为参考逐项比较索引、码字、属性、协方差和多视角渲染。
它也包含导出时的二值哈希和特征量化，不把未经导出处理的浮点模型当作解码基准。

## 验证方式

`tests/test_ecvq.py` 使用独立行为断言检查码字分配、码率、梯度、缓存、
初始化和死码字重激活；另有用例区分 L2 与平方距离，覆盖空输入和极低概率。
测试不再动态执行第三方参考源码，不再宣称与冻结参考逐项一致。

集成验证覆盖 55N 和 mask 分母、几何项相对于图像损失的权重、原始四元数空间、STE 梯度隔离、两个优化器参数覆盖、
K-means 只执行一次、码率阶段切换、重激活时机、空/单点/重复四元数、
未使用码字、拓扑变化、坐标冻结及每个阶段边界前后的续训。
CPU 继续更新按逐位一致验证；CUDA 恢复的参数和优化器本身逐位检查，继续反传允许
atol=2e-7、rtol=2e-5（原有 CUDA 原子归约并非逐位确定）。
还验证 Torch CPU/CUDA 及 tiny-cuda-nn＋原生哈希五阶段训练、推理检查点和真实训练入口续训。

`tests/test_ecvq_codec.py` 额外检查 CDF 合法性和 rANS 索引往返、裁剪/重映射、单点、未使用码字、
CPU/CUDA 特征编码、Torch/tiny-cuda-nn 网络、导出不修改训练参数、独立解码及 decoded.pt 往返，
并拒绝损坏的校验、CDF 和非法码本。训练入口测试覆盖默认自动导出。
测试不等于真实场景 35k 质量或压缩率验证。
