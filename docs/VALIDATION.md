# 实际验证记录

日期：2026-10-21。机器：共享NVIDIA RTX 4090。
机器可读记录见`docs/validation_report.json`，原始输出在`outputs/validation/`。

## 已完成

1. Python源文件编译检查；3DGS/HAC风格参数别名解析。
2. 7项核心测试在GPU可见环境全部通过，未跳过：
   * 八叉树边界、Morton顺序、重复叶拒绝、Huffman常量/全字节码本。
   * 条件Gaussian算术码流随机分布、常量和极端尾部的无损整数往返。
   * NLL尾部概率和梯度有限性。
   * Adam保留点动量迁移、新点动量清零、坐标量化后冻结。
   * 缩短五阶段反向传播、CPU码流、压缩前后渲染一致性。
   * checkpoint恢复后Adam和Python/NumPy/PyTorch随机状态一致。
   * HAC CUDA与Torch参考哈希网格的输出和参数梯度一致，含边界位置。
3. 全13场景正式命令生成、相机/点云读取及训练/测试无交集检查通过；场景统计在`docs/datasets_checked.json`。
4. 真实Garden图像完成缩短五阶段CUDA端到端检查：
   * 50步，阶段边界3/15/25/40，最终50；不是35000步正式配置。
   * 图像宽160像素，初始2000个SfM点，最多5000点，最终2604点。
   * 161个训练视图、24个测试视图，排序后每8张留一。
   * 真实ZIP大小685230字节，含全部网络/哈希/元数据/码流开销。
   * 独立进程仅凭ZIP恢复模型，属性整数符号SHA-256校验全部通过。
   * 3个视图逐像素对比，最大绝对误差4.470348358154297e−7。
   * 压缩后24个测试视图：PSNR 14.7505846、SSIM 0.2998667、VGG LPIPS 0.8427143。

这些低分辨率、少点数、50步数值仅证明代码链路可以运行和码流一致，
**不代表达到了论文质量**，不能与官方Garden 27.327 PSNR直接比较。

5. 额外用正式base默认参数、完整138766个SfM点、images_4原分辨率1297×840
   完成前20步训练，未限制点数，loss和梯度正常；输出在`outputs/validation/garden_default_20/`。
   这是原始规模初始化/训练检查，尚未进入全部正式阶段。

首次验证使用本机已有HAC环境：Python3.8.12、Torch2.1.2+cu118及已编译扩展。
记录在`outputs/validation/environment_hac.json`。

## 独立论文运行环境：已安装并验证

`/mnt/dataset/lsh/xwj/Smol-GS/.env`：Python3.11.4、Torch2.4.1+cu124、
torchvision0.19.1+cu124、CUDA运行库12.4。
四个CUDA扩展均从本目录`submodules/`源码重新编译，加载路径全部指向`.env`。
使用本机CUDA12.2的nvcc编译，运行库为12.4；没有更改系统CUDA或其他项目环境。
完整依赖冻结记录在`docs/environment_pip_freeze.txt`。

此环境下7项测试全部通过，真实Garden缩短五阶段再次完成，最终2605点。
实际码流编码后独立解码通过所有符号SHA-256；3视图渲染最大绝对差
4.3213367462158203e−7。压缩包685156字节。
24张测试视图的PSNR/SSIM/LPIPS及完整数值见
`outputs/validation/garden_torch241_50/decoded_test/metrics.json`。
这些仍为50步、160像素宽的流程检查，不能作为正式质量复现。

日志：`outputs/validation/{build,tests,train,encode,decode_eval}_torch241.log`。
独立环境信息：`outputs/validation/environment_torch241.json`。

## 尚未证明

* 13场景×35000步的完整正式实验未执行。
* 未证明各场景PSNR/SSIM/LPIPS、点数和模型大小达到论文报告值。
* 未证明训练、编解码、FPS达到H200平台论文速度。
* 作者代码尚未公开，无法验证未知实现细节或二进制格式相同。

复现时应使用`configs/base.json`和正式数据分辨率，运行`full_eval.py`；
结果来自实际模型和实际测试图像，脚本不读取官方CSV填充实测指标。

## 2026-10-21：屏幕空间增密修改后的验证

增密统计已统一为 `viewspace_points.grad[visible, :2]` 的范数。
当前环境下 11 项测试全部通过，包含真实 CUDA 渲染器及 Torch 参考渲染器的
反向传播、可见点/mask 对齐、XY 分量选择及密度控制决策。
日志：`outputs/validation/screen_adc/tests.log`。

旧三维坐标梯度实验产物及其兼容代码按用户要求删除。本次只运行测试，
没有启动新的正式训练；上面的历史验证记录不能作为修改后版本的质量指标。
