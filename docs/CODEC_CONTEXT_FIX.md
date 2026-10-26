# Truck导出一致性故障：上下文批次改变量化整数

2026-10-22，修复范围为已完成模型的导出和后续评测，不重新训练。

原运行位于`outputs/truck_sizeaware_base_20261021`。训练完成35000步，
446219个点，随后在`verify_roundtrip.py`属性核验中失败。

## 已确认原因

1. 坐标和八叉树cells逐元素相同，所有算术码流的量化整数SHA-256校验通过。
2. 检查点`attributes()`按整场景计算entropy prior；v2编码器按自适应小块重新计算。
   CUDA矩阵乘法随批次形状产生浮点差异，在舍入边界处会改变量化整数。
3. 第411790个点的第2号特征：整场景预测q约0.752013564，原始f/q约2.500000238，
   舍入为3；块内预测q约0.752013803，舍入为2。因此量化特征相差约0.752，
   该点颜色最大差0.0409933，触发属性核验。其余坐标没有改变。

详见`outputs/validation/codec_truck_sizeaware/diagnosis.json`与`diagnosis.log`。
这是上下文计算约定不一致，不是GPU显存不足，也不是训练中断。

## 修改

新包使用`smolgs-archive-v3`、`context_layout=full-scene-v1`。编码器和解码器
按完整场景计算一次prior，随后分块切片用于算术编码，保证与现有检查点推理采用相同的
计算批次。自适应字母表分割仍保留；旧v1/v2解码继续按其原块布局预测CDF。
整场景预测会增加编解码时的上下文暂存内存；不改变训练或增加模型码流中的逐点上下文数据。

没有放宽`verify_roundtrip.py`的属性或图像误差阈值。
新增CPU/CUDA回归测试模拟一步ULP的批次差异并放置半整数边界特征，
确认不同码流块大小不会改变新包量化整数。全部17项测试通过。

## 重跑与留档

`scripts/finish_truck_after_codec_fix.sh`在screen中执行：重导出最终检查点、
原标准核验、独立解码渲染全部测试图像、浮点与PNG指标、官方CSV对比。
它只对完成35000步的检查点操作。

原失败包保存在`model.before_context_fix.smolgs.zip`，原summary保存在
`summary.before_context_fix.json`；初始训练的`source/`快照不变。
修复后的源码另存到本次运行的`postprocess_source/`，对应时间与SHA-256可追溯。
训练日志追加后处理记录，详细进度另存`postprocess.log`，退出状态见`exit_code`。

## 已完成验证

重新导出的v3包通过未修改阈值的属性、图像与整数符号核验。
前三张核验视角的颜色、opacity、rotation逐元素一致，尺度最大误差1.5259e-5；
图像最大绝对误差不超过0.001946，平均绝对误差不超过4.581e-8。
全部32张测试图像已独立解码渲染并评测，PNG指标和官方CSV对比也已完成，退出码0。

最终浮点指标：PSNR=24.447070，SSIM=0.859894，LPIPS(VGG)=0.231476。
高斯数446219，整个压缩包3503254字节（3.503254 MB）。
相对官方Truck base的PSNR=25.888786，仍低1.441717 dB，不能声称达到论文质量。
运行目录下的`completion_report.json`、`paper_comparison.md`和`roundtrip.json`可核验结果。
