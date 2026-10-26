# 本机仓库参考记录

| 参考仓库与文件 | 采用的内容 | 与Smol-GS的区别 |
|---|---|---|
| `gaussian-splatting/scene/colmap_loader.py` | 原始COLMAP二进制/文本解析，保留源代码与版权 | 无 |
| `gaussian-splatting/scene/dataset_readers.py` | 相机旋转、训练测试排序和每8张留一 | 不引入后续depth/exposure扩展 |
| `HAC/scene/gaussian_model.py` | 3D+XY/XZ/YZ混合哈希、量化步长调整方式 | 去掉anchor、offset、mask；f=8，s=3 |
| `HAC/utils/encodings.py` | Binary STE、哈希层内存布局 | 保留二值±1及梯度区间，提供CPU参考 |
| `HAC/utils/encodings_cuda.py` | 原始Gaussian CUDA算术编码/解码 | 只编码Smol-GS的f、s |
| `HAC/submodules/*` | 光栅器、3邻近距离、哈希CUDA核、算术CUDA核 | 复制源码，重新编译，不复制.so |
| `HAC/arguments/__init__.py`、`HAC-plus/arguments/__init__.py` | --source_path、--model_path、--feat_dim、学习率命名 | 默认值逐项改为Smol-GS附录 |
| `HAC/run_shell_mip360.py`、`run_shell_tnt.py`、`run_shell_db.py` | 数据集场景列表、本机data路径、outputs分层组织 | 35000步、small/base/large目录 |
| `Scaffold-GS/arguments/__init__.py`、`train_mip360.sh`、`single_train.sh` | 训练入口、评估入口、ADC周期等工程习惯 | 不使用其多offset模型和锚点增长策略 |
| `LocoGS/arguments/__init__.py`、`gaussian_renderer/__init__.py` | CUDA渲染接口、renderer返回项和opacity/scale参数惯例 | 不搬入LocoGS mask及SH机制 |
| `SpatialShape-GS/README.md`、目录结构 | 编码、checkpoint、验证文档的分离方式 | 不借用其方法模型或结果 |

本项目的实际代码入口是`scene/gaussian_model.py`、`gaussian_renderer/__init__.py`、
`arguments/__init__.py`，这些目录不是仅为外观创建的包装层。
项目特有的完整码流容器和评估组织保留在`smolgs/codec.py`与`smolgs/evaluation.py`。

2026-10-21已进一步逐函数对照官方3DGS、HAC、Scaffold-GS的增密、尺度控制、
统计与Adam状态。具体故障证据、实际修改及与Smol-GS论文的差异见
[REFERENCE_ADC_AUDIT.md](REFERENCE_ADC_AUDIT.md)。对照源码及SHA-256见
`reference_audit/manifest.json`；它们只供审计，不是新增运行依赖。
