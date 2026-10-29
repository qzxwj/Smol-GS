# 坐标碰撞代表点对照

默认仍是 `--coordinate-selection first`。新增实验选项
`--coordinate-selection opacity`：在坐标冻结前，使用全部训练相机统计

`score_i = sum_k(opacity_i(camera_k) * 1[radii_i(camera_k) > 0]) / K`。

评分不读取真值图像，不使用测试相机；按固定训练相机顺序以 float64 累加。
使用冻结前坐标和 eval 模式的特征量化。`radii > 0` 是现有光栅器的投影预筛选，
不是遮挡判断或最终像素贡献。每格保留最高分，同分保留冻结前索引最小的点；
从未投影可见的格子全为零分，回退到首点。参数、Adam 状态和 VQ 索引缓存同步迁移。

评分恢复所有模块的 training 状态、Python/NumPy/Torch/CUDA RNG 和 VQ 索引缓存，
避免改变后续训练的随机数序列。训练入口将评分、可见视角数和相机名保存到
`coordinate_opacity_scores.pt`，策略写入 run.json 和检查点 metadata。
续训默认沿用该策略；冻结后禁止切换策略。

独立即时对照：

```bash
python scripts/compare_coordinate_selection.py \
  --checkpoint <30000步且未冻结的训练检查点> --output <新的输出目录>
```

两个分支使用相同源检查点、网格、保留数量和测试相机，仅改变每格选择的代表点。
保存检查点 SHA-256、代码 SHA-256、评分、可见次数、保留原索引和逐视角差值。
冻结后的可续训检查点在评估之前保存，防止 LPIPS 构造消耗 RNG 干扰对照。

此脚本只做即时评估，不更新参数。预先固定后续训练条件：PSNR 提升至少 0.05 dB，
且 SSIM 不降低、LPIPS 不升高，才继续相同 5000 步。测试指标用于这次探索性实验的
继续/停止决定，不用于评分或挑选视角；不能将结果视为独立于测试集选择的确认性结论。
它是本地实验策略，不宣称复现作者尚未公开的碰撞代表点选择方法。
