#!/usr/bin/env python3
"""Summarize completed default-weight pruning replays and counterfactuals."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.audit_compaction import readonly_view, measure, write_json
from scene.dataset_readers import load_scene
from utils.checkpoint import load_checkpoint
from utils.loss_utils import LPIPS


@torch.no_grad()
def no_size_counterfactual(path, cameras, metric):
    model, _, step, _ = load_checkpoint(path, 'cuda')
    model.eval()
    size = model.log_scales.amax(-1).exp()
    finite = (torch.isfinite(model.xyz).all(-1) & torch.isfinite(model.log_scales).all(-1)
              & torch.isfinite(size) & torch.isfinite(model.features).all(-1))
    mean_opacity = model.opacity_accum/model.vis_count.clamp_min(1)
    low = (model.vis_count > 0) & (mean_opacity < model.config.prune_opacity)
    invisible = step-model.last_seen >= model.config.invisible_patience
    result = measure(readonly_view(model, ~(low | invisible | ~finite), True), cameras, step, metric)
    return {'step': step, 'case': 'F_opacity_invisible_new_bounds', **result,
            'note': 'Same-weight immediate omission of the extra size rule; NOT a retrained policy result.'}


def decomposition(data):
    cases = data['cases']
    p = {key: value['psnr'] for key, value in cases.items()}
    return {'step': data['step'],
            'opacity_invisible_deletion_fixed_bounds': p['B_opacity_invisible_old_bounds']-p['A_before'],
            'additional_size_deletion_fixed_bounds': p['C_all_rules_old_bounds']-p['B_opacity_invisible_old_bounds'],
            'refresh_bounds_same_survivors': p['D_all_rules_new_bounds']-p['C_all_rules_old_bounds'],
            'total_immediate_change': p['D_all_rules_new_bounds']-p['A_before'],
            'effective_rule_immediate_gain': p['E_effective_size_new_bounds']-p['D_all_rules_new_bounds']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, cameras = load_scene('/root/autodl-tmp/data/tandt/truck', 'images', 1, 8, False)
    metric = LPIPS('cuda')
    report = {'lambda_opacity': 3e-7, 'lambda_rate': 5e-4, 'scope': '15k--20k only', 'arms': {},
              'notes': ['Four fixed test views are diagnostic probes, not full benchmark results.',
                        'Final and worst-event counterfactuals use all32 test views and VGG LPIPS.',
                        'Ordered PSNR differences separate deletion and changed positional encodings at identical weights.',
                        'Size/opacity/invisibility condition counts overlap. Histogram bins are mutually exclusive.',
                        'Effective/no-size candidate hit totals can count a retained point more than once; these are event hits, not unique deletions.',
                        'No point-ID lineage or 35k result is inferred beyond the measured replay.']}
    for arm in ('baseline', 'effective', 'no_size'):
        path = root/arm
        if not (path/'audit_summary.json').exists():
            continue
        summary = json.loads((path/'audit_summary.json').read_text())
        rows = [json.loads(line) for line in (path/'adc.jsonl').read_text().splitlines() if line.strip()]
        histogram = np.array([r['reason_histogram_all_codes'] for r in rows], dtype=np.int64).sum(0).tolist()
        arm_data = {'summary': summary, 'reason_histogram': histogram, 'counterfactuals': {},
                    'baseline_replay_caveat': 'Same parent and RNG state; CUDA floating-point reductions can still vary. Historical agreement is measured separately.'}
        for name in ('worst', 'final'):
            details = json.loads((path/f'{name}_counterfactual.json').read_text())
            no_size = no_size_counterfactual(path/f'{name}_before_adc.pt', cameras, metric)
            write_json(path/f'{name}_no_size_counterfactual.json', no_size)
            arm_data['counterfactuals'][name] = {
                'step': details['step'], 'decomposition': decomposition(details),
                'cases': {key: {k: v for k, v in value.items() if k != 'per_view'}
                          for key, value in details['cases'].items() if not key.startswith('F_')},
                'F_no_size': {key: value for key, value in no_size.items() if key != 'per_view'}}
        actual_case = {'baseline': 'D_all_rules_new_bounds', 'effective': 'E_effective_size_new_bounds',
                       'no_size': 'F_no_size'}[arm]
        cf_final = arm_data['counterfactuals']['final']
        cf_metric = cf_final['F_no_size'] if actual_case == 'F_no_size' else cf_final['cases'][actual_case]
        agreement = {key: cf_metric[key]-summary['metrics_20000'][key] for key in ('psnr', 'ssim', 'lpips')}
        arm_data['counterfactual_vs_actual_20k'] = agreement
        assert abs(agreement['psnr']) < 1e-4, agreement
        report['arms'][arm] = arm_data
    historical_path = Path('outputs/truck_sizeaware_base_20261021/tandt/truck/base/evaluation/iteration_20000/metrics.json')
    historical = json.loads(historical_path.read_text())
    historical = {k: v for k, v in historical.items() if k != 'per_view'}
    report['historical_baseline_20k'] = historical
    report['replay_vs_historical'] = {
        key: report['arms']['baseline']['summary']['metrics_20000'][key]-historical[key]
        for key in ('psnr', 'ssim', 'lpips', 'splats')}
    for name in ('bounds_encoding_control', 'octree_activity_control', 'gpu_octree_activity_control',
                 'baseline_candidate_profiles'):
        path = root/(name+'.json')
        if path.exists():
            report[name] = json.loads(path.read_text())
    if 'effective' in report['arms']:
        base = report['arms']['baseline']['summary']['metrics_20000']
        trial = report['arms']['effective']['summary']['metrics_20000']
        report['effective_vs_replay'] = {key: trial[key]-base[key] for key in ('psnr', 'ssim', 'lpips', 'splats')}
    if 'no_size' in report['arms']:
        base = report['arms']['baseline']['summary']['metrics_20000']
        trial = report['arms']['no_size']['summary']['metrics_20000']
        report['no_size_vs_replay'] = {key: trial[key]-base[key] for key in ('psnr', 'ssim', 'lpips', 'splats')}
    write_json(root/'report.json', report)
    lines = ['# Truck：15k–20k 剪枝归因检查', '',
             '默认权重固定为 λ_opacity=3e-7、λ_rate=5e-4。所有训练止于20k，未开展35k压缩结果评测。', '',
             '## 完整32视角的20k结果', '',
             '| 运行 | 点数 | PSNR | SSIM | LPIPS |', '|---|---:|---:|---:|---:|']
    for label, values in [('原历史base', historical)]+[
        (arm, data['summary']['metrics_20000']) for arm, data in report['arms'].items()]:
        lines.append(f'| {label} | {values["splats"]:,} | {values["psnr"]:.6f} | '
                     f'{values["ssim"]:.6f} | {values["lpips"]:.6f} |')
    lines += ['', '## 条件归因', '',
              '条件计数可以重叠；baseline中的被删除点不会再次出现。effective与no_size中保留的尺寸候选会反复命中，因此候选累计数不是独立删点数。', '']
    for arm, data in report['arms'].items():
        s = data['summary']; h = data['reason_histogram']
        lines += [f'### {arm}', '',
                  f'- 实际删除：{s["actual_removed"]:,}。低opacity条件命中：{s["low_opacity_total"]:,}；'
                  f'不可见条件命中：{s["invisible_total"]:,}；控制器尺寸条件命中：{s["controller_oversized_total"]:,}。',
                  f'- 仅尺寸条件命中：{s["size_only_total"]:,}；其中全部219个训练视角的实际轴尺度均未超限：'
                  f'{s["size_only_physically_small_total"]:,}。',
                  f'- 互斥组合：仅opacity={h[1]:,}，仅不可见={h[2]:,}，opacity+不可见={h[3]:,}，'
                  f'仅尺寸={h[4]:,}，opacity+尺寸={h[5]:,}，不可见+尺寸={h[6]:,}，三者={h[7]:,}。',
                  f'- 全部50次诊断保持训练RNG不变：{s["rng_preserved_at_all_events"]}。', '',
                  '| 检查点/事件 | 删低opacity/不可见点 ΔPSNR | 再删尺寸点 ΔPSNR | 刷新包围盒 ΔPSNR | 总变化 |',
                  '|---|---:|---:|---:|---:|']
        for name, c in data['counterfactuals'].items():
            d = c['decomposition']
            lines.append(f'| {name} / {c["step"]} | {d["opacity_invisible_deletion_fixed_bounds"]:.6f} | '
                         f'{d["additional_size_deletion_fixed_bounds"]:.6f} | '
                         f'{d["refresh_bounds_same_survivors"]:.6f} | {d["total_immediate_change"]:.6f} |')
        c = data['counterfactuals']['final']
        lines += ['', '末次ADC的同权重反事实；**不是重新训练结果**：', '',
                  '| 条件 | 点数 | PSNR | SSIM | LPIPS |', '|---|---:|---:|---:|---:|']
        for label, values in list(c['cases'].items())+[('F_no_size', c['F_no_size'])]:
            lines.append(f'| {label} | {values["splats"]:,} | {values["psnr"]:.6f} | '
                         f'{values["ssim"]:.6f} | {values["lpips"]:.6f} |')
        lines += ['']
    if 'octree_activity_control' in report:
        lines += ['## 包围盒对照的解释与附带发现', '',
                  'CPU独立检查确认：20k剪枝前模型全部451987个点的MLP_oct第二个ReLU层均为零，'
                  '位置嵌入输出恒定。15k和原35k检查点各65536个抽样点也出现相同现象。',
                  '因此，本次刷新包围盒的即时影响接近零，不能推论位置编码更新在健康模型中也没有影响。',
                  '这只确认当前网络活动异常；尚未改变初始化、激活或学习率，也没有证明修复后能达到论文指标。',
                  '详见octree_activity_control.json和bounds_encoding_control.json。', '']
        if 'gpu_octree_activity_control' in report:
            gpu = report['gpu_octree_activity_control']
            lines += [f'CUDA复核：全部{gpu["evaluated_points"]}个点的第二层ReLU正值数为'
                      f'{gpu["second_relu_positive_elements"]}，最大激活前数值为'
                      f'{gpu["maximum_second_layer_preactivation"]:.6f}，输出范围为'
                      f'{gpu["embedding_range_max"]:.6f}。该异常不只是CPU/GPU舍入差异。', '']
    if 'baseline_candidate_profiles' in report:
        g = report['baseline_candidate_profiles']['groups']
        lines += ['## 仅尺寸删点的属性', '',
                  f'控制器最大轴中位数为{g["size_controller"]["quantiles"][1]:.6f}，'
                  f'全部训练视角实际最大轴中位数为{g["size_effective"]["quantiles"][1]:.6f}，'
                  '阈值约0.584792。它们多数是略超当前尺寸阈值的点，不是此前上万尺度的异常点。',
                  f'窗口平均opacity中位数为{g["size_opacity"]["quantiles"][1]:.6f}；'
                  '其中0表示本窗口无观测，不能解释为测得opacity为零。',
                  f'到训练相机平均中心的距离中位数为{g["size_distance"]["quantiles"][1]:.6f}。', '']
    lines += ['## 证据边界', '',
              '同权重即时反事实能定位一次剪枝的损伤，不能代替改变策略后重新训练的结果。',
              'effective分支只改变compaction中的尺寸判据，其余训练设置相同；没有恢复论文随机克隆策略。',
              'F_no_size是每个快照的即时反事实，不是35k结果；若存在no_size分支，其20k实际训练结果另列在总表中。',
              '每点记录见candidates_*.npz；原15k点ID可追溯。详细视角结果见各counterfactual JSON。',
              '历史结果与重放的差异、所有条件计数和对比指标见report.json。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
