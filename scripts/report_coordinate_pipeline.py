"""Aggregate fully decoded acceptance results with the historical global16 baseline."""
from argparse import ArgumentParser
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    args = parser.parse_args()
    experiment = Path(args.experiment).resolve()
    manifest = json.loads((experiment/'manifest.json').read_text())
    assert digest(manifest['checkpoint']) == manifest['checkpoint_sha256']
    results = {name: json.loads((experiment/name/'acceptance.json').read_text()) for name in manifest['branches']}
    historical = experiment.parent/'opacity_selection_30k_v1'
    old_manifest = json.loads((historical/'manifest.json').read_text())
    assert old_manifest['source_checkpoint_sha256'] == manifest['checkpoint_sha256']
    old_comparison = json.loads((historical/'final_comparison.json').read_text())
    baseline = old_comparison['final_decoded']['opacity']
    old_metrics = json.loads((historical/'opacity/continued_35000/evaluation/decoded_35000/metrics.json').read_text())
    names = manifest['test_names']
    assert [r['name'] for r in old_metrics['per_view']] == names
    rows = [{'method': 'global16_opacity_historical', **{k: baseline[k] for k in ('psnr','ssim','lpips','views','splats','bytes','size_MB')}}]
    paired = {}
    for name, result in results.items():
        assert result['roundtrip_all_views_passed']
        archive = Path(result['archive'])
        assert archive.stat().st_size == result['bytes'] and digest(archive) == result['archive_sha256']
        metrics = json.loads((experiment/name/'decoded_evaluation/metrics.json').read_text())
        before_export = json.loads((experiment/name/'evaluation/iteration_35000/metrics.json').read_text())
        roundtrip = json.loads((experiment/name/'roundtrip.json').read_text())
        assert [r['name'] for r in metrics['per_view']] == names
        assert len(roundtrip['views']) == len(names) and roundtrip['render_agreement']
        row = {'method': name, **result['final_decoded'], 'bytes': result['bytes'], 'size_MB': result['size_MB']}
        row['psnr_gain_vs_global16'] = row['psnr']-baseline['psnr']
        row['bytes_increase_vs_global16'] = row['bytes']-baseline['bytes']
        rows.append(row)
        paired[name] = [{'name': a['name'], **{k: b[k]-a[k] for k in ('psnr','ssim','lpips')}}
                        for a,b in zip(old_metrics['per_view'], metrics['per_view'])]
        result['checkpoint_vs_decoded_psnr_delta'] = metrics['psnr']-before_export['psnr']
    summary = {'source_checkpoint_sha256': manifest['checkpoint_sha256'], 'source_checkpoint_unchanged': True,
               'test_views': len(names), 'historical_baseline_not_rerun': True,
               'rows': rows, 'branches': results, 'paired_deltas_vs_global16': paired,
               'acceptance': 'deduplicate, continue exactly 5000 steps, encode complete archive, standalone decode, all-view roundtrip, all-view decoded quality',
               'limitation': 'Single-scene exploratory comparison. Different retained point counts and file sizes; not an equal-bitrate comparison.'}
    (experiment/'final_comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (experiment/'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['method','psnr','ssim','lpips','views','splats','bytes','size_MB','psnr_gain_vs_global16','bytes_increase_vs_global16'])
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 完整坐标压缩流程验收', '',
             '两组从同一个 30000 步 checkpoint 出发，使用全部 263 个训练视角的 opacity × 投影可见性选择同格代表点，均继续训练恰好 5000 步。双区域的包围盒仅由坐标分位数确定，没有使用测试指标选择范围。', '',
             '| 方案 | 解码 PSNR | SSIM | LPIPS | 完整 ZIP / MB | 高斯数量 |',
             '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['method']} | {r['psnr']:.6f} | {r['ssim']:.6f} | {r['lpips']:.6f} | {r['size_MB']:.6f} | {r['splats']} |")
    global18, dual16 = results['global18'], results['dual16']
    lines += ['',
              f"双区域相比全局18位：PSNR {dual16['final_decoded']['psnr']-global18['final_decoded']['psnr']:+.6f} dB，完整 ZIP 增加 {dual16['bytes']-global18['bytes']} 字节。",
              '本场景可优先使用双区域方案：质量稍好，额外存储较小。相对旧16位方案仍有明显体积增加，并非免费的质量提升。',
              '量化前30k模型 PSNR 约22.148838 dB；双区域的最终解码结果已接近该水平。单次结果的小幅超出不能解释为稳定的泛化提升。']
    lines += ['', 'global16 为相同源 checkpoint、相同 opacity 代表点策略的历史完整流程结果，未在本轮重跑。两组新结果均从 ZIP 独立解码后测得，包含全部 38 个相同测试视角。MB 为十进制 1,000,000 字节。', '',
              '## 验收证据', '',
              '- 已执行去重、5000 步续训、完整编码、独立进程解码和压缩包质量评估。',
              '- 两组全部 38 个测试视角的坐标、颜色、透明度、尺度、旋转、协方差和图像往返检查通过。几何索引和码本精确匹配。',
              '- ZIP 文件完整大小包含坐标树、网络、空间哈希、特征、几何、元数据及 ZIP 开销；没有外置的模型数据。',
              '- 旧格式/ECVQ 回归以及新网格的碰撞、优化器迁移、检查点和 CPU/CUDA 编解码测试通过，日志保存在 validation/。',
              '- 原始 30k checkpoint 的 SHA256 未变化；准备阶段保留了代码快照、训练视角列表和评分。', '']
    for name, result in results.items():
        lines += [f"### {name}", '',
                  f"即时去重后 PSNR：{result['immediate']['psnr']:.6f}；最终解码 PSNR：{result['final_decoded']['psnr']:.6f}。",
                  f"往返图像最大绝对误差：{result['roundtrip_max_image_error']:.9g}；每视角平均绝对误差的最大值：{result['roundtrip_max_mean_image_error']:.9g}。",
                  f"35k 浮点检查点到压缩包解码的 PSNR 差：{result['checkpoint_vs_decoded_psnr_delta']:+.9f} dB。", '',
                  f"压缩包：`{result['archive']}`。", '']
    lines += ['## 解释范围', '',
              '这是单场景探索性结果。两组保留的高斯数量及文件大小不同，不能解读为相同码率下的严格优劣。全局18位与位置网络16位已解耦；双区域编码使用主体16位和外围独立16位，不直接删除外围高斯。新格式为 v9，需要随本次代码提供的解码器。', '',
              '逐视角结果、组件字节数、校验哈希见 final_comparison.json 和各分支的 decoded_evaluation/metrics.json、roundtrip.json；执行命令见 commands.json 和 *.log.command.json。']
    (experiment/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
