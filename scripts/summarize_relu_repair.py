#!/usr/bin/env python3
"""Read-only consolidation of ReLU repair evidence and full Truck controls."""
import csv
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/truck_relu_repair_20261022'


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def latest_row(path):
    if not path.exists():return None
    for line in reversed(path.read_text().splitlines()):
        try:return json.loads(line)
        except json.JSONDecodeError:continue
    return None


def run_info(folder):
    model=folder/'tandt/truck/base';code=(folder/'exit_code').read_text().strip() if (folder/'exit_code').exists() else None
    summary=read(model/'summary.json')
    result={'root':str(folder),'exit_code':code,'latest_training':latest_row(model/'training.jsonl'),
            'summary':summary,'gates':[read(p) for p in sorted(folder.glob('gate_*.json'),key=lambda p:int(p.stem.split('_')[1]))]}
    result['status']=('training' if result['latest_training'] else 'waiting') if code is None else ('complete' if code=='0' and summary and summary['all_stages_completed'] else 'rejected_or_stopped')
    decoded=read(model/'test/ours_35000/metrics.json')
    if decoded:
        result['decoded_35000']={k:decoded[k] for k in ('views','splats','psnr','ssim','lpips')}
        result['decoded_35000']['lpips_protocol']=decoded.get('lpips_protocol',decoded.get('lpips_status'))
    archive=model/'model.smolgs.zip'
    if archive.exists():result['archive_size_MB']=archive.stat().st_size/1e6
    verification=read(model/'roundtrip.json')
    if verification:result['roundtrip_passed']=verification.get('render_agreement',False)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'official_snapshot/results/TanksAndTemples/truck.csv').open() as stream:
        paper=next(row for row in csv.DictReader(stream) if row['Submethod']=='base')
    paper={'psnr':float(paper['PSNR']),'ssim':float(paper['SSIM']),'lpips':float(paper['LPIPS']),
           'splats':int(paper['#Gaussians']),'size_MB':int(paper['Size [Bytes]'])/1e6}
    control=run_info(ROOT/'outputs/truck_default_nosize_20261022')
    single=run_info(ROOT/'outputs/truck_relu_center05_20261022')
    dual=run_info(ROOT/'outputs/truck_relu_dual_center_20261022')
    gain=run_info(ROOT/'outputs/truck_relu_gain8_20261022')
    sfm=run_info(ROOT/'outputs/truck_relu_sfm_means_20261022')
    phase=run_info(ROOT/'outputs/truck_relu_phase_repair_20261022')
    phase['phase_status']=read(Path(phase['root'])/'status.json')
    if not phase['latest_training']:
        phase['latest_training']=latest_row(Path(phase['root'])/'matched_control/training.jsonl')
        if phase['latest_training']:
            phase['training_branch']='matched_control'
            if phase['exit_code'] is None:phase['status']='training'
    phase_center=run_info(ROOT/'outputs/truck_relu_phase_center_20261022')
    phase_center['phase_status']=read(Path(phase_center['root'])/'status.json')
    phase_center['matched_21000']=read(Path(phase_center['root'])/'matched_21000.json')
    if not phase_center['latest_training']:
        phase_center['latest_training']=latest_row(Path(phase_center['root'])/'matched_control/training.jsonl')
        if phase_center['latest_training']:
            phase_center['training_branch']='matched_control'
            if phase_center['exit_code'] is None:phase_center['status']='training'
    updated_png=read(ROOT/'outputs/truck_default_nosize_20261022/updated_lpips_png/results.json')
    active=next((r for r in (phase_center,phase,gain,sfm,dual) if r['status']=='training'),None)
    base=ROOT/'outputs/truck_octree_fix_20261022'
    activities=[]
    for label,path in [
      ('原ReLU / 7k',base/'old_7k_point_activity.json'),
      ('Kaiming候选 / 7k',base/'new_7k_point_activity.json'),
      ('输入中心化 / 7k',ROOT/'outputs/truck_relu_center05_20261022/activity_7000.json')]:
        q=read(path)
        if q:activities.append({'label':label,'evidence':str(path),'step':q['step'],'groups':q['groups']})
    dual_acts=sorted((ROOT/'outputs/truck_relu_dual_center_20261022').glob('activity_*.json'),key=lambda p:int(p.stem.split('_')[1]))
    dual_path=dual_acts[-1] if dual_acts else ROOT/'outputs/truck_relu_dual_center_20261022/interim_activity.json'
    q=read(dual_path)
    if q:activities.append({'label':f'两层仿射中心化 / {q["step"]}步','evidence':str(dual_path),'step':q['step'],'groups':q['groups']})
    for label,run,pilot in [('等价缩放初始化',gain,'gain8'),('SfM逐位/逐通道均值',sfm,'sfm_means')]:
        folder=Path(run['root']);paths=sorted(folder.glob('activity_*.json'),key=lambda p:int(p.stem.split('_')[1]))
        path=paths[-1] if paths else ROOT/f'outputs/truck_relu_conditioning_pilots_20261022/{pilot}/activity_1000.json'
        q=read(path)
        if q:activities.append({'label':f'{label} / {q["step"]}步','evidence':str(path),'step':q['step'],'groups':q['groups']})
    effects=[]
    for label,candidate in [('20k重置',phase),('20k当前点云中心化',phase_center)]:
        paths=sorted(Path(candidate['root']).glob('activity_*.json'),key=lambda p:int(p.stem.split('_')[1]))
        if paths:
            q=read(paths[-1]);activities.append({'label':f'{label} / {q["step"]}步','evidence':str(paths[-1]),'step':q['step'],'groups':q['groups']})
    for candidate in (dual,gain,sfm,phase,phase_center):
        for p in sorted(Path(candidate['root']).glob('embedding_effect_*/effect.json')):
            e=read(p);e['experiment_root']=candidate['root'];effects.append(e)
    report={'updated_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'user_constraint':'保留论文ReLU；不再启动LeakyReLU训练；保持两项默认权重和论文名义学习率',
            'paper_base':paper,'full_pruning_control':control,'single_center_candidate':single,
            'dual_center_candidate':dual,'gain8_candidate':gain,'sfm_means_candidate':sfm,
            'phase_repair_candidate':phase,'updated_lpips_control_png':updated_png,
            'phase_center_candidate':phase_center,
            'active_run':active,'point_activity':activities,'same_weight_embedding_effects':effects,
            'caveats':['Affine shifts are foldable into biases and add no learned parameters, but change Adam parameter coordinates.',
                       'Both affine centering and positive-homogeneous initialization scaling are independent choices, not confirmed author code.',
                       'Activity gates are engineering diagnostics, not paper hyperparameters or new loss terms.',
                       'Same-weight constant-mean interventions do not replace a retrained no-PE comparison.',
                       'Near-camera-mean group is a geometric grouping, not a visibility or foreground segmentation.']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    lines=['# Truck：保留论文ReLU的修复检查','',f'更新时间：{report["updated_at"]}。','',
      '用户已明确选择保留论文ReLU。所有当前训练保持λ_opacity=3e-7、λ_rate=5e-4及论文名义学习率，不启用LeakyReLU。','',
      '## 已完成的35k剪枝修正参照','',
      '该参照只修正整理阶段的额外尺寸剪枝，保留已知退化的位置网络，不能作为位置分支已修复的证据。','',
      '| 模型 | PSNR | SSIM | LPIPS | 高斯数量 | 完整压缩包MB |','|---|---:|---:|---:|---:|---:|',
      f'| 论文base | {paper["psnr"]:.6f} | {paper["ssim"]:.6f} | {paper["lpips"]:.6f} | {paper["splats"]:,} | {paper["size_MB"]:.6f} |']
    if 'decoded_35000' in control:
        d=control['decoded_35000']
        values=({'psnr':updated_png['ours_35000']['PSNR'],'ssim':updated_png['ours_35000']['SSIM'],'lpips':updated_png['ours_35000']['LPIPS']} if updated_png else d)
        lines += [f'| 剪枝修正参照（32视角，'+('新版LPIPS/解码PNG' if updated_png else '历史协议')+f'） | {values["psnr"]:.6f} | {values["ssim"]:.6f} | {values["lpips"]:.6f} | {d["splats"]:,} | {control["archive_size_MB"]:.6f} |','',
                  f'解码一致性核验通过：{control.get("roundtrip_passed")}；仍差{paper["psnr"]-values["psnr"]:.6f} dB，未宣称复现到论文指标。']
        lines += ['',f'历史浮点参照原LPIPS协议：{d["lpips_protocol"]}。后续统一使用更新后的Inria直接RGB输入协议，旧LPIPS不与新结果混用。']
    if updated_png:
        d=updated_png['ours_35000']
        lines += ['',f'同一批解码PNG按新版重算：PSNR={d["PSNR"]:.6f}，SSIM={d["SSIM"]:.6f}，LPIPS={d["LPIPS"]:.6f}。PNG与浮点渲染指标略有不同。',
                  f'证据：{ROOT}/outputs/truck_default_nosize_20261022/updated_lpips_png/results.json。']
    lines += ['', '## 逐点检查：避免被少量背景活跃点误导','',
      '| 模型/时刻 | 全部点数 | 全体非零点比例 | 相机附近组非零点比例 | 附近组最大通道标准差 |',
      '|---|---:|---:|---:|---:|']
    for a in activities:
        g=a['groups'];x=g['all'];n=g['near_camera_mean']
        lines.append(f'| {a["label"]} | {x["points"]:,} | {x["active_point_fraction"]:.6%} | {n["active_point_fraction"]:.6%} | {max(n["embedding_std_per_channel"]):.6f} |')
    lines += ['', '附近组定义为距训练相机均值不超过2倍固定场景尺度；不等同于可见或前景。',
      '单层输入中心化7k在附近组仅193/582932点保持非零，故按预先记录的规则拒绝继续。全局输出范围大并不能证明主体区域编码有效。','',
      '## 保留ReLU的兼容候选','',
      '第一层输入减去0.5，第二层输入减去初始SfM点对应隐藏特征的逐通道均值；均做偏置补偿以保持初始函数。',
      '`W(x-c)+(b+Wc)=Wx+b`。训练后也可以将固定中心折叠回普通仿射层；仍是48→64→64→8、两个ReLU、相同数量的可学习参数。',
      '固定中心来自初始化点云，不使用测试图像指标；它改变优化器参数坐标，不能称为已确认的作者实现。',
      '这个固定0.5方案也在7k的附近组出现退化，未放行35k。后续逐位中心来自初始SfM的真实位均值，不假定每位均值都为0.5。',
      '另一个候选仅改变ReLU网络的初始化尺度：W1、b1、W2乘8，b2乘64，W3除64，b3保持。由ReLU正齐次性，完整初始函数保持相同；CPU/CUDA测试验证初始输出逐位相同。',
      '等价缩放不增加前向归一化操作或学习参数，但改变初始参数尺度及相对更新幅度。它属于未公开初始化细节的独立试验，不是作者代码的确认值。',
      '每个阶段检查全部点和相机附近组的活动；门槛是诊断停止规则，不是新增训练损失或论文超参数。']
    last=active.get('latest_training') if active else None
    if last:lines += ['',f'最新训练步：{last["step"]}/35000；状态：{active["status"]}。训练日志中的逐帧PSNR不作为最终测试结果。']
    if effects:
        lines += ['', '## 相同权重的位置嵌入均值替换检查','',
          '仅将随位置变化的嵌入替换为全场景均值；保留几何、特征、视角输入及所有权重。它是即时干预，不是重新训练的无位置编码模型。','',
          '| 时刻 | 原网络PSNR | 全局常量 ΔPSNR | 仅附近组常量 ΔPSNR |','|---|---:|---:|---:|']
        for e in effects:
            near=e.get('normal_minus_near_constant')
            lines.append(f'| {e["step"]} | {e["normal"]["psnr"]:.6f} | {e["normal_minus_constant"]["psnr"]:.6f} | '+(f'{near["psnr"]:.6f}' if near else '未测')+' |')
    lines += ['', '## 终端与证据','',
      '`screen -r smolgs_truck_relu_phase_center`查看当前20k点云中心化对照；Ctrl+A再按D退出观察。',
      f'活动日志：[run.log]({active["root"]}/run.log)。' if active else '当前没有已确认的活动训练，查看各实验exit_code与日志。',
      '最新33项测试通过，涵盖初始函数等价、ReLU缩放、修复前后输出/RNG/其他参数不变、Adam状态处理、中心化向量保存/恢复、检查点/压缩往返，以及更新后LPIPS协议。',
      f'测试日志：[tests_phase_center.log]({ROOT}/outputs/validation/relu_center/tests_phase_center.log)。',
      '源码与SHA256均在对应实验的source目录。尚未完成的阶段不能用作最终质量或压缩率结论。']
    lines += ['', '## 20k函数不变的阶段诊断', '',
      '原始20k全部506153点的第二层ReLU均为零，位置嵌入是输出偏置。重置两个隐藏层、清零最后权重、保留最后偏置，要求三视角渲染和属性逐位不变，并保持其余参数及全局RNG。',
      '该一次性修复不是论文确认的步骤。原检查点和修复检查点分别继续到21k；若位置活动通过诊断门槛，再继续25k、30k、35k和完整压缩评测。',
      '后续执行使用冻结源码和新版LPIPS，两项权重与名义学习率均不变。',
      f'不做中心化的阶段重置状态：{json.dumps(phase.get("phase_status"),ensure_ascii=False)}。',
      '不做中心化的21k修复只提高0.009835dB，全部点活动率1.36%，位置均值替换ΔPSNR仅0.0000049dB，已停止。',
      '下一候选对当前20k点云计算固定逐位/逐通道均值；配置使用checkpoint来源标记，区别于初始SfM均值。仍要求修复时三视角逐位一致。',
      f'当前点云中心化阶段状态：{json.dumps(phase_center.get("phase_status"),ensure_ascii=False)}。']
    if phase_center['matched_21000']:
        m=phase_center['matched_21000'];c=m['control'];r=m['repair']
        lines += ['', '| 21k同条件对照（新版LPIPS，32视角） | PSNR | SSIM | LPIPS |',
          '|---|---:|---:|---:|',f'| 原模型 | {c["psnr"]:.6f} | {c["ssim"]:.6f} | {c["lpips"]:.6f} |',
          f'| 当前点云中心化修复 | {r["psnr"]:.6f} | {r["ssim"]:.6f} | {r["lpips"]:.6f} |',
          '21k整体画质基本相同，不能将活动恢复称作质量提升。相同权重下仅附近组嵌入均值替换使PSNR下降0.015465dB，说明该分支开始影响渲染；仍需后续阶段验证。',
          '两次未修复控制重跑的21k PSNR也相差约0.0068dB；CUDA渲染反传含并行累加，不把这一量级的小差异作为可靠收益。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(OUT/'report.md'),'active_status':active['status'] if active else None,'latest_step':last['step'] if last else None},ensure_ascii=False))

if __name__=='__main__':main()
