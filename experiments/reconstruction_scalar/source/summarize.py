from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'legacy_runtime/source/scripts'))
from restoration_metrics import quality
ids=[2018,43033,235098];seeds=[101,202,303];methods=['FROST_original','SExp','Bernstein','Bernstein_QMC','Bernstein_stratified','LocalTT33_stratified']
allrows=[]
for iid in ids:
 archive=np.load(HERE/'legacy_runtime/runs'/f'image_{iid}.npz');truth=archive['truth']
 for seed in seeds:
  allrows.append({'image_id':iid,'method':'FROST_original','seed':seed,'L':4096,**quality(archive[f'FROST_L4096_S{seed}'],truth,'reconstruction'),'provenance':'archived original predictions; seed 101 replayed exactly in stage 2'})
 for method in methods[1:]:
  for seed in seeds:
   row=json.loads((HERE/'results'/f'{iid}_{method}_S{seed}.json').read_text());allrows.append({k:row[k] for k in ['image_id','method','seed','L','psnr','ssim','mse']})
summary=[]
for iid in ids:
 for method in methods:
  rows=[r for r in allrows if r['image_id']==iid and r['method']==method]
  s={'image_id':iid,'method':method,'n_seeds':len(rows)}
  for metric in ['psnr','ssim','mse']:
   s[metric+'_mean']=float(np.mean([r[metric] for r in rows]));s[metric+'_SD']=float(np.std([r[metric] for r in rows],ddof=1))
  summary.append(s)
with (HERE/'results/metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['image_id','method','seed','L','psnr','ssim','mse','provenance']);w.writeheader();w.writerows(allrows)
(HERE/'results/summary.json').write_text(json.dumps(summary,indent=2))
comparisons=[]
for iid in ids:
 new=next(r for r in summary if r['image_id']==iid and r['method']=='LocalTT33_stratified')
 for control in ['FROST_original','SExp','Bernstein','Bernstein_QMC','Bernstein_stratified']:
  old=next(r for r in summary if r['image_id']==iid and r['method']==control)
  a=sorted([r for r in allrows if r['image_id']==iid and r['method']=='LocalTT33_stratified'],key=lambda r:r['seed']);b=sorted([r for r in allrows if r['image_id']==iid and r['method']==control],key=lambda r:r['seed'])
  comparisons.append({'image_id':iid,'control':control,'delta_psnr_mean':new['psnr_mean']-old['psnr_mean'],'delta_ssim_mean':new['ssim_mean']-old['ssim_mean'],'paired_psnr_differences':[x['psnr']-y['psnr'] for x,y in zip(a,b)],'paired_psnr_wins':sum(x['psnr']>y['psnr'] for x,y in zip(a,b))})
(HERE/'results/paired_comparisons.json').write_text(json.dumps(comparisons,indent=2))
sel=json.loads((HERE/'results/selected.json').read_text())
meannew=np.mean([r['psnr_mean'] for r in summary if r['method']=='LocalTT33_stratified']);meanold=np.mean([r['psnr_mean'] for r in summary if r['method']=='FROST_original'])
text='''FROST 算法改进实验：探索性阶段报告

结论
已得到显著改善的局部状态 TT 实验版，但没有实现“所有图像、所有指标均超过 SC 对照”的目标。
参数与采样器用校准验证分数选择，没有按三张图的重构 PSNR 选模型。
改进版超过 SExp 的三图 PSNR/SSIM；相对原 Bernstein，三图平均 PSNR 中两张略高、一张略低；相对同样经过分层采样改进的 Bernstein，三张图平均 PSNR/SSIM 均略低。
不能将这些结果写成 FROST 全面优于 SC 方法，也不能将数值期望诊断列为实际有限流性能。

实际三种子结果（L=4096；均值 ± 样本标准差）
'''
for iid in ids:
 text+=f'\n图像 {iid}\n'
 for method in methods:
  r=next(x for x in summary if x['image_id']==iid and x['method']==method)
  text+=f"  {method:25s} PSNR {r['psnr_mean']:.4f} ± {r['psnr_SD']:.4f} dB; SSIM {r['ssim_mean']:.6f} ± {r['ssim_SD']:.6f}\n"
text+=f'''\n跨三张图的描述性平均 PSNR：原 FROST {meanold:.4f} dB；局部状态 TT {meannew:.4f} dB；差值 {meannew-meanold:.4f} dB。不是数据集置信区间，不是独立模型重复训练。

实际完成
- 检查原 FROST 误差：在保留其他有限流运算时，以原 TT 期望替代采样只带来有限提升；精确 softmax 诊断显示主要损失确实来自原注意力算子。
- 保留原几何稳态 FSM 的改进初始化/训练，以及不同地址数、TT rank、编码、合成分数增强的局部状态 TT 候选。所有候选和训练历史保留。
- 实现实际有限事件的 Sobol TT 采样、按隐状态分组的分层 TT 采样；也给 Bernstein 加入 Sobol 和分层版本。
- 固定 12 个独立图像模型产生校准激活、4 个不同模型产生验证激活。三张展示图不用于这些参数的拟合或选择。训练只使用分数的 softmax 目标，不读取这三张图的目标像素。
- 新模型与对照复用相同冻结 Transformer 权重、输入、裁剪、非注意力随机算子。图像模型本身仍各自拟合目标图像，不是未见图像恢复。
- 比较使用 stream seeds 101/202/303；有限验证选择使用 192 个验证分数行和 seeds 1123/2246/3369。
- 选择基于有限流概率 RMSE；早期逐模型 checkpoint 选择基于期望概率 RMSE。开发中观察到采样误差主导后增加有限流选择，属于探索性协议调整，不宣称预注册。
- 中间 33 地址、rank 16 的图像消融先于最终有限验证选择运行；该消融的 PSNR 未作为选择目标。最终独立验证也选中了同一 checkpoint。

选定算法
1. 对每个长度为 16 的 score row 排序、减最大值，将差值裁剪到 [-8,0]。
2. 映射到 33 个均匀地址的相邻两点；分数的小数部分控制在两地址间的概率选择。
3. 使用 rank 16 的逐行归一化 TT 核。按当前隐状态分组，使用随机偏移的分层阈值产生实际状态选择和类别事件。
4. 每一级均守恒 L 个粒子；最终类别计数总和恰为 L。解码后的计数/L 接入原 SC AV 运算。

重要算法变更
这保留了多变量 TT 联合类别表示，但已替换原先的 16 状态饱和随机游走输入 FSM；它不是仅重新训练了原版 FROST。
新编码使用数值分数的粗地址与小数部分；分层采样涉及分组计数、整数除法和阈值控制。不能将其开销当作免费，也不能声称保持原门级接口、burn-in 或逐周期调度。
选定模型：{sel['checkpoint']}，SHA256 {sel['checkpoint_sha256']}。
原 TT 参数 16,512；新模型 127,248，为原来的 {127248/16512:.4f} 倍。
按 float32 参数数组计算，原纯参数为 64.5 KiB，新为 497.0625 KiB；不是 FPGA BRAM 实测。C++ CDF 表有额外填充及 64-bit 端点表示。
本阶段没有运行新的 RTL、综合、布局布线、功耗或能耗实验。任何旧版 FROST PPA 都不能移植到此版本。

边界和后续解释
这些对照是经典 SC 指数原语与数值计数归一化的适配，其他网络模块仍来自原 FROST；不是完整复现已发表加速器。
Bernstein 的新分层版本只是适配原语的额外强化控制，不能冒称文献原算法。
相同 L 不保证相同随机源数量、执行周期、面积或能耗；elapsed_CPU_seconds 只是仿真墙钟时间。
独立校准用的 12 个图像和 4 个验证图像应在未来独立图像评估中排除，或单独注明角色；不可将整个 200 图池都重新称为未参与算子开发的测试集。
只有三个目标和三个流种子，细小的 PSNR 差异不足以证明普遍优势。本轮不修改论文主稿、理论定义或硬件图，以免混用原版与新机制。
目前适合的结论是：局部状态编码、校准数据覆盖与分层采样能显著缩小原 FROST 的重构质量差距；全面优于强 SC 对照及资源优势仍未得到验证。

复现
包中保存全部新源代码、候选 checkpoint、校准分数、固定运行所需原始模型、实际预测、配对结果和校验记录。
README.txt 提供命令。所有失败/较差候选都保留；诊断结果的文件名明确含 DIAGNOSTIC 或 diagnosis。
'''
(HERE/'REPORT_zh.txt').write_text(text)
(HERE.parent/'FROST_algorithm_improvement_report.txt').write_text(text)
print(text[:5500])
if __name__=='__main__':pass
