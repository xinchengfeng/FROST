from __future__ import annotations
import hashlib,json,math,platform
from pathlib import Path
import numpy as np,pandas as pd
from study import ROOT

def tab(headers,rows):
 return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)
def ms(d,k,scale=1,dec=4):return f'{d[k].mean()*scale:.{dec}f} ± {d[k].std()*scale:.{dec}f}'

def main():
 R=ROOT/'results';m=pd.read_csv(R/'main_metrics.csv');s=pd.read_csv(R/'stream_metrics.csv')
 b=pd.read_csv(R/'baseline_metrics.csv');signed=pd.read_csv(R/'signed_tt_diagnostic.csv')
 occ=pd.read_csv(R/'signed_occupancy_diagnostic.csv');invar=pd.read_csv(R/'invariance_metrics.csv')
 stress=pd.read_csv(R/'stress_metrics.csv');quant=pd.read_csv(R/'core_quantization_metrics.csv')
 validation=json.loads((R/'validation.json').read_text())
 d=m[(m.kind=='strict')&(m.n==16)&(m.states==4)]
 rankrows=[]
 for rank in [2,4,8,16,32]:
  x=d[d['rank']==rank];rankrows.append([rank,int(x.parameters.iloc[0]),ms(x,'rmse'),ms(x,'tv'),ms(x,'top1',100,2)])
 dims=[]
 for n in [4,8,16,32,64]:
  x=m[(m.kind=='strict')&(m.n==n)&(m.states==4)&(m['rank']==8)]
  ub=b[(b.n==n)&(b.method=='uniform')].iloc[0]
  dims.append([n,ms(x,'rmse'),ms(x,'tv'),ms(x,'top1',100,2),f'{ub.tv:.4f}'])
 st=s[(s.group=='main')&(s['rank']==8)&(s.init=='center')&(s.burn==64)]
 streamrows=[]
 for L in [16,64,256,1024,2048]:
  x=st[(st.length==L)&(st.estimator=='bits')]
  streamrows.append([L,L+64,f'{x.rmse.mean():.5f}',f'{x.rmse_to_analytic.mean():.5f}',f'{x.tv.mean():.5f}',f'{100*x.zero_mass.mean():.3f}%'])
 streamnoise=st[st.estimator=='bits'].groupby('length').rmse_to_analytic.mean();select=streamnoise[streamnoise.index>=64]
 slope=np.polyfit(np.log(select.index),np.log(select),1)[0]
 floor=np.sqrt(st.groupby('seed').first().mse_approx.mean())
 baserows=[[x.method,f'{x.rmse:.6f}',f'{x.tv:.6f}',f'{100*x.top1:.2f}%'] for _,x in b[b.n==16].iterrows()]
 diagrows=[]
 for rank in [16,32]:
  x=signed[signed['rank']==rank];o=occ[occ['rank']==rank]
  diagrows.append([rank,ms(x,'test_rmse',dec=6),ms(x,'test_negative_fraction',100,2),ms(x,'state_sample_negative_fraction',100,2),ms(o,'occupancy_negative_fraction',100,2)])
 stressrows=[]
 for family,x in stress[stress['rank']==16].groupby('family'):
  stressrows.append([family,ms(x,'tv'),ms(x,'top1',100,2),f'{100*x.clip_row_fraction.mean():.2f}%'])
 legacy=m[(m.kind=='legacy')&(m['rank']==16)];r16=d[d['rank']==16]
 errors=(st.rmse**2-st.rmse_to_analytic**2-st.mse_approx-st.cross_term).abs().max()
 report=f'''# FROST Softmax 系统实验与实现审计

本报告来自当前会话实际执行的 CPU 数值实验。完成 70 次主实验训练、10 次实值 TT 诊断训练及 1 次独立先导运行；主结果均使用 5 个初始化种子。没有使用历史报告中的数值填充新结果。

## 1. 首要发现与结果适用范围

找回的 `run_frost_experiments.py` 中，`TTAttentionOperator.forward` 返回精确 `torch.softmax`；`stochastic_forward` 平均 TT logits 后仍调用精确 Softmax。旧结果能证明带 Softmax 输出层的模型可拟合目标，但不能证明去除了 Softmax。标量模型也在 TT 期望之后使用 sigmoid。

本轮新增的严格实现将每个 TT 状态切片约束为非负、逐行和为 1 的矩阵。编译后的固定核在推理时只进行矩阵收缩和状态选择，没有依赖当前输入的输出 Softmax；核的归一化属于离线参数处理。所有状态输出在数学上都是概率分布，可进行 categorical 发射。这个构造是论文一般实值 TT 的**受限参考子类**，不是已确认与原版 FROST 完全一致的实现。

结果显示：该构造能运行，但当前精度不够强，不能据此宣称 FROST 已优于常规 Softmax 近似。实值 TT 的平均输出可以拟合得更好，但采样到的状态系数中存在大量负值，不能直接充当稿件定义中的伯努利/类别概率。采用双极编码、仿射再缩放或后归一化可能提供其他实现路线，但其动态范围、随机误差和额外开销需要另行验证。

本轮仿真明确模拟 FSM 状态转移和类别输出事件；TT 收缩使用 FP32。未模拟 TT 乘加的门级随机电路、硬件 RNG、有限精度比较器或 RTL 调度，因此不报告芯片面积、功耗或硬件加速比。

## 2. 实验协议

| 项目 | 设置 |
| --- | --- |
| 主任务 | 学习完整 n 维 Softmax 概率行 |
| 输入维度 | n=4,8,16,32,64；每个维度分别训练，非单模型可变长度泛化 |
| TT rank | n=16,N=4 时 r=2,4,8,16,32 |
| FSM 状态数 | n=16,r=8 时 N=2,4,8 |
| 输入编码 | clip((z+4)/8,0,1)，范围预先固定，不做均值去除 |
| 训练数据 | 32,768 行 iid N(0,1)，数据种子 10000+n |
| 验证数据 | 独立 4,096 行，种子 20000+n |
| 主测试数据 | 独立 8,192 行，种子 30000+n |
| 模型种子 | 11,22,33,44,55 |
| 训练 | Adam，batch=256，4,000 steps；按验证 KL 选 checkpoint |
| 学习率 | 严格 TT / dense CPT 为 0.03；legacy 为 0.003；末 40% 余弦衰减 |
| 诊断训练 | 实值 TT 直接概率回归，MSE，Adam 0.002，单独按验证 MSE 选 checkpoint |
| 主位流实验 | 独立 128 行，5 个模型种子，每模型 8 次位流重复 |
| 位流长度 | L=16,32,64,128,256,512,1024,2048 |
| 初始化 | 中间状态+0/64 周期 burn-in；稳态初始化只作为理想对照 |
| RNG 相关性诊断 | 3 个模型种子、64 行、4 次重复；共享均匀随机数概率 0/0.25/0.5/1；burn-in=256 |
| 分布压力测试 | 每分布 4,096 行，Gaussian 多尺度、uniform、t(3)、单峰、双峰近并列、合成 QK |

主指标为逐元素 RMSE、逐行 TV=0.5 sum(abs(p-q))、Jensen--Shannon divergence（自然对数）、top-1 一致率和概率约束。平分最大值的预测以 1/并列数计 top-1，避免给均匀预测不合理的 argmax 优势。表中的 ± 为 5 个模型初始化的样本标准差；不是 95% 置信区间，也不把位流重复当成独立模型种子。

有限位流可以出现零概率，故原始 KL 可能为无穷大；没有通过偷偷加伪计数使它变好看。位流结果主要使用 TV、JS、RMSE 和零输出类别对应的目标概率质量。

## 3. Rank 与参数预算

以下为理想稳态期望输出，尚不含有限位流误差。n=16,N=4。

{tab(['rank','TT 参数数','RMSE','TV','Top-1 (%)'],rankrows)}

rank 增大改善拟合，但在本实现和优化预算下仍有明显误差。rank=16 的 RMSE 为 {ms(r16,'rmse',dec=6)}，TV 为 {ms(r16,'tv',dec=6)}。参数数仅表示存储的核元素数量，不等同于硬件面积。

## 4. 维度扩展：不能仅凭 RMSE 判断

固定 N=4,r=8。

{tab(['n','RMSE','TV','Top-1 (%)','Uniform TV'],dims)}

随着维度增长，单个目标概率通常缩小，因此逐元素 RMSE 可以下降，而整行 TV 和 top-1 同时恶化。不能将 RMSE 下降解释为架构随维度增大仍保持相同精度。

## 5. 有限位流与误差分解

本表配置为 n=16,N=4,r=8，中间状态初始化，额外 64 周期 burn-in。每行汇总 5 个模型、每模型 8 次位流重复，使用同一独立 128 行测试集。与上表的 8,192 行测试集不是同一批输入。

{tab(['L','实际总周期 L+64','总 RMSE → Softmax','采样 RMSE → TT 期望','TV → Softmax','目标质量落在零输出类别'],streamrows)}

这批输入上的理想近似误差水平约为 {floor:.6f}。L>=64 区间，采样 RMSE 对 L 的 log--log 拟合斜率为 {slope:.4f}，符合 RMSE 约按 L^(-1/2) 下降，而不是 RMSE 按 1/L 下降。总误差会逐渐受到函数近似误差限制。

同时记录三种量：解析 TT 期望、FSM 轨迹上概率向量的平均（Rao--Blackwellized estimator）、实际 categorical 输出频率。概率平均只是消除输出发射噪声的分析对照，并不代表最终一位/类别输出硬件。

逐次验证 MSE_total = MSE_approx + MSE_sampling + cross_term，最大数值残差为 {errors:.3g}。不能在存在暂态或偏差时直接删除交叉项。

## 6. 必须保留的常规对照

n=16，共用主测试集。下表是数值误差对照，不是相同硬件成本下的竞争排名。

{tab(['方法','RMSE','TV','Top-1'],baserows)}

- `logit_*bit_exact_softmax` 只量化输入 logits，后面仍用精确 Softmax；不等于完整整数 Softmax。
- `ibert_polynomial_float` 使用 [I-BERT Eq. 13--14](https://arxiv.org/pdf/2101.01321) 的指数近似，多项式与归一化均在浮点数中计算；不冒充官方整数实现。
- LUT 对 max-shift 后 [-16,0] 的指数进行分段线性插值，再用浮点求和及除法。低于 -16 的项置零。
- 输入裁剪 oracle 只用于分离编码范围造成的误差。

本轮参考 TT 在纯数值拟合精度上明显落后于这些算术近似。没有硬件综合数据，无法判断面积或能耗折衷。

## 7. 对旧结构与实值 TT 的诊断

在本轮同一训练与测试协议下，rank=16 的 TT+精确 Softmax 对照 RMSE 为 {ms(legacy,'rmse',dec=6)}。这说明在输出端保留目标函数会大幅改变实验含义。原归档 rank=12 checkpoint 的独立复测另存于 `prior_checkpoint_audit.json`，未与重新训练的五种子统计混合。

另训练一般实值 TT 直接回归概率，不使用输出 Softmax。它的结果如下：

{tab(['rank','平均输出 RMSE','测试预测为负 (%)','均匀状态抽样系数为负 (%)','ID 占用加权状态系数为负 (%)'],diagrows)}

每模型分别评估 65,536 个均匀抽样状态，以及 65,536 个按测试输入稳态分布抽样的状态。抽样比例不是对全部 N^M 状态的穷举结论；但发现负值已足以否定这些权重可以不加处理地作为概率发射参数。

这是探索性诊断：实值 TT 使用 MSE、严格概率 TT 使用交叉熵，因此不能仅把两者差值归因为一种约束，也不能据此证明任一类模型的最优误差。它证明的是：较低的平均拟合误差与合法的状态概率必须分别检查。

## 8. 分布与结构压力测试

下面使用严格概率 TT 的 n=16,N=4,r=16 配置。输入分布变宽时，编码范围裁剪本身会造成误差，必须与模型误差分开分析。

{tab(['输入族','TV','Top-1 (%)','含裁剪行比例'],stressrows)}

合成 QK 来自随机 Gaussian query/key、d_k=64，并非真实训练 Transformer 的中间激活。common-shift 测试使用 [-1,1] 输入加 [-2,2] 的公共偏移，整个测试都不触发裁剪。置换等变性也单独测量。未以模型训练或数据增强强制这些性质，因此结果用于揭示结构风险，不宣称已经解决。

## 9. 核验与论文使用建议

数学与实现核验：小维度显式状态枚举与 TT 期望的最大差 {validation['enumeration_max_error']:.3g}；梯度有限差分误差 {validation['gradient_error']:.3g}；独立长位流 Monte Carlo 与解析期望相符；categorical 频率和为 1。完整记录见 `validation.json`。

推荐将 Softmax 部分围绕以下问题组织：
1. 概率表示是否有效：无 runtime Softmax、状态输出合法、有限频率归一化。
2. 函数拟合是否准确：rank、状态数、维度与常规近似对照。
3. 随机计算误差来自哪里：近似、FSM 状态采样、类别发射、暂态分别分析。
4. 方法何时失效：输入尺度、峰值、平移、置换、共享 RNG、核概率精度。

建议主文展示误差来源和关键精度--资源权衡；将完整种子结果、量化、相关性与收敛曲线放入补充材料。当前参考实现未达到可用于宣称高精度替换的水平；优先确定与正式 FROST 一致且可采样的参数化，再进入 attention 输出和整网精度验证。不能将此参考子类的不足表述为所有 FROST/HI-SMURF 方法的不可行性。

## 10. 复现与文件

`README.md` 提供一键顺序；`protocol.json`、`training_jobs.json` 和 `stream_jobs.json` 保存设置；`results/` 保存逐种子 CSV、日志和关键预测数组；`checkpoints/` 保存模型；`figures/` 包含 PNG 和矢量 PDF。`provenance/` 保存旧实现相关类和原 checkpoint 以支持审计。所有训练、评估、绘图脚本均附带，原文件保持不变。
'''
 (ROOT/'FROST_softmax_report.md').write_text(report)
 latex=rf'''% Numerical reference study; not final FROST hardware results.
\subsection{{Softmax operator evaluation}}
We evaluated a probability-preserving reference parameterization of the finite-state tensor-train operator. Every state-selected core slice was constrained to be non-negative and row stochastic, ensuring that the contracted output was a probability vector. Core normalization was performed offline; no input-dependent output softmax was used during reference inference. This parameterization is a restricted subclass of the general real-valued tensor-train formulation.

For each input dimension $n\in\{{4,8,16,32,64\}}$, we generated 32,768 training rows, 4,096 validation rows and 8,192 independent test rows with logits drawn from $\mathcal{{N}}(0,1)$. Logits were encoded using $x=\operatorname{{clip}}((z+4)/8,0,1)$, while reference probabilities were computed from the original logits using FP32 softmax. Five independently initialized models were trained for each configuration. Checkpoints were selected using validation divergence, and test inputs were not used for model selection. Reported uncertainty denotes the sample standard deviation across the five model initializations.

At $n=16$ and $N=4$, increasing the rank from 8 to 16 reduced the mean test RMSE from {d[d['rank']==8].rmse.mean():.6f} to {r16.rmse.mean():.6f}, and total variation from {d[d['rank']==8].tv.mean():.6f} to {r16.tv.mean():.6f}. These errors refer to analytical steady-state outputs and exclude finite-bitstream noise. The results characterize this reference parameterization and do not establish the best achievable accuracy of a general tensor-train operator.

Finite-window evaluation used explicit saturating FSM transitions and categorical output events, with FP32 evaluation of the state-selected tensor-train contraction. For each of five models, eight independent stream realizations were evaluated on 128 independent score rows. The principal configuration used 64 burn-in cycles followed by $L\in\{{16,32,64,128,256,512,1024,2048\}}$ measured cycles. Probability-vector averaging along the same FSM trajectories was retained as a Rao--Blackwellized diagnostic to separate state-sampling error from output-emission noise. The sampling RMSE exhibited a fitted log--log slope of {slope:.3f} for $L\geq64$, whereas total error approached the analytical approximation level of approximately {floor:.5f} on this test subset.

Additional tests examined logit-scale shifts, non-Gaussian inputs, near-tied maxima, permutation equivariance, common-shift invariance, input-stream dependence and offline core-probability quantization. Arithmetic references included piecewise-linear exponential lookup and the polynomial approximation in I-BERT, evaluated in floating-point arithmetic. These comparisons assess numerical approximation rather than matched-cost hardware efficiency. No gate-level TT arithmetic or hardware energy measurements were performed in this study.
'''
 (ROOT/'methods_and_results.tex').write_text(latex)
 captions='''Figure 1. Systematic Softmax evaluation of a probability-preserving FSM--TT reference. (a) Dimension sweep at N=4,r=8; each dimension has a separately trained model. (b) Rank sweep; the TT+exact-Softmax audit control retains runtime Softmax and is not a Softmax replacement. (c) FSM-state sweep. (d) Finite-window RMSE to FP32 Softmax with 64 extra burn-in cycles; the probability-averaging estimator is a diagnostic, not emitted categorical frequencies. The IID categorical curve uses exact target probabilities and is a sampling reference, not an implemented accelerator. (e) Distribution stress tests, including input-clipping effects. (f) Common-shift invariance without clipping. Points/error bars show means/sample SD over five model initializations; stream repetitions are first averaged within each model. Main tests use 8,192 rows; stress/invariance tests use 4,096; stream tests use 128 rows and eight stream repetitions per model.

Figure 2. Sampling and implementation diagnostics. (a) State-only versus state-plus-output sampling under ideal stationary initialization. (b) Center reset with zero/64 burn-in cycles versus ideal stationary initialization. (c) Probability of sharing a uniform RNG draw across input dimensions: this is not the Pearson correlation coefficient. Three model seeds and four stream repetitions per seed are used; burn-in=256. (d) Offline probability-grid projection with step 2^-B; each core row remains normalized by largest-remainder count allocation. B is probability resolution, not a complete hardware storage-width specification. Dotted lines show unquantized FP32-core error. TT contractions are evaluated in FP32 throughout the bitstream simulations.
'''
 (ROOT/'figure_captions.txt').write_text(captions)
 readme='''# FROST Softmax reference study

Read FROST_softmax_report.md before using any numbers in a manuscript. This archive contains a new strict row-stochastic-core TT reference, controls retaining exact Softmax, and an unconstrained signed-TT diagnostic. They are explicitly different models. No hardware area/power result is claimed.

## Reproduce

Use Python 3.12+ and the recorded package versions in environment.json. CPU execution is sufficient. Run from this directory:

```bash
python study.py validate
python run_training.py
python evaluate.py
python streams.py
python signed_diagnostic.py
python validate_signed_occupancy.py
python figures_and_summary.py
python build_report.py
```

The 70 primary models are skipped when checkpoints already exist. To train them from scratch, move checkpoints to a separate backup directory first. Existing results are overwritten by evaluation scripts. Checkpoints are loaded with weights_only=True.

Main results use five fixed initialization seeds and shared, disjoint train/validation/test data. Signed TT diagnostics use a different loss and are exploratory. State-probability violations are reported rather than repaired by an unreported output normalization. Finite-stream KL may be infinite; CSVs preserve that fact.

The archival prior checkpoint uses its original exact-Softmax forward path, preserved in provenance/previous_model_definitions.py. It is a historical audit control, not new strict-model evidence.

Figures are supplied as PNG and vector PDF. methods_and_results.tex is a factual draft of the current reference experiment, not a claim that the final FROST design has been validated. protocol.json records the original plan; post-audit signed-TT diagnostics and offline core quantization are separately identified in protocol_additions.json.
'''
 (ROOT/'README.md').write_text(readme)
 additions={'exploratory_diagnostic':'10 signed TT probability-regression runs (r16/r32, five seeds), MSE validation selection; added to check whether low mean error implies valid state emission',
 'offline_quantization':'4/8/12-bit probability grid; largest-remainder row normalization; evaluated without retraining',
 'negative_state_tests':'65536 uniform state codewords + 65536 stationary-occupancy samples per model; violations refute direct probability emission, not all possible signed SC encodings',
 'scope':'Masking and real pretrained-network logits not evaluated. Synthetic QK is labeled synthetic.'}
 (ROOT/'protocol_additions.json').write_text(json.dumps(additions,indent=2))
 print('FROST_softmax_report.md, methods_and_results.tex, captions and README written.')

if __name__=='__main__':main()
