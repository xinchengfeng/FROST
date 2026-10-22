import json,hashlib,zipfile,platform,sys
from pathlib import Path
import pandas as pd,numpy as np,torch

ROOT=Path(__file__).resolve().parent

def main():
    free=pd.read_csv(ROOT/'results'/'pilot_metrics.csv');free['kind']='free_signed'
    bounded=pd.concat([pd.read_csv(ROOT/'results'/'bounded_metrics.csv'),pd.read_csv(ROOT/'results'/'bounded_extension_metrics.csv')],ignore_index=True)
    categorical=pd.read_csv(ROOT/'results'/'categorical_metrics.csv')
    allrows=pd.concat([free,bounded,categorical],ignore_index=True)
    allrows.to_csv(ROOT/'results'/'all_configurations.csv',index=False)
    streams=pd.read_csv(ROOT/'results'/'direct_stream_metrics.csv')
    ss=streams[streams['mode']=='categorical_bits'].groupby('L')[['rmse','half_row_l1','sampling_rmse']].agg(['mean','std'])
    summary=ss.copy();summary.columns=['_'.join(k) for k in summary.columns]
    summary.to_csv(ROOT/'results'/'stream_summary.csv')
    env=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform(),gpu_used=False)
    (ROOT/'environment.json').write_text(json.dumps(env,indent=2))
    protocol=json.loads((ROOT/'protocol.json').read_text())
    protocol.update(total_trained_configurations=len(allrows),model_initialization_seeds=[11],
        no_final_performance_claim='Exploratory single-initialization study; no held-out Transformer task or hardware results.',
        phases=['six free signed TT configurations','six bounded scalar-head configurations','three additional bounded sorted configurations','three joint categorical TT configurations'],
        bounded_head_constraint='Nonnegative row-stochastic internal cores and scalar terminal coefficients in [0,1]; does not guarantee a simplex across output channels.',
        joint_categorical_constraint='All core slices row-stochastic with final axis indexing outputs; every represented state output vector lies on the simplex in exact arithmetic.',
        compilation='Softmax/sigmoid on static PARAMETERS during training/compilation only; compiled input forward has neither function.',
        free_readout='q=(1+raw-mean(raw))/n; r is the rank of each base TT, not the rank of the combined post-readout CPT.',
        sorting='Binary score sorting, centering and inverse output permutation; an explicit preprocessing cost, not a piecewise function approximation.',
        test_reuse='Same 4096 test rows across phases; adaptive exploration, not a pristine final confirmation set.',
        finite_scope='One validation-selected joint CPT model, three stream seeds, 128 new rows, 256 burn-in cycles plus L sampling cycles. Separate weighted-state diagnostic is not a bitstream output.')
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    def simple_table(df):
        out='| 表示 | 编码 | N | 每个基础 TT 的 rank | 参数数 | 测试 RMSE |\n|---|---|---:|---:|---:|---:|\n'
        for _,r in df.iterrows():out+=f"| {r['kind']} / {r['heads']} | {r['encoding']} | {int(r['states'])} | {int(r['rank'])} | {int(r['parameters'])} | {r['rmse']:.7f} |\n"
        return out
    stream_table='| L | RMSE，均值 ± SD | 平均行 TV |\n|---:|---:|---:|\n'
    for length,v in ss.iterrows():stream_table+=f"| {length} | {v[('rmse','mean')]:.6f} ± {v[('rmse','std')]:.6f} | {v[('half_row_l1','mean')]:.6f} |\n"
    report=r'''# 直接使用 HI-SMURF 拟合多变量 Softmax：试验与边界

## 核心回答

可以直接拟合。分段指数近似不是 HI-SMURF 的必要组成部分。本轮删除了此前的分段指数实现，直接学习整个 logit 向量到 Softmax 输出的映射。

对每个输入变量建立 FSM。给定编码后的输入 x，使用对应的稳态概率 π。每个输出分量写成

\[
\widehat p_i(\mathbf z)
=\sum_{\mathbf s}\left[\prod_m\pi_m(s_m\mid x_m)\right]W_{\mathbf s,i},
\]

其中 W 用 TT 表示。所有输入变量均参与每个输出分量的计算。精确 Softmax 仅用于训练标签，未作为运行时输出激活。

若每个状态对应的输出满足 W[s,i] ≥ 0 且 ∑i W[s,i] = 1，那么其期望也自动是概率分布。这样，归一化性质存在于训练得到的 CPT 中，不需要在运行时再调用 Softmax。

## 实现与测试范围

本轮是依据草稿中的 FSM/TT 公式建立的直接对照实验，区分以下三种参数化；它们都没有分段线性指数插值。

1. **free_signed**：自由实值 TT 直接回归概率。通过线性读出 q=(1+raw−mean(raw))/n 使输出之和为 1；不保证非负。parallel/separate 表示为每个输出建立基础标量 TT，每个 TT 都读取完整输入向量。这里记录的 r 是基础 TT 的 rank；线性组合后的完整 CPT 不必仍是 rank r。
2. **bounded**：每个输出使用独立标量 HI-SMURF，内部参数以非负概率转移形式约束，末端标量系数处于 [0,1]。每个状态输出都能作为伯努利概率，但不同输出分量之和不自动为 1。
3. **categorical**：一个联合 TT 表示完整向量输出，所有状态输出向量都满足非负且总和为 1。它是一个受约束的非负 TT 子类，不代表一般 HI-SMURF 所能达到的全部精度。

bounded/categorical 在训练及一次性编译时，对静态核参数使用 Softmax 或 sigmoid 来建立概率约束。编译后的推理只读已固定的核；没有对输入或运行时输出执行这些非线性函数。代码检查验证了这一点。

实验统一使用 16 维 N(0,1) logits、16,384 条训练行、2,048 条验证行、4,096 条测试行，数据种子分别为 110016、120016、130016。模型初始化种子统一为 11，训练 2,000 或 3,000 步，以验证 MSE 选择各配置的 checkpoint。本轮共训练 18 个配置，属于单初始化探索，不是最终五种子论文实验。

输入改动只有仿射编码、逐行去均值，以及可选的排序和输出逆置换。排序利用 Softmax 的置换等变性，将不同排列映射到统一顺序；仍然由 TT 直接学习所有 logit 的耦合关系。排序是额外的二进制预处理，硬件代价尚未评估。

## 自由实值 TT：数值拟合与 rank

''' + simple_table(free) + r'''
separate/center 模式下，rank 从 2 增至 8 时，RMSE 从约 0.00630 降至 0.00500，同时参数数从 3,840 增至 58,368。这个结果支持“基础标量 TT 的 rank 收益较小”的局部观察，但不能直接比较不同输出共享方式的同名 rank。

排序后 rank 4 的数值 RMSE 达到约 0.00135。然而，测试输出约 0.043% 的元素为负，对实际稳态占用采样的状态系数约 39.9% 为负。因此不能将这个结果作为合法伯努利 CPT 或完整随机硬件的精度。零截断后再归一化的误差只作为诊断记录，未作为主方案。

## 状态概率有界的标量头

''' + simple_table(bounded) + r'''
这些配置的状态系数严格受结构限制，保证在 [0,1] 内。但输出分量之和只通过训练逼近 1。最佳测试 RMSE 约 0.00761，平均绝对行和误差约 0.01627。由于原始输出行未严格归一化，CSV 的 half_row_l1 不能直接称为两个概率分布间的 TV。

## 完整概率向量约束的联合 TT

''' + simple_table(categorical) + r'''
这个版本同时保持非负和归一化，直接逼近多变量 Softmax。N=16 时，rank 4 和 8 的解析测试 RMSE 分别约 0.01394 和 0.01296。浮点行和误差约 1e-6；数学结构保证其在精确算术中为 1。

这证明本轮构造可以直接实现联合概率映射，同时表明所用概率约束和输入编码仍有明显精度代价。不能据此认定所有 HI-SMURF 必须有同样的误差，也不能把自由实值拟合的低误差与受约束版本混成同一项结果。

## 实际 FSM 和输出位流

在三个 categorical 配置中根据验证误差选择 N=16、rank 8。对另外 128 条高斯输入（seed=140016），使用 3 个独立位流种子，运行真实反射随机游走及分类输出事件。FSM 从中央状态开始，先预热 256 周期，再采样 L 周期。因此总周期是 256+L。

''' + stream_table + r'''
每周期按状态 CPT 产生一个类别事件，有限输出计数归一化后行和精确为 1。TT 核收缩仍使用浮点计算，本轮没有模拟核乘加的门级随机算术。

该直接版本尚未复现此前“很短的位流就具有极高精度”的观察。解析收缩的误差与实际有限位流误差已分别记录；未将前者当作后者。加权状态均值另存为 weighted_diagnostic，它没有输出 Bernoulli/分类发射噪声，不能与真实位流结果混用。

## 校验与复现

小维度枚举 27 个状态的期望，与 TT 稳态收缩的最大差异约 6e-8。检查通过了状态系数的非负及和约束、实际分类计数行和、编译后禁止 Softmax/sigmoid/exp 的推理路径。Top-1 在出现输出并列时使用分数计分；它不是下游任务准确率。

```bash
python direct_study.py
python bounded_study.py
python bounded_study.py --jobs bounded_extension_jobs.json
python categorical_study.py
python check_and_simulate.py
python build_report.py
```

依赖 Python、NumPy、pandas 和 PyTorch CPU；版本保存在 environment.json。results/ 包含逐配置指标、原始预测、位流结果与协议，checkpoints/ 包含全部 18 个训练模型及验证曲线。

本轮后续方向应继续围绕直接 HI-SMURF：改进 CPT 的全局概率约束和输入编码，并在保持相同表示及成本口径的前提下验证有限位流。此前范围缩减的分段指数方法保留为外部对照，不作为这一直接架构的性能证明。
'''
    (ROOT/'HI_SMURF_direct_softmax_report.md').write_text(report)
    (ROOT/'README.md').write_text('''# Direct HI-SMURF-style multivariate Softmax study

No piecewise exponential interpolation. Read HI_SMURF_direct_softmax_report.md
and protocol.json before interpreting the results. Three representations are
deliberately distinguished: unrestricted signed TT regression; bounded scalar
Bernoulli heads without exact row normalization; and a joint simplex-valued
TT with valid categorical state outputs. These are exploratory reference
parameterizations, not a claim to reproduce every detail of the original paper.

Run the six commands in the report to reproduce all 18 single-seed training
configurations and the separate 3-seed finite-stream simulation. Constants-only
parameter compilation is separate from runtime inference. No GPU is required.
''')
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*'))
        if p.is_file() and '__pycache__' not in str(p) and p.name!='manifest_sha256.json' and '.openai-' not in p.name}
    (ROOT/'manifest_sha256.json').write_text(json.dumps(manifest,indent=2))
    path=ROOT.parent/'HI_SMURF_direct_softmax_experiments.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for f in list(manifest)+['manifest_sha256.json']:z.write(ROOT/f,arcname=ROOT.name+'/'+f)
    print(json.dumps(dict(configurations=len(allrows),archive=str(path),bytes=path.stat().st_size),indent=2))

if __name__=='__main__':main()
