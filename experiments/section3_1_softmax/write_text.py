"""Write paper text and captions from measured numerical summaries."""
from pathlib import Path
import json,hashlib,zipfile
import pandas as pd

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'source_data'

def main():
    n=json.loads((DATA/'numerical_summary.json').read_text())
    s=pd.read_csv(DATA/'stream_summary.csv').set_index(['mode','L'])
    a=s.loc[('categorical_bits',64)];b=s.loc[('categorical_bits',1024)]
    text=rf'''\subsection{{Direct multivariate approximation of softmax}}
\label{{sec:softmax_approximation}}

We first evaluated whether HI-SMURF could directly reproduce the coupled
probability mapping of softmax. For this experiment, we used a joint-output
conditional probability tensor whose state-dependent output vectors are
constrained to the probability simplex. The tensor-train (TT) representation
therefore predicts all output probabilities jointly. Input scores were sorted
and mean-centred before stochastic encoding, and the output permutation was
subsequently inverted. Exact softmax provided the reference probabilities for
training and evaluation; the compiled operator used neither an exponential
approximation nor a runtime output softmax.

We trained three configurations on 16,384 independent Gaussian score vectors
of dimension 16, using 2,048 validation vectors for checkpoint selection and
4,096 separate test vectors for evaluation. Each configuration was trained
once. The validation-selected model, with $N=16$ states per finite-state
machine and TT rank $r=8$, achieved a test root-mean-square error (RMSE) of
$1.296\times10^{{-2}}$ and a mean row total variation distance of
$5.640\times10^{{-2}}$ (Fig.~\ref{{fig:softmax_direct}}a).
The median and 95th-percentile row total variation distances were 0.0512 and
0.1026, respectively (Fig.~\ref{{fig:softmax_direct}}b). Residual errors
were not uniform across the probability range: larger reference probabilities
tended to be underestimated. The state-output constraints preserved
non-negativity and normalization, with a maximum test row-sum deviation of
$1.13\times10^{{-6}}$ in floating-point evaluation.

Validation error decreased over the recorded optimization steps for all three
configurations (Fig.~\ref{{fig:softmax_direct}}c). At fixed rank $r=4$,
increasing the state count from $N=8$ to $N=16$ reduced test RMSE from 0.01426
to 0.01394, while increasing the number of trainable parameters from 2,336 to
4,672. Increasing the rank from 4 to 8 at $N=16$ further reduced RMSE to
0.01296, using 16,512 parameters (Fig.~\ref{{fig:softmax_direct}}d).
Thus, the latter comparison yielded a 7.0\% reduction in error for a
3.53-fold increase in parameter count. Within the configurations examined,
additional TT capacity produced a modest improvement in approximation accuracy.

We next evaluated finite-stream operation on 128 additional score vectors,
using independent Bernoulli input streams and three random sequence seeds.
The finite-state machines were initialized centrally and advanced for 256
burn-in cycles before collecting $L$ categorical output events. The output
RMSE decreased from
$({a.rmse_mean*100:.3f}\pm{a.rmse_std*100:.3f})\times10^{{-2}}$ at $L=64$ to
$({b.rmse_mean*100:.3f}\pm{b.rmse_std*100:.3f})\times10^{{-2}}$ at $L=1024$
(mean $\pm$ sample standard deviation across sequence seeds;
Fig.~\ref{{fig:softmax_direct}}e). The analytic steady-state RMSE on these
same 128 vectors was $1.397\times10^{{-2}}$. Relative to this analytic
output, the sampling RMSE decreased from 0.0425 to 0.0114
(Fig.~\ref{{fig:softmax_direct}}f). Averaging the conditional probability
vectors along the same state trajectories, without categorical emission,
gave lower sampling errors. This diagnostic separates the additional
variability introduced by output emission from the error already present in
the finite-state trajectories. The finite-stream evaluation retained
floating-point TT contractions and did not include gate-level arithmetic or
hardware timing measurements.
'''
    (ROOT/'section3_1_softmax.tex').write_text(text)
    caption=r'''\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/Fig3_softmax_combined.pdf}
\caption{\textbf{Direct multivariate softmax approximation with a
probability-constrained HI-SMURF operator.}
\textbf{a}, Analytic predictions versus reference probabilities for the
validation-selected model ($N=16$, $r=8$), comprising 65,536 entries from
4,096 test vectors. Colour indicates the number of entries in each hexagonal
bin; the dashed line denotes identity.
\textbf{b}, Empirical cumulative distribution of row total variation
distances for the same test set. Markers indicate the median and 95th percentile.
\textbf{c}, Validation RMSE over optimization for three state-count and
rank configurations. Each curve represents one training initialization.
\textbf{d}, Test RMSE versus the number of trainable parameters for these
configurations. No uncertainty across training initializations is estimated.
\textbf{e}, Finite-stream output RMSE on 128 additional vectors. Open markers
show individual sequence seeds; filled markers and error bars show the mean
and sample standard deviation across three seeds for a fixed trained model.
The dashed line denotes analytic error on these same vectors. Sampling length
$L$ excludes 256 burn-in cycles, giving a total of $256+L$ cycles.
\textbf{f}, RMSE relative to the analytic model output, comparing categorical
emission with the conditional-probability average along the same state
trajectories. The latter is a diagnostic readout, not a binary output stream.
The dotted $L^{-1/2}$ guide is anchored to the mean categorical sampling error
at $L=64$ and is not a fitted scaling law. Data and error bars follow
\textbf{e}. Source data are supplied with the figure.}
\label{fig:softmax_direct}
\end{figure}
'''
    (ROOT/'section3_1_figure_caption.tex').write_text(caption)
    methods=r'''\subsection{Softmax approximation experiments}
\label{methods:softmax_approximation}

\paragraph{Data and encoding.}
Training, validation and test sets contained 16,384, 2,048 and 4,096
independently generated vectors, respectively, with 16 entries sampled from
$\mathcal{N}(0,1)$. NumPy generator seeds were 110016, 120016 and 130016.
Scores were stored in single precision. The reference target was the exact
softmax of the original score vector. For model input, each row was sorted
in descending order and its mean was subtracted. The centred scores
$\widetilde z_m$ were encoded as
\begin{equation}
x_m=\operatorname{clip}\left(\frac{\widetilde z_m+4}{8},0,1\right).
\end{equation}
The sorting permutation was inverted at the output. The target probabilities
were not recomputed from clipped inputs.

\paragraph{Joint probability tensor.}
We used a constrained, non-negative TT parameterization for which each
state-selected core slice is row stochastic. The TT boundary dimensions
were 1 and 16, and all internal bond dimensions were $r$. For every joint
state, contracting these slices produces a non-negative output vector whose
entries sum to one. This is a restricted subclass of general real-valued
TT tensors. State probabilities were evaluated analytically as
\begin{equation}
\pi_m(s\mid x_m)=
\frac{x_m^s(1-x_m)^{N-1-s}}
{\sum_{u=0}^{N-1}x_m^u(1-x_m)^{N-1-u}},
\qquad s=0,\ldots,N-1.
\end{equation}
Independent input streams imply a product joint state distribution
conditional on the input vector. The expected output was obtained by
contracting the TT cores with these state distributions.

\paragraph{Optimization and model selection.}
We evaluated $(N,r)=(8,4),(16,4),(16,8)$, corresponding to 2,336, 4,672
and 16,512 trainable parameters. Each configuration used one initialization
(PyTorch seed 11), the same training data, a minibatch size of 256 and 3,000
Adam updates. The objective was $16^2$ times the mean squared probability
error. The initial learning rate was 0.03 and followed a cosine decay to
0.003 over the final 40\% of updates. Gradient norms were clipped to 5.
Validation error was evaluated every 200 updates, and the checkpoint with
the lowest validation MSE was retained. The $N=16,r=8$ configuration was
selected by validation error for detailed probability and stream analyses.
These configurations followed an exploratory architecture study; the test
set was reused across that study and is not an untouched confirmatory set.
Core probability constraints were applied to static trainable parameters
during optimization and compilation. The compiled operator did not apply
softmax, sigmoid or exponentiation to runtime inputs or outputs.

\paragraph{Finite-stream simulation.}
An additional 128 Gaussian vectors were generated with seed 140016.
For each vector, independent Bernoulli bits drove saturating increment/decrement
FSM transitions. All FSMs started at state $\lfloor(N-1)/2\rfloor$ and
ran for 256 burn-in cycles before output collection. Three sequence seeds
(101, 202 and 303) were evaluated at $L=64,256,1024$. At each sampled
cycle, the compiled TT was contracted at the current state and a categorical
output event was drawn. Dividing category counts by $L$ produced the
estimated softmax vector, with an exact unit row sum. A second accumulator
averaged the conditional probability vectors along the same trajectories
without drawing output events; this was used only as a diagnostic. TT
contractions used single-precision arithmetic, so the simulation does not
constitute a gate-level stochastic implementation.

\paragraph{Metrics and statistics.}
For $B$ rows of dimension $n=16$, the aggregate RMSE was
\begin{equation}
\mathrm{RMSE}=\left[\frac{1}{Bn}\sum_{b=1}^{B}\sum_{i=1}^{n}
(\widehat p_{bi}-p_{bi})^2\right]^{1/2}.
\end{equation}
The total variation distance of row $b$ was
$\mathrm{TV}_b=\frac12\sum_i|\widehat p_{bi}-p_{bi}|$.
Sampling RMSE used the analytic output of the same model on the same rows
as its reference. Error bars indicate the sample standard deviation across
the three sequence seeds; they do not quantify training-initialization or
dataset uncertainty. Individual sequence measurements are plotted. The
empirical cumulative distribution uses all 4,096 test rows, and all 65,536
probability entries are included in the hexagonal-bin plot. Entries from
the same row are coupled and are not treated as independent replicates.
'''
    (ROOT/'section3_1_softmax_methods.tex').write_text(methods)
    notes='''# Section 3.1: direct multivariate Softmax approximation

## Deliverables

- figures/Fig3_softmax_combined.pdf: one six-panel Figure 3, 180 mm wide.
- figures/Fig3ab_softmax_fidelity.*: probability predictions and row-error CDF.
- figures/Fig3cd_model_capacity.*: optimization and parameter-count comparison.
- figures/Fig3ef_bitstream_length.*: actual output and sampling error.
- Every figure is supplied as PDF, editable SVG, and 400-dpi PNG.
- section3_1_softmax.tex: main Results subsection, placed under your Section 3.
- section3_1_figure_caption.tex: figure environment and complete caption.
- section3_1_softmax_methods.tex: reproducible Methods description.
- source_data/: numerical data underlying every panel.

## Manuscript use

Keep figures/ relative to the manuscript, load graphicx, and insert the Results
subsection and caption. The figure number follows your manuscript counter;
the filenames use Fig3 as a provisional name. Put the detailed Methods text in
the Methods section. There is no need to manually force subsection number 3.1.

The figure styles use white backgrounds, consistent sans-serif text, accessible
colours, lowercase panel letters and vector export. The Nature Communications
guidance also calls for a clear description of error bars and figure source data:
https://www.nature.com/ncomms/submit/how-to-submit

## Evidence boundaries

The main figures use all three joint categorical TT configurations from the
previous direct experiment. Free signed coefficient models and independently
bounded scalar heads do not satisfy the same joint probability constraints and
are not combined into the main accuracy or rank curves. Those controls remain
available in the earlier complete direct-experiment archive.

There is one training initialization per configuration. The three stream seeds
are repeated sequence realizations for one fixed model on the same 128 rows.
Error bars represent their sample SD, not independent model training or a
confidence interval. L excludes the 256-cycle burn-in. The analytic reference
in panel e is calculated on those same 128 rows, rather than the 4,096-row test
set. The L^-1/2 line in panel f is a reference guide, not a fit to three points.

The stated results describe this constrained reference implementation. They do
not establish a general accuracy limit for HI-SMURF, demonstrate short-stream
high accuracy, or measure complete Transformer/hardware performance. The
text describes the visible underestimation of high-probability entries.

## Reproducibility

To regenerate all figures from the supplied source CSVs:

```bash
python render_figures.py
python write_text.py
```

Dependencies: numpy, pandas, matplotlib; Nimbus Sans is used if installed.
prepare_source_data.py additionally requires torch and the sibling original
frost_direct_hi_smurf directory; it extracts the existing results without
retraining. provenance/input_sha256.json identifies its input checkpoints and
prediction files. The supplied CSVs make the rendering package standalone.
'''
    (ROOT/'README.md').write_text(notes)
    chinese='''建议将三组图合并为一张六面板 Figure 3，用于 Section 3.1。

图 a–b：主模型的概率预测与整行误差分布。正文可报告 RMSE 0.01296、平均 TV 0.05640，同时说明大概率项存在低估。
图 c–d：三个合法联合 CPT 配置的训练轨迹与参数量。N=16 时 rank 4 到 8 的误差下降 7.0%，参数增至 3.53 倍；不据此声称所有输入维度都对 rank 不敏感。
图 e–f：真实分类输出位流，以及相对于同一模型稳态输出的采样误差。标注 L 之外还需 256 个预热周期，误差棒为固定模型的三个位流种子之间的样本标准差。

正文、图注和 Methods 已分别生成英文 LaTeX。主图没有混用自由实值 TT 的低误差或前一轮分段指数方案的结果。
'''
    (ROOT/'reading_notes_zh.txt').write_text(chinese)
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*'))
        if p.is_file() and '__pycache__' not in str(p) and p.name!='manifest_sha256.json' and '.openai-' not in p.name}
    (ROOT/'manifest_sha256.json').write_text(json.dumps(manifest,indent=2))
    archive=ROOT.parent/'Section3_1_Softmax_Figures_and_Text.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in list(manifest)+['manifest_sha256.json']:z.write(ROOT/name,arcname=ROOT.name+'/'+name)
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'files':len(manifest)+1},indent=2))

if __name__=='__main__':main()
