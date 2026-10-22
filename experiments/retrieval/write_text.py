"""Build manuscript text and a complete reproducibility archive from results."""
from pathlib import Path
import io,json,hashlib,zipfile
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'source_data'

def main():
 s=pd.read_csv(DATA/'summary_metrics.csv').set_index(['config','L'])
 d=pd.read_csv(DATA/'depth_per_model.csv')
 def val(config='calibrated_full',L=256,metric='centered_logit_relative_error',stat='mean'):
  return float(s.loc[(config,L),metric+'_'+stat])
 def depth(config,metric='query_relative_error'):
  return d[(d.config==config)&(d.L==256)&(d.layer==4)][metric].mean()
 text=r'''\subsection{End-to-end simulation of a stochastic Transformer}
\label{sec:full_transformer}

We next integrated the stochastic operators into a complete four-block
encoder-only Transformer. The network included learned input and positional
embeddings, two attention heads per block, output projections, residual
connections, pre-layer normalization, feed-forward networks with GELU
activations, a final normalization layer and a classification head. We used
a controlled associative-retrieval task: 15 tokens presented distinct keys
with independently assigned four-class values, and a final query token
requested the value associated with one stored key. Three independently
trained floating-point models reached 100\% accuracy on the 128 held-out
sequences used for stochastic evaluation.

All main linear stages and attention products were evaluated using explicit
sign--magnitude bitstreams. GELU used an eight-state probability gate, whereas
layer normalization combined stochastic variance estimation with a four-state
reciprocal-square-root gate. We examined direct transfer of the softmax
operator from Section~\ref{sec:softmax_approximation} and a simple
activation-calibration procedure. Calibration updated the same-size
HI-SMURF table using reference attention scores from separate input sequences,
with its encoding range selected on an independent calibration-validation set.
The Transformer weights remained fixed, and the HI-SMURF operator retained
16 states, TT rank 8 and 16,512 parameters. Calibration used reference
operator outputs without task labels.

Calibration reduced errors throughout the network
(Fig.~\ref{fig:transformer_depth}). At $L=256$, the relative error in the
query representation after the fourth block decreased from @@GENDEPTH@@
under direct transfer to @@CALDEPTH@@ after calibration. We additionally
measured the error in each block's residual update to assess the learned
transformation separately from the identity connection. At the classifier
output, relative centred-logit error decreased from
@@GENLOG@@ $\pm$ @@GENLOGSD@@ to @@CALLOG@@ $\pm$ @@CALLOGSD@@,
and retrieval accuracy increased from @@GENACC@@\% to @@CALACC@@\%
(Fig.~\ref{fig:transformer_performance}). Values denote means across three
trained models after averaging three stream realizations per model; reported
uncertainty is the sample standard deviation across the three model means.

Longer streams improved numerical fidelity over the tested range, with
diminishing gains within the calibrated architecture.
Increasing $L$ from 64 to 1,024 reduced relative centred-logit error from
@@LOG64@@ to @@LOG1024@@, compared with @@ANALYTIC@@ for analytic
evaluation of the calibrated finite-state operators
(Fig.~\ref{fig:transformer_performance}b). Error did not decrease monotonically:
the mean at $L=1024$ was slightly above that at $L=256$ (@@CALLOG@@),
indicating that longer streams did not remove the remaining approximation
error. Corresponding retrieval
accuracies were @@ACC64@@\% and @@ACC1024@@\%
(Fig.~\ref{fig:transformer_performance}a). The continuous-output metric
therefore resolves approximation errors that need not change the predicted
class. Stream length excludes 256 burn-in cycles for each multivariate
softmax FSM ensemble and 64 cycles for each scalar FSM stage.

Progressively extending stochastic computation from the attention cores to
all learned linear stages and then to normalization and GELU yielded relative
centred-logit errors of @@ATTN@@, @@LINEAR@@ and @@CALLOG@@, respectively,
at $L=256$ (Fig.~\ref{fig:transformer_ablation}a). These end-to-end controls
contain interacting approximation errors; their ordering does not establish
that adding stochastic stages improves intrinsic accuracy. With the calibrated
operator fixed, switching from independent arithmetic streams to Sobol
streams changed this error from @@IID@@ to @@CALLOG@@
(Fig.~\ref{fig:transformer_ablation}b). These comparisons evaluate the
complete encoder graph on a controlled synthetic task. TT contractions,
centering, residual and bias accumulation, static normalization-affine gains,
range scaling and conversion between stochastic stages remained numerical
operations; hardware area, energy and latency were not measured.
'''
 replacements={
  'GENDEPTH':f'{depth("generic_full"):.3f}','CALDEPTH':f'{depth("calibrated_full"):.3f}',
  'GENLOG':f'{val("generic_full"):.3f}','GENLOGSD':f'{val("generic_full",stat="std"):.3f}',
  'CALLOG':f'{val():.3f}','CALLOGSD':f'{val(stat="std"):.3f}',
  'GENACC':f'{100*val("generic_full",metric="accuracy"):.1f}','CALACC':f'{100*val(metric="accuracy"):.1f}',
  'LOG64':f'{val(L=64):.3f}','LOG1024':f'{val(L=1024):.3f}','ANALYTIC':f'{val("calibrated_analytic"):.3f}',
  'ACC64':f'{100*val(L=64,metric="accuracy"):.1f}','ACC1024':f'{100*val(L=1024,metric="accuracy"):.1f}',
  'ATTN':f'{val("attention_only"):.3f}','LINEAR':f'{val("linear_attention"):.3f}','IID':f'{val("calibrated_iid"):.3f}'}
 for k,v in replacements.items():text=text.replace('@@'+k+'@@',v)
 assert '@@' not in text
 (ROOT/'section3_3_transformer.tex').write_text(text)

 captions=r'''% Requires graphicx. Each core figure has its own a,b panels.
% Depth figure: after the paragraph reporting fourth-block and final-output errors.
\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/Fig_transformer_depth.pdf}
\caption{\textbf{Propagation of approximation error through a four-block Transformer.}
\textbf{a}, Relative error of the query-token representation after each encoder
block at $L=256$. \textbf{b}, Relative error of the query-token residual update,
defined as the difference between the representations entering and leaving
each complete encoder block. Evaluating updates measures the learned
transformation separately from the copied residual signal.
Direct transfer uses the generic HI-SMURF checkpoint from Section 3.1.
Calibration uses reference activation data to update an operator of the same
state count and TT rank while leaving Transformer weights fixed.
The analytic control uses stationary-state expectations and exact surrounding
arithmetic, including exact variances, with the same calibrated nonlinear
operators. Open markers indicate three trained-model means, each obtained
from three stream realizations for finite-stream configurations. Filled markers
and error bars denote the mean and sample standard deviation across these
three model means. Each run uses the same 128 held-out sequences.}
\label{fig:transformer_depth}
\end{figure}

% Performance figure: after the stream-length paragraph.
\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/Fig_transformer_performance.pdf}
\caption{\textbf{End-to-end retrieval accuracy and numerical fidelity.}
\textbf{a}, Four-class associative-retrieval accuracy of the complete stochastic
Transformer versus bitstream length. The floating-point reference achieves
100\% accuracy on the same test sequences; chance performance is 25\%.
\textbf{b}, Relative error of the centred classification logits, evaluated
over all 128 test sequences. Subtracting each logit vector's mean removes a
common offset that does not affect class probabilities. The dashed analytic
reference is the mean over the three calibrated models.
Open markers show results averaged over three stream realizations for each
of three independently trained models. Filled markers and error bars denote
the mean and sample standard deviation across model means; stream realizations
are nested repetitions. Zero observed accuracy variation does not imply a
population confidence bound. Each stochastic phase uses $L$ sampling cycles;
the softmax and scalar FSM burn-in periods of 256 and 64 cycles, respectively,
are excluded from the horizontal axis.}
\label{fig:transformer_performance}
\end{figure}

% Ablation figure: after the final Results paragraph.
\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/Fig_transformer_ablation.pdf}
\caption{\textbf{Operator coverage and simple calibration in the complete network.}
\textbf{a}, Relative centred-logit error at $L=256$ when stochastic computation
covers the attention cores, additionally all learned linear stages, and
additionally layer normalization and GELU. The final configuration evaluates
the full computational graph. All three configurations use calibrated
HI-SMURF and Sobol arithmetic; operations outside each indicated subset are
evaluated exactly. \textbf{b}, Full-graph error for direct transfer with Sobol
arithmetic, calibrated operators with independent streams, and calibrated
operators with Sobol streams. FSM driving streams remain independent
Bernoulli streams in every configuration. Faint lines connect the same trained
model across conditions. Open markers show three model means, each averaging
three stream realizations; filled markers and error bars show the mean and
sample standard deviation across these model means. No Transformer weights
are updated during calibration.}
\label{fig:transformer_ablation}
\end{figure}
'''
 (ROOT/'section3_3_figure_captions.tex').write_text(captions)
 methods=r'''\subsection{Complete Transformer simulation}
\label{methods:full_transformer}

\paragraph{Architecture and associative-retrieval task.}
We implemented an encoder-only pre-layer-normalized Transformer with four
blocks, 16 tokens, model width 32, two attention heads of width 16, FFN width
64 and GELU activations. Inputs were 37-dimensional vectors containing a
32-way key indicator, a four-way value indicator and a query flag.
For each sequence, 15 distinct keys were sampled uniformly without replacement
from 32 possibilities; values were independently sampled from four classes.
The final token copied one randomly selected stored key, set the query flag
and contained no value information. Its target was the value associated with
that key. The network used a learned 37-to-32 input projection, learned
positional embeddings, four complete encoder blocks, final LayerNorm and a
32-to-4 classifier on the final query token. The reference model contains
36,100 parameters. There are 18 linear stages when each fused QKV projection
is counted as one stage, equivalent to 26 component projections if Q, K and V
are counted separately. There are nine LayerNorm operations and four GELU
operations. Attention is bidirectional and unmasked; dropout is disabled.
This benchmark does not include a decoder or cross-attention.

\paragraph{Reference training and data separation.}
Three models were trained independently with initialization seeds 11, 22
and 33. Each used 1,800 AdamW updates with batch size 64, learning rate 0.002,
weight decay 0.01 and gradient-norm clipping at 1. The learning rate followed
a cosine schedule ending at 0.0002. A fresh synthetic batch was generated at
each update. Checkpoints were selected on 512 validation sequences generated
with seed 310001, using accuracy and then cross-entropy to break ties.
Development sets used seeds 320001 and 320003. Calibration used 128 training
sequences with seed 330001 and 128 separate validation sequences with seed
330002. The test set comprised 128 sequences generated with seed 340001,
fixed before stochastic evaluation and shared across all comparisons.
Task-label integrity and the absence of query-value information were checked.

\paragraph{HI-SMURF calibration.}
The original joint categorical operator used 16 states per input and TT rank
8, with 16,512 parameters. For each trained Transformer, reference attention
score rows were extracted from all four blocks and both heads on the
calibration inputs, yielding 16,384 rows per calibration split. We fine-tuned
the original TT core logits for 1,200 Adam steps with batch size 512, initial
learning rate 0.01, a cosine decay to 0.001 and gradient-norm clipping at 5.
The loss was 256 times the mean squared probability error against exact
softmax. The sorted, centred input encoding was evaluated with bounds 4 and
8. Checkpoints were assessed every 200 steps and selected by probability RMSE
on the separate calibration-validation set. Bound 8 was selected for all
three models. Each selected operator was shared across blocks and heads
within its model. Calibration used no task labels and did not modify any
Transformer parameters. The tested encoding bounds and all calibration
results are retained in the source data.

\paragraph{Linear and attention operations.}
All main learned linear stages generated stochastic bit products from the
actual inputs and weights, followed by signed counting and scale restoration.
Signed values used a separate sign and a magnitude bitstream. Input vectors
and weight rows used power-of-two scales covering their maximum absolute
entries. The same rules were used for IID and Sobol arithmetic. Sobol
generation followed the two-coordinate, independently digitally shifted
construction in Section 3.2, with scramble seed 7001 and 32-bit shifts.
Weight streams were shared across the eight sequences in each simulation
batch. Independent source offsets separated stages and batches.
The attention subgraph included stochastic QK products, actual multivariate
FSM trajectories and categorical events, and stochastic AV aggregation with
value centring. Softmax FSMs started centrally, used independent Bernoulli
driving streams and ran for 256 burn-in cycles before collecting $L$ events.
The learned QKV and output projections were included in full-graph simulation.

\paragraph{GELU and LayerNorm.}
GELU was implemented as $x\Phi(x)$, using an eight-state FSM to approximate
the Gaussian cumulative distribution function $\Phi$. Its conditional
probability table was fitted by bounded linear least squares on 2,001
uniformly spaced points in $[-4,4]$, with all coefficients constrained to
$[0,1]$. The encoding was $p=\mathrm{clip}((x+4)/8,0,1)$.
An independent magnitude stream for $x$ was combined with the emitted gate
bit through an AND operation and the stored sign. The CDF gate's RMSE on
interleaved validation midpoints was 0.005881.
For LayerNorm, the feature mean was computed numerically and subtracted.
Variance was estimated from independent magnitude-bit products of the
centred features, summed over features and cycles. The variance plus
$10^{-5}$ was represented as $2^e m$, with $m\in[0.5,1)$.
A four-state FSM approximated $1/\sqrt{2m}$ using another bounded table fitted
on 2,001 points; midpoint probability RMSE was 0.002158. Gate emissions were
combined with centred-feature magnitude bits, followed by exponent-scale
restoration and the learned static affine gain and bias. Both scalar FSMs
started centrally and used 64 burn-in cycles, followed by $L$ gate samples.
Their driving streams remained IID Bernoulli; arithmetic and gate-emission
uniforms used the chosen IID or Sobol source. This is a moment-based
normalization implementation with a scalar FSM reciprocal-square-root stage.

\paragraph{Controls and metrics.}
We evaluated direct transfer and calibrated full-graph models at
$L\in\{64,256,1024\}$ with stream seeds 101, 202 and 303 for each trained
model. At $L=256$, further controls used calibrated IID arithmetic, stochastic
attention cores with exact surrounding operations, and stochastic attention
plus all learned linear stages with exact normalization and GELU.
Analytic controls used the stationary-state expectations of the corresponding
HI-SMURF and scalar FSMs, with exact linear arithmetic and exact variances.
There were 87 model/configuration/stream combinations in total.
For classifier logits $z_b$, centred logits were defined by
$\bar z_b=z_b-\mathrm{mean}(z_b)\mathbf{1}$. Their relative error was
\begin{equation}
\varepsilon_z=
\sqrt{\frac{\sum_b\|\widehat{\bar z}_b-\bar z_b\|_2^2}
{\sum_b\|\bar z_b\|_2^2}}.
\end{equation}
For each block, we similarly evaluated relative error over the concatenated
query-token representations. We also evaluated the residual updates
$\Delta h_{b,\ell}=h_{b,\ell}-h_{b,\ell-1}$, using their own reference norm
in the denominator. These are positions within the same trained four-block
network, not independently trained networks of different depths.
Class accuracy, agreement with the reference, raw-logit RMSE and query-feature
cosine similarity were additionally recorded. First, three stream realizations
were averaged within each trained model. Reported means and sample standard
deviations were then computed over the three model means. Thus stream
realizations were treated as nested repetitions, not as independent model
replicates. Accuracy on this finite test set is not a population confidence
bound or a result on a real vision/language benchmark.

\paragraph{Numerical scope and verification.}
All main matrix products, attention products, normalization variance products
and scalar gate products used explicitly generated bits and signed counts.
TT state contractions, centering means, residual and bias additions, static
normalization-affine gains, power-of-two range handling and decoding and
re-encoding between stages remained numerical operations. The experiment
therefore evaluates a complete mixed-representation Transformer graph, without
claiming a continuous all-bitstream or gate-level implementation. An exact
NumPy forward pass matched the PyTorch reference to within $9.73\times10^{-7}$
in logit magnitude on a separate verification set. Residual-update consistency,
endpoint products, bounded conditional probabilities and the absence of exact
GELU evaluation in the full stochastic path were checked. Hardware synthesis,
area, energy and latency were not evaluated.
'''
 (ROOT/'section3_3_transformer_methods.tex').write_text(methods)
 coverage=[
  {'component':'Input projection and classifier','count':2,'backend':'Actual sign-magnitude bit products; bias addition numerical'},
  {'component':'QKV projections (fused)','count':4,'backend':'Actual sign-magnitude bit products'},
  {'component':'Attention output projections','count':4,'backend':'Actual sign-magnitude bit products'},
  {'component':'FFN linear layers','count':8,'backend':'Actual sign-magnitude bit products'},
  {'component':'Multi-head attention','count':4,'backend':'Two heads each: actual QK and AV streams; FSM/categorical HI-SMURF; numerical TT contractions'},
  {'component':'LayerNorm','count':9,'backend':'Numerical mean; SC variance; 4-state inverse-sqrt gate/product; numerical static affine scales'},
  {'component':'GELU','count':4,'backend':'8-state CDF gate times sign-magnitude stream'},
  {'component':'Residual connections','count':8,'backend':'Numerical accumulator'},
  {'component':'Positional embedding','count':1,'backend':'Learned numerical addition'}]
 pd.DataFrame(coverage).to_csv(DATA/'operator_coverage.csv',index=False)
 readme=f'''# Section 3.3: complete Transformer simulation

This package contains 87 measured runs of a four-block encoder-only Transformer
on a controlled synthetic associative-retrieval task. There are three trained
reference models and three stochastic-source seeds nested within each model.
The original Transformer weights are frozen during all operator calibration.

At L=256, direct-transfer accuracy is {100*val('generic_full',metric='accuracy'):.3f}%
and calibrated full-graph accuracy is {100*val(metric='accuracy'):.3f}%.
Relative centred-logit errors are {val('generic_full'):.6f} and {val():.6f}.
The frozen architecture has 36,100 parameters; the shared HI-SMURF adds 16,512
operator coefficients, with the same count before and after calibration.

## Deliverables

- Three core figures, each as vector PDF, editable SVG and 400 dpi PNG.
- Results: `section3_3_transformer.tex`.
- Captions and insertion environments: `section3_3_figure_captions.tex`.
- Full protocol: `section3_3_transformer_methods.tex` and `protocol.json`.
- Every run's predictions, labels, query features and residual updates.
- Reference and calibrated operator checkpoints and complete code.

## Reproduce

The data are generated locally; no download or external model is required.
Install NumPy, SciPy, PyTorch, pandas, matplotlib and Pillow, then run:

```
python teacher.py
python stochastic_model.py
python calibrate_softmax.py
python validate.py
python run_experiments.py
python render_figures.py
python verify_artifacts.py
python write_text.py
```

For plotting only, run `render_figures.py` on the included results. The
experiment runner resumes verified result records. Re-running from zero should
use a separate copy with the generated `source_data/runs` directory removed.
The recorded software versions are in `provenance/environment.json`.

## Interpretation and limitations

This is a complete four-block encoder graph, including projections, two-head
attention, normalization, GELU, residuals and the classifier. It is a mixed
representation numerical simulation. Source bits are actually generated, but
TT contractions, residual/bias accumulation, means, static LayerNorm gains,
range handling and interstage conversions are numerical. `operator_coverage.csv`
records this scope. Linear-stage counts group QKV as one matrix; counting the
three Q/K/V maps separately gives 26 component projections instead of 18 stages.

The task is synthetic four-class retrieval, with 128 held-out sequences shared
across conditions. It does not measure performance on a real image/language
dataset, a pretrained large model, a decoder or a hardware implementation.
The error curves follow block positions within the same four-layer models.
They must not be described as arbitrary-depth scaling experiments.
Error bars are the sample SD of three model means, each averaging three stream
realizations. They are not confidence intervals or nine independent models.

## Data and provenance

- `source_data/all_metrics.csv`: 87 run records.
- `source_data/per_model_metrics.csv`: stream-averaged model results.
- `source_data/summary_metrics.csv`: means and sample SD across model means.
- `source_data/depth_metrics.csv`: all per-block errors.
- `source_data/runs/*.npz`: complete reference and predicted logits, labels,
  query representations and residual updates.
- `source_data/test_inputs.npz`: full evaluation input and generator metadata.
- Calibration and training histories are provided, including both tested bounds.
- SHA-256 hashes in `manifest_sha256.json` identify the packaged artifacts.

## Primary references

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745).
- [SciPy Sobol generator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html).
'''
 (ROOT/'README.md').write_text(readme)
 zh=f'''本节推进到完整四层 encoder-only Transformer。主模型 36,100 参数，16 tokens，d=32，两头，FFN=64。
检索任务包含 15 个键值 token 和一个 query token，4 类；query 不包含目标值。
三个教师独立训练；正式测试 128 个序列，每个教师三个位流种子；共 87 组运行。
改进是基于参考激活的小规模 HI-SMURF 校准，状态16/rank8/参数16512保持不变，教师参数不变。
L=256 直接移植准确率 {100*val('generic_full',metric='accuracy'):.2f}%，校准后 {100*val(metric='accuracy'):.2f}%。
相对中心化logit误差 {val('generic_full'):.5f} -> {val():.5f}。
所有主线性层、QK、AV、LayerNorm方差和GELU门乘法都产生真实位流。
TT收缩、均值、残差加法、归一化静态仿射增益、缩放和阶段转换仍是数值运算，不能宣称全位流ASIC。
LayerNorm采用矩估计和4状态标量FSM；GELU采用8状态CDF门。
模型深度固定4层，逐层误差不是独立深度扩展实验。
误差棒为三个模型均值的样本标准差，各模型先平均三个位流种子；不是9个独立模型。
本节是受控合成任务，不是实际图像或语言数据集的精度结论。
'''
 (ROOT/'reading_notes_zh.txt').write_text(zh)
 files=sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest_sha256.json')
 manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 (ROOT/'manifest_sha256.json').write_text(json.dumps(manifest,indent=2))
 buf=io.BytesIO()
 with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files+[ROOT/'manifest_sha256.json']:z.writestr(str(Path(ROOT.name)/p.relative_to(ROOT)),p.read_bytes())
 out=ROOT.parent/'Section3_3_Transformer_Figures_and_Text.zip';out.write_bytes(buf.getvalue())
 with zipfile.ZipFile(out) as z:assert z.testzip() is None
 print(json.dumps(dict(archive=str(out),bytes=out.stat().st_size,files=len(files)+1),indent=2))

if __name__=='__main__':main()
