FROST: separated formation and convergence figures, L <= 4096
===========================================================

使用方法
1. 将 FROST_attention_formation.pdf 与 FROST_attention_convergence.pdf
   上传到 Overleaf 的 Images 目录。
2. 用 FROST_with_convergence.tex 替换主文件。正文、图注、交叉引用和
   L=4096 的实测数值均已更新。新图分别为 Fig. 8 和 Fig. 9，原重构图
   自动顺延为 Fig. 10；位置在分类实验之后、BSDS500 重构之前。
3. FROST_convergence_package.zip 是可直接上传的完整工程：主 .tex、
   .bib、sn-jnl.cls、sn-mathphys-num.bst 与 Images 均位于工程根目录。
   选择 FROST_with_convergence.tex 为主文件，使用 pdfLaTeX + BibTeX。
4. FROST_with_convergence_preview.pdf 为本次完整编译预览（21 页）。

排版修改
原四联图拆为两张双联图，分别展示注意力形成机制与误差收敛。
保留白色背景、Nimbus Sans 字体、青绿 FROST、橙色 base TT 以及蓝色
采样误差曲线；增大字号与面板留白。所有新增曲线、轨迹、概率快照与
误差热图的有限采样长度均不超过 4096。PDF 内的图形、热图、色条与
文字均为矢量对象，字体已嵌入；SVG 保留可编辑文字，附字体版权说明。

Experimental scope and actual measurements
-----------------------------------------
One existing frozen CIFAR-10 Transformer, model seed 11. The generic and
calibrated operators both use 16 FSM states, TT rank 8, bound 4 and 16,512
parameters. No weights or operator were refitted for these figures.
Attention scores are exact numerical Transformer scores; stochastic QK
arithmetic, AV aggregation and downstream stages are outside this diagnostic.
The plots show sampling convergence at fixed weights, rather than training
loss. The analytic stationary references are limits, not longer finite runs.

The first eight test images of the existing stratified diagnostic subset form
a pool of 64 image/block/head matrices. Eight full matrices were sampled
without replacement (PCG64 seed 20261009), retaining 128 rows spanning five
images. Matrix identifiers, official test-image indices and row identities
are included in protocol.json and attention_rows.csv. The representative
row is the upper-median reference-entropy row, chosen before sampling.

Eight stream seeds, 101, 202, ..., 808, use actual nested 4096-cycle event
prefixes after 256 burn-in cycles. Saved events were truncated and metrics
recomputed, without resampling or artificial interpolation. Displayed finite
checkpoints are 16, 32, 64, 128, 256, 512, 1024, 2048 and 4096. The simplex
groups the largest reference probability, second largest and remaining 14
keys, preserving all mass. All TV metrics use all 16 original components.

At L=4096, mean row TV is 0.0650890500962138 for FROST and
0.08151325838241627 for base TT: a 20.149124954800534% reduction.
Stream SDs are 0.0006400546626451528 and 0.0014080887493409912.
Stationary mean row TV is 0.05509013662138756 and 0.07395685647888332,
respectively. FROST mean row TV to its own stationary output decreases
from 0.21705070561743806 at L=64 to 0.029549830451078965 at L=4096.
Bands and uncertainties are one sample SD across eight stream-level means
at fixed scores and weights, rather than independent-model variation or
dataset-level confidence intervals. These results describe the selected
score rows and do not establish hardware area, energy or latency benefits.

Source data
-----------
source_data/inputs.npz: reference scores, exact probabilities, candidate images,
  normalized tokens, row selection and entropy ordering.
source_data/trace_METHOD_seedSEED.npz: actual 4096-cycle categorical events,
  checkpoint probabilities, analytic reference, row errors and microstates.
source_data/stream_metrics.csv and summary.json: all 144 method/seed/length
  metric sets and the plotted aggregate curves.
source_data/atlas_row_tv.csv: all 2304 atlas cells with row identity and SD.
source_data/simplex_checkpoint_coordinates.csv: 72 measured simplex checkpoints.
source_data/representative_states_and_events.csv: 128 displayed state/event cycles.
source_data/protocol.json: selection, seeds, RNG, burn-in and checkpoint hashes.
source_data/prefix_revision.json: event-prefix hashes and revision provenance.
source_data/verification.json: 20 exact original-sampler comparisons and
  144 independent recounts of saved event prefixes; all agree exactly.
source_data/manuscript_checks.json: eleven checked numerical claims, bibliography
  and reference checks, and preservation outside the added visualization block.
source_data/publication_checks.json: vector, font, layout and compilation checks.

Reproduction
------------
From the extracted project root, use Python with NumPy, PyTorch, SciPy,
Matplotlib, fontTools and PyMuPDF, plus a C++17 compiler for the native loop.
The supplied fixed inputs and checkpoints avoid downloading CIFAR-10 again.

python source/run_convergence.py
python source/export_source_data.py
python source/plot_convergence.py
python source/write_manuscript.py source/manuscript_original.tex
python source/format_manuscript.py
latexmk -pdf -interaction=nonstopmode -halt-on-error FROST_with_convergence.tex

The native loop accelerates the original FSM transitions and TT categorical
draws; NumPy PCG64 draws retain the original ordering. attention_core.py
includes the original Python sampler used for the equivalence checks.
truncate_prefixes.py records the present retained-prefix revision and is
idempotent on the supplied data. Native binaries, caches and LaTeX auxiliary
files are excluded. The full source is FROST_with_convergence.tex; the added
block alone is also supplied as FROST_convergence_insert.tex.

本次仅调整排版：删除三处正文内浮动体隔离，将图片设为可在页顶、页底
或浮动页排列，并将参考文献条目间距设为 4 pt。正文、公式、表格、
图注、图像、参考文献内容、字号、页边距及图片尺寸均保持一致。
