from pathlib import Path
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];s=json.loads((R/'results/stage3_summary.json').read_text());F={x['design']:x for x in s['fpga']};P={(x['design'],x['L']):x for x in s['performance']};A={(x['design'],x['L']):x for x in s['asic']}
labels={'fp32_1':'FP32-LUT (1 lane)','fp32_4':'FP32-LUT (4 lanes)','fx16':'FX16 (4 lanes)','frost':'FROST'}
colors={'fp32_1':'#526E9A','fp32_4':'#665191','fx16':'#D89027','frost':'#008B83'}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(7.25,3.2),layout='constrained')
for ax,letter in zip(axes,'ab'):
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#eeeeee',linewidth=.5,zorder=0);ax.text(-.13,1.07,letter,transform=ax.transAxes,fontweight='bold',fontsize=12)
ax=axes[0]
for n in ['fp32_1','fp32_4','fx16']:
 x=P[n,0]['FPGA_tile_us_at8MHz'];y=F[n]['comb']/1000;ax.scatter(x,y,s=38,c=colors[n],marker='s',zorder=3)
 ax.annotate(labels[n],(x,y),xytext=(5,7 if n!='fx16' else -14),textcoords='offset points',fontsize=7.1)
xx=[P['frost',L]['FPGA_tile_us_at8MHz'] for L in [64,256,1024]];yy=[F['frost']['comb']/1000]*3
ax.plot(xx,yy,'o-',color=colors['frost'],lw=1,ms=5,zorder=3)
for x,y,L in zip(xx,yy,[64,256,1024]):ax.annotate(f'L = {L}',(x,y),xytext=(0,-14 if L==256 else 8),ha='center',textcoords='offset points',fontsize=7,color=colors['frost'])
ax.text(1050,5.0,'FROST',color=colors['frost'],fontsize=8,fontweight='bold')
ax.set_xscale('log');ax.set_xlim(250,3800);ax.set_ylim(0,29);ax.set_xticks([300,700,1500,3000],[300,700,1500,3000]);ax.minorticks_off();ax.set_xlabel('Latency per 16-query tile (µs)');ax.set_ylabel('ECP5 combinational resources (10³)');ax.set_title('FPGA implementation · common 8 MHz',loc='left',pad=14)
ax=axes[1]
for n in ['fp32_1','fp32_4','fx16']:
 a=A[n,0];x=a['area_um2']/1000;y=a['tile_energy_uJ'];ax.scatter(x,y,c=colors[n],marker='s',s=38,zorder=3)
 offsets={'fp32_1':(-8,8),'fp32_4':(-8,8),'fx16':(-8,-13)};ax.annotate(labels[n],(x,y),xytext=offsets[n],ha='right',textcoords='offset points',fontsize=7.1)
x=A['frost',256]['area_um2']/1000;yy=[A['frost',L]['tile_energy_uJ'] for L in [64,256,1024]];ee=[A['frost',L]['tile_energy_uJ_sd'] for L in [64,256,1024]]
ax.errorbar([x]*3,yy,yerr=ee,fmt='o-',color=colors['frost'],lw=1,ms=5,capsize=2,zorder=3)
for y,L in zip(yy,[64,256,1024]):ax.annotate(f'FROST, L = {L}',(x,y),xytext=(8,0),va='center',textcoords='offset points',fontsize=7,color=colors['frost'])
ax.set_yscale('log');ax.set_xlim(44,99);ax.set_ylim(.25,7);ax.set_yticks([.3,.5,1,2,5],[.3,.5,1,2,5]);ax.minorticks_off();ax.set_xlabel('Mapped standard-cell area (10³ µm²)');ax.set_ylabel('Estimated energy per tile (µJ)');ax.set_title('ASIC estimate · Nangate45, 15 MHz',loc='left',pad=14)
fig.savefig(R/'FROST_hardware_stage3.pdf');fig.savefig(R/'FROST_hardware_stage3.svg');fig.savefig(R/'FROST_hardware_stage3.png',dpi=180);plt.close(fig)
# Every number below is inserted from measured simulation/tool-report data.
comb_gain=100*(1-F['frost']['comb']/F['fp32_1']['comb']);energy_gain=100*(1-A['frost',256]['tile_energy_uJ']/A['fp32_1',0]['tile_energy_uJ']);lat_gain=100*(1-P['frost',256]['tile_cycles']/P['fp32_1',0]['tile_cycles'])
fr=[]
for n in ['fp32_1','fp32_4','fx16','frost']:
 d=F[n];fr.append(f"{labels[n]} & {d['comb']:,} & {d['ff']:,} & {d['dsp']} & {d['bram18']} & {d['fmax_estimate_min_MHz']:.2f}--{d['fmax_estimate_max_MHz']:.2f} "+r'\\')
pr=[];ar=[]
for n,L in [('fp32_1',0),('fp32_4',0),('fx16',0),('frost',64),('frost',256),('frost',1024)]:
 d=P[n,L];a=A[n,L];name=labels[n]+(f', $L={L}$' if L else '')
 pr.append(f"{name} & {d['cold_row_cycles']:,} & {d['resident_row_cycles']:,} & {d['tile_cycles']:,} & {d['FPGA_tile_us_at8MHz']:.1f} "+r'\\')
 pm=lambda k: (f"${a[k]:.3f}\\pm{a[k+'_sd']:.3f}$" if n=='frost' else f"{a[k]:.3f}")
 ar.append(f"{name} & {a['area_um2']/1000:.2f} & {pm('dynamic_mW')} & {pm('leakage_mW')} & {pm('total_mW')} & {pm('tile_energy_uJ')} "+r'\\')
tex=r'''% Requires booktabs, graphicx, amsmath and amssymb. Add entries from FROST_hardware_tools.bib.
% Upload FROST_hardware_stage3.pdf to Images/.
\section{Hardware feasibility of fused probabilistic attention}
\label{sec:fpga_evaluation}

We implemented a fused attention kernel to test whether categorical sampling
can replace dense multiplication of probabilities and values within a hardware
boundary. The kernel receives a row of 16 attention scores and a matrix of
16 keys by 16 value channels, and produces one 16-component output vector.
It includes score preprocessing, coefficient lookup, stochastic-state and
random-number updates, value storage, accumulation, format conversion and
control. Query--key products, projections, feed-forward layers and
whole-network classification are outside this boundary. The evaluation
characterizes implementation cost without accuracy matching; it does not
constitute an end-to-end field-programmable gate array (FPGA)
implementation of FROST.

\paragraph{Implementations and arithmetic boundary.}
The FROST candidate uses tensor-train rank 2, 16 finite-state-machine states
per score input, 8-bit cumulative distribution function (CDF) entries and
16 burn-in updates. It was frozen before the hardware experiments.
Each categorical sample $J_t\in\{1,\ldots,16\}$ selects a value row directly,
giving the estimator
\begin{equation}
\widehat{\boldsymbol y}=\frac{1}{L}\sum_{t=1}^{L}\boldsymbol V_{J_t,:},
\label{eq:fused_categorical_attention}
\end{equation}
where $\boldsymbol V\in\mathbb{R}^{16\times16}$ is the value matrix,
$\widehat{\boldsymbol y}\in\mathbb{R}^{16}$ is the reconstructed attention
output, $t$ indexes retained samples, and $L$ is the stream length.
The hardware accumulates selected quantized values in sixteen signed
32-bit accumulators; it does not materialize a probability vector or use
dense probability--value multipliers. Sorting, inverse index mapping,
32 separately initialized xorshift32 states, input encoding and burn-in
remain inside the measured boundary. The 15 binary latent transitions each require one non-terminal
threshold; the final category transition retains 15 thresholds.
Terminal unit-CDF entries are implicit. The
resulting constant lookup data contain 944 bytes before logic mapping.
The same circuit supports $L\in\{64,256,1024\}$.

Floating-point controls use single precision (FP32), with either one
or four shared fused multiply-add (FMA) lanes implemented using Berkeley
HardFloat~\cite{hauser2019hardfloat}. Both use a 1,025-entry FP32 exponential
look-up table (LUT), with spacing $1/64$ over score differences from 0 to 16,
and compute one reciprocal per query. They accumulate unnormalized
exponential--value products and normalize the final vector, rather than
performing one division per probability. A fixed-point control (FX16)
uses four integer multiply lanes, a LUT with 16 fractional bits, widened
intermediate accumulators and one restoring reciprocal per query.
These are matched kernel implementations, not reproductions of a published
whole-Transformer accelerator.

All kernels accept the same signed 16-bit score and value words, with ten
fractional bits, and return signed 32-bit words with ten fractional bits.
FP32 arithmetic operates on the integer-unit value representation and
rounds the normalized result back to this common interface; its output
conversion is included. The full $16\times16$ value matrix is loaded once
per tile of 16 consecutive queries, while each query loads its own 16
scores. Transfers use one input word per cycle. Kernels do not overlap
successive queries. Random-seed initialization precedes the run, whereas
burn-in and output transfers are included for every query.

\paragraph{Functional verification and FPGA implementation.}
Register-transfer-level (RTL) simulations with Verilator covered 3,200 score rows and their corresponding
value matrices from the 200-image evaluation subset, without further
fitting. FROST was tested at all three stream lengths and stream seeds
4004, 5005 and 6006; deterministic controls were tested once. All 614,400
output components agreed bit for bit with independent integer or C++
floating-point references. This establishes arithmetic implementation
agreement on the evaluated inputs, not equivalence to exact softmax or
classification accuracy on hardware.

Yosys 0.33 synthesis and nextpnr-ecp5 0.6 placement and routing targeted an
LFE5U-85F FPGA, CABGA381 package, speed grade 6~\cite{yosysSoftware,nextpnrSoftware}.
All four designs completed routing with seeds 1, 2 and 3 at a common
8-MHz clock. Initial FP32 implementations did not meet the 15-MHz target;
those runs were retained, and every design was rerouted at 8 MHz.
Table~\ref{tab:fused_fpga_resources} reports post-route device resources.
The timing-derived frequency ranges are estimates from the 8-MHz runs,
not measured chip frequencies or results of a maximum-frequency search.
Board pin placement and external input/output delays were not constrained;
the results characterize routed kernels rather than a board-ready system.

\begin{table}[!htbp]
\centering
\caption{Post-route FPGA resources for fused attention kernels.}
\label{tab:fused_fpga_resources}
\small\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrr@{}}
\toprule
Implementation & COMB & FF & DSP & BRAM & \shortstack{Timing estimate\\(MHz)}\\
\midrule
'''+'\n'.join(fr)+r'''
\bottomrule
\end{tabular*}
\par\smallskip\footnotesize
COMB denotes ECP5 \texttt{TRELLIS\_COMB}, including logic used for
arithmetic and distributed memory; it is not a literal LUT4 count.
FF denotes flip-flops, digital signal processing (DSP) resources denote \texttt{MULT18X18D}, and block random-access memory (BRAM) resources denote
nominal 18-kbit \texttt{DP16KD} blocks. Resource counts were identical
across the three routing seeds; frequency ranges span all three runs.
FROST uses distributed value storage and constant logic-mapped CDFs;
zero block RAM therefore does not mean zero storage cost.
\end{table}

'''+f'''FROST used {comb_gain:.1f}\\% fewer combinational resources than the
single-lane FP32-LUT control, with no DSP or block-RAM instances, but
required more flip-flops. At $L=256$, its tile cycle count was
{lat_gain:.1f}\\% lower than that control. The four-lane FP32-LUT and FX16
controls completed tiles faster at the common clock, demonstrating an
explicit resource--latency trade-off (Table~\\ref{{tab:fused_cycles}}).
'''+r'''
\begin{table}[!htbp]
\centering
\caption{RTL-derived latency, including declared input and output transfers.}
\label{tab:fused_cycles}
\small\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr@{}}
\toprule
Implementation & \shortstack{Cold row\\(cycles)} & \shortstack{Resident row\\(cycles)} & \shortstack{16-query tile\\(cycles)} & \shortstack{Tile at 8 MHz\\($\mu$s)}\\
\midrule
'''+'\n'.join(pr)+r'''
\bottomrule
\end{tabular*}
\par\smallskip\footnotesize
A cold row includes loading all 256 value words; a resident row reuses
that matrix. Tile latency comprises one cold row and 15 resident rows.
These quantities are kernel-level timings, not image-inference latency.
With no tile overlap, sustained tile throughput is clock frequency divided
by tile cycles, assuming continuously supplied inputs.
\end{table}

\paragraph{Standard-cell power estimates.}
In a separate application-specific integrated circuit (ASIC) assessment,
the same RTL was mapped to the Nangate45
typical standard-cell library at 1.10 V and $25\,^{\circ}$C. All memories
were mapped into cells; no static random-access memory (SRAM) macros were used. OpenSTA 3.1.0 estimated
cell power from zero-delay gate-level switching activity at a common
15-MHz clock~\cite{openstaSoftware}. The activity workload was the first
16-query evaluation tile, including value loading, score loading, burn-in,
computation and output. The three stream seeds were retained for FROST.
Gate simulation matched the RTL for all 3,072 checked components and
reported zero unannotated pins in each activity analysis. Pre-layout
register timing met the 15-MHz constraint for all reported runs.

Let $C_{\mathrm{tile}}$ denote the tile cycle count, $f$ the clock frequency
in hertz, and $P_{\mathrm{cell}}$ the estimated active cell power in watts,
including characterized internal, switching and leakage components.
Estimated energy per tile is
\begin{equation}
E_{\mathrm{tile}}=P_{\mathrm{cell}}\frac{C_{\mathrm{tile}}}{f}.
\label{eq:fused_tile_energy}
\end{equation}
The values in Table~\ref{tab:fused_asic_power} use $f=15$ MHz within this
standard-cell flow. They are not FPGA power estimates and are not combined
with FPGA throughput. Clock-tree implementation, extracted interconnect
parasitics, time-resolved glitch activity, pads, host processing and
external memory are absent. The single activity tile and stream variability
do not provide a dataset-level confidence interval for power.

\begin{table}[!htbp]
\centering
\caption{Pre-layout standard-cell area and activity-based power estimates.}
\label{tab:fused_asic_power}
\fontsize{8}{9.5}\selectfont\setlength{\tabcolsep}{2.5pt}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrr@{}}
\toprule
Implementation & \shortstack{Area\\($10^3\,\mu$m$^2$)} & \shortstack{Dynamic\\(mW)} & \shortstack{Leakage\\(mW)} & \shortstack{Total\\(mW)} & \shortstack{Energy/tile\\($\mu$J)}\\
\midrule
'''+'\n'.join(ar)+r'''
\bottomrule
\end{tabular*}
\par\smallskip\footnotesize
All entries are tool estimates for mapped standard cells, not measured
silicon or board values. FROST power and energy entries are means
$\pm$ sample standard deviations across three stream seeds on one tile.
The same FROST circuit is used at all stream lengths. Deterministic
controls use the identical tile once. Area is summed mapped-cell area,
not a placed core area. The two FP32 controls include one and four
shared fused multiply--add lanes, respectively.
\end{table}

'''+f'''At $L=64$, FROST used {100*(1-F['frost']['comb']/F['fp32_4']['comb']):.1f}\\% fewer FPGA combinational resources
and completed a tile {100*(1-P['frost',64]['tile_cycles']/P['fp32_4',0]['tile_cycles']):.1f}\\% sooner than the four-lane FP32 control at
the common FPGA clock. In the separate standard-cell flow, its mapped
area and estimated tile energy were {100*(1-A['frost',64]['area_um2']/A['fp32_4',0]['area_um2']):.1f}\\% and {100*(1-A['frost',64]['tile_energy_uJ']/A['fp32_4',0]['tile_energy_uJ']):.1f}\\% lower, respectively.
These reductions describe a short-stream operating point without accuracy matching.

At $L=256$, FROST's estimated tile energy was {energy_gain:.1f}\\% lower
than the single-lane FP32-LUT control. This benefit did not coincide with
a lower mapped-cell area than that control. The four-lane FP32 and FX16
results, together with the stream-length sweep, show why one resource
count or one floating-point schedule is insufficient to establish a
general power, performance and area (PPA) advantage (Fig.~\\ref{{fig:fused_hardware_stage3}}).
'''+r'''
\begin{figure}[!htbp]
\centering
\includegraphics[width=\linewidth]{Images/FROST_hardware_stage3.pdf}
\caption{\textbf{Implementation trade-offs of fused probabilistic attention.}
\textbf{a}, ECP5 combinational resources versus latency for a 16-query tile
at the common 8-MHz FPGA clock. Connected FROST points use one programmable
circuit at different stream lengths; all other points are separate
implementations. DSP and block-RAM costs are reported separately in
Table~\ref{tab:fused_fpga_resources}.
\textbf{b}, Mapped standard-cell area versus estimated energy per tile
for the separate Nangate45, 15-MHz assessment. FROST error bars show sample
standard deviations across three stream seeds on one activity tile and
may be smaller than the symbols. Smaller values are preferable on both
axes in each panel. Neither panel assumes matched approximation error.}
\label{fig:fused_hardware_stage3}
\end{figure}

The hardware study therefore demonstrates a realizable probability-sampling
kernel and a stream-dependent implementation trade-off. It does not
establish dominance over optimized fixed-point arithmetic, whole-network
energy savings, or quality-matched hardware superiority. In particular,
the short-stream configurations were retained here to characterize hardware
cost, despite failing the earlier software selection threshold. Complete
Transformer integration, accuracy evaluation of its hardware outputs and
post-layout or board-level power measurement remain outside this evaluation.
'''
(R/'FROST_hardware_stage3_section.tex').write_text(tex)
report=['FROST 硬件优化第三阶段：实际RTL、布局布线和活动驱动功耗估计','',
'已完成硬件内核实验。本轮按你的要求暂不以精度差距作为筛选门槛；这不代表不同方法已经达到相同精度。',
'实现范围：16个key、16维value的attention归一化+value聚合。不是完整Transformer，没有FPGA上板测量。',
'', '1. 实现与验证',
'FROST沿用rank2、8-bit CDF、16次burn-in；每个类别样本直接选择value并做整数累加。16个独立驱动随机状态+16个发射随机状态、排序、参数逻辑、缓冲、输出转换均计入。',
'FP32-LUT基线分别为1/4路IEEE浮点FMA，均每行只求一次倒数；另有4路FX16强对照。所有设计接受相同Q10输入并输出Q10，value矩阵在16个query间复用。',
f"RTL与独立参考逐位核对{s['rtl_checked_words']:,}个输出，门级网表与RTL核对{s['gate_checked_words']:,}个输出，均无不一致。",'',
'2. FPGA实际布局布线（共同8MHz）',
'器件LFE5U-85F/CABGA381/speed6，三种布线seed。初始15MHz浮点方案未通过时序，四种设计统一改为8MHz重新布线，失败日志保留。',
'方法 | COMB | FF | DSP | 18-kbit BRAM | 时序推算频率范围MHz']
for n in ['fp32_1','fp32_4','fx16','frost']:
 d=F[n];report.append(f"{labels[n]} | {d['comb']} | {d['ff']} | {d['dsp']} | {d['bram18']} | {d['fmax_estimate_min_MHz']:.2f}-{d['fmax_estimate_max_MHz']:.2f}")
report+=['COMB是器件组合资源计数，不等同于LUT4；FROST的0块RAM仍包含分布式value存储和CDF常数逻辑。上述频率范围是时序报告估计，不是板上测量或最高频率扫描。','', '3. 实测仿真周期与FPGA共同频率下延迟','方法 | tile周期 | 8MHz下每tile微秒']
for (n,L),d in P.items():report.append(f"{labels[n]}{(' L='+str(L)) if L else ''} | {d['tile_cycles']} | {d['FPGA_tile_us_at8MHz']:.1f}")
report+=['一个tile包括16个query、256个value输入、256个score输入和全部输出。没有query间重叠；这些数字不能叫images/s或每张图能耗。','',
'4. 独立ASIC估计（Nangate45、1.10V、25℃、15MHz）','方法 | 面积µm² | 总功耗mW | 每tile能耗µJ']
for (n,L),d in A.items():report.append(f"{labels[n]}{(' L='+str(L)) if L else ''} | {d['area_um2']:.3f} | {d['total_mW']:.4f} ± {d['total_mW_sd']:.4f} | {d['tile_energy_uJ']:.4f} ± {d['tile_energy_uJ_sd']:.4f}")
report+=['FROST的±仅为同一tile上三个随机种子的样本标准差；确定性基线只运行一次。活动来自第一个独立评估矩阵的16个query，不是整个测试集功耗统计。',
'所有存储映射到标准单元，没有SRAM宏；面积是单元面积之和。功耗包含单元内部、切换和漏电估计，未包含布局布线寄生、时钟树、毛刺时序、pad、主机和外存。ASIC功耗不能除以FPGA吞吐率。','',
'5. 当前结论',f'L64相对四路FP32：FPGA组合资源减少{100*(1-F["frost"]["comb"]/F["fp32_4"]["comb"]):.1f}%；共同频率下tile延迟减少{100*(1-P["frost",64]["tile_cycles"]/P["fp32_4",0]["tile_cycles"]):.1f}%。在独立ASIC估计中，面积减少{100*(1-A["frost",64]["area_um2"]/A["fp32_4",0]["area_um2"]):.1f}%，能耗减少{100*(1-A["frost",64]["tile_energy_uJ"]/A["fp32_4",0]["tile_energy_uJ"]):.1f}%。这是本轮最有利的短流工作点。',f'对单路FP32，FROST的FPGA组合资源减少{comb_gain:.1f}%；L256的tile周期减少{lat_gain:.1f}%，独立ASIC估计中的每tile能耗减少{energy_gain:.1f}%。',
'但FROST标准单元面积没有低于单路浮点；四路浮点和定点对照提供不同的资源/速度/能耗折中。不能把这一阶段写成FROST在完整网络上全面PPA远优于所有baseline。',
'L64硬件点保留用于你要求的忽略精度比较，不能据此宣称已经解决第二阶段短流精度不达标的问题。完整软件网络准确率没有在本阶段重新测量。','',
'6. Overleaf使用',
'以FROST_hardware_stage3_section.tex替换原先带虚构占位数值的硬件章节；把FROST_hardware_stage3.pdf上传到Images/，将FROST_hardware_tools.bib中的条目合并到参考文献。需要booktabs、graphicx、amsmath和amssymb。',
'正文已经清楚区分FPGA实现、标准单元估计及完整网络未覆盖范围。','',
'7. 下一阶段建议',
'若继续优化，先处理随机数生成、状态寄存器、value存储和数据搬运，不能只缩小CDF参数。然后评估流水线/时钟优化并同样优化浮点和定点基线。若恢复论文的精度约束，须重新验证短流及完整网络。']
(R/'FROST_hardware_stage3_report.txt').write_text('\n'.join(report)+'\n')
(R/'FROST_hardware_tools.bib').write_text(r'''@misc{hauser2019hardfloat,
 author={John R. Hauser}, title={{Berkeley HardFloat Release 1: Verilog Modules}},
 year={2019}, howpublished={\url{https://www.jhauser.us/arithmetic/HardFloat-1/doc/HardFloat-Verilog.html}}, note={Release 1, 29 July 2019}
}
@misc{yosysSoftware,
 author={{YosysHQ}}, title={{Yosys Open SYnthesis Suite}},
 howpublished={\url{https://github.com/YosysHQ/yosys}}, note={Hardware experiments used version 0.33, git 2584903a060}
}
@misc{nextpnrSoftware,
 author={{YosysHQ}}, title={{nextpnr: portable FPGA place and route}},
 howpublished={\url{https://github.com/YosysHQ/nextpnr}}, note={Hardware experiments used nextpnr-ecp5 0.6-3build5}
}
@misc{openstaSoftware,
 author={{Parallax Software}}, title={{OpenSTA: static timing analysis}},
 howpublished={\url{https://github.com/parallaxsw/OpenSTA}}, note={Hardware experiments used version 3.1.0}
}
''')
print('Created vector figure, replacement section, bibliography and Chinese report.')
