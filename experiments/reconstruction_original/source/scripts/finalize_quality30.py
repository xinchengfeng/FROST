"""Verify saved measurements and assemble transparent, reproducible deliverables."""
import argparse,csv,hashlib,json,platform,shutil,sys,zipfile
from pathlib import Path
import numpy as np
import torch,scipy,matplotlib,PIL
from restoration_data import TASKS,center_image,make_input
from restoration_metrics import quality
from report_quality30 import collect,IDS,NAMES

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main(a):
    out,previous,data=a.output,a.previous,a.data
    rows,summary=collect(out)
    old=list(csv.DictReader((previous/'source_data/metrics_per_image.csv').open()))
    checks=[]
    for path in sorted((out/'runs').glob('*.npz')):
        arrays=np.load(path);measured=json.loads(path.with_suffix('.json').read_text())
        first=measured[0];id=first['image_id'];task=first['task'];truth=center_image(data/'test'/f'{id}.jpg')
        assert np.array_equal(arrays['truth'],truth),path
        for row in measured:
            q=quality(arrays[row['method']],truth,task)
            for k in ['psnr','ssim','mse']:assert abs(q[k]-row[k])<1e-10,(path,k)
            assert np.isfinite([row[k] for k in ['psnr','ssim','mse']]).all()
        if first['experiment']=='restoration':
            old_arrays=np.load(previous/'runs'/f'{task}_{id}.npz')
            assert np.array_equal(arrays['truth'],old_arrays['truth'])
            assert np.array_equal(arrays['input'],old_arrays['input'])
        checks.append(dict(run=path.name,rows=len(measured),metrics_recomputed=True))
    paired=[]
    for r in rows:
        if r['experiment']!='restoration':continue
        if r['method'] not in ['input','float','w8a8','analytic','frost','SC_base']:continue
        method=r['method'];oldmethod=f"frost{r['L']}" if method=='frost' else method
        prior=[z for z in old if int(z['image_id'])==r['image_id'] and z['task']==r['task'] and z['method']==oldmethod and int(z['L'])==r['L']]
        if not prior:continue
        assert len(prior)==1,(r,prior)
        p=prior[0];paired.append(dict(image_id=r['image_id'],task=r['task'],method=method,L=r['L'],
            old_psnr=float(p['psnr']),new_psnr=r['psnr'],psnr_change=r['psnr']-float(p['psnr']),
            old_ssim=float(p['ssim']),new_ssim=r['ssim'],ssim_change=r['ssim']-float(p['ssim'])))
    write_csv(out/'source_data/restoration_before_after.csv',paired)
    means=[]
    for task in TASKS:
        for method in ['float','frost']:
            group=[r for r in paired if r['task']==task and r['method']==method]
            assert len(group)==3,(task,method)
            means.append(dict(task=task,method=method,n_images=3,**{key:float(np.mean([r[key] for r in group])) for key in ['old_psnr','new_psnr','psnr_change','old_ssim','new_ssim','ssim_change']}))
    write_csv(out/'source_data/restoration_mean_before_after.csv',means)
    train_ids=sorted(int(p.stem) for p in (data/'train').glob('*.jpg'))
    val_ids=sorted(int(p.stem) for p in (data/'val').glob('*.jpg'))[:20]
    assert not(set(IDS)&(set(train_ids)|set(val_ids)))
    render_cks={id:torch.load(out/'checkpoints'/f'render_{id}.pt',weights_only=True) for id in IDS}
    for id,ck in render_cks.items():
        assert ck['training_steps']==1500 and ck['best_step']==1500
        assert (ck['state_dict']['pos']==0).all()
    rest_ck=torch.load(out/'checkpoints/restoration_teacher.pt',weights_only=True)
    rlong=[r for r in rows if r['experiment']=='render' and r['method']=='frost' and r['L']==4096]
    assert len(rlong)==9 and min(r['psnr'] for r in rlong)>30
    assert len([r for r in rows if r['experiment']=='restoration' and r['method']=='frost'])==9
    environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,Pillow=PIL.__version__,device='CPU',threads_per_process=1)
    dump(out/'source_data/environment.json',environment)
    protocol=dict(target_images=IDS,target_resolution=[64,64,3],
        preprocessing='PIL RGB; resize shorter edge to 96 px with bicubic interpolation; center 64 x 64 crop. Same exact references as original pilot.',
        metrics=dict(range=[0,1],output_clip=True,psnr='-10 log10(mean RGB squared error)',ssim='Mean over three RGB channels; Gaussian 11 x 11 sigma 1.5; population covariance; K1=0.01 K2=0.03; valid centers.',sr2_edge_shave=2,reconstruction_edge_shave=0),
        held_out_restoration=dict(test_targets_used_for_training=False,train_ids=train_ids,model_validation_ids=val_ids[:12],operator_calibration_ids=val_ids[12:20],
          architecture=rest_ck['architecture'],warm_start='Original 1500-update teacher trained on eight separate images.',additional_updates=6000,best_additional_update=rest_ck['best_step'],
          model_seed=11,batch_size=12,optimizer='AdamW',initial_lr=.0002,weight_decay=.01,gradient_clip=1,cosine_schedule_updates=6000,lr_floor_fraction=.1,
          checkpoint_selection='Highest mean validation PSNR over three tasks, measured every 250 updates; no test checkpoint selection.',
          degradation='Unclipped IID Gaussian input sigma 25/255 or 50/255, or PIL bicubic 64 -> 32 -> 64. Targets/input arrays identical to original.',noise_seed='510001 + image_id * 97 + task_index * 131',L=1024,stream_base_seed=101),
        image_specific_reconstruction=dict(targets_used_for_training=True,target_training_pixels='All 64 x 64 RGB target pixels; no held-out spatial subset.',input='256 coordinate tokens only, each 50-dimensional.',
          coordinates='16 x 16 grid; x,y=-1+2*i/15. Features [x,y,sin(pi*f_k*x),cos(pi*f_k*x),sin(pi*f_k*y),cos(pi*f_k*y)] for 12 equally spaced frequencies 1 through 8.',
          architecture=render_cks[2018]['architecture'],trainable_parameters=render_cks[2018]['trainable_parameters'],frozen_positional_embedding='All zeros; not a learned per-position image table.',
          output='48 RGB values per token, arranged into 4 x 4 patches. Constant 0.5 baseline, no input-image skip connection.',
          weights='Different fitted weights for each target image; same weights shared by Float, W8A8, SC-base and FROST within each image.',
          model_seeds={str(id):render_cks[id]['seed'] for id in IDS},training_updates=1500,checkpoint_selection='Highest target-fit PSNR at 250-step checkpoints (training fidelity, not validation). All selected update 1500.',
          optimizer='AdamW',initial_lr=.0008,weight_decay=.0001,gradient_clip=1,batch_size=1,training_hidden_jitter_std=.004,
          schedule='Cosine schedule with planned horizon 3000 and floor fraction 0.1; executed 1500 updates, not 3000.',
          stream_lengths=[64,256,1024,4096],stream_base_seeds={str(L):([101,202,303] if L==4096 else [101]) for L in [64,256,1024,4096]},
          threshold='30 dB PSNR; user-specified goal. L=4096 selected to meet it across these fitted targets.',statistics='Mean and sample s.d. (ddof=1) across three stream seeds at fixed weights. Not model-training repetitions, dataset confidence intervals, or held-out restoration.',
          inference_seed='stream_base_seed + image_id * 9001 + 11 * 17011',scope='Training-image reconstruction/fidelity illustration. Not unconditional image generation, image compression, or unseen-image restoration.'),
        operator_calibration=dict(softmax_window_N=16,tensor_train_rank=8,candidate_bounds=[2,4,8],steps_per_candidate=600,batch_size=512,
          row_split='Random 80/20 activation-row split within calibration images, capped at 8192/4096. No image-disjoint operator-validation claim.',
          reconstruction_trace_source='Each fitted target model on its coordinates; no independent target validation set.',scalar_tables='Original frozen scalar FSM tables; unchanged.'),
        stochastic_execution=dict(real_bits=True,learned_projections='Sign-magnitude AND/popcount on actual scrambled Sobol input streams, exactly packed in memory-bounded chunks.',
          attention='Actual stochastic query-key products and value aggregation; IID FSM drives and categorical TT emission sampling.',
          mixed_numeric_operations='Coordinates/Fourier input features, means, power-of-two scales, exponent bookkeeping, sorting, reflection padding, residual additions, bias, parameter tables, and clipping are numerical.',
          claims='Software numerical fidelity simulation only; no measured FPGA/ASIC energy, latency, or fully bit-serial end-to-end hardware claim.'),dataset_url='https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz')
    dump(out/'source_data/protocol.json',protocol)
    dump(out/'source_data/verification.json',dict(runs=checks,all_metrics_recomputed=True,identical_original_restoration_targets_and_inputs=True,
      restoration_test_training_ids_disjoint=True,reconstruction_pos_frozen_zero=True,n_FROST_4096_reconstruction_runs=9,
      minimum_reconstruction_psnr=min(r['psnr'] for r in rlong),minimum_reconstruction_ssim=min(r['ssim'] for r in rlong),
      maximum_exact_graph_difference=max(r['exact_graph_max_abs_difference'] for r in rows),
      packed_original_kernel_exact_comparison=json.loads((out/'source_data/packed_kernel_verification.json').read_text())))
    bibliography=(previous/'FROST_references_with_pilot.bib').read_text()+'''\n@inproceedings{tancik2020fourfeat,
  title = {Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains},
  author = {Tancik, Matthew and Srinivasan, Pratul P. and Mildenhall, Ben and Fridovich-Keil, Sara and Raghavan, Nithin and Singhal, Utkarsh and Ramamoorthi, Ravi and Barron, Jonathan T. and Ng, Ren},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {33},
  year = {2020},
  url = {https://papers.nips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html}
}
@inproceedings{loshchilov2019decoupled,
  title = {Decoupled Weight Decay Regularization},
  author = {Loshchilov, Ilya and Hutter, Frank},
  booktitle = {International Conference on Learning Representations},
  year = {2019},
  url = {https://openreview.net/forum?id=Bkg6RiCqY7}
}
'''
    (out/'FROST_references_quality30.bib').write_text(bibliography)
    summary_map={(r['image_id'],r['method'],r['L']):r for r in summary}
    table=[]
    for id,name in zip(IDS,NAMES):
        native=summary_map[id,'float',0];quant=summary_map[id,'w8a8',0];sc=summary_map[id,'SC_base',4096];f=summary_map[id,'frost',4096]
        table.append(f"{name} (\\#{id}) & {native['psnr_mean']:.2f} & {quant['psnr_mean']:.2f} & {sc['psnr_mean']:.2f} $\\pm$ {sc['psnr_std']:.2f} & {f['psnr_mean']:.2f} $\\pm$ {f['psnr_std']:.2f} & {f['ssim_mean']:.4f} $\\pm$ {f['ssim_std']:.4f} \\\\")
    restoration_lines=[]
    for r in means:
        if r['method']=='frost':
            label={'denoise25':'Denoising, $\\sigma=25/255$','denoise50':'Denoising, $\\sigma=50/255$','sr2':'Bicubic $2\\times$ restoration'}[r['task']]
            restoration_lines.append(f"{label} & {r['old_psnr']:.2f} & {r['new_psnr']:.2f} & {r['old_ssim']:.4f} & {r['new_ssim']:.4f} \\\\")
    latex=r'''% Requires graphicx, amsmath and booktabs. Compile with FROST_references_quality30.bib.
% This is a new supplementary experiment; retain the original held-out pilot separately.
\subsection{Image-specific coordinate reconstruction}
We examined whether finite stochastic streams can preserve a high-fidelity image representation once the numerical model has fitted the target. We used the same three Berkeley Segmentation Data Set 500 (BSDS500) reference images as the restoration pilot\cite{arbelaez2011contour}. Each reference is the unchanged $64\times64$ red--green--blue (RGB) center crop after bicubic resizing of the shorter image edge to 96 pixels. A separate model was fitted to all target pixels of each image. Thus these are training-image reconstruction measurements, rather than held-out denoising, super-resolution or unconditional image-generation results.

Let $x_i=-1+2i/15$ and $y_j=-1+2j/15$, where $i,j\in\{0,\ldots,15\}$ index a $16\times16$ grid of output patches. We use the fixed frequencies $f_k=1+7(k-1)/11$, where $k\in\{1,\ldots,12\}$. Motivated by Fourier feature mappings\cite{tancik2020fourfeat}, each coordinate pair is encoded as the following 50-component vector:
\begin{equation}
\phi(x_i,y_j)=\left[x_i,y_j,\left\{\sin(\pi f_kx_i),\cos(\pi f_kx_i),\sin(\pi f_ky_j),\cos(\pi f_ky_j)\right\}_{k=1}^{12}\right].
\end{equation}
The 256 coordinate tokens enter a two-block, pre-normalization windowed Transformer of width 64, with four attention heads and a feed-forward width of 128. Attention operates on $4\times4$ token windows, with a two-token reflected shift in the second block. Each token predicts the 48 RGB components of one $4\times4$ output patch. The additional positional embedding is frozen to zero, and the image-output baseline is a constant 0.5; inference receives coordinates only. Model weights are nevertheless fitted to the reference image, so the reference information is present in the weights.

Each model was trained for 1,500 updates using adaptive moment estimation with decoupled weight decay (AdamW)\cite{loshchilov2019decoupled}, an initial learning rate of $8\times10^{-4}$, weight decay $10^{-4}$ and gradient norm clipping at 1. The cosine learning-rate schedule had a planned horizon of 3,000 updates and a floor fraction of 0.1; only its first 1,500 updates were executed. Training adds independent zero-mean Gaussian perturbations with standard deviation 0.004 after each block; these perturbations are absent during evaluation. The checkpoint with the highest target-fit peak signal-to-noise ratio (PSNR), checked every 250 updates, was selected; all three selected checkpoints were at update 1,500. Tensor-train (TT) attention operators with window length 16 and rank 8 were calibrated on each fitted model's score rows using a random 80/20 row split, with candidate score bounds 2, 4 and 8 and 600 optimization steps per candidate. This row split does not constitute independent-image validation. The original scalar finite-state machine (FSM) tables were retained.

Within each image, all implementations share the same fitted Transformer weights. Float is the numerical model; the 8-bit weights and 8-bit activations reference (W8A8) uses symmetric quantization in linear projections. The stochastic-computing (SC) control uses the original, uncalibrated TT attention table, whereas FROST uses the model-calibrated TT table. Both SC implementations use real finite streams for learned products, query--key products and attention-value aggregation, with FSM-driven attention and scalar nonlinear operators. Coordinate encoding, normalization means and scales, sorting, residual additions, bias, padding, table access and output clipping retain numerical operations. These results assess a mixed numerical/stochastic software simulation and do not establish hardware energy or latency advantages.

Let $T_{u,v,c}$ denote a reference component and $R_{u,v,c}$ its reconstructed component after clipping to $[0,1]$, where $u,v\in\{0,\ldots,63\}$ index pixels and $c\in\{1,2,3\}$ indexes RGB channels. Define the mean squared error $E$ over all components. With unit peak value,
\begin{equation}
E=\frac{1}{64\cdot64\cdot3}\sum_{u=0}^{63}\sum_{v=0}^{63}\sum_{c=1}^{3}(R_{u,v,c}-T_{u,v,c})^2,
\qquad \mathrm{PSNR}=-10\log_{10}E.
\end{equation}
The structural similarity index measure (SSIM)\cite{wang2004ssim} is computed separately for each RGB channel and then averaged, using an $11\times11$ Gaussian window of standard deviation 1.5, population covariance, unit data range and constants $K_1=0.01$ and $K_2=0.03$. Only valid window centers enter SSIM; no border pixels are removed before reconstruction PSNR.

We evaluated stream lengths $L\in\{64,256,1024,4096\}$, where $L$ is the number of bits per stream. At $L=4096$, base seeds 101, 202 and 303 were run for both SC and FROST; shorter lengths use base seed 101. Reported means and sample standard deviations describe stream variability for fixed weights, not variation across independently trained models or dataset confidence intervals. All nine FROST image--seed combinations at $L=4096$ exceeded 30 dB; the minimum was 31.15 dB. Float and W8A8 remained more accurate, and these three fitted targets do not support claims about unseen-image restoration or compression.

\begin{table}[!htbp]
\centering
\caption{Training-image reconstruction fidelity. PSNR is in dB. SC and FROST use $L=4096$ and report mean $\pm$ sample standard deviation across three stream seeds at fixed weights. Float and W8A8 are deterministic references. Every reference image participates in fitting its own model.}
\label{tab:frost_coordinate_reconstruction}
\resizebox{\linewidth}{!}{\begin{tabular}{lrrrrr}
\toprule
Fitting target & Float & W8A8 & SC (base TT) & FROST & FROST SSIM \\
\midrule
'''+'\n'.join(table)+r'''
\bottomrule
\end{tabular}}
\end{table}

\begin{figure}[!htbp]
\centering
\includegraphics[width=\linewidth]{figures/FROST_reconstruction_30dB.pdf}
\caption{\textbf{Image-specific coordinate reconstruction under finite stochastic streams.}
\textbf{a--c}, Unchanged reference crops and reconstructions for the snow scene, bird and roller-coaster images. All models are fitted to their respective reference; base seed 101 is displayed, whereas Table~\ref{tab:frost_coordinate_reconstruction} aggregates three seeds at $L=4096$. Numbers are PSNR in dB and RGB SSIM. Blue boxes mark the same $16\times16$ region of interest (ROI) across methods, chosen in the original pilot. Each image is accompanied by its enlarged ROI and the mean absolute RGB error in that ROI, shown on a common $[0,0.1]$ color scale. Teal FROST scores indicate PSNR at least 30 dB; orange indicates a lower PSNR. Labels, boxes and lines are vector objects; the measured image and error-map pixels are embedded raster data. These are training-image reconstruction results.}
\label{fig:frost_coordinate_reconstruction}
\end{figure}

\begin{figure}[!htbp]
\centering
\includegraphics[width=\linewidth]{figures/FROST_reconstruction_streams.pdf}
\caption{\textbf{Finite-stream reconstruction fidelity on fitted targets.}
\textbf{a}, PSNR as a function of stream length. The black dashed line marks 30 dB.
\textbf{b}, RGB SSIM. Colored dashed lines in both panels show analytical FROST expectations for the calibrated operators. At $L=4096$, markers and bars are the mean and sample standard deviation of three fixed stream seeds; shorter lengths contain one seed. The models and targets are fixed across stream lengths. This is a representation-fidelity experiment, not a held-out restoration benchmark.}
\label{fig:frost_coordinate_streams}
\end{figure}

\paragraph{Separate rerun of the original held-out restoration conditions.}
We also retained the original Gaussian-noise standard deviations $\sigma=25/255$ and $\sigma=50/255$, together with bicubic downsampling and upsampling by a factor of two. The three target images and all noisy/bicubic input arrays are identical to those of the original pilot. The original teacher was fine-tuned for 6,000 additional updates on 200 separate BSDS500 training images, with model-checkpoint selection on 12 separate validation images and operator calibration on eight further validation images. Target test images were not used for this training or selection. FROST was reevaluated with $L=1024$ and base seed 101; two pixels per border were removed before both metrics only in the two-fold restoration condition. The three-image arithmetic means below remain below 30 dB. The fitted-image measurements above therefore cannot replace the held-out restoration results.
\begin{table}[!htbp]
\centering
\caption{Matched original-condition FROST rerun. Values are arithmetic means over the same three held-out images at $L=1024$, rather than results from the coordinate-fitting experiment.}
\label{tab:frost_restoration_rerun}
\begin{tabular}{lrrrr}
\toprule
Task & Prior PSNR & New PSNR & Prior SSIM & New SSIM \\
\midrule
'''+'\n'.join(restoration_lines)+r'''
\bottomrule
\end{tabular}
\end{table}
'''
    (out/'FROST_quality30_insert.tex').write_text(latex)
    facts=[]
    for id,name in zip(IDS,NAMES):
        f=summary_map[id,'frost',4096]
        facts.append(f"{name} #{id}: PSNR {f['psnr_mean']:.4f} ± {f['psnr_std']:.4f} dB; SSIM {f['ssim_mean']:.6f} ± {f['ssim_std']:.6f}; minimum seed PSNR {f['psnr_min']:.4f} dB.")
    mean_lines=[f"{r['task']} FROST L=1024: PSNR {r['old_psnr']:.4f} -> {r['new_psnr']:.4f} dB; SSIM {r['old_ssim']:.6f} -> {r['new_ssim']:.6f}." for r in means if r['method']=='frost']
    readme='''FROST 三张图片重新实验：30 dB 单图重建 + 原条件复测

关键区别
此次得到超过 30 dB 的结果属于“逐图拟合的坐标重建”：每张图单独训练一个模型，训练时使用全部目标像素，推理时只输入坐标。它展示 FROST 有限比特流对已拟合图像的重建保真度，不能写成未知图像去噪、超分辨率、无条件图片生成或压缩实验。目标图像的信息存在模型权重中。原修复任务也实际重跑，但尚未达成 30 dB；两组结果绝不能合并为同一基准。

所有实验仍使用先前相同的三张 BSDS500 图片和 64×64 RGB 中心裁剪，不是原图全分辨率。参考图先将短边 bicubic 缩放到 96 像素。未减小原去噪强度、替换输入、对齐输出直方图、缩小指标区域或改动评分公式。

单图重建结果（L=4096；三个固定随机流种子；均值 ± 样本标准差）
'''+ '\n'.join(facts)+'''
三个种子下九次 FROST 输出均超过 30 dB；SSIM 是无单位指标，30 dB 阈值仅适用于 PSNR。原生 Float 和 W8A8 的保真度更高，不应据此宣称 FROST 超过这些数值参考。

原条件复测（三张图片均值；L=1024；仍是不参与训练的测试图片）
'''+ '\n'.join(mean_lines)+'''
训练扩充到官方 200 张独立训练图片，原 1,500 步模型之后继续 6,000 步；测试仍只有上述三张图片。训练图片不是本次新增的大规模测试集。模型选择用 12 张独立验证图片；TT 校准用另 8 张验证图片中的随机 80/20 激活行划分，不是图片级独立的算子验证。

图和论文文件
figures/FROST_reconstruction_30dB.pdf/svg/png：NC 风格对比，Float、W8A8、原 TT 随机计算控制、FROST L=1024 和 L=4096，固定种子 101。全部标注为训练图重建，给出相同 ROI 及共用色标的误差图。图中文字是可编辑矢量；照片和实测误差值本身为像素数据，不能变成真正的照片矢量图。
figures/FROST_reconstruction_streams.pdf/svg/png：L=64/256/1024/4096 的 PSNR/SSIM，30 dB 线、分析期望参考，以及 L=4096 的三种子误差棒。短流长度只有一个种子，误差棒只描述固定权重下随机流变化。
FROST_quality30_insert.tex：可插入实验部分的英文正文、公式、完整图注和两张表。它明确保留原条件复测结果。依赖 graphicx、amsmath、booktabs；将 figures/ 复制到论文目录后插入。
FROST_references_quality30.bib：原完整参考文献加经作者主页和正式会议页面核验的 Fourier features 引文。此处使用规则频率的 Fourier 编码 + Transformer，不是复现该论文的 MLP 实验。
source_data/reconstruction_summary.csv：单图重建汇总；all_metrics.csv：全部独立记录；restoration_before_after.csv 和 restoration_mean_before_after.csv：原条件逐图和均值配对比较。
source_data/protocol.json：参数、数据划分、种子、指标和实验范围；verification.json：从所有 npz 输出重新计算指标及输入一致性检查。模型、实测输出、绘图代码和数据均在完整 ZIP 内。

复现（完整 ZIP 解压后的根目录）
python -m venv .venv
pip install numpy scipy torch pillow matplotlib
所有脚本在 source/scripts/，FROST 原始算子在 source/original/。随附 CPU 实验环境版本记录；程序限制每进程一个 CPU 线程。

使用已有权重重测一个目标：
python source/scripts/run_quality30.py --data data --output . --previous previous --stage evaluate --kind render --id 2018 --task reconstruction --L 4096 --stream 101
依次选择 id=2018,43033,235098，L=64,256,1024,4096；L=4096 运行 stream=101,202,303，其他流长度只运行 101。已存在同名 npz 的实验会跳过；希望重新生成时请改用空输出目录并复制 checkpoints/，而非覆盖原始证据。
python source/scripts/run_quality30.py --data data --output fresh_measurements --previous previous --stage evaluate --kind restoration --id 2018 --task denoise25 --L 1024 --stream 101
原修复任务使用 id 上述三值、task=denoise25,denoise50,sr2 的九种组合。先将已有 checkpoints/ 复制到 fresh_measurements/。
python source/scripts/report_quality30.py --stage figures --output . --previous previous

从头重新训练（请先创建一个空输出目录，如 new_experiment；不要复制 resume 检查点）：
python source/scripts/run_quality30.py --data data --output new_experiment --previous previous --stage train_restoration --steps 6000
python source/scripts/run_quality30.py --data data --output new_experiment --previous previous --stage train_render --id 2018 --steps 1500
后一个命令对另外两张图分别执行。然后分别校准四个模型：
python source/scripts/run_quality30.py --data data --output new_experiment --previous previous --stage calibrate --kind restoration --id 2018
python source/scripts/run_quality30.py --data data --output new_experiment --previous previous --stage calibrate --kind render --id 2018
render 校准对另外两个 id 分别执行。随后执行上面的评价组合。重训可能因软件版本产生小数值差异，随附权重和实测输出是此处数字的直接依据。

方法和限制
每个单图模型实际训练 1,500 步，不是 3,000 步；余弦学习率按原计划 3,000 步跨度运行前半段。宽度 64、2 层、4 头、FFN 宽度 128；训练隐藏扰动标准差 0.004，测试关闭。每个 50 维坐标 token 输出 4×4 RGB patch；额外位置向量冻结为零，无输入图像残差捷径。训练和选择检查点都用目标拟合度，不是未知图片泛化。
投影、QK、AV 以及归一化方差使用原 FROST 真实随机比特序列，经精确 bit packing 后以 AND/popcount 计算。随机 TT 注意力采样与 FSM 标量算子保留原实现。packed_kernel_verification.json 的 L=64/256 随机小张量上，与原未打包计算的差为零。均值、缩放、Fourier 编码、排序、残差、偏置、表访问、裁剪等仍为数值操作，不能宣称已测端到端纯随机硬件、速度或能耗。
TT 校准以激活行随机 80/20 划分；单图模型的 TT 校准行来自自身拟合目标，无独立目标验证集。L=4096 是为达到本轮目标选择的长度，不是未知测试集上的独立选择结果。增加流长度降低随机误差，但校准算子仍有近似偏差。
PSNR：单位峰值、全 RGB MSE；SSIM：每 RGB 通道 11×11 高斯窗 sigma=1.5、总体协方差、K1=0.01、K2=0.03，随后取通道均值。只有原 sr2 条件在评分前去除四边各 2 像素；单图重建没有 PSNR 边界裁剪。输出统一裁剪到 [0,1]；无锐化、超采样美化或结果选择性隐藏。

数据来源：https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
ZIP 包含实际使用的 200 张训练图、20 张验证图和 3 张目标图，及文件 SHA256 清单；源图保留 JPEG 原始字节。保留了三张原先 pilot 的输出、源指标和初始检查点，用于逐项配对验证。字体使用先前相同的 Nimbus Sans，嵌入 PDF；授权说明随附 fonts/。
'''
    (out/'README_quality30.txt').write_text(readme)
    # Assemble source paths matching the scripts' relative module resolution.
    source=out/'source';(source/'scripts').mkdir(parents=True,exist_ok=True);(source/'original/checkpoints').mkdir(parents=True,exist_ok=True)
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,source/'scripts'/p.name)
    for name in ['attention_core.py','stochastic_model.py','calibrate.py','vision.py','data.py']:
        shutil.copy2(Path(__file__).parents[1]/'original'/name,source/'original'/name)
    for name in ['softmax_N16_r8.pt','scalar_tables.json']:shutil.copy2(Path(__file__).parents[1]/'original/checkpoints'/name,source/'original/checkpoints'/name)
    (out/'previous/source_data').mkdir(parents=True,exist_ok=True);(out/'previous/runs').mkdir(parents=True,exist_ok=True);(out/'previous/checkpoints').mkdir(parents=True,exist_ok=True)
    shutil.copytree(previous/'fonts',out/'previous/fonts',dirs_exist_ok=True)
    for name in ['roi_coordinates.json','metrics_per_image.csv']:shutil.copy2(previous/'source_data'/name,out/'previous/source_data'/name)
    shutil.copy2(previous/'FROST_references_with_pilot.bib',out/'previous/FROST_references_with_pilot.bib')
    shutil.copy2(previous/'checkpoints/restorer_seed11.pt',out/'previous/checkpoints/restorer_seed11.pt')
    for task in TASKS:
        for id in IDS:shutil.copy2(previous/'runs'/f'{task}_{id}.npz',out/'previous/runs'/f'{task}_{id}.npz')
    for split,chosen in [('train',train_ids),('val',val_ids),('test',IDS)]:
        dest=out/'data'/split;dest.mkdir(parents=True,exist_ok=True)
        for id in chosen:shutil.copy2(data/split/f'{id}.jpg',dest/f'{id}.jpg')
    (out/'requirements.txt').write_text('numpy>=2.3\nscipy>=1.17\ntorch>=2.10\npillow>=11\nmatplotlib>=3.10\n')
    files=sorted(p for p in out.rglob('*') if p.is_file() and p.suffix!='.zip' and '__pycache__' not in p.parts and p.name!='file_manifest_sha256.json')
    dump(out/'source_data/file_manifest_sha256.json',{str(p.relative_to(out)):digest(p) for p in files})
    files.append(out/'source_data/file_manifest_sha256.json')
    with zipfile.ZipFile(out/'FROST_quality30_complete.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,p.relative_to(out))
    print(json.dumps(dict(reconstruction=[r for r in summary if r['method']=='frost' and r['L']==4096],restoration=means,verification_runs=len(checks),bundle_files=len(files),zip_bytes=(out/'FROST_quality30_complete.zip').stat().st_size),indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--previous',type=Path,required=True);ap.add_argument('--data',type=Path,required=True)
    main(ap.parse_args())
