"""Write the replacement LaTeX exclusively from verified measured aggregates."""
import argparse
import json
from pathlib import Path


TEMPLATE=r'''% Requires graphicx and booktabs. Uses the existing reconstruction figure.
\paragraph{Image-specific coordinate reconstruction.}
To probe spatially structured continuous outputs, we evaluated all 200 images
in the official test partition of the Berkeley Segmentation Data Set 500
(BSDS500)~\cite{arbelaez2011contour}, without exclusions based on
reconstruction quality. Each reference was a $64\times64$
red--green--blue (RGB) centre crop obtained after bicubic resizing of the
shorter image edge to 96 pixels. A separate model was fitted to all
reference pixels of each image. Thus, the test partition served as a fixed
evaluation image pool; its pixels were fitting targets, and these measurements
concern image-specific reconstruction rather than held-out restoration.
The architecture, fitting budget, preprocessing, calibration procedure and
stream lengths were fixed before extending the three-image pilot.

A $16\times16$ coordinate grid normalized to $[-1,1]$ was encoded into
50 features using the two raw coordinates and sine/cosine features at
12 frequencies uniformly spaced from 1 to 8~\cite{tancik2020fourfeat}.
A two-block pre-normalization Transformer, with $4\times4$-token windows,
feature width 64, four attention heads and feed-forward width 128,
predicted one $4\times4$ RGB patch per token. The second block used a
two-token reflected window shift. The additional positional embedding was
frozen to zero and the output baseline was fixed at 0.5. Inference received
encoded coordinates, with target-image information stored in the fitted weights.
Each model underwent 1,500 updates using adaptive moment estimation with
decoupled weight decay (AdamW)~\cite{loshchilov2019decoupled}, with
initial learning rate $8\times10^{-4}$, weight decay $10^{-4}$ and gradient
norm clipping at 1. The original cosine learning-rate schedule had a
3,000-update span; only its first 1,500 updates were used for every target.
Hidden-state Gaussian perturbations with standard deviation 0.004 were
applied during fitting and disabled during inference. Checkpoints were
checked every 250 updates and selected by target-fit peak signal-to-noise
ratio (PSNR).
The tensor-train (TT) attention operator was calibrated separately for each
model using a random 80/20 split of its activation rows, with the same
600-update calibration budget for each of the candidate bounds 2, 4 and 8.
This row split provides no independent-image validation.

Within each image, all implementations shared the same fitted Transformer
weights. The stochastic-computing (SC) control used the original
uncalibrated TT attention operator, whereas FROST used the calibrated
operator. The software W8A8 control quantized linear projections,
query--key product operands, and attention weights and values for value
aggregation; normalization, softmax and nonlinear activations remained
numerical. The stream length $L$ denotes the number of recorded stochastic
cycles, excluding warm-up cycles of the finite-state machines.
SC was evaluated at $L=4096$, and FROST at
$L=1024$ and $L=4096$, using base stream seeds 101, 202 and 303 for
every target. Float and W8A8 were deterministic at fixed fitted weights.
PSNR was computed from full-crop RGB mean
squared error with unit peak value after output clipping to $[0,1]$.
The structural similarity index measure
(SSIM)~\cite{wang2004ssim} was averaged across RGB channels, using an
$11\times11$ Gaussian window of standard deviation 1.5, population
covariance, unit data range and stabilization constants $K_1=0.01$
and $K_2=0.03$. No border pixels were removed before reconstruction PSNR;
SSIM used valid window centres.

For stochastic implementations, the three stream scores were first averaged
within each image. Dataset means and sample standard deviations were then
computed across these 200 image-level averages
(Table~\ref{tab:frost_coordinate_reconstruction}). At $L=4096$, FROST
achieved PSNR ${{FROST_PSNR}}\pm{{FROST_PSNR_SD}}$~dB and SSIM
${{FROST_SSIM}}\pm{{FROST_SSIM_SD}}$.
The mean PSNR exceeded or equalled 30~dB for {{PASS_IMAGES}} of 200 images
({{PASS_PERCENT}}\%); {{PASS_RUNS}} of the 600 individual stream runs
reached this threshold, with a minimum run PSNR of {{MIN_RUN}}~dB.
Relative to the uncalibrated SC control at the same length, FROST improved
image-averaged PSNR by {{GAIN_SC}}~dB (95\% paired image-bootstrap confidence
interval, {{GAIN_SC_LO}}--{{GAIN_SC_HI}}~dB; 10,000 resamples).
At $L=1024$, FROST obtained mean PSNR {{FROST1024_PSNR}}~dB and SSIM
{{FROST1024_SSIM}}, compared with {{SC_PSNR}}~dB and {{SC_SSIM}} for
the SC control at $L=4096$.
Increasing $L$ from 1,024 to 4,096 increased mean FROST PSNR from
{{FROST1024_PSNR}} to {{FROST_PSNR}}~dB, a paired gain of {{GAIN_LENGTH}}~dB.
Float and W8A8 retained higher mean fidelity, with mean PSNRs of
{{FLOAT_PSNR}} and {{W8A8_PSNR}}~dB, respectively, indicating residual
approximation and sampling errors in FROST.
Figure~\ref{fig:frost_reconstruction} retains the three images selected
for the original pilot as visual examples; the quantitative conclusions
use all 200 targets. Here, table standard deviations describe variation
across target-specific fits after averaging streams. They are distinct
from within-model stream variability and from the independently trained
model variation in the classification experiments.

\begin{table}[!htbp]
\caption{Reconstruction fidelity across all 200 BSDS500 evaluation images.}
\label{tab:frost_coordinate_reconstruction}
\label{tab:frost_bsds500_reconstruction}
\centering
\fontsize{8}{9.5}\selectfont
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.1}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcrrrr@{}}
\toprule
Implementation & $L$ & PSNR (dB) & SSIM
& \shortstack{PSNR loss\\vs Float (dB)}
& \shortstack{Images with\\PSNR $\geq30$ dB} \\
\midrule
{{TABLE_ROWS}}
\bottomrule
\end{tabular*}
\par\vspace{3pt}
\begin{minipage}{\linewidth}
\fontsize{8}{9.5}\selectfont
PSNR and SSIM are means $\pm$ sample standard deviations across 200
images. For SC and FROST, each image contributes the mean of its three
base stream seeds (101, 202 and 303); the 600 runs are not treated as
600 independent images. Float and W8A8 each contribute one deterministic
score per fitted image. PSNR loss is Float PSNR minus implementation
PSNR, averaged over matched images. The last column counts images whose
PSNR, averaged over streams where applicable, is at least 30~dB.
Every target participates in fitting its own model; no target is excluded.
The error terms describe between-image variation, not confidence intervals
or repeated independent training of the same target. Image identifiers,
all individual scores and within-image stream standard deviations are
provided in the accompanying source data.
\end{minipage}
\end{table}

\begin{figure}[!htbp]
\centering
\includegraphics[width=\linewidth]{Images/FROST_reconstruction.pdf}
\caption{\textbf{Image-specific reconstruction with finite stochastic streams.}
\textbf{a--c}, Reference crops and reconstructions of the snow scene
(BSDS500 image 2018), bird and reeds (43033), and roller coaster (235098).
These three targets were selected for the original pilot and retained as
visual examples in the 200-image evaluation. Each model was fitted to all
pixels of its reference image and received encoded coordinates during
inference. Columns compare the reference, Float, W8A8, the SC control
with the base TT operator at $L=4096$, and FROST at $L=1024$ and
$L=4096$. Base stream seed 101 is displayed. Numbers give PSNR in dB
and RGB SSIM. Blue boxes identify the same $16\times16$ region of
interest (ROI) across implementations; below each crop are the enlarged
ROI and its mean absolute RGB error on a common $[0,0.1]$ colour scale.
Teal FROST scores indicate PSNR at least 30 dB, and orange scores
indicate lower PSNR. Figure numbers are individual seed scores;
Table~\ref{tab:frost_coordinate_reconstruction} reports the full 200-image
aggregates after averaging three stream seeds within each target.
These are fitted-image reconstruction results.}
\label{fig:frost_reconstruction}
\label{fig:frost_coordinate_reconstruction}
\end{figure}

Together, the classification and reconstruction experiments assess
decision fidelity and spatially structured output fidelity under
stochastic conversion. Both are software simulations with numerical
centering, range handling, residual and bias additions, and stage
conversions; reconstruction also uses numerical coordinate encoding.
They measure the behaviour of a mixed-representation Transformer graph,
without establishing hardware area, energy or latency advantages.
'''


def main(root):
    data=json.loads((root/'source_data/dataset_summary.json').read_text())
    verification=json.loads((root/'source_data/results_verification.json').read_text())
    assert verification['images']==200 and verification['all_valid']
    assert verification['independently_recomputed_outputs']==2400
    groups={(r['method'],r['L']):r for r in data['groups']}
    frost=groups['FROST',4096];short=groups['FROST',1024]
    floating=groups['Float',0];w8=groups['W8A8',0]
    assert floating['psnr_image_mean']>frost['psnr_image_mean']
    assert w8['psnr_image_mean']>frost['psnr_image_mean']
    gains={r['comparison']:r for r in data['paired_comparisons']}
    gain=gains['FROST4096_minus_SC4096'];length=gains['FROST4096_minus_FROST1024']
    tokens={
        'FROST_PSNR':f"{frost['psnr_image_mean']:.2f}",
        'FROST_PSNR_SD':f"{frost['psnr_between_image_sd']:.2f}",
        'FROST_SSIM':f"{frost['ssim_image_mean']:.4f}",
        'FROST_SSIM_SD':f"{frost['ssim_between_image_sd']:.4f}",
        'PASS_IMAGES':str(frost['images_with_mean_psnr_ge30']),
        'PASS_PERCENT':f"{frost['images_with_mean_psnr_ge30']/2:.1f}",
        'PASS_RUNS':str(frost['raw_runs_psnr_ge30']),
        'MIN_RUN':f"{frost['min_run_psnr']:.2f}",
        'GAIN_SC':f"{gain['mean_gain_db']:.2f}",
        'GAIN_SC_LO':f"{gain['image_bootstrap_ci95'][0]:.2f}",
        'GAIN_SC_HI':f"{gain['image_bootstrap_ci95'][1]:.2f}",
        'FROST1024_PSNR':f"{short['psnr_image_mean']:.2f}",
        'FROST1024_SSIM':f"{short['ssim_image_mean']:.4f}",
        'SC_PSNR':f"{groups['SC_base',4096]['psnr_image_mean']:.2f}",
        'SC_SSIM':f"{groups['SC_base',4096]['ssim_image_mean']:.4f}",
        'GAIN_LENGTH':f"{length['mean_gain_db']:.2f}",
        'FLOAT_PSNR':f"{floating['psnr_image_mean']:.2f}",
        'W8A8_PSNR':f"{w8['psnr_image_mean']:.2f}",
    }
    rows=[]
    for method,L,label in [('Float',0,'Float'),('W8A8',0,'W8A8'),
                           ('SC_base',4096,'SC (base TT)'),('FROST',1024,'FROST'),('FROST',4096,'FROST')]:
        r=groups[method,L];stream=str(L) if L else '--'
        rows.append(f"{label} & {stream} & ${r['psnr_image_mean']:.2f}\\pm{r['psnr_between_image_sd']:.2f}$ "
            f"& ${r['ssim_image_mean']:.4f}\\pm{r['ssim_between_image_sd']:.4f}$ "
            f"& {r['psnr_loss_vs_float_mean']:.2f} & {r['images_with_mean_psnr_ge30']}/200 \\\\")
    tokens['TABLE_ROWS']='\n'.join(rows)
    result=TEMPLATE
    for key,value in tokens.items():result=result.replace('{{'+key+'}}',value)
    assert '{{' not in result
    (root/'FROST_BSDS500_reconstruction.tex').write_text(result)
    print('Wrote verified replacement LaTeX')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root.resolve())
