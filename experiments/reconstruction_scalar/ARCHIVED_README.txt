Fig. 10: multiple SC controls and experimental FROST-LT
====================================================

Deliverables
  figure/FROST_reconstruction_SC.pdf  - final 180 x 170 mm figure
  figure/FROST_reconstruction_SC.svg  - editable vector layout
  figure/FROST_reconstruction_SC.png  - preview
  Fig10_replacement.tex              - local methods/results + replacement figure
  Fig10_SC_references.bib            - four verified reference entries
  REPORT_zh.txt                      - results, scope and next-stage instructions
  results/fig10_source_data.csv      - 54 individual image/stream scores
  results/fig10_summary.json         - 18 three-seed aggregates
  results/fig10_paired_comparisons.json - differences against all five controls

The PDF/SVG preserve text, lines and annotation geometry as vectors. The real
64 x 64 source and reconstruction pixels are embedded raster images; their
native information is preserved. No image-generation model, denoising or
detail enhancement was used. ROI choices were fixed from the references.
Blue ROI boxes and every error map use the same location within an image;
all errors share the [0,0.06] scale. The displayed image seed is always 101.
Printed metrics use seeds 101, 202 and 303, with sample standard deviations.

IMPORTANT SCOPE
FROST-LT denotes a NEW experimental local-state TT variant. It is not the
original 16-state random-walk FSM architecture. It has 127,248 TT parameters
instead of 16,512. Its helper operations and memory footprint require new
hardware evaluation. Original FROST PPA results cannot be assigned to it.
This is fitted-image reconstruction, not unseen-image restoration.
The 200-image table in the manuscript describes the ORIGINAL configuration;
the present new-baseline experiment covers only the three stated images.

Baseline definitions
- SExp-FSM: existing Li et al.-derived SExp control, 16 states, gain 2,
  256 discarded warm-up updates; exp(0) endpoint bypass; decoded count
  normalization. Implementation and calibration are in the prior protocol.
- Horner-SC (new in this stage): Parhi & Liu-inspired NAND/Horner exponent,
  gap delta = clip(max(z)-z,0,8), x = delta/8. For each output event, compute
  eight independently driven copies of T_m(x), AND their outputs, then count.
  T_m(x) = sum_{k=0}^m (-x)^k/k!. Each copy starts h=1 and updates
  h = NOT(x_bit AND coefficient_bit AND h), k=m,...,1;
  P(x_bit=1)=x, P(coefficient_bit=1)=1/k. Different draws are used for all
  factors and stages. Counts are divided by their row sum numerically.
  The five degrees [3,4,5,6,8] were compared at L4096 on the same independent
  192 validation score rows and seeds [1123,2246,3369]. Degree 5 was selected
  by mean squared probability error before any new Horner image evaluation.
  Its 8 factors and 5 levels use 80 RNG draws per scalar output cycle in
  this simulator; this is not a claim of one hardware cycle per event.
- Bernstein-SC: degree 8 constrained polynomial approximating exp(-8x),
  existing nine coefficients. Actual input Bernoulli bits, population
  counting and coefficient bits are used before numerical count normalization.
- Bernstein-LD: same polynomial with a fixed scrambled Sobol net and
  independent per-scalar digital shifts; 9 dimensions for the 8 input bits
  and one coefficient draw. This is an adaptation of the low-discrepancy
  SC principle, not a reproduction of a published softmax accelerator.
  The cited DATE2014 design uses Halton sequences; this adapter uses Sobol.
- Bernstein-Strat: same polynomial with conditional stratification at each
  population-count transition and coefficient emission. This additional
  enhanced control was retained even when it outperformed FROST-LT.

Horner and SExp use xorshift32(13,17,5) with nonzero per-scalar states;
32-bit thresholds use ceil(p*2^32). The ideal independence assumption is
checked statistically, rather than asserted as an exact property of a PRNG.
All methods retain the same frozen models and non-attention stochastic
kernels, including their numerical helper operations. Equal L does not
mean equal RNG counts, wall time, hardware latency, area or energy.

Literature screening
Implemented primitive families: Li et al. TC2014; Qian/Riedel DAC2008;
Parhi/Liu TETC2019. The low-discrepancy principle follows Alaghi/Hayes DATE2014.
The published TETC full text was not accessible from its publisher during
this run; author-institution metadata and the explicit NAND/Horner equations
in the authors' related open-access Scientific Reports 2018 paper were read.
The Horner factorization and threshold simulator here are an explicitly
documented adaptation, not claimed verbatim published implementation.
Also examined: Yuan et al., GLSVLSI2017 SC softmax regression; Sadi/Mahani,
Integration2021; ASCEND, DATE2024. These complete pipelines were not
reproduced and do not appear as measured baselines in this figure. The
archived ASCEND-formula diagnostic is numerical, not an SC baseline.
Full citation identifiers and access URLs are in results/literature.json.

Reproduction (Python 3, C++17 compiler; dependencies in requirements.txt)
Run in this folder. For training provenance, see previous_README.txt.
  python -c "import sys;sys.path.insert(0,'source');import sc_baselines;sc_baselines.build()"
  python source/local_tt.py
  python legacy_runtime/source/scripts/fast_finite.py
  python source/horner.py
  python source/validate_horner.py
  python source/evaluate.py --method Horner --workers 3
  python source/evaluate.py --method SExp
  python source/evaluate.py --method Bernstein
  python source/evaluate.py --method Bernstein_QMC
  python source/evaluate.py --method Bernstein_stratified
  python source/evaluate.py --method LocalTT33_stratified --checkpoint checkpoints/local33_r16_max8_aug.pt
  # For figure regeneration, install Arial or point to a licensed local font directory:
  FROST_FIGURE_FONT_DIR=/path/to/Arial/fonts python source/make_figure.py

evaluate.py reuses existing completed per-image results. To rerun predictions,
use a COPY of this package and move the desired *_S*.npz/.json pairs out of
results first. Preserve this package as the immutable experimental record.
Training/selection scripts overwrite their own named outputs. The historical
source/summarize.py and previous_* files describe the earlier improvement
stage. For the CURRENT figure, use make_figure.py and fig10_* outputs.

Native .so files and Python caches are excluded from the archive. No network
access is needed after dependencies are installed: reference crops, frozen
teacher checkpoints, calibrated TT checkpoints, calibration scores and
prediction arrays are included. Diagnostics are identified as such and
are not used in this figure. No new synthesis or hardware measurements ran.

Overleaf
1. Upload figure/FROST_reconstruction_SC.pdf into Images/.
2. Follow the local replacement instructions in Fig10_replacement.tex.
3. Upload Fig10_SC_references.bib at the root and append its basename to
   your existing bibliography command, or merge its entries by DOI into
   the existing .bib. Keep only one entry per paper and adjust keys if merged.
4. The document should still assign number 10 if the old figure is replaced
   in place. Do not retain both old and new figure environments.
5. The diagram was designed at 180 mm width; the current 166 mm manuscript
   text width is suitable. Avoid squeezing it into a narrow single column.

No blanket licence is assigned to archived user inputs or checkpoints.
See previous_LICENSE_STATUS.txt. The stored paper URLs are references;
third-party paper PDFs are not bundled.

Figure audit (17 October 2026)
- The 54 individual metrics were recalculated and matched the existing records.
- All 105 checked prediction/model/data files retained identical SHA256 hashes.
- Arial regular/bold are embedded TrueType subsets; text remains editable.
  Font files are not distributed in this package. Install Arial or provide its
  directory with FROST_FIGURE_FONT_DIR when regenerating the figure.
- Figure size: 180 x 170 mm; font sizes 6.2-8 pt at this size, or 5.72-7.38 pt
  at the manuscript's 166 mm width. Use a full-width figure.
- Actual full-crop pixels remain 64 x 64. Exact 8x pixel replication embeds
  them as 512 x 512; 16 x 16 ROI/error pixels use 16x replication. This fixes
  unnecessary intermediate sampling; it does NOT create higher-resolution
  experimental data. No interpolation, denoising or enhancement is used.
- The duplicated title/context moved into the caption. All controls remain.
- Caption defines stream replicates, acronyms, ROI, shared error saturation,
  integer pixel display and gains calculated from unrounded means.
- Teal/bold identify FROST-LT, not the winning method.
Official guidance consulted:
https://www.nature.com/ncomms/submit/how-to-submit
https://www.nature.com/documents/ncomms-submission-guide.pdf
The quick-guide PDF is dated 3 May 2019; the online how-to-submit page was
checked as well. This figure audit does not certify full-manuscript compliance.
