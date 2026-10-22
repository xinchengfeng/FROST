# Section 3.2: stochastic attention core

This package contains actual measured behavioral simulations. No figures use
invented points. Main result at L=256: baseline RMSE 0.313545356, improved RMSE
0.109164814, reduction 65.184%. There are 170 configuration/seed runs.

## Main figures and text

- `figures/Fig_attention_fidelity.*`: complete-output fidelity and stage controls.
- `figures/Fig_attention_efficiency.*`: sequential ablation and stream length.
- `figures/Fig_attention_robustness.*`: input regimes and value centring.
- `figures/Fig_attention_dimension_supplement.*`: optional head-dimension control.
- All figures include vector PDF, editable SVG, and 400 dpi PNG versions.
- `section3_2_attention.tex`: Results subsection.
- `section3_2_figure_captions.tex`: captions and LaTeX insertion environments.
- `section3_2_attention_methods.tex`: full protocol and metric definitions.

## Reproduce

Run in an environment with Python, NumPy, SciPy, PyTorch, pandas and matplotlib:

```
python validate.py
python run_experiments.py
python render_figures.py
python write_text.py
```

The runner verifies and resumes existing per-run results. To rerun from zero,
use a separate copy and remove its generated `source_data/runs` directory first.
Replotting needs only the included source data. There is no network dependency
and no additional training. The exact package versions are in provenance.
Inputs can be recovered from their NPZ files or deterministically regenerated
from the recorded seeds. No prior project files are needed.

## Interpretation

The experiment covers QK -> HI-SMURF -> AV from supplied Q,K,V. It does not
include learned projections, multiple-head output projection, a complete
Transformer block, real-task inference, masking or changes in token count.
All three stages explicitly use random bits. TT contractions and stage
decoding/re-encoding remain floating-point numerical operations. This is not
a gate-level, fully bitstream-preserving hardware implementation.

Five stream/digital-shift seeds quantify randomness for one fixed trained
model. No uncertainty over training initializations or statistical superiority
over external SC implementations is claimed. Relative errors on value-shift
tests are not used as the headline, since large offsets inflate denominators.
The offset plot instead reports absolute RMSE. Sharper and correlated inputs
retain appreciable error; these limitations are intentionally shown.

The experimental protocol was written before held-out evaluation. An initial
workspace I/O interruption was resumed from 86 verified result checkpoints;
84 missing runs completed afterwards with the same algorithm and seeds.
The note and original I/O error are retained in provenance. Workers now
regenerate deterministic inputs and artifacts use single-buffer writes.

## Data

- `source_data/all_metrics.csv`: all 170 run-level records.
- `source_data/summary_metrics.csv`: mean and sample SD across five seeds.
- `source_data/analytic_metrics.csv`: analytic reference controls and input clipping.
- `source_data/Fig_*.csv`: plot-level source data.
- `source_data/inputs_*.npz`: complete synthetic input tensors.
- `source_data/runs/*_per_matrix.npz`: per-matrix metrics from each run.
- `source_data/fidelity_*.npz`: complete outputs for predetermined scatter run.
- `source_data/calibration_gain.csv`: unsuccessful gain calibration, retained.
- `checkpoints/softmax_N16_r8.pt`: frozen checkpoint from Section 3.1.

Source data and captions distinguish actual stochastic outputs from analytic
and stage-isolation controls. SHA-256 hashes are listed in `manifest_sha256.json`.
