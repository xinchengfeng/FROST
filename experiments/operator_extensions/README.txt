FROST - NC manuscript revision (10 October 2026)

OVERLEAF
Upload FROST_NC_Overleaf.zip as a new project. Select FROST_NC_main.tex as
the main document and pdfLaTeX as compiler. To compile Supplementary
Information, select FROST_NC_Supplementary.tex instead. BibTeX is handled
by Overleaf. Both source documents are complete and use
FROST_references_NC.bib. All required image files are included.

NEW VECTOR FIGURES
Fig. 1 / section 2.1: Images/FROST_execution.pdf
  Actual pre-normalization ordering and latent categorical sampling.
Fig. 2 / section 2.2: Images/FROST_dimension_transfer.pdf
  Three dimensions, four input families, sharp-calibration transfer and
  a fixed-checkpoint finite-stream follow-up.
Fig. 3 / section 2.3: Images/FROST_sampler_precision.pdf
  CDF word length, independent/shared latent-stage draws and conditional
  probability allocation. Heatmap cells and every plot element in these
  three PDFs are vectors. Fonts are embedded; SVG sources are included.
  Existing photograph panels retain their native raster image content.

ACTUALLY EXECUTED NEW STUDIES
- n=16/32/64, N=16 states, rank=8, fit seeds 11/22/33.
  9 separately optimized base fits + 9 sharp-calibrated checkpoints.
  Common dimension-resized archived warm start; not unrelated random
  initializations. 16,384 train / 2,048 validation / 4,096 test rows per
  family; 700 base updates and 400 calibration updates.
- 8/12/16-bit CDF words with 7/11/15 fraction bits; 512 conditional state
  tuples and 128 fixed CIFAR-10 score rows. 8 stream seeds and recorded
  L=64/256/1024/4096. FSM drive schedules and emission draws are paired.
- Exact shared-uniform conditional-law integration and a finite-stream
  shared-source negative control. Unit mass is not treated as proof of
  a correct probability distribution.
- Secondary range-only controls: same base kernels, bound 4 changed to
  8, no refitting or test-driven parameter selection.
- Secondary dimension-stream diagnostic: base fit seed 11, 128 fixed
  Gaussian test rows per n, eight stream realizations, no refitting.
- 2,400 existing BSDS500 predictions independently rescored against
  the original reference crops, including original image hash checks.
  The 200 image-specific models were not retrained for this check.
- Saved sampler events independently reproduce all 160 prefix records;
  core TV bounds, parameter counts and selected checkpoints verified.

NOT RERUN IN THIS REVISION
Complete MNIST/Fashion-MNIST/CIFAR-10 classification and associative
retrieval training/evaluation are retained as author-reported results
from the supplied manuscript. Their original implementation is included
for reference; the complete old classifier run archive is not duplicated
here. No FPGA/ASIC synthesis, board measurement or energy experiment is
performed. Hardware placeholders are absent from the revised manuscript.

REPRODUCE THE NEW OPERATOR STUDIES
Unzip FROST_NC_software.zip. In the extracted directory, create a Python
3.12 environment. Install the pinned numerical dependencies from
requirements.txt, and install CPU PyTorch using its official index:
  pip install -r requirements.txt
  pip install torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu

Run in order (the scripts locate checkpoints relative to their parent):
  python source/run_dimension_study.py --workers 5
  python source/evaluate_range_control.py
  python source/run_dimension_streams.py
  python source/run_sampler_study.py
  python source/verify_revision.py
  python source/plot_revision.py

These regenerate selected operator weights, held-out predictions,
categorical-event traces, CSV/JSON summaries and the three vector PDFs.
Tests and scoring use supplied frozen checkpoints/input rows and do not
download image datasets. The larger BSDS500 source bundle from the
previous reconstruction study supplies data/test/, runs/ and checkpoints/
needed by source/archive_bsds/scripts/summarize_bsds500.py. To rescore
that bundle, run that script with --root pointing to its extracted root.
Only BSDS500 metric/protocol/verification records are duplicated here.

DATA AND STATISTICS
source_data/new_experiment_protocol.json distinguishes the fixed primary
protocol from secondary controls. dimension_summary.json and all
dimension_*_fit.json store fitting and selection logs; dimension_*.npz
store fixed test rows and per-row outcomes. Train/validation rows are
generated from the recorded seeds. Three-fit standard deviations concern optimized
operator replicates sharing one resized warm start. Stream standard
deviations concern random realizations at fixed inputs/weights.
sampler_conditional_outputs.npz stores exact diagnostic conditional laws;
sampler_events_seed*.npz stores actual emitted categories. The fixed
input source is prior_inputs.npz. BSDS500 aggregation first averages
three streams within an image; image bootstrap uses 200 images as the
resampling unit. No quality-based image exclusions or fabricated results
are added. revision_verification.json records checks and dependency
versions; source_data/pdf_font_repairs.json records PDF font fixes.

FONTS
FROST Nimbus Sans is a uniquely named copy of the provided Nimbus Sans
TrueType outlines, used to force consistent Matplotlib embedding.
Appearance matches the earlier plots. The font license/copyright notice
is in fonts/. PDF files embed the fonts; install the provided fonts if
editing SVG text in a vector editor. Archived CFF-in-TrueType PDF font
containers were corrected without changing glyph IDs or plot data.

AUTHOR ACTIONS BEFORE JOURNAL SUBMISSION
Confirm the retained classification/retrieval results against your own
complete raw runs. Add verified author contributions, funding and
competing-interest declarations; none have been invented. Deposit code,
checkpoints and source data in a public repository and replace the local
package statements in Data/Code availability with permanent links/DOIs.
The author-only original hardware draft is a planning backup containing
old illustrative values; it is excluded from the Overleaf and software
packages and is not a manuscript result. Review the negative transfer
findings rather than presenting calibration as universal generalization.
These revisions strengthen the evidence and reporting but do not imply
a guaranteed Nature Communications editorial or acceptance outcome.
