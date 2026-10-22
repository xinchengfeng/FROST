# Original FROST reconstruction

This directory contains the original coordinate-reconstruction implementation
used as the numerical/uncalibrated/original-FROST references in **Table 2** and
the original three-target pilot in **Supplementary Fig. S4**. The refined
FROST-LT results and scalar comparisons have separate sibling directories.

Every target image is fitted individually; this is fitted-image reconstruction,
not held-out restoration or unconditional image generation. Historical scripts
also contain exploratory restoration functions, which are not claimed as
additional experiments in the current paper.

## Dependencies and data

Use the repository Python environment, with a C++17 compiler (`g++`). The archived
data release records the exact experimental environment. No fitted checkpoint,
original benchmark photograph, compiled extension or large output is committed.

Obtain the official BSDS500 image files from the provider and extract
`FROST_Experimental_Records_20261022.zip` outside this repository. The following
command installs local working data for all three reconstruction directories:

```bash
python experiments/reconstruction_original/prepare_data.py \
  --records /path/to/FROST_Experimental_Records \
  --bsds-images /path/to/BSDS500/data/images
```

This verifies reference-crop hashes and public-file hashes before restoring
the omitted `truth` arrays into local NPZ working copies. It never modifies
the downloaded records or source photographs. Existing different working files
are rejected unless `--overwrite` is explicitly supplied. It also restores
legacy replay anchors from existing archived predictions; this is data layout
adaptation, not new inference. Keep these installed files out of Git.

## Build the finite kernels

From this directory:

```bash
python source/scripts/fast_finite.py
```

The `packed_kernels.py` and `finite_kernels.cpp` implementations retain the
original stochastic products, normalisation and feed-forward kernels.

## Recompute the archived original reconstruction statistics

```bash
python source/scripts/summarize_bsds500.py --root .
```

This verifies the original JPEG hashes, crop equality, saved predictions and
per-run metrics, then rewrites the local aggregate CSV/JSON files. It does not
train models or perform new inference. Use a disposable working copy if you
want to keep the downloaded outputs untouched.

## Re-run inference or fit models

```bash
python source/scripts/evaluate_bsds500.py --root . --workers 2
```

This reuses existing `runs/image_ID.npz` records. To perform actual inference
again, use a **separate working copy**, retain checkpoints, reference data,
`source_data/protocol.json`, `source_data/exact_kernel_verification.json` and
the three `runs/reference_previous_ID.npz` anchors, and move only the desired
`runs/image_ID.npz`/`.json` pairs out of that copy. Do not delete the archives.

The training entry point is:

```bash
python source/scripts/benchmark_bsds500.py --root . --stage fit --workers 2
```

It skips existing teachers/operators. A fresh fitting directory must have the
same data/protocol and empty fitting checkpoint/output locations. The script
uses 1,500 updates for each of the 200 targets; this is expensive and is not a
quick-start check. Do not use an arbitrary invocation of the historical
`run_quality30.py` instead of this protocol when reproducing the 200-image run.

## Historical pilot figure

`source/scripts/plot_examples_bsds500.py --root .` regenerates the original
three-image panel (now Supplementary Fig. S4). That historical plot expects
legitimately installed/copied Nimbus Sans Regular/Bold TTFs at
`fonts/NimbusSans-Regular.ttf` and `fonts/NimbusSans-Bold.ttf`; font binaries are
not redistributed in this code release. Plot layout may differ if typography
is substituted. The data and numerical scripts do not require fonts.

`ARCHIVED_README_BSDS500.txt` retains the original package instructions and old
figure numbering. This README and the main repository figure map define the
current manuscript mapping.
