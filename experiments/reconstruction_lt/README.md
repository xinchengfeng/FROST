# FROST-LT: 200 fitted BSDS500 images

This directory implements **Table 2's FROST-LT rows** and the corresponding
200-image results. One frozen local-state TT probability operator is shared
across all 200 previously fitted Transformer models. The selected operator
has 33 local addresses, TT rank 16 and 127,248 parameters; it uses conditional
stratification. It is distinct from the original random-walk FROST operator.
Its reconstruction quality does not transfer original-FROST hardware costs.

The image pool has 12 operator-fitting images, 4 operator-validation images
and 184 remaining images, as recorded in `protocol.json`. Every target's own
Transformer was fitted to that target's pixels. The two retained lengths and
three streams produce 1,200 FROST-LT outputs, not 1,200 independent images.

## Install data

From the repository root, run the sibling preparation helper:

```bash
python experiments/reconstruction_original/prepare_data.py \
  --records /path/to/FROST_Experimental_Records \
  --bsds-images /path/to/BSDS500/data/images --targets lt
```

The helper maps public `reconstruction_LT/` records into this directory,
original teachers/operators into `runtime/checkpoints/`, archived baseline
predictions into `inputs/`, and the nine existing scalar-comparison FROST-LT
anchors into `provenance/`. It restores reference pixels only after matching
the recorded crop hashes. The public selected checkpoint SHA-256 is
`ab16cbb1ec099cf701931540472161f889a8b748f9af9989a507e85e71f59074`.

## Build, replay and verify

From this directory, with Python dependencies installed and `g++` available:

```bash
python -c "import sys; sys.path[:0]=['source','runtime/source/scripts']; import local_tt,fast_finite; local_tt.build(); fast_finite.build()"
python evaluate_dataset.py --pilot --workers 1
python summarize_dataset.py
```

The evaluation command **reuses already saved outputs**. `summarize_dataset.py`
recomputes metrics for all saved FROST-LT and baseline predictions, checks
prediction and checkpoint hashes, and reproduces image-level aggregates and
paired bootstrap intervals. It overwrites only local summary files.

For actual new finite-stream inference use a separate working copy, move the
desired `results/image_ID_FROST_LT_L*_S*.npz`/`.json` pairs out of that copy,
and run:

```bash
python evaluate_dataset.py --pilot --workers 1
python evaluate_dataset.py --workers 2
python summarize_dataset.py
```

The pilot contains nine L4096 jobs and enforces exact equality to the archived
three-target FROST-LT outputs when its anchor files are present. Full inference
can be lengthy. These commands do not fit additional Transformer weights.

## Operator fitting provenance

`source/train_candidates.py` also supplies the `GridTT` class needed at
inference. Its training entry point is retained from the earlier scalar-control
development package; use the **sibling reconstruction_scalar directory's**
training instructions for a fresh selected-operator fit and candidate selection.
Do not treat the LT wrapper as a second independent operator-training study.

`ARCHIVED_README.txt` uses the old Fig.10 label for what is now Fig.9. This
directory does not contain a FROST-LT hardware implementation.
