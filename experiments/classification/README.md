# FROST: validation across image datasets (Section 3.4)

This bundle records actual CPU experiments on MNIST, Fashion-MNIST and
CIFAR-10. It evaluates a compact custom patch Transformer, not a pretrained
state-of-the-art vision model. Each dataset has separately trained teachers
and separately calibrated HI-SMURF operators.

## Experimental design

- Four pre-LayerNorm encoder blocks, 16 patch tokens, feature width 32,
  two heads, FFN width 64, GELU, residual connections, final LN, mean pooling
  and a 10-class head. Input images retain all pixels.
- Three independently trained model seeds: 11, 22, 33.
- Training/teacher-validation/operator-calibration/calibration-validation
  indices are disjoint. No test data are used to choose weights or ranges.
- Headline comparisons use all 10,000 official test images per dataset and
  model: floating point, simulated W8A8, analytic FROST, actual FROST L256.
- A prespecified class-balanced 1,000-image test subset is used for L64/1024,
  generic-operator transfer and extra stream realizations. L256 seed101
  subset predictions are drawn from the complete test results.
- Main means and SDs use three independently trained models, each with one
  full-test stochastic realization. Separate nested stream summaries average
  seeds101/202/303 within each model, then average across models.
- Inference is a mixed-representation numerical simulation. Main learned
  products, QK, AV, LN variance and GELU gates use actual bits; surrounding
  scaling, means, residual/bias addition and conversions remain numerical.
  Hardware area, energy, latency and throughput have not been measured.

## Outputs

- `section3_4_results.tex`: measured Results text.
- `section3_4_methods.tex`: complete experimental methods.
- `figure_captions.tex`: three two-column `figure*` environments.
- `results_table.tex`: main full-test comparison.
- `figures/`: editable SVG, vector PDF and 400-dpi PNG exports.
- `source_data/all_metrics.csv`: per-model/configuration/split metrics.
- `source_data/headline_results.csv`: complete-test means and model SDs.
- `source_data/runs/`: all predictions, reference logits, labels, indices
  and shard metadata, including diagnostic evaluations.
- `source_data/stream_repeats_*.csv`: separately labelled nested repetitions.
- `checkpoints/`: all trained models, calibrated operators and scalar tables.
- `data/*_metadata.npz`: fixed indices, labels and normalization statistics.
- `provenance/`: environment, archive hashes, validation and run receipts.
- `protocol.json`: fixed experiment settings.

## Reproduction

Python 3.12 with NumPy >=2.0 (`bitwise_count`), SciPy, pandas, scikit-learn,
Matplotlib and CPU PyTorch. Exact executed versions are in
`provenance/environment.json`. No torchvision or GPU is required.

The distributable archive excludes original image archives and decoded
images; verified public download instructions are included in `data.py`.
Run these commands from the extracted directory:

```bash
python3 data.py
python3 vision.py
python3 calibrate.py
python3 validate.py
python3 evaluate.py --workers 8
python3 render_figures.py
python3 write_results.py
python3 verify_results.py
```

Existing checkpoints and completed prediction shards are reused. To rerun
inference from scratch, move `source_data/runs/` to a backup location before
running `evaluate.py`. To retrain, similarly move teacher and calibrated
checkpoint files to a backup location. Do not delete the generic
`softmax_N16_r8.pt` or scalar tables, which are inherited operator inputs.

The packed AND/popcount implementation is checked against explicit signed
bit products. Ancestral sampling of the row-stochastic tensor cores is
checked against their contracted conditional categorical distribution.
The analytic controls use expectations and must not be reported as actual
finite-bitstream runs.

## Dataset sources

- MNIST mirror used by PyTorch: https://ossci-datasets.s3.amazonaws.com/mnist/
- Fashion-MNIST authors: https://github.com/zalandoresearch/fashion-mnist
- CIFAR-10 authors: https://www.cs.toronto.edu/~kriz/cifar.html

Dataset citation entries are provided in `dataset_references.bib`.
