FROST-LT: fixed-checkpoint extension to the 200-image BSDS500 pool
================================================================

This experiment reruns the optimized Fig.10 local-state TT operator using
all 200 original frozen image-specific Transformer models. It does not
train new image models or change the selected operator, sampler or clipping.

The selected operator is checkpoints/local33_r16_max8_aug.pt, SHA256
ab16cbb1ec099cf701931540472161f889a8b748f9af9989a507e85e71f59074.
It uses 33 adjacent local addresses, TT rank 16 and conditional stratification.
The original random-walk TT operator is loaded by the legacy model constructor
but replaced before inference. Other stochastic kernels are unchanged.
TT probabilities are represented using the same quantized integer thresholds
as the archived Fig.10 implementation. No expectation or additive-noise
approximation replaces finite-stream evaluation.

Protocol
--------
Every image-specific model was fitted to all pixels of its own target.
Inputs at inference are the original 50 coordinate/Fourier features.
Reference targets remain the original 64x64 RGB crops. Retained lengths
are 1024 and 4096; base stream seeds are 101, 202 and 303, with the archived
image-dependent seed offset. All image IDs and parameters are in protocol.json.
The same operator is used for every image and both lengths.

The earlier operator development used activation rows from 12 of these
image models for fitting and 4 for validation. Thus all-200 statistics
are descriptive fitted-image pool statistics, not an independent-image
test of operator generalization. Summary files additionally report the
184 images absent from operator fitting/selection, and the 181 remaining
after excluding the three previously displayed pilot examples.

Outputs are clipped to [0,1] and saved as float32, exactly as in Fig.10.
PSNR uses all RGB crop pixels with unit peak. SSIM uses the archived
11x11 Gaussian window (sigma 1.5), population covariance and valid centres.
Metrics are calculated on each output first, then averaged across seeds
within each image. Dataset SD is the sample SD across image means. Bootstrap
intervals resample paired images, not individual stream runs.

Provenance
----------
inputs/ contains the original target and saved baseline predictions for
each image. Float, W8A8, original FROST and uncalibrated TT table values are
recomputed from these archived arrays; they are not new baseline runs.
runtime/checkpoints/ contains all frozen teachers and original operators.
provenance/ retains the prior operator selection, split, nine pilot arrays,
input hashes and original/current environments.
results/ contains newly evaluated FROST-LT predictions and individual metrics.
summary/ contains all per-run scores, image means and paired statistics.
The nine original Fig.10 runs at L4096 must reproduce pixel-for-pixel before
the full extension is accepted; evaluate_dataset.py enforces this check.

Reproduce
---------
Dependencies: Python 3.12, NumPy, SciPy, PyTorch, Pillow, and a C++17 compiler.
Run from this package directory:

python -c "import sys; sys.path[:0]=['source','runtime/source/scripts']; import local_tt,fast_finite; local_tt.build(); fast_finite.build()"
python evaluate_dataset.py --pilot --workers 3
python evaluate_dataset.py --workers 7
python summarize_dataset.py
python write_replacement.py

Completed outputs are reused. For a fresh rerun, work on a copy and remove
only its results/ predictions and JSON records, leaving provenance intact.
The summary script verifies all 1200 new prediction metrics and hashes.

Scope
-----
These are actual CPU simulations of finite stochastic computations.
They are not FPGA measurements, inference latency measurements, independent
retraining replicates, held-out image restoration or unconditional generation.
FROST-LT has 127248 TT parameters versus the original 16512. Its hardware
cost has not been evaluated; original FROST hardware results do not transfer.
The prior Fig.10 scalar controls remain three-image results and cannot be
relabelled as 200-image comparisons.
