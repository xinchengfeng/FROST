# Direct HI-SMURF-style multivariate Softmax study

No piecewise exponential interpolation. Read HI_SMURF_direct_softmax_report.md
and protocol.json before interpreting the results. Three representations are
deliberately distinguished: unrestricted signed TT regression; bounded scalar
Bernoulli heads without exact row normalization; and a joint simplex-valued
TT with valid categorical state outputs. These are exploratory reference
parameterizations, not a claim to reproduce every detail of the original paper.

Run the six commands in the report to reproduce all 18 single-seed training
configurations and the separate 3-seed finite-stream simulation. Constants-only
parameter compilation is separate from runtime inference. No GPU is required.
