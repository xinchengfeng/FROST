# Section 3.3: complete Transformer simulation

This package contains 87 measured runs of a four-block encoder-only Transformer
on a controlled synthetic associative-retrieval task. There are three trained
reference models and three stochastic-source seeds nested within each model.
The original Transformer weights are frozen during all operator calibration.

At L=256, direct-transfer accuracy is 71.354%
and calibrated full-graph accuracy is 99.913%.
Relative centred-logit errors are 0.810975 and 0.079338.
The frozen architecture has 36,100 parameters; the shared HI-SMURF adds 16,512
operator coefficients, with the same count before and after calibration.

## Deliverables

- Three core figures, each as vector PDF, editable SVG and 400 dpi PNG.
- Results: `section3_3_transformer.tex`.
- Captions and insertion environments: `section3_3_figure_captions.tex`.
- Full protocol: `section3_3_transformer_methods.tex` and `protocol.json`.
- Every run's predictions, labels, query features and residual updates.
- Reference and calibrated operator checkpoints and complete code.

## Reproduce

The data are generated locally; no download or external model is required.
Install NumPy, SciPy, PyTorch, pandas, matplotlib and Pillow, then run:

```
python teacher.py
python stochastic_model.py
python calibrate_softmax.py
python validate.py
python run_experiments.py
python render_figures.py
python verify_artifacts.py
python write_text.py
```

For plotting only, run `render_figures.py` on the included results. The
experiment runner resumes verified result records. Re-running from zero should
use a separate copy with the generated `source_data/runs` directory removed.
The recorded software versions are in `provenance/environment.json`.

## Interpretation and limitations

This is a complete four-block encoder graph, including projections, two-head
attention, normalization, GELU, residuals and the classifier. It is a mixed
representation numerical simulation. Source bits are actually generated, but
TT contractions, residual/bias accumulation, means, static LayerNorm gains,
range handling and interstage conversions are numerical. `operator_coverage.csv`
records this scope. Linear-stage counts group QKV as one matrix; counting the
three Q/K/V maps separately gives 26 component projections instead of 18 stages.

The task is synthetic four-class retrieval, with 128 held-out sequences shared
across conditions. It does not measure performance on a real image/language
dataset, a pretrained large model, a decoder or a hardware implementation.
The error curves follow block positions within the same four-layer models.
They must not be described as arbitrary-depth scaling experiments.
Error bars are the sample SD of three model means, each averaging three stream
realizations. They are not confidence intervals or nine independent models.

## Data and provenance

- `source_data/all_metrics.csv`: 87 run records.
- `source_data/per_model_metrics.csv`: stream-averaged model results.
- `source_data/summary_metrics.csv`: means and sample SD across model means.
- `source_data/depth_metrics.csv`: all per-block errors.
- `source_data/runs/*.npz`: complete reference and predicted logits, labels,
  query representations and residual updates.
- `source_data/test_inputs.npz`: full evaluation input and generator metadata.
- Calibration and training histories are provided, including both tested bounds.
- SHA-256 hashes in `manifest_sha256.json` identify the packaged artifacts.

## Primary references

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745).
- [SciPy Sobol generator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html).
