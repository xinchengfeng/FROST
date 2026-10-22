# Section 3.1: direct multivariate Softmax approximation

## Deliverables

- figures/Fig3_softmax_combined.pdf: one six-panel Figure 3, 180 mm wide.
- figures/Fig3ab_softmax_fidelity.*: probability predictions and row-error CDF.
- figures/Fig3cd_model_capacity.*: optimization and parameter-count comparison.
- figures/Fig3ef_bitstream_length.*: actual output and sampling error.
- Every figure is supplied as PDF, editable SVG, and 400-dpi PNG.
- section3_1_softmax.tex: main Results subsection, placed under your Section 3.
- section3_1_figure_caption.tex: figure environment and complete caption.
- section3_1_softmax_methods.tex: reproducible Methods description.
- source_data/: numerical data underlying every panel.

## Manuscript use

Keep figures/ relative to the manuscript, load graphicx, and insert the Results
subsection and caption. The figure number follows your manuscript counter;
the filenames use Fig3 as a provisional name. Put the detailed Methods text in
the Methods section. There is no need to manually force subsection number 3.1.

The figure styles use white backgrounds, consistent sans-serif text, accessible
colours, lowercase panel letters and vector export. The Nature Communications
guidance also calls for a clear description of error bars and figure source data:
https://www.nature.com/ncomms/submit/how-to-submit

## Evidence boundaries

The main figures use all three joint categorical TT configurations from the
previous direct experiment. Free signed coefficient models and independently
bounded scalar heads do not satisfy the same joint probability constraints and
are not combined into the main accuracy or rank curves. Those controls remain
available in the earlier complete direct-experiment archive.

There is one training initialization per configuration. The three stream seeds
are repeated sequence realizations for one fixed model on the same 128 rows.
Error bars represent their sample SD, not independent model training or a
confidence interval. L excludes the 256-cycle burn-in. The analytic reference
in panel e is calculated on those same 128 rows, rather than the 4,096-row test
set. The L^-1/2 line in panel f is a reference guide, not a fit to three points.

The stated results describe this constrained reference implementation. They do
not establish a general accuracy limit for HI-SMURF, demonstrate short-stream
high accuracy, or measure complete Transformer/hardware performance. The
text describes the visible underestimation of high-probability entries.

## Reproducibility

To regenerate all figures from the supplied source CSVs:

```bash
python render_figures.py
python write_text.py
```

Dependencies: numpy, pandas, matplotlib; Nimbus Sans is used if installed.
prepare_source_data.py additionally requires torch and the sibling original
frost_direct_hi_smurf directory; it extracts the existing results without
retraining. provenance/input_sha256.json identifies its input checkpoints and
prediction files. The supplied CSVs make the rendering package standalone.
