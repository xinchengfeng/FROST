# Final compact hardware figures

Run from the repository root:

```bash
python hardware/stage6/reproduce_figures.py --output-dir hardware/stage6/Images --qa-dir hardware/stage6/qa
```

Outputs retain the manuscript names `FROST_hardware_stage3.pdf` and
`FROST_hardware_tradeoffs.pdf`. Both are single-page 180-mm-wide vector PDFs
with embedded TrueType fonts. The scripts check label bounding boxes and
absence of raster images and write QA reports/PNGs outside version control.
They read numeric values directly from `data/stage4_summary.json` and
`data/stage5_summary.json`; no coordinates are recovered from a PDF.

Dependencies: Python, ReportLab, PyMuPDF, Matplotlib. DejaVu Sans is the
portable default. For the exact Arial layout, set `FROST_FIGURE_FONT_DIR` to a
licensed local directory containing `Arial.TTF` and `Arialbd.TTF`; fonts are
not redistributed. A font fallback is a documented layout adaptation and
does not change experiment values. Plotting does not run new hardware tests.

Area is Nangate45 mapped cell area. FPGA resources are post-route ECP5
counts. Energy is the mean pre-layout estimate across three stream seeds
on one fixed tile; error bars are sample standard deviations. Those seeds
are not independent images or between-input confidence intervals. The
FROST FPGA and ASIC implementations are distinct target-specific variants.
`legacy/` retains the original Stage6 layout/manuscript assembly scripts;
they are historical, not portable entrypoints.
