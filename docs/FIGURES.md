# Reproducing publication figures

The current main text contains ten numbered figures, several assembled from two
PDF files. The filenames below match the manuscript. Figure rendering is separate
from rerunning the experiments: this command reads recorded measurements or
preserves existing author-generated vector paths. It does not train new models or
run synthesis, place-and-route, or power estimation.

From the repository root, install the repository requirements, then run:

```bash
python figures/reproduce.py --figures 1-8,10 --output build/figures
```

Figs 1–8 and 10 can be rendered with the small data/templates already in the code
release. The CSV and JSON files in `figures/data/` are unmodified copies of the
recorded figure inputs. Their original and packaged SHA256 values are listed in
`figures/SOURCE_PROVENANCE.json`. Full experimental records and experimental
rerun entrypoints are documented separately in the repository.

To draw one group and retain intermediate plots and built-in validation records:

```bash
python figures/reproduce.py --figures 7 --output build/fig7 --keep-work
```

The default is `--figures 1-8`; ranges and comma-separated lists are accepted.
Each output directory contains `figure_build_manifest.json` and `source_audit/`
with numerical/geometry checks emitted by the plotting scripts. The normal
workflow writes generated output under `build/`, which should not be committed.

## Figure mapping and source provenance

| Current figure | PDF filename(s) | Source and what the renderer does |
| --- | --- | --- |
| 1 | `Fig_FROST_overview.pdf` | `source/draw_schematics.py`, overview schematic; diagram, not experimental observations |
| 2 | `Fig_FROST_representation.pdf` | Same source, representation schematic |
| 3 | `Fig_FROST_operator.pdf` | Same source, operator schematic |
| 4 | `Fig_FROST_block.pdf` | Same source, complete-block schematic |
| 5 | `Softmax_approximation.pdf`, `Softmax_bitstream.pdf` | `source/draw_softmax.py`; redraws recorded CSV observations and summaries in `data/fig5/`; verifies probability errors against the archived JSON |
| 6 | `FROST_attention_formation.pdf`, `FROST_attention_convergence.pdf` | `source/restyle_convergence.py`; changes palette/text of original author vector templates; original measured paths, uncertainty bands, and geometry are preserved |
| 7 | `Fig_attention_fidelity.pdf`, `Fig_attention_efficiency.pdf` | `source/draw_attention_fidelity.py` and `source/draw_attention_efficiency.py`; redraw b/c from the actual 40/20 observations in `data/fig7/`; preserve original a/d template panels unchanged |
| 8 | `Fig_transformer_performance.pdf`, `Fig_transformer_ablation.pdf` | `source/restyle_retrieval.py`; preserves measured main-plot vectors, updates style, and draws a magnified inset using original model-level CSV means and sample standard deviations in `data/fig8/` |
| 9 | `FROST_reconstruction_SC.pdf` | Reconstruction package, described below; validates archived RGB reconstruction arrays and scores before rendering |
| 10 | `FROST_hardware_stage3.pdf`, `FROST_hardware_tradeoffs.pdf` | Delegates to `hardware/stage6/reproduce_figures.py`; uses actual archived Stage4/5 summary JSON, not values digitized from a PDF |

The vector PDFs in `figures/templates/` are author-produced source assets for
style-preserving redraws. They are **layout templates containing the original
measured paths**, not substitutes for raw experimental records. Some historical
CSV filenames retain the figure numbers used when the experiment was first run;
use the mapping above for current numbering. The current manuscript uses its
`\FROSTpanelrow` LaTeX macro to overlay the final a–d panel labels on Fig7;
`Fig_attention_efficiency.pdf` retains its original internal a/b letters, which
the manuscript overlays as c/d. This preserves the manuscript asset contract.

Fig7 displays means and sample standard deviations across five stream seeds.
Coincident points stay at their true coordinates. Its paired display and zoom
change readability, not observations. Fig8 checks the retained mean curves against
CSV summaries. Fig10 is a kernel-level hardware comparison; the plotting code does
not convert it into a full-network or quality-matched hardware result.

## Fonts

No font binaries are distributed. The default uses Matplotlib's open-source
DejaVu Sans. Existing untouched vector-template panels keep their embedded
original fonts. The build manifest explicitly records font substitution; exact
text metrics can therefore differ from the publication PDFs.

For the original publication font choices, obtain the fonts under a suitable
license and provide their directories explicitly:

```bash
python figures/reproduce.py --figures 1-8,10 \
  --font-dir /path/to/arial_fonts \
  --diagram-font-dir /path/to/source_sans_3_fonts \
  --output build/figures
```

The Arial directory must contain `Arial.TTF`, `Arialbd.TTF`, `Ariali.TTF`, and
`Arialbi.TTF`. The schematic directory must contain the Source Sans 3 `.ttf`
files, including the styles needed by the diagrams. Environment variables
`FROST_FIGURE_FONT_DIR` and `FROST_DIAGRAM_FONT_DIR` provide the same defaults.
Plots and annotations are drawn as vector paths/text; the renderer checks that
Figs 1–8/10 contain no raster images. Actual RGB photos in Fig9 necessarily remain
raster pixels within the PDF; its text and graphical overlays remain vector.

## Reconstruction figure and supplementary pilot

Restore the experimental records and obtain BSDS500 separately, following the
reconstruction package documentation. BSDS500 images are not redistributed in the
code repository. Prepare the scalar-comparison targets before rendering Fig9:

```bash
python experiments/reconstruction_original/prepare_data.py \
  --records /path/to/FROST_Experimental_Records \
  --bsds-images /path/to/BSDS500/data/images \
  --targets scalar
python figures/reproduce.py --figures 9 --output build/figures
```

The equivalent direct plotting command, run inside
`experiments/reconstruction_scalar/`, is:

```bash
python source/make_figure.py
```

That script recomputes the 54 displayed scores from the restored image arrays and
checks them against the archived scores. Its historical internal metadata may
refer to Fig10; `FROST_reconstruction_SC.pdf` is the current Fig9.

The original three-image reconstruction pilot is supplementary, not a replacement
for the 200-image original-FROST evaluation. After restoring its records, the
historical supplementary plotting script is
`experiments/reconstruction_original/source/scripts/plot_examples_bsds500.py`.
Run it from that package with `--root .`; consult its README for the expected
`fonts/` and data paths. Other experiment plotting scripts remain with their
respective experimental source packages.

## Verification included in this release

`figures/VALIDATION.json` records the completed lightweight redraw checks. All
14 PDFs covering current Figs1–8 and Fig10 were generated successfully using the
portable DejaVu Sans fallback. The checks verified zero raster image objects,
retained panel geometry, true mean/standard-deviation statistics, and unchanged
retained template panels where applicable. These checks validate the plotting
pipeline and preservation of recorded inputs; they do not constitute new
experimental runs.
