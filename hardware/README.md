# FROST hardware source and reproduction map

This directory retains the evaluated RTL, finite-precision inputs, independent
reference models, testbenches, synthesis/routing/power scripts and small public
cell-library dependencies. It does not contain a complete Transformer FPGA
accelerator. The current paper hardware evidence concerns a **16-key ×
16-value-channel score-to-output kernel**, with 16 queries/tile and L=64.

## Current-paper map

| Paper item | Exact sources / data | Role |
|---|---|---|
| Fig. 10, resource and energy panels; Tables 3–4 | `stage6/data/stage4_summary.json`, `stage6/data/stage5_summary.json` | Final summary values from saved implementation reports |
| FROST FPGA | `stage3/rtl/top_perm.v`, associated frontend and pipeline includes; `stage3/source/stage3_route.py` | ECP5 target-specific `perm` circuit |
| FROST standard cell | `stage5/rtl/top_hot_reuse.v`, `frost_frontend_reuse.v`, `frost_hot_reuse.v` | Nangate45 target-specific `hot_reuse` circuit |
| Four scalar-SC controls | `stage5/rtl/top_{sexp16,sexp256,bern8,horner5}.v`, `sc_count_attention.v` | Literature-derived scalar primitives + shared integer count-weighted value aggregation |
| Final hardware PDFs | `stage6/reproduce_figures.py` | Plots from report JSON, without rerunning EDA |
| Earlier FP32/FX16 comparisons | `stage3/`, `legacy/` | Retained history, not the final four scalar-SC comparison |

`perm` and `hot_reuse` are distinct RTL implementations with equivalent saved
outputs and cycles. Both use the original fixed-coefficient rank-2 FROST kernel,
not the rank-16 FROST-LT reconstruction model. The scalar controls are adapted
primitives, not reproductions of complete published accelerators. Equal L does
not establish matched approximation error or task accuracy. Figures and numbers
must be interpreted with the scope recorded in the summary JSON.

## Quick verification and plotting

From the repository root:

```bash
python hardware/smoke_check.py --json hardware/smoke_check_result.json
python hardware/stage6/reproduce_figures.py
```

The first command needs Python, NumPy and g++ (C++17). It compiles two independent
reference models in a temporary directory and checks 3,840 saved output words
across five designs and three stream seeds. It also checks the four SC models'
cycle counts. It is **not** a new RTL/gate simulation or new PPA measurement.
The second command needs ReportLab, PyMuPDF and Matplotlib. See `stage6/README.md`
for optional publication-font selection.

## Data overlay

Small memory/input files and 16-row smoke fixtures are already included. Saved
implementation outputs are kept in the accompanying data release, not mixed
into the source repository. The repository's data-preparation command overlays:

| Data-release path | Repository destination |
|---|---|
| `FROST_Hardware_Records/stage3/` | `hardware/stage3/` |
| `FROST_Hardware_Records/stage5/` | `hardware/stage5/` |
| `FROST_Hardware_Records/stage6/` | `hardware/stage6/` |
| Three hardware activity ZIPs, containing final 8-MHz waves | `hardware/stage5/reports/power/` |

After applying the overlay, `python hardware/audit_records.py` checks all 15
saved gate/RTL output pairs, cycles, netlist/library hashes, recorded tool
metadata and energy statistics without EDA. Add `--check-waves` to hash the
separate activity files too. This is an archived-consistency audit, not a fresh
simulation.

Reports/netlists/CSV outputs are needed to audit the archived full run. All 15
final `.vcd.gz` waveforms are required by `stage5_aggregate.py` for waveform hash
checks; they are optional for plotting and the quick functional check. Stage5
also retains `prior_frost/` comparisons used by the aggregation assertions.
Do not delete those records from a full audit. Original package manifests cover
the original archives, including deliberately excluded build artifacts, and are
not manifests of this cleaned GitHub tree.

## EDA reproduction

Read `toolchain/README.md`, `stage3/README_GITHUB.md` and
`stage5/README_GITHUB.md`. Tools are not vendored as binaries. Build/run commands
write local `build/`, `reports/`, `results/` and `corner_work/` directories; use a
fresh checkout if preserving an overlaid archived run. Recorded tool versions
and hashes are in `toolchain/*.json`. Different versions can change mapping,
routing and estimated power; retain new reports as a distinct run.

Recorded final conditions: ECP5 LFE5U-85F, CABGA381, speed 6, three placement
seeds (1,2,3), 8-MHz constraint; Nangate45 typical, 1.1 V, 25 °C, 8 MHz,
stream seeds 4004, 5005, 6006. Saved Stage3 power at 15 MHz is historical; final
SC/FROST energy comparisons use the newly simulated Stage5 8-MHz activity.

The power boundary includes RNG, constant tables, value/score loads, burn-in,
conversion and output transfer. Reset and the 32-cycle seed setup are excluded.
Standard-cell power is a pre-layout zero-delay estimate without interconnect,
clock-tree, physical glitches, pads, board or host costs. No board measurement
or complete-network hardware accuracy is claimed.

## Source preservation and licenses

RTL and constant tables are unchanged from the recovered archives. Adaptations
only resolve tool paths, create output directories, permit source regeneration
of the existing FROST flattened netlist, audit saved tool hashes without
bundling an executable, and provide portable plots. See `SOURCE_PROVENANCE.json`
for source-level hashes and `THIRD_PARTY_NOTICES.md` for retained third-party
licenses. No blanket license has been selected for author-created code; see
the repository's license-status document before public licensing.
