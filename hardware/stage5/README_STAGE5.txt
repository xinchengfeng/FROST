FROST hardware validation, stage 5
================================

Purpose
-------
Add real standard-cell gate-activity power estimates for the four stochastic
computing (SC) controls implemented and verified in stage 4. Keep the existing
optimized FROST circuit unchanged and remeasure its activity at the new common
clock of 8 MHz. Figure 11, Figure 12 and manuscript
integration are a separate next stage.

Evidence levels
---------------
1. Stage 4: four SC RTL implementations matched an independent C++ model on
   626,688 output words and their output-cycle counts (3,200 evaluation rows
   plus 64 corner rows, 16 outputs per row, 3 seeds per design).
2. Stage 5: each mapped SC netlist is simulated on the first 16-query tile
   for seeds 4004, 5005 and 6006. Each gate run is checked against both the
   RTL and the independent functional model: 3,072 newly checked gate words.
   The much larger RTL verification count must NOT be called a gate check.
3. FROST is also simulated anew at 8 MHz: 768 more gate-checked words. The
   15-MHz stage-3 outputs are retained in prior_frost/standard_cell and checked
   for exact cycle/output agreement. Frozen inputs, seed files, cell library
   and analysis-binary hashes are also checked for consistency.
4. ASIC area is synthesis-mapped Nangate45 cell area. FPGA resources are
   separately routed ECP5 results from stage 4. Neither is a board measurement.

Frozen experiment boundary
--------------------------
16 keys, 16 value channels, 16 queries per tile; stream length L=64.
Input scores and values are signed 16-bit Q10; outputs are signed 32-bit Q10.
The tile includes 256 value loads, score loads, random generation, finite-state
burn-in, accumulation, normalization/conversion and output transfer.
Reset and 32 seed-loading cycles are excluded for every design. The first
frozen evaluation tile is used without input selection or parameter retuning.

The SC designs are:
  sexp16:   16-state SExp, gain G=2, burn-in=16;
  sexp256:  16-state SExp, gain G=2, burn-in=256;
  bern8:    degree-8 Bernstein stochastic exponentiation;
  horner5:  eight stochastic fifth-order Horner factors.
All SC controls use a shared, efficient integer count-weighted value backend:
four reused multipliers and one reciprocal per query. They do not re-sample
stochastic AV. These are declared implementations of published arithmetic
principles, not reproductions of complete published accelerators. RNGs are
finite-state xorshift32 sources; independent seeds do not guarantee ideal
statistical independence. See README_STAGE4.txt and inputs/sc_design_protocol.json.

FROST is the fixed-coefficient rank-2 kernel, L=64, burn-in=16. It is NOT the
rank-16 FROST-LT software reconstruction model. Standard-cell power uses the
hot_reuse variant; the separate FPGA resource evaluation uses perm. Their
verified outputs/cycles agree, but these are different target-specific RTL.

Power conditions and interpretation
-----------------------------------
Nangate45 typical corner, 1.1 V, 25 C, 8 MHz. OpenSTA 3.1.0 with identical
library and executable for FROST and all controls. Input transition 0.05 ns.
Verilator produces zero-delay gate activity with 1-ps VCD timestamps. The
clock period is 125000 ps. Analysis begins at 35*125000 ps and ends at
(35+tile_cycles)*125000 ps. All relevant pins must be activity-annotated,
and timing must meet 8 MHz. The initial 15-MHz proposal failed SC timing
checks (critical-path arrivals 80.19-100.23 ns); those diagnostic reports
are retained. 8 MHz was selected before the final power comparison, and
all five designs use freshly generated 8-MHz waveforms. Earlier 15-MHz
SExp16 power diagnostics are excluded from the final aggregation.
Timing checks cover constrained internal register paths. External input/output
delay and pad loading are not specified, so this is not a timing-closed chip
interface or a routed ASIC implementation.

These are PRE-LAYOUT TOOL ESTIMATES. Wire parasitics, clock-tree insertion,
physical glitches, pads, board power, host cost and external-memory cost are
excluded. Do not label them FPGA power, measured chip power or full-network
energy. Energy per tile (microjoules) = power (watts) * tile_cycles / 8.
There is no inference-throughput claim based on these one-tile experiments.

Means and sample standard deviations use three random-stream seeds on ONE
fixed input tile. They are not confidence intervals across inputs or models.
Equal L does not establish equal output accuracy. Diagnostic fidelity results
remain in results/fidelity_diagnostics.json; no output-accuracy parity is
claimed. FROST uses fewer DSPs but more FPGA COMB/FF resources than SExp.
Keep both this qualification and the underlying resource counts in reporting.

Files
-----
results/stage5_summary.json       authoritative completed power aggregation
results/stage5_standard_cell.csv full precision numeric summary
reports/power/                   raw VCD activity, SHA256 manifests, commands,
                                 cell-power reports, timing and annotation
reports/asic/                    mapped and flattened gate netlists
results/gate_power_*.csv         public gate outputs and per-query cycles
results/*_golden.csv             independent functional-reference outputs
prior_frost/standard_cell/       prior 15-MHz FROST reference and raw evidence
literature/                      existing manuscript citation keys and sources
PACKAGE_CONTENTS_SHA256.json     exact file hashes inside this checkpoint

Reproduction
------------
From the stage5 directory, with dependencies restored to the sibling toolchain:

  python3 source/restore_stage3_tools.py
  python3 source/stage2_flow.py build hot_reuse sexp16 sexp256 bern8 horner5 --jobs 2
  python3 source/stage3_exec.py python3 source/build_gate_simulators.py \
    --build build/gate --designs hot_reuse sexp16 sexp256 bern8 horner5
  python3 source/stage5_power.py wave hot_reuse sexp16 sexp256 bern8 horner5 --jobs 2
  python3 source/stage5_power.py sta hot_reuse sexp16 sexp256 bern8 horner5 --jobs 2
  python3 source/stage5_aggregate.py

The restore script records exact package sources and checksums; it may need
network access to obtain compiler/simulation packages. Large compiled build
objects are omitted from the checkpoint. Raw gate waveforms and all result
reports are retained. FROST's full independent replay is also available in
the previously delivered FROST_hardware_stage3_verified_checkpoint.zip.

Completed results at the common 8-MHz point
-------------------------------------------
All 3,840 gate output words and all per-query cycle counts matched.
All 15 final activity reports had zero unannotated pins. All five mapped
designs met internal timing. Below: three-seed means on one frozen tile.

Design       area (um2)    cycles   mean power (mW)   energy (uJ/tile)
hot_reuse     54287.142    2592       1.448622284   0.469353620
sexp16        58866.598    4128       1.678396443   0.866052565
sexp256       59306.296    7968       1.653116390   1.646503925
bern8         79736.160    3872       2.278358443   1.102725486
horner5       94827.404    7968       2.895750571   2.884167569

FROST energy reduction versus sexp16 / sexp256 / bern8 / horner5:
45.805% / 71.494% / 57.437% / 83.727%
