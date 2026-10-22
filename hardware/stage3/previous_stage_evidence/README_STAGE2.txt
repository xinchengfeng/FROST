FROST hardware implementation optimization -- stage 2 checkpoint
================================================================

This is a reproducible experiment checkpoint, not a new PPA paper figure.
The original manuscript figures and all existing numerical experiments remain
unchanged. The next stage must measure the new netlists before changing them.

Scope
-----
Fixed-coefficient attention score-to-output kernel, 16 keys, 16 value channels,
16-state FSMs, TT rank 2, L=64, 16 burn-in updates, 32 xorshift32 states,
signed-16 inputs and 22-bit accumulation. This is neither a complete Transformer
accelerator nor a hardware implementation of reconstruction FROST-LT (rank 16).

What was actually run
--------------------
1. Reinstalled verified Ubuntu Yosys 0.33 / ABC and Verilator 5.020 packages
   into a project-local prefix. The package archives and SHA256 hashes are kept.
2. Rebuilt and reran the original l64c kernel. ECP5 mapped cell counts and
   Nangate45 cell counts/area exactly reproduce the archived reports.
3. Built 9 candidate rewrites plus the original. Every version was verified
   against a freshly executed integer C++ model, archived RTL words and cycles:
   3 seeds x 3200 rows x 16 outputs = 153600 normal output words per version.
4. Checked 3 seeds x 64 boundary rows x 16 = 3072 additional words per version,
   including signed-16 extrema, alternating extrema, tied scores and random data.
   All valid output words and recorded cycles match. No accuracy was inferred
   from hardware area, and no new image/classification accuracy was measured.
5. Verified all 32 terminal CDF rows x 128 possible random draws (4096 pairs)
   when replacing cumulative comparator counting with a monotone decoder.
6. Synthesized all 10 versions for ECP5, and 7 versions for Nangate45. All
   candidates, including regressions, are retained in synthesis_comparison.csv.

Selected implementations (target-specific)
-----------------------------------------
FPGA candidate: perm
  - Prefix insertion decisions avoid priority-index redecoding.
  - The permutation aliases the sorted order registers; a redundant 64-bit
    copy and its write/decode logic are removed.
  - Terminal CDF uses a priority decoder; inferred distributed value RAM stays.
  - Yosys LUT4 cells: 4427 -> 4161 (-6.0086%).
  - TRELLIS_FF cells: 2036 -> 1972 (-3.1434%).
  - Both versions retain 64 TRELLIS_DPR16X4 memory cells, zero DSP and zero BRAM.
  These are mapping counts, not nextpnr packed/post-route COMB utilization.

ASIC candidate: hot_reuse
  - Prefix insertion and the same permutation aliasing.
  - Dead score registers are overwritten with encoded thresholds after their
    last use. The FSMs consume these thresholds only after frontend completion.
  - A shared one-hot address selects values across the 16 lanes.
  - Same fixed CDF constants, random streams, accumulator and schedule.
  - Nangate45 mapped cell area: 58135.896 -> 54287.142 square micrometres
    (-6.6203%). This is pre-layout library cell area, not placed core area.

The one-hot value representation costs substantially more FPGA resources and is
therefore excluded from that target. The FPGA and ASIC selected netlists differ;
they must not be presented as one physical implementation. Their valid output
words and measured query/tile cycles are identical on the verification corpus.

Paths
-----
  rtl/: all original and candidate RTL, plus the original numeric controls
  mem/, inputs/: frozen tables, values, score rows, seeds and corner vectors
  baseline/: original port-level reference outputs
  results/verification.json: all per-version/per-seed exact comparisons
  results/corner_verification.json: boundary comparisons
  results/synthesis_comparison.csv: every completed mapping result
  results/stage2_summary.json: selection and machine-readable scope
  reports/{build,ecp5,asic}/: commands, logs, mapped netlists and source hashes
  paper/FROST_stage1_references_fixed.tex: complete 37-reference manuscript
  ../toolchain/: verified distribution archives for local restoration

Reproduce from the extracted checkpoint root
-------------------------------------------
  cd stage2
  python3 source/restore_toolchain.py
  python3 source/stage2_flow.py build l64c perm hot_reuse
  python3 source/stage2_verify.py l64c perm hot_reuse
  python3 source/stage2_flow.py ecp5 l64c perm
  python3 source/stage2_flow.py asic l64c hot_reuse

Requires a compatible Linux x86-64 host (Ubuntu 24.04 runtime or equivalent),
Python with NumPy, g++, make and dpkg-deb. Simulation build directories are not
archived; these commands recreate them. Original scripts are retained, but use
stage2_flow.py to recognize the new variants.

Pending work
------------
The new variants have NOT undergone place-and-route, gate-level equivalence,
activity-based power analysis, board measurements or quality-matched baseline
evaluation. Old power/Fmax numbers cannot be assigned to the new netlists.
The broader request to use matched SC hardware controls still requires actual
SC implementations and their reports; numeric controls must not be relabelled SC.
Next: perform those checks in bounded stages, then regenerate Fig.11/Fig.12 and
integrate only the confirmed results into the complete LaTeX.
