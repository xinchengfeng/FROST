# Stage5: current scalar-SC comparison

Run from `hardware/stage5` after installing the recorded toolchain (see
`../toolchain/README.md`) and preparing the data overlay for prior comparison
records. These commands regenerate reports; use a fresh copy to retain the
archived evidence. Workload/memory paths are relative to this directory and
are intentionally preserved.

```bash
python source/stage2_flow.py build hot_reuse sexp16 sexp256 bern8 horner5 --jobs 2
python source/verify_sc.py sexp16 sexp256 bern8 horner5 --rows 3200
python source/stage2_flow.py ecp5 sexp16 sexp256 bern8 horner5 --jobs 2
python source/stage4_route.py
python source/stage2_flow.py asic hot_reuse sexp16 sexp256 bern8 horner5 --jobs 2
python source/stage4_aggregate.py
python source/stage4_flatten.py
python source/stage3_exec.py python3 source/build_gate_simulators.py --build build/gate --designs hot_reuse sexp16 sexp256 bern8 horner5
python source/stage5_power.py wave hot_reuse sexp16 sexp256 bern8 horner5 --jobs 2
python source/stage5_power.py sta hot_reuse sexp16 sexp256 bern8 horner5 --jobs 2
python source/stage5_aggregate.py
```

FROST's current FPGA `perm` result is in Stage3, not a Stage5 `hot_reuse`
FPGA synthesis. `stage4_aggregate.py` combines new SC routes with the Stage3
saved FROST reference. To update the FROST FPGA result, reproduce Stage3's
`perm` flow and update the prior report overlay explicitly, rather than mix
unlabelled runs.

Verification uses 3,200 evaluation queries and 64 corner queries for each
of four SC controls and three stream seeds. Full RTL verification covers
626,688 output words and exact cycle checks. The smaller gate-power workload
is the first frozen 16-query tile: five designs × three seeds × 256 words =
3,840 checked output words. `diagnose_sc.py` preserves the separate fidelity
diagnostics; equal stream lengths do not equate output quality.

The current definitions are `sexp16` (16-state SExp with 16 burn-in updates),
`sexp256` (same primitive with 256 burn-in updates), `bern8` (degree-8
Bernstein) and `horner5` (8 factors of degree-5 Horner). The count-to-value
backend uses four reused integer multipliers and a reciprocal per query.
Scores/values are signed 16-bit Q10, outputs signed 32-bit Q10. Frozen
coefficient cases and memory tables remain in `rtl/` and `mem/`.

`stage4_flatten.py` now applies the original net-renaming operation to
`hot_reuse` as well, so the pre-existing FROST flattened netlist can be rebuilt
from source. This does not alter arithmetic or controller RTL.
`stage5_aggregate.py` uses `FROST_OPENSTA`'s executable hash when supplied;
otherwise it hashes the restored recorded executable if present or compares
archived report metadata against the recorded binary hash in tool provenance.
The latter is a metadata consistency check, not executable verification.

The original `package_stage*.py` and `restore_stage3_tools.py` are historical
archive utilities. They are not release/setup entrypoints; the latter expects
original source tarballs and can create a system nextpnr database symlink.
Use installed tools or the explicit environment overrides instead.
