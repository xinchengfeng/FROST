# Stage3: FROST FPGA and earlier numerical controls

The current-paper FROST FPGA implementation is `perm`. From this directory,
with the recorded toolchain installed:

```bash
python source/stage2_flow.py build perm --jobs 1
python source/stage2_flow.py ecp5 perm --jobs 1
PYTHONPATH=source python -c "from stage3_route import run; [run(('perm',s)) for s in (1,2,3)]"
```

The route command writes seeds 1/2/3 at the original 8-MHz constraint on
ECP5 LFE5U-85F CABGA381 speed6. External I/O delays are unconstrained; the
saved timing is an internal-path estimate, not measured board Fmax.
`source/stage2_verify.py` supplies the larger original functional comparison
when its required `baseline/` data overlay and all declared designs are built.
The compiler-only repo-wide smoke check is a separate lightweight route.

The original full Stage3 scripts also cover `l64c`, `hot_reuse`, `fx16`,
`fp32_1` and `fp32_4`, including HardFloat-based numerical controls. Their
15-MHz standard-cell power estimates are retained for provenance, but the
final scalar-SC comparison uses Stage5's newly simulated 8-MHz records.
`stage3_aggregate.py` expects the full original set, so it is not needed to
rebuild only `perm`. Read the saved `README_STAGE3.txt` for the full historical
workflow, and `../toolchain/README.md` for portable tool selection.

The original `package_stage3.py` and `restore_stage3_tools.py` are archival
utilities, not recommended installation entrypoints. No board bitstream,
board power measurement or complete-network accelerator is included.
