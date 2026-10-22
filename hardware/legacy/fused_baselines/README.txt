FROST hardware optimization stage 3

Scope: actual synthesizable fused normalization/value aggregation RTL (16 keys, 16 value channels), not a full Transformer or a board experiment.
Designs: rank-2 FROST categorical sampler with direct integer gather accumulation; FX16 with 4 integer multiply lanes; FP32-LUT with 1 or 4 IEEE binary32 FMA lanes and ONE reciprocal per query.
All use identical signed16 Q10 input words and signed32 Q10 outputs. Value matrices are loaded once per tile of 16 queries. RNG seed loading precedes the workload; burn-in is included per query. Clock, reset, handshake, buffers, conversions and numerical score handling are in the synthesis boundary. There is no inter-query overlap.

The user waived quality matching for this hardware feasibility stage. The previous selected candidate was preserved; no fitting or post-test parameter changes occurred. Raw RTL output errors remain in results/rtl_output_quality.csv. Full-network FPGA accuracy was not measured.

Primary FPGA flow: Yosys 0.33, nextpnr-ecp5 0.6-3build5, LFE5U-85F CABGA381 speed6. Four designs, routing seeds 1/2/3 at 8 MHz. Initial FP32 routes at 15 MHz failed; those logs are retained. No board pinout or external I/O delays are constrained. Resources and timing therefore characterize placed-and-routed IP cores, not a board-ready implementation. TRELLIS_COMB is the ECP5 combinational resource count, not a literal LUT4 count. DP16KD counts ECP5 nominal18-kbit blocks, not36-kbit equivalents. Post-route timing estimates are not measured chip frequencies or a frequency sweep.

Secondary ASIC assessment: Yosys standard-cell mapping to Nangate45 typical (library included with license). Every memory maps to cells; there are no SRAM macros. Gate simulation supplies zero-delay switching activity to OpenSTA at15MHz, using the first independent evaluation matrix (16 queries including value load). This is one activity workload, not200 independent power measurements. FROST uses L64/256/1024 and seeds4004/5005/6006; deterministic baselines run once. Activity annotation and gate-vs-RTL output checks are required. Estimated cell power includes characterized internal, switching and leakage components but excludes routed parasitics, clock trees, pads, external memory and glitch timing. ASIC power must not be divided by FPGA throughput.

Reproduction
1. Install g++, Python3 with numpy, Verilator, Yosys and nextpnr-ecp5 with85k support; for power install OpenSTA supporting JSON power reports.
2. Run bash reproduce_fpga.sh. BUILD_DIR optionally selects build location. This builds and verifies all3200queries, synthesizes allfourcores and routes each with3seeds at8MHz.
3. Set STA to your OpenSTA executable and run bash reproduce_asic.sh. Large zero-delay VCD traces are regenerated. They are omitted from the portable archive; their hashes and sizes are supplied.
4. Run python3 source/aggregate.py and python3 source/make_deliverables.py (matplotlib required) to refresh tables, section and figure.

inputs/ and mem/ hold the exact selected experimental inputs and exported coefficients. mem/generation is fixed; no entire CIFAR download or training is required.
third_party/ retains source licenses. Reports are unmodified tool outputs except report summaries produced by scripts. MANIFEST.json gives integrity hashes. Build executables, huge temporary netlists and waveform files may be excluded as described in the manifest; all needed regeneration sources are included.
