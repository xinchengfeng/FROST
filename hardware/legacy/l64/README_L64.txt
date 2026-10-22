FROST L64 output-preserving hardware optimization

Final design: rtl/top_l64c.v + rtl/frost_l64c.v + rtl/frost_frontend.v + rtl/pipeline_retime.vh.
Unchanged: TT rank2, 16 states/input, 16 burn-in updates, CDF entries, all random seeds/streams, signed16 Q10 inputs and signed32 Q10 outputs.
New implementation: retime local FSM/drive-RNG updates by pipeline stage; specialize accumulators to 22 bits for 64 samples; detect sample completion from the contiguous valid burst; use six-bit launch counter.

Revisions retained:
  retime: stage-local updates only
  l64: retiming plus fixed-length arithmetic
  l64c: final controller simplification
All three were synthesized, routed with three seeds, and evaluated with gate-level power activity. Full ablation numbers are in results/L64_summary.json.

Checks:
  Final design: 153600 regular output words matched both independent C++ and original RTL, with unchanged cycles.
  Additional signed16 boundary tests: 3072 final output words matched original; constant min/max values checked analytically.
  Across three variants: 460800 regular output words and 2304 gate/RTL words, no mismatch.

FPGA: Yosys 0.33, nextpnr-ecp5 0.6; LFE5U-85F, CABGA381, speed6; common8MHz, route seeds1/2/3. External IO delays and board pin assignments unconstrained.
ASIC estimate: Nangate45 typical1.10V25C, 15MHz, OpenSTA3.1.0; all memory mapped to standard cells, one16-query tile, seeds4004/5005/6006. No physical clock tree, parasitics or time-resolved glitches. These are not measured FPGA watts.
Baseline arithmetic designs and reports were retained unchanged from stage3 with the same input tile and process settings. They were not reoptimized in this revision. No accuracy matching between arithmetic representations is asserted.

Reproduction (tools must be on PATH; run from this directory):
  mkdir -p build
  g++ -O2 -std=c++17 source/fused_sampler.cpp -o build/golden_frost
  python source/flow.py build --designs frost retime l64 l64c --build build/rtl
  python source/verify_L64.py --build build/rtl --golden build/golden_frost
  python source/verify_corners.py --original build/rtl/frost/Vdut --optimized build/rtl/l64c/Vdut --work build/corners
  python source/flow.py synth --designs retime l64 l64c
  python source/flow.py route --designs retime l64 l64c --freq 8 --seeds 1 2 3
  python source/run_asic_synthesis.py --designs retime l64 l64c
  python source/flatten_L64.py
  python source/build_gate_simulators.py --designs retime l64 l64c --build build/gate
  python source/run_gate_power.py --designs retime l64 l64c --lengths 64 --gate-build build/gate --rtl-build build/rtl --sta /path/to/OpenSTA/sta
  python source/aggregate_L64.py
  python paper/source/make_figures.py

The archive omits reproducible compressed gate waveforms and transient build artifacts. Waveform sizes and SHA256 hashes are recorded in reports/omitted_waveforms_L64.json. The gate-power command regenerates them. Published figures use the frozen results copied to paper/source/L64_summary.json.
