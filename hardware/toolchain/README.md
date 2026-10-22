# Hardware toolchain

The source release excludes compiled EDA tools and operating-system packages.
The original Ubuntu package metadata (including download paths and SHA256)
and OpenSTA/CUDD source commits are retained in this directory. Install tools
in your own environment; full reruns were **not** performed during GitHub
packaging because EDA executables were unavailable.

Recorded main tools: Yosys 0.33 (`0.33-5build2`), ABC from the matching package,
Verilator 5.020 (`5.020-1`), nextpnr-ecp5 0.6 and its ECP5 database, Icarus
Verilog 12.0, OpenSTA 3.1.0. The package JSON gives exact distro builds and
all supplementary packages. Ubuntu24.04 x86-64 matches the original runtime;
Python3, NumPy, C++17 g++, make, CMake, flex/bison, Tcl8.6 development headers,
Eigen3 and zlib development headers are required for building relevant tools.

Useful primary upstreams:

- Yosys: https://github.com/YosysHQ/yosys
- Verilator: https://github.com/verilator/verilator
- nextpnr: https://github.com/YosysHQ/nextpnr (ECP5 and chip database required)
- OpenSTA: https://github.com/parallaxsw/OpenSTA
- CUDD: https://github.com/ivmai/cudd

Exact OpenSTA source commit:
`4f8d3ae6bd3da537f8c4e819c2b327545294fd4d`.
Exact CUDD source commit: `f54f533303640afd5dbe47a05ebeabb3066f2a25`.
The retained archive SHA256 hashes refer to the original packaged source
tarballs, not arbitrary GitHub-generated archives of those commits. A new
build is not expected to reproduce the original executable byte hash; retain
its version/hash with new runs. Build settings are in
`opensta_configure_command.json` and the original `restore_stage3_tools.py`.

The portable experiment scripts prefer tools on PATH. Optional overrides:

```bash
export FROST_TOOLCHAIN_ROOT=/absolute/path/to/restored/toolchain
export FROST_ABC=/absolute/path/to/yosys-abc
export FROST_OPENSTA=/absolute/path/to/sta
```

`FROST_TOOLCHAIN_ROOT` expects `native/usr/bin` etc. when using extracted
packages. If absent, normal system tools are used. ECP5 routing also requires
nextpnr to locate its installed chip database; normal installation avoids
editing system directories. The original `restore_stage3_tools.py` is retained
for history but may create `/usr/share/nextpnr` and expects original source
tarballs, so is not the general installation entrypoint.

Original binary/source hashes are provenance, not evidence of a new run.
The lightweight `../smoke_check.py` only needs g++/NumPy and performs functional
reference replay; it cannot establish synthesis, timing, power or routing.
