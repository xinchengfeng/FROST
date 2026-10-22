# Reproduction guide

Run commands from the indicated module directory unless stated otherwise. Original experiment code retains experiment-specific modules and configurations. Do not merge similarly named Python files across experiments.

## Quick checks

From the repository root:

```bash
python tools/check_repository.py
python tools/doctor.py
python experiments/check_software_kernels.py
python experiments/reconstruction_scalar/smoke_native.py
python hardware/smoke_check.py
```

The last three commands perform small functional native/numerical checks; consult their output for dependency-related skips. These checks are not full network training or hardware synthesis. The release preparation report at `metadata/VALIDATION.json` lists exactly what was tested.

## Environment

Python 3.12 and NumPy >=2.0 are recommended for the archived software workflows. Install the compatible dependencies in `requirements.txt` and PyTorch separately using the official CPU wheel instructions. Packed stochastic routines use NumPy 2 APIs. Local native kernels require a C++17 compiler (`g++` in the tested Linux environment). Windows users can use WSL2 for these workflows.

Per-module requirements and archived environment JSON files record original versions; the root requirements are a compatibility baseline, not a fabricated universal lockfile. A present-day environment should be checked against small fixtures before comparing numerical results.

## Paper mapping and entry points

| Current manuscript | Working directory / guidance |
|---|---|
| Figs1-4 | `figures/`; see `docs/FIGURES.md` |
| Fig5; SupplementaryS1 | `experiments/section3_1_softmax`, with sibling `frost_direct_hi_smurf`; selected operator records and original readme |
| Fig6 | `experiments/convergence`; `python source/run_convergence.py --verify-only` after record restoration |
| Fig7; SupplementaryS2 | `experiments/attention`; `validate.py`, `run_experiments.py`, `render_figures.py` |
| Fig8; SupplementaryS3 | `experiments/retrieval`; teacher, calibration, evaluation and figure scripts in its original README |
| Table1 | `experiments/classification`; data, teacher, calibration, full-test and diagnostic scripts in its original README |
| Table2; original-FROST/S4 | `experiments/reconstruction_original`; read its preparation/reproduction README |
| Table2; FROST-LT | `experiments/reconstruction_lt`; shared operator and frozen 200-image models; see its README |
| Fig9 | `experiments/reconstruction_scalar`; six methods, three targets, three stream seeds; see its README |
| Fig10; Tables3-4 | `hardware/stage3`, `stage5`; declared toolchain and exact inputs in hardware README |
| Final figure layouts | `figures/`; historical scripts may retain original figure numbers |

## Data-dependent workflows

Follow `docs/DATA.md` to restore the exact data-release records. Complete the reconstruction reference preparation before rescoring. Scripts often resume existing runs; regenerating a summary from saved outputs is different from executing new predictions, and both differ from fitting a new model.

Training, calibration, checkpoint selection, range handling, retained stream length, burn-in and seeds must match the corresponding archived protocol. Preserving frozen weights is essential when reproducing comparisons. Refer to the manuscript for each experiment's statistical unit.

The hardware workflow requires Verilator, Yosys, nextpnr-ecp5, OpenSTA and the documented ECP5 support data/tool versions. These executables are not bundled. C++ reference replay is useful independent evidence, but it is not a new RTL simulation or physical implementation.

## Historical material

Earlier manuscript writers, packaging scripts, alternative configurations and plots are retained as provenance and labelled in their module documentation. They can contain historical paths and may overwrite legacy outputs. They are not the supported entry points for generating the current manuscript. Use the portable tools and figure runner documented here.
