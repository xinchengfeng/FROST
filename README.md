# FROST

Research code for **End-to-end stochastic Transformer inference with tensorized finite-state operators**.

FROST combines stochastic arithmetic with tensor-train finite-state probability mappings. Its joint categorical attention events preserve normalization and directly select value rows for accumulation. This repository collects the implementation and experiment scripts associated with the manuscript, including the local-state refinement FROST-LT and the evaluated scalar stochastic controls.

## Start here

1. Use Python 3.12 for the archived software workflows. Linux/WSL2 is recommended for C++ kernels and hardware flows; publishing this repository from Windows does not require WSL.
2. Create a virtual environment and install the dependencies in `requirements.txt`. Install a compatible PyTorch CPU build using the official PyTorch installation instructions if it is not already available.
3. Run `python tools/check_repository.py` for source checks and `python tools/doctor.py` to inspect optional runtime dependencies.
4. Read [docs/DATA.md](docs/DATA.md) to restore the separately archived research data, and [docs/REPRODUCE.md](docs/REPRODUCE.md) for the experiment entry points.
5. Read the module README before launching a long experiment. Many original scripts reuse completed runs or train models when required files are absent.

```bash
python -m venv .venv
# Linux / WSL2:
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/check_repository.py
python tools/doctor.py
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. Python source checks work without numerical packages; the actual research workflows require their documented dependencies.

## Repository contents

| Directory | Purpose |
|---|---|
| `experiments/frost_direct_hi_smurf` | Direct joint softmax fitting, calibration and finite-stream sampling |
| `experiments/section3_1_softmax` | Selected softmax operator evaluation |
| `experiments/frost_softmax_study` | Extended softmax configurations and selection provenance |
| `experiments/attention` | Stochastic QK, attention and value-aggregation experiments |
| `experiments/retrieval` | Four-block associative retrieval and ablations |
| `experiments/classification` | MNIST, Fashion-MNIST and CIFAR-10 evaluations |
| `experiments/convergence` | Finite-prefix categorical attention diagnostics |
| `experiments/operator_extensions` | Additional range, dimension and sampler diagnostics |
| `experiments/reconstruction_original` | Original 200-image fitted reconstruction study |
| `experiments/reconstruction_lt` | FROST-LT extension to the 200-image pool |
| `experiments/reconstruction_scalar` | Separate three-target comparison with five scalar SC controls |
| `hardware/stage3`, `hardware/stage5` | Evaluated FROST kernel, scalar controls and hardware flows |
| `hardware/stage6`, `figures` | Publication figure scripts and final layout preparation |
| `hardware/legacy`, `experiments/figure_audit` | Earlier workflows retained as explicitly labelled provenance |
| `tools` | Data restoration, dependency inspection and repository validation |

The experiment-specific `attention_core.py`, `data.py` and requirements files are distinct versions. Keep the module directory structures intact.

## Data and reproduction scope

Large checkpoints, generated predictions, implementation reports and switching traces are distributed through the accompanying data archive rather than Git. The data DOI and final repository URL have not been supplied in this preparation and are deliberately not fabricated. Add the actual persistent identifiers to `docs/DATA.md` and `CITATION.cff` after publication.

The archive filenames, exact directory mapping and a restoration command are documented in `docs/DATA.md`. Original benchmark image pixels are obtained from the providers; reference checksums and preprocessing metadata allow their reconstruction. See the reconstruction preparation helper for restoring local reference arrays.

Reconstruction is fitted-image reconstruction, not held-out image restoration. The scalar reconstruction controls cover three targets. Hardware results cover a separately configured original-FROST attention kernel, not FROST-LT full-network hardware; standard-cell power is a pre-layout estimate. Software and hardware configurations must not be interchanged without validation.

## Validation

The preparation checks source integrity, Python syntax, selected small numerical/native-kernel checks and documented figure-generation paths. `metadata/VALIDATION.json` records exactly which checks were executed and which tool-dependent workflows were not re-run. Packaging is not a new training, FPGA implementation or power-measurement experiment.

## Citation, authorship and license

`CITATION.cff` identifies the research software. Xincheng Feng is the implementation/experiment contact. Manuscript authorship and research guidance are acknowledged separately in `AUTHORS.md`.

A project-wide license for author-controlled code has not been assigned in this preparation. See `LICENSE_STATUS.txt` before describing the repository as open-source. Included third-party components retain their original notices and licenses; those are not replaced by a future FROST license. Data licensing and software licensing are separate choices.

AI-assisted tools, including OpenAI Codex/ChatGPT, were used in implementation assistance, visualization, repository organization and consistency checking. The research authors remain responsible for reviewing and approving the released materials.

For GitHub upload instructions, see `docs/GITHUB_UPLOAD_zh.txt`.
