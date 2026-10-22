# Software experiment entry points

These directories retain the original working layouts and scientific implementations. Large datasets, predictions and checkpoints belong to the accompanying data release. Restore those data using the repository instructions before running a script that loads them. Run commands below **from the stated experiment directory**, in a writable copy. Reproduction scripts can overwrite their own saved metrics and outputs.

| Current manuscript content | Directory | Typical entry points |
| --- | --- | --- |
| Fig.5 and Supplementary Fig.S1: direct joint-softmax fitting | `frost_direct_hi_smurf/` and `section3_1_softmax/` | `categorical_study.py`, `check_and_simulate.py`, `prepare_source_data.py` |
| Fig.6: finite-stream attention-event convergence | `convergence/` | `source/run_convergence.py`, `source/export_source_data.py` |
| Fig.7 and Supplementary Fig.S2: stochastic attention and range/centering controls | `attention/` | `validate.py`, `run_experiments.py`, `verify_artifacts.py` |
| Fig.8 and Supplementary Fig.S3: synthetic associative retrieval and ablations | `retrieval/` | `teacher.py`, `calibrate_softmax.py`, `run_experiments.py`, `verify_artifacts.py` |
| Table1: MNIST, Fashion-MNIST and CIFAR-10 | `classification/` | `data.py`, `vision.py`, `calibrate.py`, `evaluate.py`, `verify_results.py` |
| Earlier operator-capacity/range/sampler explorations | `operator_extensions/` | `source/run_dimension_study.py`, `source/run_sampler_study.py`, `source/evaluate_range_control.py` |
| Earlier systematic probability-TT study and diagnostics | `frost_softmax_study/` | `study.py`, `run_training.py`, `evaluate.py`, `streams.py` |
| Earlier figure/source consistency audit | `figure_audit/` | `source/verify_stage3.py`, `source/audit_vector_assets.py` |

Original filenames and some script-produced captions retain earlier figure numbers. The **current** paper mapping is the table above and the repository-level mapping, not an old generated `.tex` filename. The final publication figure scripts are documented separately by the root README. Legacy manuscript writers, package builders and rendering scripts are retained as provenance; they are not required to run the numerical kernels.

## Check the software interfaces without full experiments

From the repository root:

```bash
python experiments/check_software_kernels.py
```

This uses small synthetic values to check balanced deterministic partitions, invertible patch extraction, native finite-state transitions and categorical-event/count agreement against a NumPy reference. It compiles a temporary C++ library and does not download datasets or modify research outputs. If PyTorch is installed, it also checks joint categorical normalization, the classification graph and parameter count, and retrieval graph output shape. Missing optional dependencies are reported as skipped checks. A pass is not a reproduction of the paper's trained accuracy or hardware PPA.

## Main reproduction paths

### Direct joint softmax

From `frost_direct_hi_smurf/`:

```bash
python categorical_study.py
python check_and_simulate.py
```

The first command fits the documented configurations and writes checkpoints. To reuse the archived frozen fits, restore the data and start with `check_and_simulate.py`. Exploratory free/signed and bounded variants remain in `direct_study.py` and `bounded_study.py`; they must not be substituted silently for the selected categorical model. From sibling `section3_1_softmax/`, `python prepare_source_data.py` regenerates the numerical exports. It intentionally reads `../frost_direct_hi_smurf/`.

### Attention kernel

From `attention/`, after restoring the frozen softmax checkpoint:

```bash
python validate.py
python run_experiments.py
python verify_artifacts.py
```

The runner resumes verified saved jobs. It uses 128 fixed matrices and stream/digital-shift seeds from the archived protocol. QK, the FSM/TT mapping and AV remain distinct simulation stages; numerical helper operations remain part of this software experiment. CPU execution time is not a hardware result.

### Retrieval

From `retrieval/`, frozen-checkpoint evaluation:

```bash
python run_experiments.py
python verify_artifacts.py
```

To fit new teachers and recalibrate, first run `python teacher.py` and `python calibrate_softmax.py`. Those operations replace local fitted artifacts and are not needed to inspect the archived predictions. `calibration_selection.json` identifies the selected operator for each of the three trained model seeds.

### Image classification

From `classification/`:

```bash
python data.py
python evaluate.py --stage all --workers 2
python render_figures.py
python verify_results.py
```

`data.py` downloads the original benchmarks and reconstructs the exact partitions and image arrays. Frozen model/operator checkpoints and metadata should be restored before evaluation. The commands above cover all three datasets. To run CIFAR-10 alone, add `--datasets cifar10` to `evaluate.py`; the original `verify_results.py` is a complete-suite check and requires all 810 prediction/metric shards, prepared image arrays and the three legacy classification figure PNGs. `render_figures.py` produces those legacy assets; it does not replace the final publication layouts documented at the repository root. `vision.py` trains the three models for each dataset; `calibrate.py` fits the probability operators. Full finite-stream evaluation is much slower than the interface checks. The archived full-test configuration has 10,000 images and the diagnostic subset is separate. Reduce worker count when memory is limited, without changing the scientific configuration.

The archived CIFAR downloader requires the upstream server to honor HTTP Range requests. If the server does not, download the official `cifar-10-binary.tar.gz` separately into `data/raw/cifar10/`, retaining the archived MD5 check, then rerun `data.py`. No alternate image preprocessing is introduced by this fallback.

### Convergence

From `convergence/`, after restoring the fixed scores, protocols and checkpoints:

```bash
python source/run_convergence.py --verify-only
python source/run_convergence.py
python source/export_source_data.py
```

The existing fixed `source_data/inputs.npz` contains the scores needed for this diagnostic and avoids downloading CIFAR-10 again. Public copies omit original benchmark pixels. To recreate inputs from source images, first prepare classification data and use:

```bash
python source/run_convergence.py --data-dir ../classification/data
```

All finite checkpoints are nested prefixes of the saved 4,096-cycle event trajectory after burn-in; they are not independently resampled curves or training-loss histories.

## Public data archive paths

The corresponding detailed-record archive uses these prefixes:

| Archive prefix | Experiment directory |
| --- | --- |
| `operators/softmax_selected/` | `section3_1_softmax/` |
| `operators/softmax_selection/` | `frost_direct_hi_smurf/` |
| `operators/softmax_extended/` | `frost_softmax_study/` |
| `operators/attention/` | `attention/` |
| `operators/retrieval/` | `retrieval/` |
| `operators/convergence/` | `convergence/` |
| `classification/` | `classification/` |

`operator_extensions/` is a separate historical study, not the destination for `operators/softmax_extended/`. Its extra saved numerical results were not required for the final main-paper claims and remain in the historical full-artifact archive. The embedded `operator_extensions/source/archive_bsds/` files are historical copies, not the current FROST-LT entry point.

## Dependencies and preserved provenance

Core numerical experiments use Python, NumPy, SciPy, pandas and CPU PyTorch. Plotting uses Matplotlib/Pillow; older publication helpers additionally use fontTools, PyMuPDF and pypdf. Native convergence uses a C++17 compiler. Module-specific requirements and environment records preserve the original configurations; the repository root documents the supported installation route.

Original `manifest_sha256.json` files describe the **full original experiment archives**, which included data and outputs intentionally stored separately from GitHub. They are retained as provenance and should not be interpreted as GitHub-package completeness manifests. Use the current repository release manifest for code-integrity checks. Do not pool duplicate exports or intermediate summaries as independent experimental observations.
