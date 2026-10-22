# Restoring the separately archived data

The data and software records have different roles. This repository contains source code, configuration, small functional fixtures and figure inputs. The companion data release contains frozen model weights, large predictions, per-run records, reports and switching traces.

**Data DOI:** not yet supplied to this preparation. Add the actual version-specific DOI here after the data record is published. The private `/uploads/...` editing URL is not a public data citation.

## Available archives

- `FROST_Source_Data_20261022.zip`: compact CSV/JSON source data for the current figures and tables.
- `FROST_Experimental_Records_20261022.zip`: checkpoints, predictions, fixed inputs and experiment records.
- `FROST_Hardware_Records_20261022.zip`: original hardware inputs, implementation/verification outputs and reports.
- `FROST_Hardware_Waveforms_part01_20261022.zip`, `part02`, `part03`: all 15 final 8-MHz gate activity traces. These are ordinary independent ZIPs, not multipart fragments requiring special extraction software.

Use the current 20261022 data release, not the earlier uncurated `FROST_Artifacts.zip` overlay instructions found in historical READMEs.

## Restore software records

From the repository root, first preview the operation (replace the example paths with actual downloaded files):

```bash
python tools/restore_data.py \
  --experiment-records /path/FROST_Experimental_Records_20261022.zip \
  --source-data /path/FROST_Source_Data_20261022.zip
```

Add `--apply` to copy verified data into the expected module directories. The helper verifies the release manifest and refuses to overwrite differing existing files unless `--overwrite` is explicitly supplied. Use a separate reproduction checkout if you want to preserve an unchanged source-only upload folder.

The mapping is:

| Public records prefix | Repository destination |
|---|---|
| `operators/softmax_selection` | `experiments/frost_direct_hi_smurf` |
| `operators/softmax_selected` | `experiments/section3_1_softmax` |
| `operators/softmax_extended` | `experiments/frost_softmax_study` |
| `operators/attention` | `experiments/attention` |
| `operators/retrieval` | `experiments/retrieval` |
| `operators/convergence` | `experiments/convergence` |
| `classification` | `experiments/classification` |
| `reconstruction_original` | `experiments/reconstruction_original` |
| `reconstruction_LT` | `experiments/reconstruction_lt` |
| `reconstruction_scalar` | `experiments/reconstruction_scalar` |
| Source Data ZIP contents | `local_data/source_data` |

The optional historical `operator_extensions` workflow has separate original data; it is not silently mapped onto the distinct systematic softmax study. See its README before using it.

## Restore hardware records

```bash
python tools/restore_data.py \
  --hardware-records /path/FROST_Hardware_Records_20261022.zip \
  --waveforms /path/FROST_Hardware_Waveforms_part01_20261022.zip \
              /path/FROST_Hardware_Waveforms_part02_20261022.zip \
              /path/FROST_Hardware_Waveforms_part03_20261022.zip \
  --apply
```

The records are restored under `hardware/stage3`, `stage5` and `stage6`. Large traces are optional for inspecting published summaries; they are needed for exact replay of archived switching-activity power inputs. Tool installation and physical implementation are separate operations.

## Reference images

Original MNIST/Fashion-MNIST/CIFAR-10/BSDS500 image corpora are acquired from their providers. Public reconstruction NPZ copies omit `truth` arrays, and convergence records omit original image `pixels`; all generated predictions and other retained arrays are unchanged. Reference hashes and sample IDs identify the expected arrays.

Use the reconstruction preparation helper and its README to recreate locally required `truth` arrays from official BSDS500 JPGs. This restores local inputs rather than modifying public research measurements. Figure 6 already includes saved score/event arrays; recomputing the score-extraction stage requires the corresponding official CIFAR-10 images.

Do not commit restored full datasets, model weights, waveforms or large experiment results to the GitHub source repository. The top-level `.gitignore` covers standard generated outputs; the restoration helper also records newly restored files in local Git exclusions when a Git checkout is present.
