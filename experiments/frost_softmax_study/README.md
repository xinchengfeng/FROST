# FROST Softmax reference study

Read FROST_softmax_report.md before using any numbers in a manuscript. This archive contains a new strict row-stochastic-core TT reference, controls retaining exact Softmax, and an unconstrained signed-TT diagnostic. They are explicitly different models. No hardware area/power result is claimed.

## Reproduce

Use Python 3.12+ and the recorded package versions in environment.json. CPU execution is sufficient. Run from this directory:

```bash
python study.py validate
python run_training.py
python evaluate.py
python streams.py
python signed_diagnostic.py
python validate_signed_occupancy.py
python figures_and_summary.py
python build_report.py
```

The 70 primary models are skipped when checkpoints already exist. To train them from scratch, move checkpoints to a separate backup directory first. Existing results are overwritten by evaluation scripts. Checkpoints are loaded with weights_only=True.

Main results use five fixed initialization seeds and shared, disjoint train/validation/test data. Signed TT diagnostics use a different loss and are exploratory. State-probability violations are reported rather than repaired by an unreported output normalization. Finite-stream KL may be infinite; CSVs preserve that fact.

The archival prior checkpoint uses its original exact-Softmax forward path, preserved in provenance/previous_model_definitions.py. It is a historical audit control, not new strict-model evidence.

Figures are supplied as PNG and vector PDF. methods_and_results.tex is a factual draft of the current reference experiment, not a claim that the final FROST design has been validated. protocol.json records the original plan; post-audit signed-TT diagnostics and offline core quantization are separately identified in protocol_additions.json.
