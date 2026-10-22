FROST algorithm improvement: exploratory software study

Outcome: reconstruction improves substantially relative to archived FROST,
but the selected variant does not dominate the SC controls. See REPORT_zh.txt.
No new hardware PPA was measured. This is a NEW input-state encoding and
sampler; historical FROST hardware measurements cannot be reused for it.

Environment: Python 3.12, PyTorch 2.5.1+cpu, NumPy/SciPy and scikit-image
versions in requirements.txt; C++17 compiler. Run from this directory.

Build native finite-event kernels:
  python -c "import sys;sys.path.insert(0,'source');import sc_baselines;sc_baselines.build()"
  python source/local_tt.py
  python legacy_runtime/source/scripts/fast_finite.py

Reproduce the selected model from the saved independent calibration scores:
  python source/train_candidates.py --name local33_r16_max8_aug --basis local --states 33 --rank 16 --bound 8 --encoding max --augment --steps 12000
The selected checkpoint is the minimum expected-probability validation RMSE
within that run. Architecture/sampler selection used finite stream validation:
  python source/select_finite.py

Verify actual threshold logic and count conservation:
  python source/verify_local.py checkpoints/local33_r16_max8_aug.pt
  python source/verify_stratified.py checkpoints/local33_r16_max8_aug.pt

Reproduce the three-seed image results:
  python source/evaluate.py --method SExp
  python source/evaluate.py --method Bernstein
  python source/evaluate.py --method Bernstein_QMC
  python source/evaluate.py --method Bernstein_stratified
  python source/evaluate.py --method LocalTT33_stratified --checkpoint checkpoints/local33_r16_max8_aug.pt
  python source/summarize.py

Existing completed result files are reused by evaluate.py. For a fresh run,
work in a copy and move its per-image results to a separate archive first.
Training/selection/report scripts overwrite their own named outputs. Keep the
packaged results unchanged if an exact record of this experiment is required.

Other completed training configurations (add --name matching the file stem):
 geometric16_r8_b4: --basis geometric --states 16 --rank 8 --bound 4 --steps 2000
 local32_r16_b8: --basis local --states 32 --rank 16 --bound 8 --steps 2500 --init random
 local32_r16_identity: --basis local --states 32 --rank 16 --bound 8 --steps 10000
 local64_r32_identity: --basis local --states 64 --rank 32 --bound 8 --steps 10000
 geometric16_r16_identity: --basis geometric --states 16 --rank 16 --bound 8 --fresh --steps 6000
 geometric16_r8_identity: --basis geometric --states 16 --rank 8 --bound 4 --fresh --steps 6000
 local33_r16_max8: --basis local --states 33 --rank 16 --bound 8 --encoding max --steps 8000
 local65_r16_max12: --basis local --states 65 --rank 16 --bound 12 --encoding max --steps 8000
 local33_r32_max8_aug: --basis local --states 33 --rank 32 --bound 8 --encoding max --augment --steps 12000
 local17_r16_max8_aug: --basis local --states 17 --rank 16 --bound 8 --encoding max --augment --steps 12000
 local65_r16_max8_aug: --basis local --states 65 --rank 16 --bound 8 --encoding max --augment --steps 12000

A final antithetic sampler was tested only on independent validation scores,
performed worse, and was not used for image comparisons. Its source and
validation record are retained; it has no image-performance or hardware claim.

Data boundaries:
- calibration_protocol.json lists 12 calibration image IDs and 4 disjoint
  validation image IDs. Their frozen teachers and scores are included.
- the three pilot IDs never entered candidate fitting or selection losses.
- all image teachers were originally fitted to their own target images;
  these are fitted-image reconstructions, not held-out image restoration.
- 101/202/303 are stream seeds at fixed weights, not independent training runs.
- original FROST numbers are recomputed from archived prediction arrays;
  stage 2 reproduced seed 101 arrays exactly. Controls with seed 101 also
  reproduced their stage 2 arrays exactly in this stage.

Sampler definitions:
LocalTT uses adjacent coarse-grid addresses with fractional Bernoulli input
selection, followed by learned row-stochastic TT transitions. QMC uses distinct
Sobol coordinates. StratifiedTT groups particles by latent state, creates
actual fractional-threshold state selections, and applies stratified CDF
thresholds to emit new latent states. Every stage conserves L particles.
No analytic softmax is evaluated inside these finite-event inference methods.
This changes the original saturating random-walk FSM input interface. Sorting,
max subtraction, address calculation, integer division for stratum thresholds,
grouping, count decoding and all other original numerical helpers remain.
They must be included in a future hardware implementation boundary.

The baseline adapters retain scalar SC exponentiation and numerical count
normalization. QMC and stratified Bernstein are additional adapted controls,
not claimed reproductions of published complete accelerators.

Files containing DIAGNOSTIC or diagnosis report expectation/numerical
substitutions to localize errors. They are NOT finite SC results.
CPU elapsed times are NOT hardware latency, power or energy measurements.

Core parameter count: original 16,512; selected prototype 127,248. This is
not mapped gate area or BRAM usage. The new model's hardware cost is unknown.

Source references for the controls remain those documented in the stage 2
package: Li et al., IEEE TC 2014, DOI 10.1109/TC.2012.231; Qian & Riedel,
DAC 2008, DOI 10.1109/DAC.2008.4555898. New prototype modifications in this
package have not been established as novel relative to the wider literature.
See LICENSE_STATUS.txt. No blanket license is assigned to archived inputs.
