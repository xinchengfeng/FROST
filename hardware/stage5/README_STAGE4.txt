FROST hardware study: stage 4, implemented SC controls
====================================================

This checkpoint adds actual RTL for three scalar stochastic-computing (SC)
exponentiation families. All four hardware configurations use L=64 and the
same 16-key, 16-value-channel score-to-output boundary as stage-3 FROST.
They are adaptations of published primitives to this kernel, not reproductions
of complete published accelerators. No numbers from literature are used as
our implementation measurements.

Implemented controls
--------------------
sexp256: 16-state saturating SExp FSM, gain G=2, 256 discarded updates and
64 retained output bits. This matches the primitive parameters selected for
the manuscript's earlier software experiment.
sexp16: the same primitive with 16 discarded updates. This additional control
tests whether long baseline burn-in alone explains a cycle advantage.
bern8: degree-8 Bernstein polynomial, using the previously selected nine
coefficients. Eight successive xorshift draws produce Bernoulli inputs; their
population count selects a coefficient, which drives the emitted bit.
horner5: eight fifth-order NAND-Horner factors. The eight factors operate in
parallel while five recursion levels reuse the logic across five cycles per
output bit. Their AND implements the range-factored exponential approximation.

The source relationships are Li et al. (IEEE TC 2014), Qian and Riedel
(DAC 2008), and Parhi and Liu (IEEE TETC 2019), respectively. See literature/
SC_hardware_references.bib and literature/sources.json for identifiers and
primary-source links. We wrote the RTL and reference model in this stage;
they are not code copied from the authors' hardware implementations.

Numerical and stochastic interfaces
----------------------------------
Input scores and values are signed 16-bit integers with 10 fractional bits.
For each score row, d=max(score)-score is clipped to [0,8] in real units.
Bernstein and Horner use x=d/8. SExp uses p=min(1, 1/2+d/(4G)), with
G=2, initial state 7, and emits one when the updated state is below 14.
The d=0 SExp component is forced to exp(0)=1; Bernstein and Horner already
have that endpoint property. Consequently the row count sum D is in [64,1024].

The RNG is xorshift32(13,17,5), with thresholds compared against its upper
16 bits. All 32 initial state words are loaded through the same public ports.
SExp uses 16 states (the unused 16 RNG slots are optimized away); Bernstein
and Horner use two separately seeded RNG states per key, 32 total. Bernstein
unrolls eight input draws plus one coefficient draw per output bit. Horner
unrolls eight input and eight coefficient draws per recursion cycle. This
unrolling is real logic included in synthesis. Draws are pseudo-random;
mathematical independence is not guaranteed by a deterministic PRNG.
Bernstein and Horner coefficient thresholds use ceil(2^16*c), including the
explicit 65536 threshold for probability one. All constants are archived.

The scalar output counts c_j are accumulated over 64 emitted bits. The
back end computes the count-weighted numerator S_d=sum_j(c_j*V_jd), using
four reused integer multipliers. It performs one 32-cycle restoring division
per query to obtain R=floor(2^30/D), then produces

    output_d = floor((S_d*R + 2^29)/2^30).

This shares normalization across all 16 output channels. It avoids computing
16 separate normalized attention probabilities. The accumulator is signed
26-bit, the reciprocal is unsigned 25-bit, and the final product is signed
52-bit. Counts, data loading, RNGs, reciprocal, products, buffers and control
are inside the measured boundary. No floating-point operation occurs in RTL.

These are SC-exponentiation / integer-aggregation controls. Their back end
uses the conditional weighted average of decoded counts, rather than the
extra stochastic AV resampling used in Fig.10's software graph. Therefore
these hardware results are not an implementation or accuracy evaluation of
the complete reconstruction models. FROST's original rank-2 hardware kernel
and the rank-16 reconstruction FROST-LT model also remain distinct.

Protocol and evidence
---------------------
The source and coefficients were fixed before synthesis. Methods were not
selected by their measured area or latency. Both SExp warm-up configurations
are retained, including whichever one is less favourable to the FROST story.
The retained stream length is equal, but random draw counts, internal cycles,
encoding precision and reconstruction accuracy are not assumed equal.

An independent C++ model uses direct integer division and direct mathematical
loops for each primitive; it does not copy the RTL controller or restoring
divider. On 3200 frozen query rows and 64 directed corner rows for each of
three seeds, all four RTL configurations exactly match its output words and
public-port cycles. Total checked words: 4*3*(3200+64)*16 = 626688.
Corner cases include equal scores, tied extremes, clipping, signed-16 maximum
and minimum values, alternating extremes and random full-range data.

Each tile loads 256 value words and processes 16 queries with shared values.
Per-query cycles are 162 + SC generation cycles; per-tile cycles include a
further 256 value-load cycles. SExp generation cycles are burn-in+64;
Bernstein uses 64 and Horner uses 320. The simulator measures these cycles;
the reference model independently checks their predicted values.
Reset and one-time seed initialization are excluded, as in stage 3.

Every design is synthesized using the same Yosys/ABC version and Nangate45
typical library as stage 3, and mapped/routed to the same ECP5 device using
nextpnr at an 8-MHz constraint with seeds 1,2,3. Mapped standard-cell area
is not placed-and-routed ASIC area. FPGA counts are post-route TRELLIS_COMB,
TRELLIS_FF, MULT18X18D, DP16KD and TRELLIS_RAMW, not interchangeable with
the Yosys LUT4 count. Frequency is an internal timing estimate of that
implementation, not measured board Fmax. I/O remains unconstrained.

All source data, raw CSV outputs and corner cases are retained. Descriptive
fidelity diagnostics relative to numerical softmax are also provided, so an
equal-L comparison cannot be mistaken for an equal-accuracy comparison.
These diagnostics were not used to retune the primitives or choose a winner.
Hardware SC power and energy are NOT yet evaluated in this stage. They must
be measured from new gate-level activity, not inferred from area or cycles.

Reproduction
------------
Keep stage4/ and toolchain/ as siblings when unpacking. Tools are the versions
recorded in the previous stage. Package metadata and hashes are included;
the restoration helper downloads exact Ubuntu packages when not present.
An Ubuntu 24.04-compatible x86-64 runtime, Python 3/numpy, g++, make,
dpkg-deb, zlib development headers and network access are prerequisites.

From stage4/:
  python3 source/restore_stage3_tools.py
  python3 source/stage2_flow.py build sexp16 sexp256 bern8 horner5
  python3 source/verify_sc.py sexp16 sexp256 bern8 horner5
  python3 source/stage2_flow.py asic sexp16 sexp256 bern8 horner5
  python3 source/stage2_flow.py ecp5 sexp16 sexp256 bern8 horner5
  python3 source/stage4_route.py
  python3 source/stage4_aggregate.py

The tool restoration helper may need to expose the ECP5 chip database at
/usr/share/nextpnr because the recorded Ubuntu binary uses that fixed path.
It never replaces an existing chip database. Use the machine's normal package
installation process if creating that path is unavailable.

For diagnostics, the archived stage-3 FROST comparison outputs are copied
under prior_frost/; diagnose_sc.py uses them as its original FROST reference.
The next stage will generate and check mapped gate activity for these SC
controls, estimate power under the same standard-cell assumptions, and then
prepare the manuscript's comparison figures and LaTeX integration.
