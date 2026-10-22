"""Aggregate SC hardware measurements; do not generate unmeasured energy."""
from pathlib import Path
import csv
import hashlib
import json
import statistics

ROOT=Path(__file__).resolve().parents[1]
names=['sexp16','sexp256','bern8','horner5']
verification=json.loads((ROOT/'results/rtl_verification.json').read_text())
assert len(verification)==24
assert sum(r['checked_words'] for r in verification)==626688
assert all(r['output_mismatches']==r['cycle_mismatches']==0 for r in verification)
prior=json.loads((ROOT/'prior_frost/stage3_summary.json').read_text())
fpga_ref=next(r for r in prior['fpga'] if r['design']=='perm')
asic_ref=next(r for r in prior['standard_cell_power'] if r['design']=='hot_reuse')
records=[]
for name in names:
    vr=[r for r in verification if r['design']==name]
    cycles={r['tile_cycles'] for r in vr};assert len(cycles)==1
    area=json.loads((ROOT/f'reports/asic/{name}_stat.json').read_text())['modules']['\\dut']['area']
    routed=[json.loads((ROOT/f'reports/route/{name}_s{s}.json').read_text()) for s in [1,2,3]]
    u=[{k:v['used'] for k,v in r['utilization'].items()} for r in routed]
    assert u[0]==u[1]==u[2]
    f=[next(iter(r['fmax'].values()))['achieved'] for r in routed]
    assert min(f)>=8
    for seed in [1,2,3]:
        rec=json.loads((ROOT/f'reports/route/{name}_s{seed}_manifest.json').read_text())
        assert rec['exit_code']==0 and rec['netlist_sha256']==hashlib.sha256((ROOT/f'reports/ecp5/{name}.json').read_bytes()).hexdigest()
    record=dict(design=name,stream_length=64,tile_cycles=cycles.pop(),mapped_cell_area_um2=area,
                fpga={k:u[0][k] for k in ['TRELLIS_COMB','TRELLIS_FF','MULT18X18D','DP16KD','TRELLIS_RAMW']},
                routed_internal_timing_MHz=f,common_clock_MHz=8,
                checked_words=sum(r['checked_words'] for r in vr),gate_verification='pending',
                power_W=None,energy_per_tile_uJ=None)
    record['tile_service_us_at_8MHz']=record['tile_cycles']/8
    record['FROST_cell_area_reduction_percent']=100*(1-asic_ref['mapped_cell_area_um2']/area)
    record['FROST_cycle_reduction_percent']=100*(1-asic_ref['tile_cycles']/record['tile_cycles'])
    record['FROST_cycle_speedup']=record['tile_cycles']/asic_ref['tile_cycles']
    records.append(record)
result=dict(status='RTL verification, synthesis and FPGA routing complete; SC gate-activity power pending',
            boundary='16-key x 16-value-channel score-to-output kernel; one tile contains 16 queries, with L=64. Not a complete Transformer.',
            controls='Literature-derived stochastic exponentiation followed by a shared integer count-weighted AV; not reproductions of complete published accelerators.',
            statistics='Three RNG seeds for output checks; three placement seeds for timing. No same-accuracy claim.',
            reference_note='The FPGA FROST reference is perm; the standard-cell reference is hot_reuse. Their outputs and cycles match but the RTL implementations differ.',
            FROST_reference=dict(fpga=fpga_ref,standard_cell_area_um2=asic_ref['mapped_cell_area_um2'],tile_cycles=asic_ref['tile_cycles']),
            SC_results=records,checked_words=626688,
            caveats=['Area is mapped standard-cell area, not routed ASIC area.',
                     'Timing estimates are internal paths in 8-MHz-constrained FPGA implementations; I/O is unconstrained.',
                     'No SC power or energy has yet been computed. Do not combine these areas with old FP32 or FROST power.',
                     'FROST has more FPGA COMB and FF resources than the SExp controls, though fewer DSP blocks and fewer cycles.',
                     'Diagnostic output fidelity differs; equal recorded L does not establish accuracy parity.'])
(ROOT/'results/stage4_summary.json').write_text(json.dumps(result,indent=2)+'\n')
with (ROOT/'results/stage4_comparison.csv').open('w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['design','SC_L','area_um2','tile_cycles','tile_us_at_8MHz','TRELLIS_COMB','TRELLIS_FF','MULT18X18D','DP16KD','TRELLIS_RAMW','timing_MHz_seed1','timing_MHz_seed2','timing_MHz_seed3'])
    w.writerow(['FROST_target_specific',64,asic_ref['mapped_cell_area_um2'],asic_ref['tile_cycles'],asic_ref['tile_cycles']/8]+[fpga_ref[k] for k in ['TRELLIS_COMB','TRELLIS_FF','MULT18X18D','DP16KD','TRELLIS_RAMW']]+fpga_ref['internal_timing_estimate_MHz']['values'])
    for r in records:
        w.writerow([r['design'],64,r['mapped_cell_area_um2'],r['tile_cycles'],r['tile_service_us_at_8MHz']]+list(r['fpga'].values())+r['routed_internal_timing_MHz'])
print(json.dumps(records,indent=2))
