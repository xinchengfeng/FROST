"""Aggregate only completed, newly validated stage-3 runs; no placeholder data."""
from pathlib import Path
import csv
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def mean_sd(values):
    return {'mean': statistics.mean(values),
            'sample_sd': statistics.stdev(values) if len(values)>1 else None,
            'n': len(values), 'values': values}

def main():
    route=[]
    for name in ['l64c','perm','fx16','fp32_1','fp32_4']:
        reports=[json.loads((ROOT/f'reports/route/{name}_s{s}.json').read_text()) for s in [1,2,3]]
        counts=[{k:v['used'] for k,v in r['utilization'].items()} for r in reports]
        assert counts[0]==counts[1]==counts[2]
        f=[next(iter(r['fmax'].values()))['achieved'] for r in reports]
        assert min(f)>=8
        c=counts[0]
        route.append(dict(design=name, TRELLIS_COMB=c['TRELLIS_COMB'],
                          TRELLIS_FF=c['TRELLIS_FF'], MULT18X18D=c['MULT18X18D'],
                          DP16KD=c['DP16KD'], TRELLIS_RAMW=c['TRELLIS_RAMW'],
                          internal_timing_estimate_MHz=mean_sd(f), common_clock_MHz=8))
    power=[]
    for name in ['l64c','hot_reuse','fx16','fp32_1','fp32_4']:
        seeds=[4004,5005,6006] if name in ['l64c','hot_reuse'] else [4004]
        records=[json.loads((ROOT/f'reports/power/{name}_seed{s}_result.json').read_text()) for s in seeds]
        area=json.loads((ROOT/f'reports/asic/{name}_stat.json').read_text())['modules']['\\dut']['area']
        for r in records:
            assert r['gate_rtl_output_mismatches']==r['gate_rtl_cycle_mismatches']==0
            assert r['checked_words']==256 and r['rows']==16
            assert r['netlist_sha256']==digest(ROOT/f'reports/asic/flat_{name}.v')
            assert 'unannotated     0' in r['activity_annotation']
            t=r['power_W']['Total']
            # OpenSTA stores power totals as float; allow two float epsilons.
            assert abs(t['total']-sum(t[k] for k in ['internal','switching','leakage']))<t['total']*2.4e-7
            assert abs(r['energy_uJ']-t['total']*r['tile_cycles']/15)<1e-12
            assert 'slack (MET)' in (ROOT/f"reports/power/{name}_seed{r['seed']}_timing.txt").read_text()
        assert len(set(r['tile_cycles'] for r in records))==1
        power.append(dict(design=name, mapped_cell_area_um2=area, clock_MHz=15,
                          tile_cycles=records[0]['tile_cycles'],
                          tile_latency_us=records[0]['tile_latency_us'],
                          total_power_mW=mean_sd([r['power_W']['Total']['total']*1000 for r in records]),
                          internal_power_mW=mean_sd([r['power_W']['Total']['internal']*1000 for r in records]),
                          switching_power_mW=mean_sd([r['power_W']['Total']['switching']*1000 for r in records]),
                          leakage_power_mW=mean_sd([r['power_W']['Total']['leakage']*1000 for r in records]),
                          energy_per_tile_uJ=mean_sd([r['energy_uJ'] for r in records]),
                          checked_gate_words=sum(r['checked_words'] for r in records),
                          seeds=seeds))
    old,new=power[:2]
    improvement={k:100*(1-new[k]['mean']/old[k]['mean']) for k in ['total_power_mW','internal_power_mW','switching_power_mW','leakage_power_mW','energy_per_tile_uJ']}
    improvement['mapped_cell_area_percent']=100*(1-new['mapped_cell_area_um2']/old['mapped_cell_area_um2'])
    improvement['FPGA_TRELLIS_COMB_percent']=100*(1-route[1]['TRELLIS_COMB']/route[0]['TRELLIS_COMB'])
    improvement['FPGA_TRELLIS_FF_percent']=100*(1-route[1]['TRELLIS_FF']/route[0]['TRELLIS_FF'])
    comparisons={x['design']:{'energy_reduction_percent':100*(1-new['energy_per_tile_uJ']['mean']/x['energy_per_tile_uJ']['mean']),
                              'cell_area_reduction_percent':100*(1-new['mapped_cell_area_um2']/x['mapped_cell_area_um2'])} for x in power[2:]}
    result=dict(status='stage 3 complete; figures and manuscript integration pending',
                scope='Fixed-coefficient rank-2 score-to-weighted-value kernel; 16 keys and 16 value channels; L=64, burn-in=16. Not a complete Transformer or the reconstruction FROST-LT model.',
                fpga_variant='perm', asic_variant='hot_reuse',
                power_scope='Nangate45 typical 1.1 V, 25 C; mapped standard-cell, zero-delay activity, pre-layout estimate. No wire parasitics, clock-tree insertion, glitches, pads, board or host costs.',
                activity_scope='First frozen evaluation tile: 16 query rows, including 256 value loads, score loads, burn-in, conversion and output transfer. Reset and random-source seed initialization excluded.',
                timing_scope='FPGA internal timing estimates from 8-MHz-constrained routed designs; I/O unconstrained. Not measured board Fmax or a frequency search.',
                uncertainty_scope='Power sample SD varies stream seeds on one fixed tile, not input-population uncertainty. Route SD varies placement seeds. Deterministic controls have n=1 power run.',
                baselines='FP32-LUT (1 or 4 FMA lanes) and FX16. These are numerical controls, not SC baselines; hardware SC comparisons remain pending.',
                quality_scope='Optimizations preserve the existing integer kernel outputs. No new accuracy-parity or end-to-end accuracy claim; archived candidate quality limitations remain applicable.',
                fpga=route,standard_cell_power=power,optimization_reductions=improvement,
                optimized_vs_numeric_controls=comparisons)
    (OUT/'stage3_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'stage3_fpga.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['design','TRELLIS_COMB','TRELLIS_FF','MULT18X18D','DP16KD','TRELLIS_RAMW','timing_MHz_seed1','timing_MHz_seed2','timing_MHz_seed3'])
        for r in route:
            w.writerow([r[k] for k in ['design','TRELLIS_COMB','TRELLIS_FF','MULT18X18D','DP16KD','TRELLIS_RAMW']]+r['internal_timing_estimate_MHz']['values'])
    with (OUT/'stage3_standard_cell.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['design','area_um2','cycles_per_tile','latency_us_at_15MHz','power_mW_mean','power_mW_seed_sd','energy_uJ_mean','energy_uJ_seed_sd','n_stream_seeds','gate_words_checked'])
        for r in power:
            w.writerow([r['design'],r['mapped_cell_area_um2'],r['tile_cycles'],r['tile_latency_us'],r['total_power_mW']['mean'],r['total_power_mW']['sample_sd'],r['energy_per_tile_uJ']['mean'],r['energy_per_tile_uJ']['sample_sd'],r['energy_per_tile_uJ']['n'],r['checked_gate_words']])
    print(json.dumps({'optimization_reductions':improvement,'optimized_vs_numeric_controls':comparisons},indent=2))

if __name__=='__main__':
    main()
