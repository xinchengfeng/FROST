"""Combine fresh SC and FROST gate activity at a common, timing-feasible 8 MHz.

No power is inferred from cell counts or borrowed from a different circuit.
The single fixed input tile and three seed repetitions are kept explicit.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import re
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [4004, 5005, 6006]
NAMES = ['sexp16', 'sexp256', 'bern8', 'horner5']
EXPECTED_CYCLES = dict(hot_reuse=2592, sexp16=4128, sexp256=7968, bern8=3872, horner5=7968)

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()

def stats(values):
    return dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values), n=len(values), values=values)

def csv_arrays(p):
    a = np.genfromtxt(p, delimiter=',', names=True)
    w = np.stack([a[f'word{i}'] for i in range(16)], -1).astype(np.uint32)
    return a, w

def read_design(name, base):
    area = json.loads((base/f'reports/asic/{name}_stat.json').read_text())['modules']['\\dut']['area']
    records = []
    for seed in SEEDS:
        stem = f'{name}_seed{seed}'
        r = json.loads((base/f'reports/power/{stem}_result.json').read_text())
        assert r['gate_rtl_output_mismatches'] == r['gate_rtl_cycle_mismatches'] == 0
        assert r['checked_words'] == 256 and r['rows'] == 16
        assert r['tile_cycles'] == EXPECTED_CYCLES[name]
        assert r['clock_MHz'] == 8
        assert r['netlist_sha256'] == digest(base/f'reports/asic/flat_{name}.v')
        assert r['waveform_sha256'] == digest(base/f'reports/power/{stem}.vcd.gz')
        assert r['liberty_sha256'] == digest(ROOT/'third_party/nangate45/NangateOpenCellLibrary_typical.lib')
        # Archived report verification does not require redistributing an executable.
        # An explicitly supplied/rebuilt binary is hashed when available.
        sta=Path(os.environ.get('FROST_OPENSTA', ROOT.parent/'toolchain/opensta_build/sta'))
        if sta.is_file():
            expected_sta=digest(sta)
        else:
            provenance=json.loads((ROOT.parent/'toolchain/stage3_tool_provenance.json').read_text())
            expected_sta=provenance['OpenSTA']['binary_sha256']
        assert r['sta_binary_sha256'] == expected_sta
        assert re.search(r'unannotated\s+0\b', r['activity_annotation'])
        assert 'slack (MET)' in (base/f'reports/power/{stem}_timing.txt').read_text()
        assert r['begin_ps'] == 35*125000
        assert r['end_ps'] == (35+r['tile_cycles'])*125000
        p = r['power_W']['Total']
        assert abs(p['total']-sum(p[k] for k in ['internal', 'switching', 'leakage'])) < p['total']*2.4e-7
        assert abs(r['energy_uJ'] - p['total']*r['tile_cycles']/8) < 1e-12
        a, w = csv_arrays(base/f'results/gate_power_{stem}.csv')
        b, q = csv_arrays(base/f'results/rtl_power_{stem}.csv')
        assert np.array_equal(w, q) and np.array_equal(a['cycles'], b['cycles'])
        if name in NAMES:
            g, v = csv_arrays(ROOT/f'results/{name}_evaluation_seed{seed}_golden.csv')
            assert np.array_equal(w, v[:16]) and np.array_equal(a['cycles'], g['cycles'][:16])
            assert r['gate_golden_output_mismatches'] == r['gate_golden_cycle_mismatches'] == 0
        records.append(r)
    out = dict(design=name, mapped_cell_area_um2=area, clock_MHz=8,
               tile_cycles=EXPECTED_CYCLES[name], tile_latency_us=EXPECTED_CYCLES[name]/8,
               checked_gate_words=768, seeds=SEEDS,
               origin='new stage-5 gate runs at 8 MHz; unchanged RTL/netlist')
    for key, field in [('total', 'total_power_mW'), ('internal', 'internal_power_mW'),
                       ('switching', 'switching_power_mW'), ('leakage', 'leakage_power_mW')]:
        out[field] = stats([r['power_W']['Total'][key]*1000 for r in records])
    out['energy_per_tile_uJ'] = stats([r['energy_uJ'] for r in records])
    return out

def main():
    frost = read_design('hot_reuse', ROOT)
    controls = [read_design(n, ROOT) for n in NAMES]
    stage4 = json.loads((ROOT/'results/stage4_summary.json').read_text())
    stage3 = json.loads((ROOT/'prior_frost/stage3_summary.json').read_text())
    original = next(r for r in stage3['standard_cell_power'] if r['design']=='hot_reuse')
    assert frost['mapped_cell_area_um2'] == original['mapped_cell_area_um2']
    assert frost['tile_cycles'] == original['tile_cycles']
    for seed in SEEDS:
        a,w = csv_arrays(ROOT/f'results/gate_power_hot_reuse_seed{seed}.csv')
        b,q = csv_arrays(ROOT/f'prior_frost/standard_cell/results/gate_power_hot_reuse_seed{seed}.csv')
        assert np.array_equal(w,q) and np.array_equal(a['cycles'],b['cycles'])
    comparisons = {}
    for r in controls:
        s4 = next(s for s in stage4['SC_results'] if s['design']==r['design'])
        assert s4['mapped_cell_area_um2'] == r['mapped_cell_area_um2']
        assert s4['tile_cycles'] == r['tile_cycles']
        comparisons[r['design']] = dict(
            cell_area_reduction_percent=100*(1-frost['mapped_cell_area_um2']/r['mapped_cell_area_um2']),
            cycle_reduction_percent=100*(1-frost['tile_cycles']/r['tile_cycles']),
            power_reduction_percent=100*(1-frost['total_power_mW']['mean']/r['total_power_mW']['mean']),
            energy_reduction_percent=100*(1-frost['energy_per_tile_uJ']['mean']/r['energy_per_tile_uJ']['mean']),
            energy_ratio_SC_over_FROST=r['energy_per_tile_uJ']['mean']/frost['energy_per_tile_uJ']['mean'])
    result = dict(
        status='Stage 5 complete: mapped SC gate outputs, activity and pre-layout power verified; final figures and manuscript pending',
        boundary='16-key, 16-value-channel score-to-weighted-value kernel. One tile is 16 queries. L=64. Not a complete Transformer.',
        clock_selection='15 MHz failed all SC mapped timing diagnostics. 8 MHz was then selected for every design before final power comparisons; all five are simulated anew at 8 MHz.',
        power_conditions='Nangate45 typical library: 1.1 V, 25 degrees C, 8 MHz; identical OpenSTA binary, library, inputs and seed files.',
        timing_scope='Mapped internal register paths meet the common clock. External I/O delays and pad loads are unspecified; this is not routed ASIC timing closure.',
        activity_scope='First frozen evaluation tile; 256 value loads, score loads, burn-in, conversion and output transfer included; reset and 32-cycle seed setup excluded.',
        excluded_costs='Pre-layout zero-delay activity excludes wire and clock-tree parasitics, physical glitches, pads, board, host and external-memory cost.',
        statistics='Mean and sample SD over three stream seeds on the same one input tile; not between-input confidence intervals.',
        controls='Implemented literature-derived SExp, Bernstein and Horner stochastic exponentiation with shared integer count-weighted value aggregation; not whole published accelerators or stochastic AV.',
        FROST_scope='Fixed-coefficient rank-2 kernel with burn-in=16. ASIC uses hot_reuse, FPGA uses perm; they are distinct target-specific RTL variants with equivalent outputs and cycles. These costs do not describe the rank-16 FROST-LT reconstruction model.',
        quality_scope='Equal stream length does not imply matched output fidelity. Diagnostic RMSE is retained in fidelity_diagnostics.json. No accuracy-parity or end-to-end advantage is established.',
        FPGA_caveat='FROST uses no DSP blocks, but the SExp controls use fewer FPGA COMB and FF resources. ASIC power cannot be relabelled as FPGA power.',
        new_SC_gate_words_checked=3072, new_FROST_gate_words_checked=768, total_new_gate_words_checked=3840,
        standard_cell_power=[frost]+controls,
        optimized_FROST_vs_SC=comparisons,
        FPGA_results_source='stage4_summary.json',
        data_provenance='Each raw waveform, mapped netlist, power report and activity-window manifest is retained with SHA256.')
    (ROOT/'results/stage5_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    with (ROOT/'results/stage5_standard_cell.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['design','cell_area_um2','tile_cycles','latency_us_at_8MHz','power_mW_mean','power_mW_sample_sd','energy_uJ_mean','energy_uJ_sample_sd','stream_seeds','gate_words'])
        for r in result['standard_cell_power']:
            writer.writerow([r['design'],r['mapped_cell_area_um2'],r['tile_cycles'],r['tile_latency_us'],r['total_power_mW']['mean'],r['total_power_mW']['sample_sd'],r['energy_per_tile_uJ']['mean'],r['energy_per_tile_uJ']['sample_sd'],3,r['checked_gate_words']])
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
