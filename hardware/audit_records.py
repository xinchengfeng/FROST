#!/usr/bin/env python3
"""Audit the overlaid final hardware reports without needing EDA executables.

Waveform hashing is optional because activity traces are separate large files.
This validates archived consistency, not a new gate simulation or power run.
"""
import argparse, hashlib, json, statistics
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def arrays(p):
    a=np.genfromtxt(p,delimiter=',',names=True)
    w=np.stack([a[f'word{i}'] for i in range(16)],-1).astype(np.uint32)
    return a,w

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-waves',action='store_true')
    p.add_argument('--json',type=Path)
    args=p.parse_args()
    base=ROOT/'stage5'
    summary=json.loads((ROOT/'stage6/data/stage5_summary.json').read_text())
    prov=json.loads((ROOT/'toolchain/stage3_tool_provenance.json').read_text())
    records=[]
    for s in summary['standard_cell_power']:
        name=s['design'];energy=[]
        for seed in [4004,5005,6006]:
            stem=f'{name}_seed{seed}'
            path=base/f'reports/power/{stem}_result.json'
            if not path.is_file():
                raise FileNotFoundError(f'{path}: prepare the hardware-records data overlay first')
            r=json.loads(path.read_text())
            assert r['clock_MHz']==8 and r['tile_cycles']==s['tile_cycles']
            assert r['gate_rtl_output_mismatches']==r['gate_rtl_cycle_mismatches']==0
            assert r['checked_words']==256
            assert r['netlist_sha256']==digest(base/f'reports/asic/flat_{name}.v')
            assert r['liberty_sha256']==digest(base/'third_party/nangate45/NangateOpenCellLibrary_typical.lib')
            assert r['sta_binary_sha256']==prov['OpenSTA']['binary_sha256']
            if args.check_waves:
                assert r['waveform_sha256']==digest(base/f'reports/power/{stem}.vcd.gz')
            a,w=arrays(base/f'results/gate_power_{stem}.csv')
            b,v=arrays(base/f'results/rtl_power_{stem}.csv')
            assert np.array_equal(w,v) and np.array_equal(a['cycles'],b['cycles'])
            if name!='hot_reuse':
                g,q=arrays(base/f'results/{name}_evaluation_seed{seed}_golden.csv')
                assert np.array_equal(w,q[:16]) and np.array_equal(a['cycles'],g['cycles'][:16])
            assert int(a['cycles'].sum())==s['tile_cycles']
            e=r['power_W']['Total']['total']*s['tile_cycles']/8
            assert abs(e-r['energy_uJ'])<1e-12
            energy.append(e)
            records.append({'design':name,'seed':seed,'words':256,'output_mismatches':0,'cycle_mismatches':0})
        assert abs(statistics.mean(energy)-s['energy_per_tile_uJ']['mean'])<1e-12
        assert abs(statistics.stdev(energy)-s['energy_per_tile_uJ']['sample_sd'])<1e-12
    out={'status':'pass','scope':'archived report consistency audit; no new EDA run',
         'checked_output_words':sum(r['words'] for r in records),'waveform_hashes_checked':args.check_waves,
         'sta_binary_check':'archived metadata matches recorded provenance; executable not redistributed',
         'checks':records}
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True)
        args.json.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
