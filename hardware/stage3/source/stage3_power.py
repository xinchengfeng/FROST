"""Gate activity, exact port checks, and a separate pre-layout power estimate.

The workload boundary is one 16-query tile, including value/score loading,
burn-in, numerical conversion, and output transfer; reset and seed setup are
excluded. Zero-delay switching does not include physical glitches or wire/CTS
parasitics. This is not FPGA/board power or a whole-Transformer measurement.
"""
from pathlib import Path
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import time
import numpy as np
from stage2_flow import ROOT, ENV, NATIVE

if (NATIVE/'share/tcltk/tcl8.6').is_dir():
    ENV['TCL_LIBRARY'] = str(NATIVE/'share/tcltk/tcl8.6')
OUT = ROOT/'reports/power'
OUT.mkdir(parents=True,exist_ok=True)
FROST = ['l64c', 'hot_reuse']
import os
STA = Path(os.environ.get('FROST_OPENSTA') or shutil.which('sta', path=ENV['PATH']) or str(NATIVE.parent.parent/'opensta_build/sta')).resolve()


def read_csv(path):
    z = np.genfromtxt(path, delimiter=',', names=True)
    w = np.stack([z[f'word{i}'] for i in range(16)], -1).astype(np.uint32)
    return z, w


def run_wave(job):
    name, seed = job
    stem = f'{name}_seed{seed}'
    length = 64 if name in FROST else 256  # Numeric controls ignore this input.
    gate = ROOT/'build/gate'/name/'Vdut'
    rtl = ROOT/'build'/name/'Vdut'
    if not rtl.exists() and name in FROST:
        # The initial run reused the already verified stage-2 RTL executable.
        rtl = ROOT.parent/'stage2'/'build'/name/'Vdut'
    gcsv = ROOT/f'results/gate_power_{stem}.csv'
    rcsv = ROOT/f'results/rtl_power_{stem}.csv'
    wave = OUT/f'{stem}.vcd.gz'
    cmd = [str(gate), 'mem/scores.hex', f'mem/seed{seed}.hex', str(gcsv), str(length), '16', '/dev/stdout']
    t = time.monotonic()
    with (OUT/f'{stem}_sim.log').open('w') as log:
        process = subprocess.Popen(cmd, cwd=ROOT, env=ENV, stdout=subprocess.PIPE, stderr=log)
        with gzip.open(wave, 'wb', compresslevel=1) as out:
            shutil.copyfileobj(process.stdout, out, 1024*1024)
        result = process.wait()
    if result:
        raise RuntimeError(f'Gate simulation failed: {stem}')
    subprocess.run([str(rtl), 'mem/scores.hex', f'mem/seed{seed}.hex', str(rcsv), str(length), '16'],
                   cwd=ROOT, env=ENV, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    a,w = read_csv(gcsv)
    b,q = read_csv(rcsv)
    assert len(a)==16 and np.array_equal(w,q) and np.array_equal(a['cycles'],b['cycles']), stem
    with gzip.open(wave, 'rt') as f:
        header = []
        for line in f:
            header.append(line)
            if '$enddefinitions' in line:
                break
    assert re.search(r'\$timescale\s+1ps\s+\$end', ''.join(header)), stem
    cycles = int(a['cycles'].sum())
    rec = dict(design=name, seed=seed, stream_length=64 if name in FROST else None,
               gate_rtl_output_mismatches=0, gate_rtl_cycle_mismatches=0,
               checked_words=int(w.size), rows=16, tile_cycles=cycles,
               begin_ps=35*200000//3, end_ps=(35+cycles)*200000//3,
               netlist_sha256=hashlib.sha256((ROOT/f'reports/asic/flat_{name}.v').read_bytes()).hexdigest(),
               waveform_sha256=hashlib.sha256(wave.read_bytes()).hexdigest(),
               waveform_bytes=wave.stat().st_size, command=cmd, seconds=time.monotonic()-t)
    (OUT/f'{stem}_wave_manifest.json').write_text(json.dumps(rec, indent=2)+'\n')
    print('Verified gate waveform: '+stem, flush=True)


def run_sta(job):
    name,seed = job
    stem = f'{name}_seed{seed}'
    rec = json.loads((OUT/f'{stem}_wave_manifest.json').read_text())
    tcl = f'''read_liberty third_party/nangate45/NangateOpenCellLibrary_typical.lib
read_verilog reports/asic/flat_{name}.v
link_design dut
create_clock -name clk -period 66.6666666666667 [get_ports clk]
set_input_transition 0.05 [all_inputs]
read_vcd -scope TOP/dut -begin_time {rec['begin_ps']} -end_time {rec['end_ps']} reports/power/{stem}.vcd.gz
report_units
report_activity_annotation > reports/power/{stem}_annotation.txt
report_power -format json -digits 10 > reports/power/{stem}_power.json
report_checks -path_delay max -group_path_count 1 > reports/power/{stem}_timing.txt
exit
'''
    path=OUT/f'{stem}.tcl'
    path.write_text(tcl)
    cmd=[str(STA),'-no_splash','-no_init','-exit',str(path)]
    t=time.monotonic()
    with (OUT/f'{stem}_sta.log').open('w') as f:
        process=subprocess.run(cmd,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT)
    log=(OUT/f'{stem}_sta.log').read_text()
    assert process.returncode==0 and 'Error:' not in log, stem+' OpenSTA error'
    annotation=(OUT/f'{stem}_annotation.txt').read_text()
    assert re.search(r'unannotated\s+0\b',annotation), stem+' incomplete annotation'
    power=json.loads((OUT/f'{stem}_power.json').read_text())
    rec.update(power_W=power, clock_MHz=15, tile_latency_us=rec['tile_cycles']/15,
               energy_uJ=power['Total']['total']*rec['tile_cycles']/15,
               activity_annotation=annotation,
               sta_binary_sha256=hashlib.sha256(STA.read_bytes()).hexdigest(),
               liberty_sha256=hashlib.sha256((ROOT/'third_party/nangate45/NangateOpenCellLibrary_typical.lib').read_bytes()).hexdigest(),
               sta_command=cmd,sta_seconds=time.monotonic()-t)
    (OUT/f'{stem}_result.json').write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps(dict(design=name,seed=seed,power_mW=power['Total']['total']*1000,
                         energy_uJ=rec['energy_uJ'],tile_cycles=rec['tile_cycles'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['wave','sta'])
    p.add_argument('designs',nargs='+')
    p.add_argument('--jobs',type=int,default=2)
    a=p.parse_args()
    jobs=[(n,s) for n in a.designs for s in ([4004,5005,6006] if n in FROST else [4004])]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        list(pool.map(run_wave if a.mode=='wave' else run_sta,jobs))
