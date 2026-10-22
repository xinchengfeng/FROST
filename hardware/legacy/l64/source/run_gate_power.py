"""Obtain real gate activities and OpenSTA cell-power estimates.

Power is from Nangate45 characterization and mapped zero-delay activities;
not from FPGA measurements, CPU timing, or guessed wattage.
"""
from pathlib import Path
import argparse,csv,gzip,json,shutil,subprocess,time,re
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--gate-build',type=Path,required=True);p.add_argument('--rtl-build',type=Path,required=True);p.add_argument('--sta',required=True);p.add_argument('--designs',nargs='+',default=['frost','fx16','fp32_1','fp32_4']);p.add_argument('--lengths',type=int,nargs='+',default=[256]);args=p.parse_args()
GATE=args.gate_build.resolve();RTL=args.rtl_build.resolve();OUT=ROOT/'reports/asic';OUT.mkdir(exist_ok=True)
def read(path):
 a=np.genfromtxt(path,delimiter=',',names=True);assert len(a)==16
 return a,np.stack([a[f'word{i}'] for i in range(16)],-1).astype(np.uint32)
manifest=OUT/('power_'+'_'.join(args.designs)+'_manifest.json')
records=json.loads(manifest.read_text()) if manifest.exists() else []
for name in args.designs:
 for seed,L in [(s,L) for s in ([4004,5005,6006] if name in ['frost','retime','l64','l64c'] else [4004]) for L in (args.lengths if name in ['frost','retime','l64','l64c'] else [256])]:
  stem=f'{name}_L{L}_seed{seed}';gatecsv=ROOT/'results'/f'gate_power_{stem}.csv';rtlcsv=ROOT/'results'/f'rtl_power_{stem}.csv';wave=OUT/f'{stem}.vcd.gz';temp=wave.with_suffix('.gz.tmp');t=time.time()
  cmd=[str(GATE/name/'Vdut'),'mem/scores.hex',f'mem/seed{seed}.hex',str(gatecsv),str(L),'16','/dev/stdout']
  with (OUT/f'{stem}_simulation.log').open('w') as log:
   proc=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=log)
   with gzip.open(temp,'wb',compresslevel=1) as f:shutil.copyfileobj(proc.stdout,f,1024*1024)
   code=proc.wait()
  if code:raise RuntimeError(f'Gate simulation failed: {stem}')
  temp.replace(wave)
  subprocess.run([str(RTL/name/'Vdut'),'mem/scores.hex',f'mem/seed{seed}.hex',str(rtlcsv),str(L),'16'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  a,w=read(gatecsv);b,q=read(rtlcsv)
  if not np.array_equal(w,q) or not np.array_equal(a['cycles'],b['cycles']):raise AssertionError('Gate/RTL mismatch '+stem)
  # OpenSTA 3.1's VCD begin/end API takes integer ticks in the VCD timescale,
  # rather than Liberty nanoseconds. The recorded VCD timescale is 1 ps.
  begin_ps=35*200000//3
  end_ps=(35+int(a['cycles'].sum()))*200000//3
  with gzip.open(wave,'rt') as f:
   header=[]
   for line in f:
    header.append(line)
    if '$enddefinitions' in line:break
  assert re.search(r'\$timescale\s+1ps\s+\$end',''.join(header)), 'Unexpected VCD time units'
  netlist=f'reports/asic/flat_{name}.v' if (OUT/f'flat_{name}.v').exists() else f'reports/asic/{name}.v'
  script=f'''read_liberty third_party/nangate45/NangateOpenCellLibrary_typical.lib
read_verilog {netlist}
link_design dut
create_clock -name clk -period 66.6666666666667 [get_ports clk]
set_input_transition 0.05 [all_inputs]
read_vcd -scope TOP/dut -begin_time {begin_ps} -end_time {end_ps} reports/asic/{stem}.vcd.gz
report_units
report_activity_annotation > reports/asic/{stem}_annotation.txt
report_power -format json -digits 10 > reports/asic/{stem}_power.json
report_checks -path_delay max -group_path_count 1 > reports/asic/{stem}_timing.txt
exit
'''
  tcl=OUT/f'{stem}.tcl';tcl.write_text(script)
  with (OUT/f'{stem}_sta.log').open('w') as f:r=subprocess.run([args.sta,'-no_splash','-no_init','-exit',str(tcl)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  if r.returncode:raise RuntimeError('OpenSTA failed '+stem)
  if not (OUT/f'{stem}_power.json').exists():raise RuntimeError('Missing power report '+stem)
  if 'Error:' in (OUT/f'{stem}_sta.log').read_text():raise RuntimeError('OpenSTA command error '+stem)
  power=json.loads((OUT/f'{stem}_power.json').read_text())
  annotation=(OUT/f'{stem}_annotation.txt').read_text()
  if not re.search(r'unannotated\s+0\b',annotation):raise RuntimeError('Incomplete activity annotation '+stem)
  records=[r for r in records if not (r['design']==name and r['seed']==seed and r['L']==(L if name in ['frost','retime','l64','l64c'] else 0))]
  records.append({'design':name,'L':L if name in ['frost','retime','l64','l64c'] else 0,'seed':seed,'checked_words':int(w.size),'gate_rtl_mismatches':0,'mean_cycles':float(a['cycles'].mean()),'rows':16,'frequency_MHz':15,'waveform_compressed_bytes':wave.stat().st_size,'power_report':power,'seconds':time.time()-t})
  (OUT/'power_run_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
  print(f'Gate outputs verified and power estimated: {stem}, {time.time()-t:.1f}s, {wave.stat().st_size/1e6:.1f} MB compressed',flush=True)
