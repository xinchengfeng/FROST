"""Simulate the mapped standard-cell netlist, retaining real signal activity."""
from pathlib import Path
import argparse,json,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--designs',nargs='+',default=['frost','fx16','fp32_1','fp32_4']);args=p.parse_args();BUILD=args.build.resolve();BUILD.mkdir(parents=True,exist_ok=True)
records=[]
for name in args.designs:
 netlist=ROOT/'reports/asic'/f'flat_{name}.v'
 if not netlist.exists():netlist=ROOT/'reports/asic'/f'{name}.v'
 depth='2' if netlist.name.startswith('flat_') else '99'
 # Every flattened net is traced, including underscore names. Child-cell
 # aliases duplicate those nets and are unnecessary for full pin annotation.
 cmd=['verilator','--cc','--exe','--build','-j','3','--trace','--trace-underscore','--trace-depth',depth,'-Wno-fatal','--prefix','Vdut','--top-module','dut','--output-split','24000','--output-split-cfuncs','1000','--Mdir',str(BUILD/name),'-MAKEFLAGS','VK_PCH_I_FAST= VK_PCH_I_SLOW=','-CFLAGS','-O0',str(netlist),str(ROOT/'third_party/nangate45/cells_sim.v'),str(ROOT/'source/sim_main.cpp')]
 t=time.time();log=ROOT/'reports/asic'/f'{name}_gate_build.log';tmp=log.with_suffix('.log.tmp')
 with tmp.open('w') as f:r=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 # Retry only incomplete zero-byte compilation artifacts; RTL is unchanged.
 for attempt in range(4):
  if r.returncode==0:break
  incomplete=[q for q in (BUILD/name).glob('*.o') if q.stat().st_size==0]
  if not incomplete:break
  for q in incomplete:q.unlink()
  with tmp.open('a') as f:
   f.write('\nRetrying incomplete compilation artifacts: '+','.join(q.name for q in incomplete)+'\n');f.flush()
   r=subprocess.run(['make','-C',str(BUILD/name),'-f','Vdut.mk','-j','1'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 tmp.replace(log);records.append({'design':name,'exit_code':r.returncode,'seconds':time.time()-t,'command':cmd})
 (ROOT/f'reports/asic/gate_build_{name}_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
 if r.returncode:raise RuntimeError(f'Gate netlist compilation failed: {name}')
 print(f'Built standard-cell gate simulator {name}, {time.time()-t:.1f}s',flush=True)
