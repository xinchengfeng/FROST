from pathlib import Path
import argparse,subprocess,time,json,os,concurrent.futures,hashlib
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('mode',choices=['build','synth','route']);p.add_argument('--designs',nargs='+',default=['frost','fx16','fp32_1','fp32_4']);p.add_argument('--build',type=Path,default=ROOT/'build');p.add_argument('--seeds',type=int,nargs='+',default=[1,2,3]);p.add_argument('--freq',type=float,default=8);p.add_argument('--workers',type=int,default=2);a=p.parse_args();a.build=a.build.resolve();a.build.mkdir(parents=True,exist_ok=True)
def sources(n):
 src=[f'rtl/top_{n}.v'];inc=['rtl']
 if n=='frost':src+=['rtl/frost_frontend.v','rtl/frost_fused.v']
 elif n=='fx16':src+=['rtl/fx16_fused.v']
 else:
  hf='third_party/HardFloat-1/source';inc+=[hf,hf+'/RISCV'];src+=['rtl/fp32_fused.v']+[hf+'/'+f for f in ['HardFloat_primitives.v','HardFloat_rawFN.v','isSigNaNRecFN.v','fNToRecFN.v','recFNToFN.v','addRecFN.v','divSqrtRecFN_small.v','mulAddRecFN.v','iNToRecFN.v','recFNToIN.v','RISCV/HardFloat_specialize.v']]
 return src,inc
def run(cmd,log):
 t=time.time()
 with (ROOT/'reports'/log).open('w') as f:r=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(log)
 return round(time.time()-t,2)
def job(n):
 src,inc=sources(n);record={'design':n,'sources':{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in src}}
 if a.mode=='build':
  cmd=['verilator','--cc','--exe','--build','-j','3','--trace','-Wno-fatal','--prefix','Vdut','--top-module','dut','--Mdir',str(a.build/n),'-CFLAGS','-O2']+['-I'+str(ROOT/i) for i in inc]+[str(ROOT/s) for s in src]+[str(ROOT/'source/sim_main.cpp')]
  record['seconds']=run(cmd,f'build_{n}.log');record['command']=cmd
 elif a.mode=='synth':
  script='read_verilog '+' '.join('-I'+i for i in inc)+' '+' '.join(src)+f'; synth_ecp5 -top dut -json reports/fpga_{n}.json; tee -o reports/stat_{n}.json stat -json'
  (ROOT/'reports'/f'{n}.ys').write_text(script+'\n');record['seconds']=run(['yosys','-Q','-s',f'reports/{n}.ys'],f'synth_{n}.log')
 else:
  record['runs']=[]
  for seed in a.seeds:
   stem=f'{n}_f{a.freq:g}_s{seed}'
   cmd=['nextpnr-ecp5','--85k','--package','CABGA381','--speed','6','--json',f'reports/fpga_{n}.json','--freq',str(a.freq),'--seed',str(seed),'--lpf-allow-unconstrained','--report',f'reports/route_{stem}.json','--textcfg',f'reports/route_{stem}.config']
   seconds=run(cmd,f'route_{stem}.log');record['runs'].append({'seed':seed,'frequency_MHz':a.freq,'command':cmd,'seconds':seconds});print('Routed',stem,seconds,flush=True)
 (ROOT/'reports'/f'{a.mode}_{n}_manifest.json').write_text(json.dumps(record,indent=2)+'\n');print('Completed',a.mode,n,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:list(pool.map(job,a.designs))
