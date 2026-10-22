"""Map the SAME kernels into a characterized standard-cell library.

This is a secondary pre-layout, fixed-coefficient ASIC assessment. ROMs are
implemented as combinational standard-cell networks, not as SRAM macros.
Its power must never be relabelled FPGA power or board power.
"""
from pathlib import Path
import argparse,json,subprocess,time,hashlib
ROOT=Path(__file__).resolve().parents[1]
LIB='third_party/nangate45/NangateOpenCellLibrary_typical.lib'
p=argparse.ArgumentParser();p.add_argument('--designs',nargs='+',default=['frost','fx16','fp32_1','fp32_4']);args=p.parse_args()
OUT=ROOT/'reports/asic';OUT.mkdir(parents=True,exist_ok=True)
records=[]
for name in args.designs:
 sources=[f'rtl/top_{name}.v'];inc='-Irtl '
 if name in ['frost','retime','l64','l64c']:sources+=['rtl/frost_frontend.v','rtl/frost_'+({'frost':'fused'}.get(name,name))+'.v']
 elif name=='fx16':sources+=['rtl/fx16_fused.v']
 else:
  hf='third_party/HardFloat-1/source';inc+=f'-I{hf} -I{hf}/RISCV '
  sources+=['rtl/fp32_fused.v']+[hf+'/'+n for n in ['HardFloat_primitives.v','HardFloat_rawFN.v','isSigNaNRecFN.v','fNToRecFN.v','recFNToFN.v','addRecFN.v','divSqrtRecFN_small.v','mulAddRecFN.v','iNToRecFN.v','recFNToIN.v','RISCV/HardFloat_specialize.v']]
 script=f'read_verilog {inc}'+ ' '.join(sources)+f'; hierarchy -check -top dut; synth -top dut -flatten -noabc; dfflibmap -liberty {LIB}; abc -liberty {LIB}; clean; write_verilog -noattr -noexpr reports/asic/{name}.v; tee -o reports/asic/{name}_stat.json stat -liberty {LIB} -json'
 (OUT/f'{name}.ys').write_text(script+'\n');t=time.time()
 log=OUT/f'{name}_synthesis.log';tmp=log.with_suffix('.log.tmp')
 with tmp.open('w') as f:r=subprocess.run(['yosys','-Q','-s',str(OUT/f'{name}.ys')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 tmp.replace(log)
 records.append({'design':name,'exit_code':r.returncode,'seconds':time.time()-t,'library_sha256':hashlib.sha256((ROOT/LIB).read_bytes()).hexdigest(),'source_sha256':{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources}})
 (OUT/f'synthesis_{name}_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
 if r.returncode:raise RuntimeError(f'ASIC synthesis failed: {name}')
 print(f'Mapped {name} to Nangate45 in {time.time()-t:.1f}s',flush=True)
