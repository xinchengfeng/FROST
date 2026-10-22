from pathlib import Path
import argparse,subprocess,numpy as np,csv,json,time
R=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--rows',type=int,default=3200);a=p.parse_args();B=a.build.resolve();B.mkdir(parents=True,exist_ok=True)
subprocess.run(['g++','-O2','-std=c++17',str(R/'source/fused_sampler.cpp'),'-o',str(B/'golden_frost')],check=True)
subprocess.run(['g++','-O2','-std=c++17','-ffp-contract=off',str(R/'source/golden_fp32.cpp'),'-o',str(B/'golden_fp32')],check=True)
lut=np.array([int(x,16) for x in (R/'mem/exp_fp32.hex').read_text().split()],np.uint32);lut.tofile(R/'inputs/exp_fp32.bin')
subprocess.run([str(B/'golden_fp32'),'inputs/scores.bin','inputs/values.bin','inputs/exp_fp32.bin','results/golden_fp32.bin'],cwd=R,check=True)
fp=np.fromfile(R/'results/golden_fp32.bin',np.int32).reshape(-1,16)[:a.rows]
data=np.load(R/'inputs/evaluate_rows.npz');q=data['score_words'][:a.rows].astype(np.int64);v=data['value_words'][data['matrix_index'][:a.rows]].astype(np.int64)
ix=np.minimum((q.max(-1,keepdims=True)-q+8)>>4,1024);e=np.rint(np.exp(-ix/64)*65536).astype(np.int64);rec=(2**31)//e.sum(-1)
fx=(np.einsum('rk,rkd->rd',e,v)*rec[:,None]+2**30)>>31
results=[]
for name,L,seed in [('fx16',256,4004),('fp32_1',256,4004),('fp32_4',256,4004)]+[('frost',L,seed) for L in [64,256,1024] for seed in [4004,5005,6006]]:
 stem=f'{name}_L{L}_seed{seed}';path=R/'results'/f'{stem}.csv';t=time.time()
 subprocess.run([str(B/name/'Vdut'),'mem/scores.hex',f'mem/seed{seed}.hex',str(path),str(L),str(a.rows)],cwd=R,check=True)
 z=np.genfromtxt(path,delimiter=',',names=True);words=np.stack([z[f'word{i}'] for i in range(16)],-1).astype(np.uint32).view(np.int32)
 if name=='frost':
  subprocess.run([str(B/'golden_frost'),'inputs/scores.bin','inputs/values.bin','inputs/cdf_scaled.bin',f'inputs/seed{seed}.bin',f'results/{stem}_counts.bin',f'results/{stem}_sums.bin',str(a.rows),'2',str(L),'16',f'results/{stem}_seedout.bin'],cwd=R,check=True)
  sums=np.fromfile(R/'results'/f'{stem}_sums.bin',np.int32).reshape(-1,16);ref=(sums.astype(np.int64)+L//2)//L
 else:ref=fx if name=='fx16' else fp
 bad=np.argwhere(words!=ref)
 if len(bad):
  np.savez_compressed(R/'results'/f'{stem}_mismatch.npz',words=words,reference=ref)
  raise AssertionError((stem,len(bad),bad[:5].tolist(),words[tuple(bad[0])],ref[tuple(bad[0])]))
 cycles=z['cycles'].astype(np.int64);tile=cycles[:len(cycles)//16*16].reshape(-1,16).sum(-1)
 rec={'design':name,'L':L,'seed':seed,'queries':a.rows,'checked_words':int(words.size),'mismatches':0,'cold_row_cycles':int(cycles[0]),'resident_row_cycles_min':int(cycles[np.arange(len(cycles))%16!=0].min()),'resident_row_cycles_max':int(cycles[np.arange(len(cycles))%16!=0].max()),'tile16_cycles_mean':float(tile.mean()),'seconds':time.time()-t}
 results.append(rec);print(rec,flush=True)
 (R/'results/rtl_verification.json').write_text(json.dumps(results,indent=2)+'\n')
