from pathlib import Path
import argparse, json, shutil, subprocess, numpy as np
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--optimized',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args()
W=a.work.resolve();W.mkdir(parents=True,exist_ok=True);shutil.copytree(R/'mem',W/'mem',dirs_exist_ok=True)
rng=np.random.default_rng(73021)
scores=rng.integers(-32768,32768,(64,16),dtype=np.int32)
scores[:16]=0;scores[16:32]=np.arange(16)*1024-8192
vals=rng.integers(-32768,32768,(4,16,16),dtype=np.int32)
vals[0]=-32768;vals[1]=32767;vals[2]=np.where((np.indices((16,16)).sum(0)%2)==0,-32768,32767)
for name,x in [('scores',scores),('values',vals)]:
 (W/f'mem/{name}.hex').write_text(''.join(f'{int(v)&65535:04x}\n' for v in x.flat))
np.savez_compressed(R/'inputs/corner_vectors.npz',scores=scores,values=vals)
recs=[]
for seed in [4004,5005,6006]:
 out=[]
 for name,exe in [('original',a.original),('optimized',a.optimized)]:
  target=R/f'results/corner_{name}_seed{seed}.csv'
  subprocess.run([str(exe.resolve()),'mem/scores.hex',f'mem/seed{seed}.hex',str(target.resolve()),'64','64'],cwd=W,check=True)
  z=np.genfromtxt(target,delimiter=',',names=True);ww=np.stack([z[f'word{i}'] for i in range(16)],-1).astype(np.uint32).view(np.int32);out.append((z,ww))
 assert np.array_equal(out[0][1],out[1][1]) and np.array_equal(out[0][0]['cycles'],out[1][0]['cycles'])
 assert (out[1][1][:16]==-32768).all() and (out[1][1][16:32]==32767).all()
 recs.append({'seed':seed,'checked_words':int(out[1][1].size),'mismatches':0,'extreme_value_checks':'constant signed16 minimum and maximum, alternating extrema, uniform random signed16'})
(R/'results/corner_verification.json').write_text(json.dumps(recs,indent=2)+'\n')
print(json.dumps(recs,indent=2))
