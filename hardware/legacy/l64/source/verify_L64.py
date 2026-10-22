from pathlib import Path
import argparse, subprocess, numpy as np, json, time
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--golden',type=Path,required=True);p.add_argument('--designs',nargs='+',default=['retime','l64','l64c']);a=p.parse_args()
B=a.build.resolve();G=a.golden.resolve()
records=json.loads((R/'results/rtl_verification_L64.json').read_text()) if (R/'results/rtl_verification_L64.json').exists() else []
records=[x for x in records if x['design'] not in a.designs]
for seed in [4004,5005,6006]:
    stem=f'golden_L64_seed{seed}'
    subprocess.run([str(G),'inputs/scores.bin','inputs/values.bin','inputs/cdf_scaled.bin',f'inputs/seed{seed}.bin',f'results/{stem}_counts.bin',f'results/{stem}_sums.bin','3200','2','64','16',f'results/{stem}_seedout.bin'],cwd=R,check=True)
    ref=(np.fromfile(R/f'results/{stem}_sums.bin',np.int32).reshape(-1,16).astype(np.int64)+32)//64
    old=np.genfromtxt(R/f'baseline/frost_L64_seed{seed}.csv',delimiter=',',names=True)
    for name in a.designs:
        out=R/f'results/{name}_L64_seed{seed}.csv';t=time.time()
        subprocess.run([str(B/name/'Vdut'),'mem/scores.hex',f'mem/seed{seed}.hex',str(out),'64','3200'],cwd=R,check=True)
        z=np.genfromtxt(out,delimiter=',',names=True)
        words=np.stack([z[f'word{i}'] for i in range(16)],-1).astype(np.uint32).view(np.int32)
        ow=np.stack([old[f'word{i}'] for i in range(16)],-1).astype(np.uint32).view(np.int32)
        bad=np.argwhere(words!=ref)
        if len(bad): raise AssertionError((name,seed,len(bad),bad[:3].tolist(),words[tuple(bad[0])],ref[tuple(bad[0])]))
        assert np.array_equal(words,ow) and np.array_equal(z['cycles'],old['cycles'])
        records.append({'design':name,'seed':seed,'queries':len(words),'checked_words':words.size,'reference_mismatches':0,'original_kernel_mismatches':0,'cycle_mismatches':0,'tile_cycles':int(z['cycles'][:16].sum()),'seconds':time.time()-t})
        (R/'results/rtl_verification_L64.json').write_text(json.dumps(records,indent=2)+'\n')
        print(records[-1],flush=True)
