"""Secondary follow-up: finite categorical streams at increasing dimension.

Specified after inspecting the fixed-budget analytic study. No further fitting
or checkpoint selection: base fit seed 11 and the first 128 held-out Gaussian
rows are fixed for each dimension. Eight independent stream realizations.
"""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import concurrent.futures, csv, json
import numpy as np
from attention_core import FrozenHI
from run_sampler_study import stationary, metric

ROOT=Path(__file__).resolve().parents[1]

def run(n):
    hi=FrozenHI(ROOT/'checkpoints'/f'dimension_n{n}_seed11_base.pt')
    z=np.load(ROOT/'source_data'/f'dimension_n{n}_seed11_base_gaussian.npz')['scores'][:128].astype(float)
    x,order=hi.encode(z)
    cores=[c/c.sum(-1,keepdims=True) for c in hi.cores]
    cdfs=[np.cumsum(c,-1) for c in cores]
    for c in cdfs:c[...,-1]=1
    ref=np.exp(z-z.max(-1,keepdims=True));ref/=ref.sum(-1,keepdims=True)
    analytic=stationary(cores,x,order)
    rows=[]
    for seed in [101,202,303,404,505,606,707,808]:
        drive=np.random.default_rng(seed);emit=np.random.default_rng(seed+900019)
        s=np.full(x.shape,7,dtype=np.int16)
        for t in range(256):s=np.clip(s+2*(drive.random(x.shape)<x)-1,0,15)
        counts=np.zeros(x.shape,dtype=np.int64);events=[]
        for a in range(0,4096,16):
            trace=[]
            for t in range(16):
                s=np.clip(s+2*(drive.random(x.shape)<x)-1,0,15);trace.append(s.copy())
            trace=np.array(trace).reshape(-1,n)
            bond=np.zeros(len(trace),dtype=np.int64)
            for j,c in enumerate(cdfs):
                u=emit.random((len(trace),1))
                bond=(c[trace[:,j],bond]<=u).sum(-1)
            ev=bond.reshape(16,len(x));events.append(ev.astype(np.uint8))
            for t in range(16):counts[np.arange(len(x)),ev[t]]+=1
            if a+16 in [64,256,1024,4096]:
                p=np.empty_like(x);np.put_along_axis(p,order,counts/(a+16),-1)
                rows.append({'n':n,'seed':seed,'L':a+16,**metric(p,ref),
                    'sampling_tv':metric(p,analytic)['mean_row_tv'],
                    'mass_error':float(np.abs(p.sum(-1)-1).max())})
        np.savez_compressed(ROOT/'source_data'/f'dimension_n{n}_events_seed{seed}.npz',events=np.concatenate(events))
    print(f'Dimension finite streams n={n} complete.',flush=True)
    return rows

if __name__=='__main__':
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as p:
        rows=[r for g in p.map(run,[16,32,64]) for r in g]
    with (ROOT/'source_data/dimension_stream_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    groups=[]
    for n in [16,32,64]:
        for L in [64,256,1024,4096]:
            vals=[v for v in rows if v['n']==n and v['L']==L]
            g={'n':n,'L':L}
            for k in ['mean_row_tv','probability_rmse','sampling_tv']:
                g[k+'_mean']=float(np.mean([v[k] for v in vals]));g[k+'_stream_sd']=float(np.std([v[k] for v in vals],ddof=1))
            groups.append(g)
    (ROOT/'source_data/dimension_stream_summary.json').write_text(json.dumps({'groups':groups,
        'maximum_mass_error':max(v['mass_error'] for v in rows)},indent=2))
