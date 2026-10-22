import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import concurrent.futures,json,time
from pathlib import Path
from study import ROOT,train,name

if __name__=='__main__':
    seeds=[11,22,33,44,55];configs=[]
    # Sort longer dimensionality first to avoid a long tail of jobs.
    dims=[(n,4,8) for n in [64,32,16,8,4]]
    dims += [(16,4,r) for r in [2,4,16,32]]
    dims += [(16,s,8) for s in [2,8]]
    for n,s,r in dims:
        configs += [dict(kind='strict',n=n,states=s,rank=r,seed=seed,steps=4000) for seed in seeds]
    configs += [dict(kind='dense',n=4,states=4,rank=0,seed=seed,steps=4000) for seed in seeds]
    configs += [dict(kind='legacy',n=16,states=4,rank=r,seed=seed,steps=4000) for r in [8,16] for seed in seeds]
    (ROOT/'training_jobs.json').write_text(json.dumps(configs,indent=2))
    p=json.loads((ROOT/'protocol.json').read_text());p['main_model_seeds']=seeds;p['training_steps']=4000
    p['planned_axes']['rank_at_n16_s4']=[2,4,8,16,32]
    (ROOT/'protocol.json').write_text(json.dumps(p,indent=2))
    jobs=[c for c in configs if not (ROOT/'checkpoints'/f'{name(c)}.pt').exists()]
    start=time.time();print(f'Start {len(jobs)} jobs, 4 workers',flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        fs={ex.submit(train,c):c for c in jobs}
        for i,f in enumerate(concurrent.futures.as_completed(fs)):
            try:f.result()
            except Exception as err:
                print('FAILED',fs[f],repr(err),flush=True);raise
            print(f'COMPLETED {i+1}/{len(jobs)} elapsed={time.time()-start:.1f}s',flush=True)
