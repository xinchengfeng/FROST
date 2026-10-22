from __future__ import annotations
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import concurrent.futures,json,time,math
import numpy as np
import pandas as pd
import torch
from study import ROOT,load,name,logits_data,encode,measures,simulate

def one(job):
    rank,seed,init,burn,rho,group=job
    c=dict(kind='strict',n=16,states=4,rank=rank,seed=seed,steps=4000)
    model,ck=load(ROOT/'checkpoints'/f'{name(c)}.pt')
    if group=='main':size=128;rep=8;lengths=[16,32,64,128,256,512,1024,2048]
    else:size=64;rep=4;lengths=[256,1024,4096]
    z=logits_data(16,size,40016);p=z.softmax(-1);x=encode(z);q=model(x)
    start=time.time();out=simulate(model,x,lengths,rep,seed=900000+seed,init=init,burn=burn,rho=rho)
    rows=[]
    for L,estimates in out.items():
        for estimator,arr in estimates.items():
            for repeat in range(rep):
                a=arr[repeat];mt=measures(p,a);mn=measures(q,a)
                mse_approx=(q-p).square().mean().item();cross=(2*(q-p)*(a-q)).mean().item()
                rows.append(dict(rank=rank,seed=seed,init=init,burn=burn,rng_share_probability=rho,group=group,
                    length=L,total_cycles=L+burn,estimator=estimator,repeat=repeat,
                    rmse_to_analytic=mn['rmse'],tv_to_analytic=mn['tv'],
                    mse_approx=mse_approx,cross_term=cross,**mt))
    tag=f'r{rank}_seed{seed}_{init}_burn{burn}_rho{rho}_{group}'
    pd.DataFrame(rows).to_csv(ROOT/'results'/f'stream_{tag}.csv',index=False)
    if group=='main' and rank==8 and init=='center' and burn==64:
        saved={f'{est}_L{L}':v.numpy() for L,a in out.items() for est,v in a.items()}
        np.savez_compressed(ROOT/'results'/f'stream_{tag}_predictions.npz',target=p.detach().numpy(),analytic=q.detach().numpy(),**saved)
    print('STREAM',tag,round(time.time()-start,2),'s',flush=True)

def oracle():
    z=logits_data(16,128,40016);p=z.softmax(-1)
    gen=torch.Generator().manual_seed(80917)
    # IID categorical samples from exact Softmax, an ideal sampling reference.
    arr=torch.multinomial(p.repeat(8,1),2048,replacement=True,generator=gen)
    rows=[]
    for L in [16,32,64,128,256,512,1024,2048]:
        counts=torch.zeros(8*128,16);counts.scatter_add_(1,arr[:,:L],torch.ones(8*128,L))
        freq=(counts/L).reshape(8,128,16)
        for rep in range(8):rows.append(dict(length=L,repeat=rep,
            theoretical_rmse=math.sqrt((1-p.square().sum(-1).mean().item())/(16*L)),**measures(p,freq[rep])))
    pd.DataFrame(rows).to_csv(ROOT/'results'/'ideal_categorical_oracle.csv',index=False)

if __name__=='__main__':
    import math
    seeds=[11,22,33,44,55]
    jobs=[(8,s,init,burn,0,'main') for s in seeds for init,burn in [('center',0),('center',64),('stationary',0)]]
    jobs += [(16,s,'center',64,0,'main') for s in seeds]
    jobs += [(8,s,'center',256,rho,'correlation') for s in [11,22,33] for rho in [0,.25,.5,1.]]
    (ROOT/'stream_jobs.json').write_text(json.dumps(jobs,indent=2))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        fs=[ex.submit(one,j) for j in jobs]
        for f in concurrent.futures.as_completed(fs):f.result()
    oracle()
    frames=[pd.read_csv(p) for p in (ROOT/'results').glob('stream_r*.csv')]
    pd.concat(frames,ignore_index=True).to_csv(ROOT/'results'/'stream_metrics.csv',index=False)
