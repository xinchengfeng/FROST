from __future__ import annotations
import json, math, time, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from study import ROOT,load,name,logits_data,encode,measures,batched,baselines

def main():
    start=time.time();jobs=json.loads((ROOT/'training_jobs.json').read_text())
    rows=[];stress=[];inv=[];pos=[];quant=[]
    for index,c in enumerate(jobs):
        tag=name(c);m,ck=load(ROOT/'checkpoints'/f'{tag}.pt')
        n=c['n'];z=logits_data(n,8192,30000+n);p=z.softmax(-1);q=batched(m,encode(z))
        row={**c,'parameters':ck['parameters'],'best_step':ck['best_step'],**measures(p,q)}
        rows.append(row)
        if c['kind']=='strict' and n==16 and c['states']==4 and c['rank'] in [8,16,32]:
            np.savez_compressed(ROOT/'results'/f'{tag}_test_predictions.npz',logits=z.numpy(),target=p.numpy(),prediction=q.numpy())
            for family in ['sigma_0.25','sigma_0.5','normal','sigma_2','sigma_4','uniform','student_t3','one_peak','near_tie','qk_d64']:
                zs=logits_data(n,4096,50000+n,family);ps=zs.softmax(-1);qs=batched(m,encode(zs));clip=zs.clamp(-4,4).softmax(-1)
                mr=measures(ps,qs);mc=measures(clip,qs)
                stress.append({**c,'family':family,'clip_element_fraction':(zs.abs()>4).float().mean().item(),
                    'clip_row_fraction':(zs.abs()>4).any(-1).float().mean().item(),
                    'tv_vs_clipped':mc['tv'],'rmse_vs_clipped':mc['rmse'],**mr})
            rng=np.random.default_rng(60000+n);zb=torch.tensor(rng.uniform(-1,1,(4096,n)),dtype=torch.float32)
            qb=batched(m,encode(zb))
            for shift in [-2.,-1.,0.,1.,2.]:
                qa=batched(m,encode(zb+shift));inv.append({**c,'test':'shift','shift':shift,
                    'invariance_tv':.5*(qa-qb).abs().sum(-1).mean().item(),
                    'target_tv':measures(zb.softmax(-1),qa)['tv']})
            permutations=torch.tensor(np.argsort(rng.random(zb.shape),axis=1))
            qp=batched(m,encode(zb.gather(1,permutations)))
            inv.append({**c,'test':'permutation','shift':0.,'invariance_tv':.5*(qp-qb.gather(1,permutations)).abs().sum(-1).mean().item(),
                        'target_tv':measures(zb.gather(1,permutations).softmax(-1),qp)['tv']})
            for j in range(n):pos.append({**c,'position':j+1,'position_mae':(q[:,j]-p[:,j]).abs().mean().item()})
            # Offline probability-grid projection: each row has counts summing
            # exactly to 2**bits. Resolution bits are NOT an RTL memory width.
            original=m._compiled
            for bits in [4,8,12]:
                m._compiled=[]
                for core in original:
                    shape=core.shape;v=core.reshape(-1,shape[-1]).double();scaled=v*2**bits
                    counts=scaled.floor();remaining=(2**bits-counts.sum(-1)).long()
                    order=(scaled-counts).argsort(-1,descending=True)
                    extra=torch.zeros_like(counts)
                    extra.scatter_(1,order,(torch.arange(shape[-1])[None,:]<remaining[:,None]).double())
                    m._compiled.append(((counts+extra)/2**bits).reshape(shape).float())
                qq=batched(m,encode(z));quant.append({**c,'probability_resolution_bits':bits,
                    'tv_vs_float':measures(q,qq)['tv'],**measures(p,qq)})
            m._compiled=original
        if index%10==0:print('evaluated',index+1,'/',len(jobs),flush=True)
    base=[];stressbase=[]
    for n in [4,8,16,32,64]:
        z=logits_data(n,8192,30000+n);p=z.softmax(-1)
        for label,q in baselines(z).items():base.append({'method':label,'n':n,**measures(p,q)})
    for family in ['sigma_0.25','sigma_0.5','normal','sigma_2','sigma_4','uniform','student_t3','one_peak','near_tie','qk_d64']:
        zs=logits_data(16,4096,50016,family);ps=zs.softmax(-1)
        for label,qs in baselines(zs).items():stressbase.append({'method':label,'family':family,**measures(ps,qs)})
    # Re-evaluate the actual earlier checkpoint without modifying its forward path.
    spec=importlib.util.spec_from_file_location('archived',ROOT/'provenance'/'previous_model_definitions.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    ck=torch.load(ROOT/'provenance'/'frost_attention_rank12.pt',weights_only=True)
    prior=mod.TTAttentionOperator(token_count=ck['token_count'],n_states=ck['n_states'],rank=ck['rank'])
    prior.load_state_dict(ck['model_state_dict']);prior.eval()
    z=logits_data(16,8192,30016);q=batched(prior,encode(z));audit=measures(z.softmax(-1),q)
    audit.update({'description':'Actual archived checkpoint, runtime exact Softmax retained; new test distribution; not Softmax-free FROST',
                  'rank':12,'parameters':prior.parameter_count,'test_rows':8192})
    (ROOT/'results'/'prior_checkpoint_audit.json').write_text(json.dumps(audit,indent=2))
    for label,data in [('main_metrics',rows),('stress_metrics',stress),('invariance_metrics',inv),('position_metrics',pos),('core_quantization_metrics',quant),('baseline_metrics',base),('stress_baselines',stressbase)]:
        pd.DataFrame(data).to_csv(ROOT/'results'/f'{label}.csv',index=False)
    print('Evaluation complete',time.time()-start,flush=True)

if __name__=='__main__':main()
