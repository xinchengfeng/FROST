"""Unlabelled activation calibration, sharing one bounded categorical TT across blocks."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import copy, io, json, math, time, concurrent.futures
import numpy as np, torch
from torch import nn
from data import ROOT,DATASETS,load_data,patchify
from vision import load,MODEL_SEEDS
torch.set_num_threads(1)

class CalTT(nn.Module):
 def __init__(self,base,bound):
  super().__init__();self.bound=bound
  self.cs=nn.ParameterList([nn.Parameter(base['state_dict'][f'cs.{j}'].clone()) for j in range(16)])
 def forward(self,z):
  z,order=z.sort(-1,descending=True);z=z-z.mean(-1,keepdim=True)
  x=((z+self.bound)/(2*self.bound)).clamp(0,1);s=torch.arange(16,dtype=x.dtype)
  pi=x[...,None]**s*(1-x[...,None])**(15-s);pi/=pi.sum(-1,keepdim=True)
  h=torch.ones(len(z),1)
  for j,c in enumerate(self.cs):
   v=torch.einsum('bs,asr->bar',pi[:,j],c.softmax(-1));h=torch.bmm(h[:,None],v).squeeze(1)
  return torch.zeros_like(h).scatter(1,order,h)

def activation_scores(dataset,seed,split):
 m,_=load(dataset,seed);d=load_data(dataset);idx=d[split+'_indices']
 x=torch.tensor(patchify(d['images'][idx],d['mean'],d['std']));rows=[]
 with torch.no_grad():
  for a in x.split(128):
   _,_,_,tr=m(a,True);rows.extend(t['scores'].reshape(-1,16) for t in tr)
 return torch.cat(rows)

def calibrate(job):
 dataset,seed=job;selection_path=ROOT/'checkpoints'/f'calibrated_{dataset}_{seed}.pt'
 if selection_path.exists():return json.loads((ROOT/'source_data'/f'calibration_{dataset}_{seed}.json').read_text())
 start=time.time();train=activation_scores(dataset,seed,'calibration');val=activation_scores(dataset,seed,'calibration_validation')
 base=torch.load(ROOT/'checkpoints'/'softmax_N16_r8.pt',weights_only=True)
 target=train.softmax(-1);vtarget=val.softmax(-1);candidates=[];bestoverall=float('inf')
 for bound in [2,4,8]:
  torch.manual_seed(440000+seed);rng=torch.Generator().manual_seed(450000+seed)
  m=CalTT(base,bound);opt=torch.optim.Adam(m.parameters(),lr=.01);hist=[];best=float('inf')
  with torch.no_grad():initial=torch.cat([m(z) for z in val.split(1024)]).sub(vtarget).square().mean().sqrt().item()
  for step in range(1200):
   ids=torch.randint(len(train),(512,),generator=rng);loss=256*(m(train[ids])-target[ids]).square().mean()
   opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
   opt.param_groups[0]['lr']=.01*(.1+.9*.5*(1+math.cos(math.pi*(step+1)/1200)))
   if (step+1)%300==0:
    with torch.no_grad():rmse=torch.cat([m(z) for z in val.split(1024)]).sub(vtarget).square().mean().sqrt().item()
    hist.append(dict(step=step+1,validation_probability_rmse=rmse))
    if rmse<best:best=rmse;state=copy.deepcopy(m.state_dict());beststep=step+1
  record=dict(dataset=dataset,model_seed=seed,bound=bound,initial_validation_probability_rmse=initial,validation_probability_rmse=best,best_step=beststep,history=hist)
  candidates.append(record)
  if best<bestoverall:
   bestoverall=best;selected=bound
   selected_ck=dict(config={**base['config'],'bound':float(bound)},state_dict=state,dataset=dataset,model_seed=seed,best_step=beststep,validation_rmse=best)
  print('calibration',dataset,seed,bound,round(initial,5),'->',round(best,5),flush=True)
 b=io.BytesIO();torch.save(selected_ck,b);selection_path.write_bytes(b.getvalue())
 result=dict(dataset=dataset,model_seed=seed,selected_bound=selected,train_rows=len(train),validation_rows=len(val),selected_validation_probability_rmse=bestoverall,candidates=candidates,seconds=time.time()-start)
 (ROOT/'source_data'/f'calibration_{dataset}_{seed}.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
 jobs=[(d,s) for d in DATASETS for s in MODEL_SEEDS]
 with concurrent.futures.ProcessPoolExecutor(max_workers=6) as ex:rows=list(ex.map(calibrate,jobs))
 (ROOT/'source_data'/'calibration_summary.json').write_text(json.dumps(rows,indent=2))
