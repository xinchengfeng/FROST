"""Exploratory diagnostic added after main training: unrestricted signed TT.
Fits probabilities with MSE; measures statewise probability validity separately.
Not a valid categorical-emission model unless state constraints also hold.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import copy,json,time,concurrent.futures
import numpy as np,pandas as pd,torch
from torch import nn
from study import ROOT,basis,encode,logits_data,batched

class SignedTT(nn.Module):
 def __init__(self,n=16,states=4,rank=16):
  super().__init__();self.n=n;self.states=states;self.rank=rank
  first=.03*torch.randn(1,states,rank);first[:,:,0]=1
  cs=[nn.Parameter(first)]
  for m in range(n-2):cs.append(nn.Parameter(torch.eye(rank)[:,None,:].repeat(1,states,1)+.01*torch.randn(rank,states,rank)))
  last=.03*torch.randn(rank,states,n);last[0,:,:]=1/n;cs.append(nn.Parameter(last));self.cs=nn.ParameterList(cs)
 def forward(self,x):
  p=basis(x,self.states);v=torch.ones(len(x),1)
  for m,c in enumerate(self.cs):v=torch.bmm(v[:,None,:],torch.einsum('bn,anr->bar',p[:,m],c)).squeeze(1)
  return v
 def selected(self,s):
  v=self.cs[0][0,s[:,0],:]
  for m in range(1,self.n):v=torch.bmm(v[:,None,:],self.cs[m][:,s[:,m],:].permute(1,0,2)).squeeze(1)
  return v

def run(job):
 r,seed=job;torch.manual_seed(seed);model=SignedTT(rank=r);opt=torch.optim.Adam(model.parameters(),lr=.002)
 z=logits_data(16,32768,10016);p=z.softmax(-1);x=encode(z)
 zv=logits_data(16,4096,20016);pv=zv.softmax(-1);xv=encode(zv)
 gen=torch.Generator().manual_seed(seed+700000);best=float('inf');hist=[];start=time.time()
 for step in range(4000):
  idx=torch.randint(len(z),(256,),generator=gen);q=model(x[idx]);loss=256*(q-p[idx]).square().mean()
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
  if step>=2400:opt.param_groups[0]['lr']=.002*(.1+.9*.5*(1+np.cos(np.pi*(step-2400)/1600)))
  if (step+1)%100==0:
   with torch.no_grad():v=(batched(model,xv)-pv).square().mean().item()
   hist.append({'step':step+1,'val_mse':v})
   if v<best:best=v;bs=copy.deepcopy(model.state_dict());beststep=step+1
 model.load_state_dict(bs)
 torch.save({'rank':r,'seed':seed,'state_dict':bs,'history':hist},ROOT/'checkpoints'/f'signed_diagnostic_r{r}_seed{seed}.pt')
 zt=logits_data(16,8192,30016);pt=zt.softmax(-1);q=batched(model,encode(zt))
 # Sample 65536 uniformly chosen state codewords; any violation disproves global validity.
 g=torch.Generator().manual_seed(81741)
 code=torch.randint(4,(65536,16),generator=g)
 with torch.no_grad():w=torch.cat([model.selected(s) for s in code.split(512)])
 row={'rank':r,'seed':seed,'best_step':beststep,'parameters':sum(v.numel() for v in model.parameters()),
 'test_rmse':(q-pt).square().mean().sqrt().item(),
 'test_mean_row_l1':(q-pt).abs().sum(-1).mean().item(),
 'test_negative_fraction':(q<0).float().mean().item(),
 'test_mean_abs_sum_error':(q.sum(-1)-1).abs().mean().item(),
 'state_sample_negative_fraction':(w<0).float().mean().item(),
 'state_sample_invalid_row_fraction':((w<0).any(-1)|(w>1).any(-1)|((w.sum(-1)-1).abs()>1e-4)).float().mean().item(),
 'state_sample_min':w.min().item(),'state_sample_max':w.max().item(),
 'state_sample_mean_abs_sum_error':(w.sum(-1)-1).abs().mean().item(),'seconds':time.time()-start}
 print(json.dumps(row),flush=True);return row

if __name__=='__main__':
 jobs=[(r,s) for r in [16,32] for s in [11,22,33,44,55]]
 with concurrent.futures.ProcessPoolExecutor(max_workers=3) as ex:rows=list(ex.map(run,jobs))
 pd.DataFrame(rows).to_csv(ROOT/'results'/'signed_tt_diagnostic.csv',index=False)
