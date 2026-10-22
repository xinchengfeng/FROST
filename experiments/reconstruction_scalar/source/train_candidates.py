"""Prototype alternatives. Model selection uses disjoint calibration images only.
No reconstruction targets are loaded; checkpoint criteria are probability errors.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import sys,copy,time,json,math,argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
HERE=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
class GridTT(nn.Module):
 def __init__(self,states=32,rank=16,bound=8.,basis='local',base=None,init='identity',encoding='center'):
  super().__init__();self.states=states;self.rank=rank;self.bound=bound;self.basis=basis;self.encoding=encoding
  self.cs=nn.ParameterList()
  for j in range(16):
   a=1 if j==0 else rank;b=16 if j==15 else rank
   if base is not None:raw=base['state_dict'][f'cs.{j}'].clone()
   else:
    raw=torch.randn(a,states,b)*.15
    if init=='identity' and a==b and j>0:
     raw-=3.
     for k in range(a):raw[k,:,k]+=6.
    elif init=='identity' and j==15 and a%16==0:
     raw-=3.
     for k in range(a):raw[k,:,k%16]+=6.
   self.cs.append(nn.Parameter(raw))
 def prep(self,z):
  z,order=z.sort(-1,descending=True)
  if self.encoding=='max':x=((z-z[:,:1]+self.bound)/self.bound).clamp(0,1)
  else:
   z=z-z.mean(-1,keepdim=True);x=((z+self.bound)/(2*self.bound)).clamp(0,1)
  if self.basis=='local':
   t=x*(self.states-1);lo=t.floor().long().clamp(0,self.states-2);p=t-lo
   return (lo,p),order
  s=torch.arange(self.states,dtype=x.dtype)
  if self.basis=='geometric':
   logp=torch.log(x[...,None].clamp(1e-20))*s+torch.log((1-x)[...,None].clamp(1e-20))*(self.states-1-s)
  else:
   logc=torch.lgamma(torch.tensor(float(self.states)))-torch.lgamma(s+1)-torch.lgamma(self.states-s)
   logp=logc+torch.log(x[...,None].clamp(1e-20))*s+torch.log((1-x)[...,None].clamp(1e-20))*(self.states-1-s)
  return logp.softmax(-1),order
 def compute(self,data,order):
  h=torch.ones(len(order),1)
  for j,c in enumerate(self.cs):
   c=c.softmax(-1)
   if self.basis=='local':
    lo,p=data;c=c.permute(1,0,2)
    v=c[lo[:,j]]*(1-p[:,j,None,None])+c[lo[:,j]+1]*p[:,j,None,None]
   else:v=torch.einsum('bs,asr->bar',data[:,j],c)
   h=torch.bmm(h[:,None],v).squeeze(1)
  return torch.zeros_like(h).scatter(1,order,h)
 def forward(self,z):return self.compute(*self.prep(z))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--basis',default='local');ap.add_argument('--states',type=int,default=32);ap.add_argument('--rank',type=int,default=16);ap.add_argument('--bound',type=float,default=8.);ap.add_argument('--steps',type=int,default=2500);ap.add_argument('--resume',default='');ap.add_argument('--init',default='identity');ap.add_argument('--fresh',action='store_true');ap.add_argument('--encoding',default='center');ap.add_argument('--augment',action='store_true')
 args=ap.parse_args();torch.manual_seed(2026101703)
 data=np.load(HERE/'results/calibration_rows.npz');tr=torch.tensor(data['train']);va=torch.tensor(data['valid'])
 # Inject independently generated sub-grid noise into training to prevent memorization.
 rng=torch.Generator().manual_seed(2026101704)
 base=None
 if args.basis=='geometric' and not args.fresh:
  path=HERE/'legacy_runtime/source/original/checkpoints/softmax_N16_r8.pt'
  base=torch.load(path,weights_only=True)
 model=GridTT(args.states,args.rank,args.bound,args.basis,base,args.init,args.encoding)
 if args.resume:model.load_state_dict(torch.load(args.resume,weights_only=True)['state_dict'])
 opt=torch.optim.Adam(model.parameters(),lr=.025)
 vt=va.softmax(-1);best=float('inf');history=[];start=time.perf_counter()
 for step in range(args.steps):
  idx=torch.randint(len(tr),(256,),generator=rng);z=tr[idx]+torch.randn((256,16),generator=rng)*.015
  if args.augment:
   scales=torch.exp(torch.rand((64,1),generator=rng)*(math.log(5)-math.log(.1))+math.log(.1))
   z[-64:]=torch.randn((64,16),generator=rng)*scales
  p=model(z);target=z.softmax(-1)
  # KL + squared error optimize accurate probabilities across sharp and diffuse rows.
  loss=256*(p-target).square().mean()+.1*(target*(target.clamp_min(1e-8).log()-p.clamp_min(1e-8).log())).sum(-1).mean()
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
  opt.param_groups[0]['lr']=.025*(.1+.9*.5*(1+math.cos(math.pi*(step+1)/args.steps)))
  if (step+1)%250==0:
   with torch.no_grad():pred=torch.cat([model(v) for v in va.split(512)]);rmse=float((pred-vt).square().mean().sqrt());tv=float((pred-vt).abs().sum(-1).mean()/2)
   row={'step':step+1,'rmse':rmse,'tv':tv,'seconds':time.perf_counter()-start};history.append(row);print(args.name,json.dumps(row),flush=True)
   if rmse<best:
    best=rmse;ck={'config':dict(states=args.states,rank=args.rank,bound=args.bound,basis=args.basis,n=16,initialization=args.init,encoding=args.encoding,synthetic_augmentation=args.augment),'state_dict':copy.deepcopy(model.state_dict()),'validation':row,'selection':'independent calibration image probability RMSE','parameters':sum(p.numel() for p in model.parameters())}
    torch.save(ck,HERE/'checkpoints'/f'{args.name}.pt')
   (HERE/'results'/f'training_{args.name}.json').write_text(json.dumps(history,indent=2))
if __name__=='__main__':main()
