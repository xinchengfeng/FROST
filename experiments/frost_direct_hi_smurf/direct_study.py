"""Direct multivariate FSM/TT Softmax regression: exploratory numerical study.

No piecewise interpolation, exponent approximation, sigmoid, or output Softmax.
Exact stationary-state contraction is trained. A probability-sum constraint is
linear in the state coefficient tensor. Statewise positivity is AUDITED and
must not be inferred from accurate mean predictions.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import copy,json,time,argparse,concurrent.futures
import numpy as np
import pandas as pd
import torch
from torch import nn

torch.set_num_threads(1);torch.set_num_interop_threads(1)
ROOT=Path(__file__).resolve().parent

def data(n,size,seed):
    return torch.tensor(np.random.default_rng(seed).normal(size=(size,n)),dtype=torch.float32)

def basis(x,states):
    k=torch.arange(states,dtype=x.dtype)
    v=x[...,None]**k*(1-x[...,None])**(states-1-k)
    return v/v.sum(-1,keepdim=True)

class DirectTT(nn.Module):
    def __init__(self,n=16,states=4,rank=4,heads='separate',encoding='center',bound=4.):
        super().__init__();self.n=n;self.states=states;self.rank=rank
        self.heads=heads;self.encoding=encoding;self.bound=bound
        if heads=='separate':
            first=.02*torch.randn(n,1,states,rank);first[:,0,:,0]=1.
            cs=[nn.Parameter(first)]
            for _ in range(n-2):
                cs.append(nn.Parameter(torch.eye(rank)[None,:,None,:].repeat(n,1,states,1)+.01*torch.randn(n,rank,states,rank)))
            cs.append(nn.Parameter(.01*torch.randn(n,rank,states,1)))
        else:
            first=.02*torch.randn(1,states,rank);first[:,:,0]=1.
            cs=[nn.Parameter(first)]
            for _ in range(n-2):cs.append(nn.Parameter(torch.eye(rank)[:,None,:].repeat(1,states,1)+.01*torch.randn(rank,states,rank)))
            cs.append(nn.Parameter(.01*torch.randn(rank,states,n)))
        self.cs=nn.ParameterList(cs)

    def encode(self,z):
        order=None
        if self.encoding=='sorted_center':z,order=z.sort(-1,descending=True)
        if self.encoding in ['center','sorted_center']:z=z-z.mean(-1,keepdim=True)
        x=((z+self.bound)/(2*self.bound)).clamp(0,1)
        return x,order

    def raw_from_basis(self,p):
        if self.heads=='separate':
            v=torch.ones(len(p),self.n,1)
            for m,c in enumerate(self.cs):
                a=torch.einsum('bs,hasr->bhar',p[:,m],c)
                v=torch.einsum('bha,bhar->bhr',v,a)
            return v.squeeze(-1)
        v=torch.ones(len(p),1)
        for m,c in enumerate(self.cs):
            a=torch.einsum('bs,asr->bar',p[:,m],c)
            v=torch.bmm(v[:,None],a).squeeze(1)
        return v

    def simplex_sum(self,raw):
        # Linear affine map: can be absorbed in/augmented onto the TT coefficient
        # representation. Guarantees sum_i q_i=1, NOT coefficient nonnegativity.
        return (1.+raw-raw.mean(-1,keepdim=True))/self.n

    def forward(self,z):
        x,order=self.encode(z);q=self.simplex_sum(self.raw_from_basis(basis(x,self.states)))
        if order is not None:q=torch.zeros_like(q).scatter(1,order,q)
        return q

    def selected(self,s):
        p=torch.nn.functional.one_hot(s,self.states).float()
        return self.simplex_sum(self.raw_from_basis(p))

def batch(m,z,bs=256):
    with torch.no_grad():return torch.cat([m(b) for b in z.split(bs)])

def measures(p,q):
    e=(q-p).double()
    tied=q==q.max(-1,keepdim=True).values
    credit=tied.gather(1,p.argmax(-1)[:,None]).float().squeeze(1)/tied.sum(-1)
    return dict(rmse=e.square().mean().sqrt().item(),row_l1=e.abs().sum(-1).mean().item(),
        half_row_l1=.5*e.abs().sum(-1).mean().item(),
        negative_fraction=(q<0).float().mean().item(),above_one_fraction=(q>1).float().mean().item(),
        invalid_row_fraction=((q<0).any(-1)|(q>1).any(-1)).float().mean().item(),
        sum_error=(q.sum(-1)-1).abs().max().item(),
        top1=credit.mean().item(),min_value=q.min().item(),max_value=q.max().item())

def tag(c):return f"{c['heads']}_{c['encoding']}_n{c['n']}_s{c['states']}_r{c['rank']}_b{c['bound']:g}_seed{c['seed']}"

def run(c):
    start=time.time();torch.manual_seed(c['seed'])
    m=DirectTT(**{k:c[k] for k in ['n','states','rank','heads','encoding','bound']})
    z=data(c['n'],16384,110000+c['n']);p=z.softmax(-1)
    zv=data(c['n'],2048,120000+c['n']);pv=zv.softmax(-1)
    opt=torch.optim.Adam(m.parameters(),lr=.003)
    rng=torch.Generator().manual_seed(c['seed']+900000);best=float('inf');history=[]
    for step in range(c['steps']):
        idx=torch.randint(len(z),(256,),generator=rng)
        q=m(z[idx]);loss=c['n']**2*(q-p[idx]).square().mean()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
        if step>=.6*c['steps']:
            frac=(step-.6*c['steps'])/(.4*c['steps']);opt.param_groups[0]['lr']=.003*(.1+.9*.5*(1+np.cos(np.pi*frac)))
        if (step+1)%200==0 or step+1==c['steps']:
            pred=batch(m,zv);vm=(pred-pv).square().mean().item()
            history.append({'step':step+1,'validation_mse':vm})
            if vm<best:best=vm;beststate=copy.deepcopy(m.state_dict());beststep=step+1
    m.load_state_dict(beststate)
    zt=data(c['n'],4096,130000+c['n']);pt=zt.softmax(-1);qt=batch(m,zt)
    # Sampling the joint stationary distribution, conditional on independent
    # input streams, probes actually occupied states, not just uniform codes.
    x,_=m.encode(zt);pi=basis(x,m.states)
    g=torch.Generator().manual_seed(96321)
    s=torch.multinomial(pi.reshape(-1,m.states),8,replacement=True,generator=g).reshape(len(zt),m.n,8).permute(0,2,1).reshape(-1,m.n)
    with torch.no_grad():w=torch.cat([m.selected(v) for v in s.split(256)])
    clipped=qt.clamp_min(0);clipped=clipped/clipped.sum(-1,keepdim=True)
    row={**c,'parameters':sum(v.numel() for v in m.parameters()),'best_step':beststep,
        'validation_rmse':best**.5,**measures(pt,qt),
        'state_negative_fraction':(w<0).float().mean().item(),
        'state_invalid_row_fraction':((w<0).any(-1)|(w>1).any(-1)).float().mean().item(),
        'state_min':w.min().item(),'state_max':w.max().item(),
        'projected_rmse':(clipped-pt).square().mean().sqrt().item(),
        'seconds':time.time()-start}
    ck={'config':c,'state_dict':beststate,'history':history,'metrics':row}
    torch.save(ck,ROOT/'checkpoints'/f'{tag(c)}.pt')
    (ROOT/'results'/f'{tag(c)}.json').write_text(json.dumps(row,indent=2))
    np.savez_compressed(ROOT/'results'/f'{tag(c)}_predictions.npz',logits=zt.numpy(),target=pt.numpy(),prediction=qt.numpy())
    print(json.dumps(row),flush=True);return row

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--jobs',default='pilot_jobs.json');args=parser.parse_args()
    jobs=json.loads((ROOT/args.jobs).read_text())
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as ex:rows=list(ex.map(run,jobs))
    pd.DataFrame(rows).to_csv(ROOT/'results'/args.jobs.replace('_jobs.json','_metrics.csv'),index=False)
