"""Reproducible Softmax operator study. CPU reference simulation, not RTL.

The strict TT model is a NEW probability-preserving nonnegative TT variant.
State slices are row stochastic, so every contracted CPT row is a simplex.
Parameter softmax is used OFFLINE only; no runtime output softmax is required.
It is a restricted subclass of the signed TT family in the manuscript.
"""
from __future__ import annotations
import argparse, copy, hashlib, itertools, json, math, os, platform, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
DTYPE = torch.float32

def basis(x, states):
    # Polynomial form is defined at BOTH endpoints, unlike unguarded logit.
    k = torch.arange(states, device=x.device, dtype=x.dtype)
    v = x[..., None] ** k * (1-x[..., None]) ** (states-1-k)
    return v / v.sum(-1, keepdim=True)

def encode(z, bound=4.):
    return ((z+bound)/(2*bound)).clamp(0, 1)

def logits_data(n, size, seed, family='normal'):
    rng = np.random.default_rng(seed)
    if family == 'normal': z = rng.normal(size=(size,n))
    elif family.startswith('sigma_'): z = rng.normal(size=(size,n))*float(family[6:])
    elif family == 'uniform': z = rng.uniform(-math.sqrt(3),math.sqrt(3),(size,n))
    elif family == 'student_t3': z = rng.standard_t(3,(size,n))/math.sqrt(3)
    elif family == 'one_peak':
        z = rng.normal(0,.5,(size,n)); z[np.arange(size),rng.integers(n,size=size)] += 4
    elif family == 'near_tie':
        z = rng.normal(0,.2,(size,n))-2
        orders=np.argsort(rng.random((size,n)),axis=1)
        z[np.arange(size),orders[:,0]]=2
        z[np.arange(size),orders[:,1]]=1.95
    elif family == 'qk_d64':
        # Synthetic query-key rows; NOT activations from a trained Transformer.
        q=rng.normal(size=(size,64))
        k=rng.normal(size=(size,n,64))
        z=np.einsum('bi,bji->bj',q,k)/8
    else: raise ValueError(family)
    return torch.tensor(z,dtype=DTYPE)

class ProbabilityTT(nn.Module):
    def __init__(self,n,states,rank):
        super().__init__(); self.n=n;self.states=states;self.rank=rank
        dims=[1]+[rank]*(n-1)+[n]
        ps=[]
        for m in range(n):
            a=.2*torch.randn(dims[m],states,dims[m+1])
            if 0<m<n-1:
                a += 4.*torch.eye(rank)[:,None,:]
            ps.append(nn.Parameter(a))
        self.raw=nn.ParameterList(ps);self._compiled=None
    def cores(self):
        return self._compiled if self._compiled is not None else [p.softmax(-1) for p in self.raw]
    def prepare(self):
        self.eval();self._compiled=[p.detach().softmax(-1) for p in self.raw];return self
    def forward(self,x):
        p=basis(x,self.states); cs=self.cores()
        v=torch.ones((len(x),1),dtype=x.dtype,device=x.device)
        for m,c in enumerate(cs):
            a=torch.einsum('bn,anr->bar',p[:,m],c)
            v=torch.bmm(v[:,None,:],a).squeeze(1)
        return v
    def selected(self,s):
        cs=self.cores();v=cs[0][0,s[:,0],:]
        for m in range(1,self.n):
            a=cs[m][:,s[:,m],:].permute(1,0,2)
            v=torch.bmm(v[:,None,:],a).squeeze(1)
        return v

class DenseCPT(nn.Module):
    def __init__(self,n,states,rank=0):
        super().__init__();self.n=n;self.states=states;self.rank=0
        self.raw=nn.Parameter(torch.zeros(states**n,n));self._compiled=None
        self.register_buffer('codes',torch.cartesian_prod(*[torch.arange(states)]*n))
    def table(self): return self._compiled if self._compiled is not None else self.raw.softmax(-1)
    def prepare(self): self.eval();self._compiled=self.raw.detach().softmax(-1);return self
    def forward(self,x):
        p=basis(x,self.states); joint=torch.ones(len(x),len(self.codes),device=x.device)
        for m in range(self.n):joint=joint*p[:,m,self.codes[:,m]]
        return joint@self.table()
    def selected(self,s):
        idx=torch.zeros(len(s),dtype=torch.long)
        for m in range(self.n):idx=idx*self.states+s[:,m]
        return self.table()[idx]

class LegacyTT(nn.Module):
    """Same signed TT / exact runtime Softmax structure as prior code.
    Retrained on this study's shared data, not a claim to reproduce old metrics.
    """
    def __init__(self,n,states,rank):
        super().__init__();self.n=n;self.states=states;self.rank=rank
        ps=[nn.Parameter(torch.randn(1,states,rank)*.15)]
        for _ in range(n-2):
            c=torch.eye(rank)[:,None,:].repeat(1,states,1)+.02*torch.randn(rank,states,rank)
            ps.append(nn.Parameter(c))
        ps.append(nn.Parameter(torch.randn(rank,states,n)*.15))
        self.raw=nn.ParameterList(ps);self.bias=nn.Parameter(torch.zeros(n));self.temperature=nn.Parameter(torch.tensor(1.))
    def prepare(self):self.eval();return self
    def forward(self,x):
        p=basis(x.clamp(1e-5,1-1e-5),self.states)
        v=torch.ones(len(x),1)
        for m,c in enumerate(self.raw):
            a=torch.einsum('bn,anr->bar',p[:,m],c)
            v=torch.bmm(v[:,None,:],a).squeeze(1)
        return ((v+self.bias)/self.temperature.clamp(.2,5)).softmax(-1)

def create(kind,n,states,rank):return {'strict':ProbabilityTT,'dense':DenseCPT,'legacy':LegacyTT}[kind](n,states,rank)

def measures(p,q):
    p=p.double();q=q.double();e=q-p
    mid=(p+q)/2
    klterms=torch.where(p>0,p*(p.log()-q.log()),torch.zeros_like(p))
    kl=klterms.sum(-1)
    js=.5*torch.where(p>0,p*(p.log()-mid.log()),torch.zeros_like(p)).sum(-1)
    js+=.5*torch.where(q>0,q*(q.log()-mid.log()),torch.zeros_like(q)).sum(-1)
    t=p.argmax(-1)
    # Fractional credit for ties, prevents uniform predictor from accidental argmax bias.
    ties=(q==q.max(-1,keepdim=True).values)
    credit=ties.gather(1,t[:,None]).double().squeeze(1)/ties.sum(-1)
    return dict(rmse=e.square().mean().sqrt().item(),mae=e.abs().mean().item(),
        tv=(.5*e.abs().sum(-1)).mean().item(),js=js.mean().item(),
        kl=kl.mean().item(),top1=credit.mean().item(),
        row_linf_p95=torch.quantile(e.abs().max(-1).values,.95).item(),
        sum_error=(q.sum(-1)-1).abs().max().item(),min_prob=q.min().item(),
        zero_mass=(p*(q==0)).sum(-1).mean().item(),
        infinite_kl_fraction=torch.isinf(kl).double().mean().item())

def batched(model,x,bs=512):
    with torch.no_grad():return torch.cat([model(v) for v in x.split(bs)])

def name(c):return f"{c['kind']}_n{c['n']}_s{c['states']}_r{c['rank']}_seed{c['seed']}"

def train(c):
    torch.manual_seed(c['seed']);start=time.time();tag=name(c)
    model=create(c['kind'],c['n'],c['states'],c['rank'])
    z=logits_data(c['n'],32768,10000+c['n'])
    zv=logits_data(c['n'],4096,20000+c['n'])
    x=encode(z);p=z.softmax(-1);xv=encode(zv);pv=zv.softmax(-1)
    opt=torch.optim.Adam(model.parameters(),lr=.03 if c['kind'] in ['strict','dense'] else .003)
    gen=torch.Generator().manual_seed(c['seed']+700000)
    best=float('inf');beststate=None;hist=[]
    for step in range(c['steps']):
        idx=torch.randint(len(z),(256,),generator=gen)
        pred=model(x[idx]);loss=-(p[idx]*pred.clamp_min(1e-12).log()).sum(-1).mean()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if step>=int(.6*c['steps']):
            frac=(step-.6*c['steps'])/(.4*c['steps'])
            opt.param_groups[0]['lr']=(.03 if c['kind'] in ['strict','dense'] else .003)*(.1+.9*.5*(1+math.cos(math.pi*frac)))
        if (step+1)%100==0 or step==0 or step+1==c['steps']:
            with torch.no_grad():
                predv=batched(model,xv);kv=(pv*(pv.log()-predv.clamp_min(1e-12).log())).sum(-1).mean().item()
            hist.append(dict(step=step+1,train_ce=loss.item(),val_kl=kv))
            if kv<best:best=kv;beststate=copy.deepcopy(model.state_dict());beststep=step+1
    model.load_state_dict(beststate)
    saved=dict(config=c,state_dict=beststate,best_val_kl=best,best_step=beststep,
               parameters=sum(p.numel() for p in model.parameters()),seconds=time.time()-start,history=hist)
    torch.save(saved,ROOT/'checkpoints'/f'{tag}.pt')
    out={k:v for k,v in saved.items() if k not in ['state_dict','history']}
    (ROOT/'results'/f'{tag}_train.json').write_text(json.dumps(out,indent=2))
    (ROOT/'results'/f'{tag}_history.json').write_text(json.dumps(hist))
    print(json.dumps(out),flush=True);return out

def load(path):
    ck=torch.load(path,weights_only=True,map_location='cpu');c=ck['config']
    m=create(c['kind'],c['n'],c['states'],c['rank']);m.load_state_dict(ck['state_dict']);m.prepare();return m,ck

@torch.no_grad()
def simulate(model,x,lengths,repeats=4,seed=400001,init='stationary',burn=0,rho=0):
    """Actual saturating FSM trajectories + exact floating TT lookup + categorical emission.
    No gate-level simulation of TT multiplication, RNG quantization, or sampler circuitry.
    Rao-Blackwellized output averages CPT probability vectors, not emitted bits.
    """
    b,n=x.shape;x=x.repeat(repeats,1);size=len(x);g=torch.Generator().manual_seed(seed)
    if init=='stationary':
        pi=basis(x,model.states)
        s=(torch.rand((size,n,1),generator=g)>pi.cumsum(-1)).sum(-1).clamp_max(model.states-1)
    else:s=torch.full((size,n),(model.states-1)//2,dtype=torch.long)
    rb=torch.zeros(size,n);counts=torch.zeros(size,n);out={}
    for t in range(burn+max(lengths)):
        if rho:
            share=torch.rand((size,1),generator=g)<rho
            uni=torch.where(share,torch.rand((size,1),generator=g).expand(size,n),torch.rand((size,n),generator=g))
        else:uni=torch.rand((size,n),generator=g)
        s=(s+torch.where(uni<x,1,-1)).clamp(0,model.states-1)
        if t>=burn:
            w=model.selected(s)
            # floating roundoff only, no fitted-output Softmax or normalization.
            j=(torch.rand((size,1),generator=g)>w.cumsum(-1)).sum(-1).clamp_max(n-1)
            rb+=w;counts.scatter_add_(1,j[:,None],torch.ones(size,1))
            length=t-burn+1
            if length in lengths:
                out[length]={'rb':(rb/length).reshape(repeats,b,n).clone(),
                             'bits':(counts/length).reshape(repeats,b,n).clone()}
    return out

def baselines(z):
    p=z.softmax(-1);x=z-z.max(-1,keepdim=True).values
    out={'uniform':torch.ones_like(z)/z.shape[-1],
         'encoding_clipped_oracle':z.clamp(-4,4).softmax(-1)}
    for bits in [4,8]:
        levels=2**bits-1;zq=(encode(z)*levels).round()/levels*8-4
        out[f'logit_{bits}bit_exact_softmax']=zq.softmax(-1)
    # Equation 13-14 from I-BERT, evaluated in floating point; NOT integer implementation.
    ln2=math.log(2);k=torch.floor(-x/ln2);rem=x+k*ln2
    exp_approx=(.3585*(rem+1.353)**2+.344)*torch.pow(2.,-k)
    out['ibert_polynomial_float']=exp_approx/exp_approx.sum(-1,keepdim=True)
    for segments in [16,64]:
        grid=torch.linspace(-16,0,segments+1);vals=grid.exp()
        idx=((x.clamp(-16,0)+16)/16*segments).clamp(0,segments)
        i=idx.floor().long().clamp_max(segments-1);f=idx-i
        a=vals[i]*(1-f)+vals[i+1]*f
        a=torch.where(x < -16,torch.zeros_like(a),a)
        out[f'exp_lut_{segments}_float']=a/a.sum(-1,keepdim=True)
    return out

def validate():
    torch.manual_seed(17);model=ProbabilityTT(3,3,4).double()
    x=torch.rand(5,3,dtype=torch.double)
    codes=torch.cartesian_prod(*[torch.arange(3)]*3)
    pi=basis(x,3);joint=torch.ones(5,len(codes),dtype=torch.double)
    for m in range(3):joint*=pi[:,m,codes[:,m]]
    # selected uses model core dtype; analytical v dtype follows x.
    table=model.selected(codes);exact=joint@table;pred=model(x)
    gap=(exact-pred).abs().max().item()
    assert gap<1e-12
    assert (table.sum(-1)-1).abs().max()<1e-12 and table.min()>0
    testx=torch.tensor([[0.,.5,1.]])
    assert torch.allclose(basis(testx,4),torch.tensor([[[1.,0,0,0],[.25]*4,[0,0,0,1.]]]))
    loss=(pred*torch.arange(3,dtype=torch.double)).sum();loss.backward()
    idx=(0,0,0);ag=model.raw[0].grad[idx].item();eps=1e-5
    with torch.no_grad():
        old=model.raw[0][idx].item();model.raw[0][idx]=old+eps
        fp=(model(x)*torch.arange(3,dtype=torch.double)).sum().item()
        model.raw[0][idx]=old-eps
        fm=(model(x)*torch.arange(3,dtype=torch.double)).sum().item();model.raw[0][idx]=old
    assert abs(ag-(fp-fm)/(2*eps))<1e-8
    model=model.float().prepare();xs=torch.tensor([[.2,.4,.7],[.1,.5,.9]])
    stream=simulate(model,xs,[4096],repeats=12,seed=99)[4096]
    freq=stream['bits'].mean(0);target=model(xs)
    assert (freq-target).abs().max()<.02
    assert (stream['bits'].sum(-1)-1).abs().max()==0
    # Empirical check of full saturation-chain stationary law.
    res=dict(enumeration_max_error=gap,gradient_error=abs(ag-(fp-fm)/(2*eps)),
             simulation_mean_max_error=(freq-target).abs().max().item(),
             categorical_sum_error=(stream['bits'].sum(-1)-1).abs().max().item())
    (ROOT/'results'/'validation.json').write_text(json.dumps(res,indent=2));print(res)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['validate','train'])
    ap.add_argument('--kind',default='strict');ap.add_argument('--n',type=int,default=16)
    ap.add_argument('--states',type=int,default=4);ap.add_argument('--rank',type=int,default=8)
    ap.add_argument('--seed',type=int,default=11);ap.add_argument('--steps',type=int,default=2000)
    a=vars(ap.parse_args());mode=a.pop('mode')
    if mode=='validate':validate()
    else:train(a)
