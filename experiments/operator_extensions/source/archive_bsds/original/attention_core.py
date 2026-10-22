"""Reproducible behavioral SC attention core; no claim of gate-level TT logic.

QK and AV contain actual generated bits and signed integer-valued counts.
HI-SMURF contains actual saturating FSM trajectories and categorical events.
Scores and attention counts are decoded at stage boundaries. Scaling, sorting,
centering and threshold comparisons are numerical. Finite-stream emissions
sample the latent indices of row-stochastic TT cores; analytic controls and
distribution checks additionally use explicit TT contractions.
"""
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import math
import numpy as np
import torch
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parent

def reference(q,k,v):
    z=q@k.swapaxes(-1,-2)/math.sqrt(q.shape[-1])
    a=np.exp(z-z.max(-1,keepdims=True));a/=a.sum(-1,keepdims=True)
    return z,a,a@v

def dataset(size=128,n=16,d=32,seed=220001,family='normal'):
    rng=np.random.default_rng(seed)
    q=rng.normal(size=(size,n,d));k=rng.normal(size=(size,n,d))
    v=rng.normal(size=(size,n,d))
    if family=='diffuse':q*=.5
    elif family=='sharp':q*=2
    elif family=='correlated':
        q=.7*k+math.sqrt(1-.7**2)*q
        v=.7*k+math.sqrt(1-.7**2)*v
    elif family.startswith('offset_'):v+=float(family.split('_')[1])
    elif family!='normal':raise ValueError(family)
    return q,k,v

class FrozenHI:
    def __init__(self,path=None):
        path=path or ROOT/'checkpoints'/'softmax_N16_r8.pt'
        ck=torch.load(path,weights_only=True,map_location='cpu')
        self.config=ck['config'];self.n=self.config['n'];self.states=self.config['states'];self.bound=self.config['bound']
        # Static trainable-logit transforms are compiled OFFLINE here.
        self.cores=[torch.softmax(ck['state_dict'][f'cs.{j}'],dim=-1).numpy().astype(np.float64) for j in range(self.n)]
        self.cores=[np.moveaxis(c,1,0).copy() for c in self.cores] # state,left,right
        self.cdfs=[np.cumsum(c,axis=-1) for c in self.cores]

    def encode(self,z,gain=1.):
        z=np.asarray(z,dtype=np.float64)*gain
        order=np.argsort(-z,axis=-1,kind='stable')
        s=np.take_along_axis(z,order,-1);s-=s.mean(-1,keepdims=True)
        x=np.clip((s+self.bound)/(2*self.bound),0.,1.)
        return x,order

    def analytic(self,z,gain=1.):
        shape=z.shape;x,order=self.encode(z.reshape(-1,self.n),gain)
        s=np.arange(self.states)
        pi=x[...,None]**s*(1-x[...,None])**(self.states-1-s)
        pi/=pi.sum(-1,keepdims=True)
        h=np.ones((len(x),1))
        for j,c in enumerate(self.cores):
            h=np.einsum('bi,bs,sik->bk',h,pi[:,j],c,optimize=True)
        a=np.empty_like(h);np.put_along_axis(a,order,h,-1)
        return a.reshape(shape)

    def selected(self,state):
        h=np.ones((len(state),1))
        for j,c in enumerate(self.cores):
            h=np.einsum('bi,bij->bj',h,c[state[:,j]],optimize=True)
        return h

    def stream(self,z,L,seed,gain=1.,burn=256,return_diagnostic=False):
        shape=z.shape;x,order=self.encode(z.reshape(-1,self.n),gain)
        # Independent IID input/emission RNG streams. QMC is NOT used for FSM
        # drives because temporal structure can change the stationary law.
        rng=np.random.default_rng(seed);emit=np.random.default_rng(seed+900019)
        state=np.full(x.shape,(self.states-1)//2,dtype=np.int16)
        for t in range(burn):
            bits=rng.random(x.shape)<x
            state=np.clip(state+2*bits-1,0,self.states-1)
        counts=np.zeros(x.shape,dtype=np.int32);weighted=np.zeros_like(x)
        row=np.arange(len(x))
        # Compile selected TT states in bounded chunks. Time order is preserved.
        chunk=16
        for start in range(0,L,chunk):
            width=min(chunk,L-start)
            trace=np.empty((width,*state.shape),dtype=np.int16)
            for t in range(width):
                bits=rng.random(x.shape)<x
                state=np.clip(state+2*bits-1,0,self.states-1);trace[t]=state
            if return_diagnostic:
                w=self.selected(trace.reshape(-1,self.n)).reshape(width,len(x),self.n)
                weighted+=w.sum(0)
            # Each core is nonnegative and row stochastic. Ancestral draws
            # through its latent bond index reproduce exactly the contracted
            # categorical output law, without forming a full probability vector.
            states=trace.reshape(-1,self.n);bond=np.zeros(len(states),dtype=np.int64)
            for j,cdf in enumerate(self.cdfs):
                probs=cdf[states[:,j],bond]
                u=emit.random((len(states),1))
                bond=np.minimum((probs<u).sum(-1),probs.shape[-1]-1)
            ix=bond.reshape(width,len(x))
            for t in range(width):counts[row,ix[t]]+=1
        a=np.empty_like(x);np.put_along_axis(a,order,counts/L,-1)
        if return_diagnostic:
            diag=np.empty_like(x);np.put_along_axis(diag,order,weighted/L,-1)
            return a.reshape(shape),diag.reshape(shape)
        return a.reshape(shape)

def scale_power2(x,axis):
    maxabs=np.max(np.abs(x),axis=axis,keepdims=True)
    return np.where(maxabs>0,2.**np.ceil(np.log2(np.maximum(maxabs,1e-30))),1.)

def uniforms(shape,L,seed,source,coordinate):
    rng=np.random.default_rng(seed)
    if source=='iid':return rng.random((*shape,L),dtype=np.float32)
    if source!='sobol':raise ValueError(source)
    # Both operands use the same 2-D scrambled net, independent digital shifts.
    # coordinate 0 and coordinate 1 must never be used interchangeably for
    # operands of one product. Each product sees a shifted 2-D Sobol net.
    net=qmc.Sobol(d=2,scramble=True,seed=7001,bits=32).random_base2(int(math.log2(L)))
    u=(net[:,coordinate]*2.**32).astype(np.uint32)
    shift=rng.integers(0,2**32,size=shape,dtype=np.uint32)
    return np.bitwise_xor(shift[...,None],u).astype(np.float64)/2.**32

def stochastic_qk(q,k,L,seed,source='iid',encoding='bipolar'):
    b,n,d=q.shape
    sq=scale_power2(q,-1);sk=scale_power2(k,-1)
    qn=q/sq;kn=k/sk
    uq=uniforms(q.shape,L,seed+101,source,0);uk=uniforms(k.shape,L,seed+211,source,1)
    if encoding=='bipolar':
        qb=(2*(uq<(qn[...,None]+1)/2).astype(np.float32)-1)
        kb=(2*(uk<(kn[...,None]+1)/2).astype(np.float32)-1)
    elif encoding=='magnitude':
        qb=(uq<np.abs(qn[...,None])).astype(np.float32)*np.sign(qn[...,None])
        kb=(uk<np.abs(kn[...,None])).astype(np.float32)*np.sign(kn[...,None])
    else:raise ValueError(encoding)
    # Summing signed bit products equals XNOR decoding (bipolar), or AND
    # followed by a fixed sign product (sign--magnitude). No real QK products.
    count=qb.reshape(b,n,d*L)@kb.reshape(b,n,d*L).swapaxes(-1,-2)
    z=count.astype(np.float64)/L*(sq*sk.swapaxes(-1,-2))/math.sqrt(d)
    return z

def stochastic_av(a,v,L,seed,source='iid',encoding='bipolar',center=False):
    b,n,d=v.shape
    mu=v.mean(1,keepdims=True) if center else np.zeros((b,1,d))
    vc=v-mu;sv=scale_power2(vc,1);vn=vc/sv
    ua=uniforms(a.shape,L,seed+307,source,0)
    uv=uniforms(v.shape,L,seed+401,source,1)
    ab=(ua<a[...,None]).astype(np.float32)
    if encoding=='bipolar':vb=2*(uv<(vn[...,None]+1)/2).astype(np.float32)-1
    elif encoding=='magnitude':vb=(uv<np.abs(vn[...,None])).astype(np.float32)*np.sign(vn[...,None])
    else:raise ValueError(encoding)
    # Flatten (key, cycle), retaining output feature axis.
    count=ab.reshape(b,n,n*L)@vb.transpose(0,1,3,2).reshape(b,n*L,d)
    return count.astype(np.float64)/L*sv+mu

def metrics(y,hat):
    diff=hat-y
    e2=np.sum(diff**2,axis=(1,2));norm=np.sum(y**2,axis=(1,2))
    dot=np.sum(y*hat,axis=(1,2));hn=np.sum(hat**2,axis=(1,2))
    per=dict(rmse=np.sqrt(np.mean(diff**2,axis=(1,2))),rel_frob=np.sqrt(e2/np.maximum(norm,1e-30)),cosine=dot/np.sqrt(np.maximum(norm*hn,1e-30)))
    aggregate=dict(rmse=float(np.sqrt(np.mean(diff**2))),rel_frob=float(np.sqrt(e2.sum()/norm.sum())),cosine=float(per['cosine'].mean()))
    return aggregate,per

def run_core(q,k,v,model,L,seed,source='iid',encoding='bipolar',center=False,gain=1.,stage='full'):
    z,a,y=reference(q,k,v)
    if stage in ('full','qk_only'):
        zh=stochastic_qk(q,k,L,seed,source,encoding)
    else:zh=z
    if stage in ('full','hi_only'):
        ah=model.stream(zh,L,seed+500009,gain=gain)
    elif stage=='qk_only':
        # Exact softmax is permitted only in this labelled isolation control.
        ah=np.exp(zh-zh.max(-1,keepdims=True));ah/=ah.sum(-1,keepdims=True)
    else:ah=a
    if stage in ('full','av_only'):
        yh=stochastic_av(ah,v,L,seed,source,encoding,center)
    else:yh=ah@v
    agg,per=metrics(y,yh)
    agg.update(score_rmse=float(np.sqrt(np.mean((zh-z)**2))),attention_tv=float(np.mean(.5*np.abs(ah-a).sum(-1))),attention_sum_error=float(np.max(np.abs(ah.sum(-1)-1))))
    return agg,per,{'y':y,'prediction':yh,'a':a,'attention':ah,'scores':z,'sc_scores':zh}
