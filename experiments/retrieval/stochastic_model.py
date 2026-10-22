"""Full Transformer graph with explicit SC products and FSM nonlinear kernels.

Behavioral mixed-representation simulation. All main learned linear products,
QK, AV, LayerNorm variance/products and GELU gates use actual generated bits.
Centering, residual/bias accumulation, range handling, TT contraction and
static normalization-affine output scales remain numerical operations.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import math,json,io
import numpy as np,torch
from scipy.special import ndtr
from scipy.optimize import lsq_linear
from scipy.stats import qmc
from attention_core import ROOT,FrozenHI,scale_power2,uniforms,stochastic_qk,stochastic_av

def fit_scalar_tables():
    result={}
    for kind,n in [('cdf',8),('rsqrt',4)]:
        z=np.linspace(-4,4,2001) if kind=='cdf' else np.linspace(.5,1,2001)
        x=(z+4)/8 if kind=='cdf' else (z-.5)*2
        s=np.arange(n);pi=x[:,None]**s*(1-x[:,None])**(n-1-s);pi/=pi.sum(-1,keepdims=True)
        y=ndtr(z) if kind=='cdf' else 1/np.sqrt(2*z)
        w=lsq_linear(pi,y,bounds=(0,1),tol=1e-12).x
        zz=(z[:-1]+z[1:])/2;xx=(zz+4)/8 if kind=='cdf' else (zz-.5)*2
        pp=xx[:,None]**s*(1-xx[:,None])**(n-1-s);pp/=pp.sum(-1,keepdims=True)
        yy=ndtr(zz) if kind=='cdf' else 1/np.sqrt(2*zz)
        result[kind]={'states':n,'weights':w.tolist(),'grid_rmse':float(np.sqrt(np.mean((pi@w-y)**2))),
                      'midpoint_rmse':float(np.sqrt(np.mean((pp@w-yy)**2))),'domain':[float(z.min()),float(z.max())]}
    (ROOT/'checkpoints'/'scalar_tables.json').write_text(json.dumps(result,indent=2));return result

class UniformStream:
    def __init__(self,shape,L,seed,source,coordinate):
        self.shape=shape;self.rng=np.random.default_rng(seed);self.source=source
        if source=='sobol':
            self.net=(qmc.Sobol(2,scramble=True,seed=7001,bits=32).random_base2(int(math.log2(L)))[:,coordinate]*2.**32).astype(np.uint32)
            self.shift=self.rng.integers(0,2**32,size=shape,dtype=np.uint32)
    def at(self,t):
        if self.source=='iid':return self.rng.random(self.shape)
        return np.bitwise_xor(self.shift,self.net[t]).astype(np.float64)/2.**32

def linear(x,w,bias,L,seed,source='sobol',encoding='magnitude'):
    shape=x.shape;x=x.reshape(-1,x.shape[-1]);sx=scale_power2(x,-1);sw=scale_power2(w,-1)
    xn=x/sx;wn=w/sw
    ux=uniforms(x.shape,L,seed+17,source,0);uw=uniforms(w.shape,L,seed+59,source,1)
    if encoding=='bipolar':
        xb=2*(ux<(xn[...,None]+1)/2).astype(np.float32)-1
        wb=2*(uw<(wn[...,None]+1)/2).astype(np.float32)-1
    else:
        xb=(ux<np.abs(xn[...,None])).astype(np.float32)*np.sign(xn[...,None]).astype(np.float32)
        wb=(uw<np.abs(wn[...,None])).astype(np.float32)*np.sign(wn[...,None]).astype(np.float32)
    count=xb.reshape(len(x),-1)@wb.reshape(len(w),-1).T
    out=count.astype(np.float64)/L*sx*sw.T+bias
    return out.reshape(*shape[:-1],len(w))

def steady(x,table):
    n=len(table);s=np.arange(n);pi=x[...,None]**s*(1-x[...,None])**(n-1-s);pi/=pi.sum(-1,keepdims=True)
    return pi@table

def gated(x,prob_input,table,L,seed,source='sobol',burn=64):
    """FSM gate shared across broadcast dimensions; sign--magnitude AND."""
    scale=scale_power2(x,-1);norm=x/scale;sign=np.sign(norm)
    n=len(table);state=np.full(prob_input.shape,(n-1)//2,dtype=np.int16)
    rng=np.random.default_rng(seed+701)
    for _ in range(burn):state=np.clip(state+2*(rng.random(prob_input.shape)<prob_input)-1,0,n-1)
    mag=UniformStream(x.shape,L,seed+1301,source,0)
    em=UniformStream(prob_input.shape,L,seed+1709,source,1)
    count=np.zeros(x.shape,dtype=np.int32)
    for t in range(L):
        state=np.clip(state+2*(rng.random(prob_input.shape)<prob_input)-1,0,n-1)
        gate=em.at(t)<table[state]
        count+=(mag.at(t)<np.abs(norm))*gate*sign.astype(np.int32)
    return count.astype(np.float64)/L*scale

def layer_norm(x,gamma,beta,tables,L,seed,source='sobol',analytic=False):
    c=x-x.mean(-1,keepdims=True)
    if analytic:variance=(c*c).mean(-1,keepdims=True)
    else:
        scale=scale_power2(c,-1);v=np.abs(c/scale)
        u=uniforms(c.shape,L,seed+11,source,0);w=uniforms(c.shape,L,seed+29,source,1)
        squares=(u<v[...,None])&(w<v[...,None])
        variance=squares.mean(axis=(-1,-2))[...,None]*scale**2
    m,e=np.frexp(variance+1e-5)
    # 2^{-e/2}, implemented as a power-of-two rescaling plus an offline sqrt(2).
    factor=np.ldexp(np.ones_like(m),-(e//2))*np.where(e%2,1/math.sqrt(2),1.)
    p=np.clip((m-.5)*2,0,1);table=np.array(tables['rsqrt']['weights'])
    if analytic:out=c*steady(p,table)
    else:out=gated(c,p,table,L,seed+3001,source)
    return out*(math.sqrt(2)*factor)*gamma+beta

def gelu(x,tables,L,seed,source='sobol',analytic=False):
    p=np.clip((x+4)/8,0,1);table=np.array(tables['cdf']['weights'])
    if analytic:return x*steady(p,table)
    return gated(x,p,table,L,seed,source)

class SCTransformer:
    def __init__(self,teacher,hi_path=None):
        self.w={k:v.detach().numpy().astype(np.float64) for k,v in teacher.state_dict().items()}
        self.hi=FrozenHI(hi_path)
        self.tables=json.loads((ROOT/'checkpoints'/'scalar_tables.json').read_text())
    def run(self,x,L=256,seed=101,mode='full',source='sobol',encoding='magnitude'):
        w=self.w;counter=0;scope=[]
        all_linear=mode in ['full','linear_attention']
        all_nonlinear=mode in ['full','analytic']
        approximate_attention=mode in ['full','linear_attention','attention_only','analytic']
        analytic=mode=='analytic'
        def take():
            nonlocal counter
            counter+=1;return seed+counter*10009
        def lin(a,name):
            s=take();weight=w[name+'.weight'];b=w[name+'.bias']
            if all_linear:return linear(a,weight,b,L,s,source,encoding)
            return a@weight.T+b
        def norm(a,name):
            s=take()
            if all_nonlinear:return layer_norm(a,w[name+'.weight'],w[name+'.bias'],self.tables,L,s,source,analytic)
            c=a-a.mean(-1,keepdims=True);return c/np.sqrt((c*c).mean(-1,keepdims=True)+1e-5)*w[name+'.weight']+w[name+'.bias']
        x=lin(x,'embed')+w['pos'];hist=[x.copy()];updates=[]
        for layer in range(4):
            name=f'blocks.{layer}';u=norm(x,name+'.ln1');q,k,v=np.split(lin(u,name+'.qkv'),3,axis=-1)
            b,n,d=q.shape
            split=lambda a:a.reshape(b,n,2,16).transpose(0,2,1,3).reshape(b*2,n,16)
            q,k,v=map(split,[q,k,v]);s=take()
            if approximate_attention:
                z=q@k.swapaxes(-1,-2)/4 if analytic else stochastic_qk(q,k,L,s,source,encoding)
                a=self.hi.analytic(z) if analytic else self.hi.stream(z,L,s+3901)
                att=a@v if analytic else stochastic_av(a,v,L,s+5903,source,encoding,center=True)
            else:
                z=q@k.swapaxes(-1,-2)/4;a=np.exp(z-z.max(-1,keepdims=True));a/=a.sum(-1,keepdims=True);att=a@v
            att=att.reshape(b,2,n,16).transpose(0,2,1,3).reshape(b,n,d)
            r1=lin(att,name+'.out');h=x+r1;f=lin(norm(h,name+'.ln2'),name+'.fc1');s=take()
            f=gelu(f,self.tables,L,s,source,analytic) if all_nonlinear else f*ndtr(f)
            r2=lin(f,name+'.fc2');x=h+r2;hist.append(x.copy());updates.append(r1+r2)
        hidden=norm(x,'norm');logits=lin(hidden[:,-1],'head')
        return logits,hidden,hist,updates

if __name__=='__main__':print(json.dumps(fit_scalar_tables(),indent=2))
