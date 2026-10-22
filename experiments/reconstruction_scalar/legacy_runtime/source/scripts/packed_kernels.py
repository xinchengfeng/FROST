"""Memory-bounded exact bit packing, preserving original Sobol digital shifts.

AND/popcount implements the original bit products, not their expectation.
Chunking only changes materialization, not seeds or bit values.
"""
import math
import numpy as np
from scipy.stats import qmc
import stochastic_model as sm
from attention_core import scale_power2
REFERENCE_LAYER_NORM=sm.layer_norm

def packed(prob,L,seed,source,coordinate):
    shape=prob.shape;flat=prob.reshape(-1,shape[-1]);rng=np.random.default_rng(seed)
    result=np.empty((len(flat),flat.shape[-1],L//64),dtype=np.uint64)
    if source=='sobol':
        net=qmc.Sobol(2,scramble=True,seed=7001,bits=32).random_base2(int(math.log2(L)))
        net=(net[:,coordinate]*2.**32).astype(np.uint32)
        shift=rng.integers(0,2**32,size=shape,dtype=np.uint32).reshape(flat.shape)
        # For integer u, u/2^32 < p iff u < ceil(p*2^32).
        threshold=np.ceil(flat*2.**32).clip(0,2**32).astype(np.uint64)
    for start in range(0,len(flat),8):
        stop=min(start+8,len(flat))
        if source=='sobol':bits=np.bitwise_xor(shift[start:stop,:,None],net)<threshold[start:stop,:,None]
        elif source=='iid':bits=rng.random((stop-start,flat.shape[-1],L),dtype=np.float32)<flat[start:stop,:,None]
        else:raise ValueError(source)
        result[start:stop]=np.packbits(bits,axis=-1,bitorder='little').view(np.uint64)
    return result.reshape(*shape,L//64)

def linear(x,w,bias,L,seed,source='sobol',encoding='magnitude'):
    if encoding!='magnitude':raise ValueError('Packed runner uses sign-magnitude encoding.')
    shape=x.shape;x=x.reshape(-1,x.shape[-1]);sx=scale_power2(x,-1);sw=scale_power2(w,-1)
    xn=x/sx;wn=w/sw;xb=packed(np.abs(xn),L,seed+17,source,0);wb=packed(np.abs(wn),L,seed+59,source,1)
    xs=np.sign(xn).astype(np.int32);ws=np.sign(wn).astype(np.int32);count=np.zeros((len(x),len(w)),dtype=np.int32)
    for start in range(0,len(w),16):
        stop=min(start+16,len(w));p=np.bitwise_count(xb[:,None]&wb[None,start:stop]).sum(-1,dtype=np.int32)
        count[:,start:stop]=(p*xs[:,None]*ws[None,start:stop]).sum(-1,dtype=np.int32)
    y=count.astype(np.float64)/L*sx*sw.T+bias
    return y.reshape(*shape[:-1],len(w))

def stochastic_qk(q,k,L,seed,source='sobol',encoding='magnitude'):
    if encoding!='magnitude':raise ValueError(encoding)
    b,n,d=q.shape;sq=scale_power2(q,-1);sk=scale_power2(k,-1);qn=q/sq;kn=k/sk
    qb=packed(np.abs(qn),L,seed+101,source,0);kb=packed(np.abs(kn),L,seed+211,source,1)
    qs=np.sign(qn).astype(np.int32);ks=np.sign(kn).astype(np.int32);count=np.empty((b,n,n),dtype=np.int32)
    for i in range(n):
        p=np.bitwise_count(qb[:,i,None]&kb).sum(-1,dtype=np.int32)
        count[:,i]=(p*qs[:,i,None]*ks).sum(-1,dtype=np.int32)
    return count.astype(np.float64)/L*(sq*sk.swapaxes(-1,-2))/math.sqrt(d)

def stochastic_av(a,v,L,seed,source='sobol',encoding='magnitude',center=False):
    if encoding!='magnitude':raise ValueError(encoding)
    b,n,d=v.shape;mu=v.mean(1,keepdims=True) if center else np.zeros((b,1,d));vc=v-mu;sv=scale_power2(vc,1);vn=vc/sv
    ab=packed(a,L,seed+307,source,0);vb=packed(np.abs(vn),L,seed+401,source,1);vs=np.sign(vn).astype(np.int32)
    count=np.empty((b,n,d),dtype=np.int32)
    for i in range(n):
        p=np.bitwise_count(ab[:,i,:,None]&vb).sum(-1,dtype=np.int32)
        count[:,i]=(p*vs).sum(1,dtype=np.int32)
    return count.astype(np.float64)/L*sv+mu

def layer_norm(x,gamma,beta,tables,L,seed,source='sobol',analytic=False):
    if analytic:return REFERENCE_LAYER_NORM(x,gamma,beta,tables,L,seed,source,True)
    c=x-x.mean(-1,keepdims=True);scale=scale_power2(c,-1);v=np.abs(c/scale)
    xb=packed(v,L,seed+11,source,0);yb=packed(v,L,seed+29,source,1)
    count=np.bitwise_count(xb&yb).sum((-1,-2),dtype=np.int64)
    variance=count[...,None].astype(np.float64)/(L*c.shape[-1])*scale**2
    m,e=np.frexp(variance+1e-5)
    factor=np.ldexp(np.ones_like(m),-(e//2))*np.where(e%2,1/math.sqrt(2),1.)
    p=np.clip((m-.5)*2,0,1);table=np.array(tables['rsqrt']['weights'])
    out=sm.gated(c,p,table,L,seed+3001,source)
    return out*(math.sqrt(2)*factor)*gamma+beta
