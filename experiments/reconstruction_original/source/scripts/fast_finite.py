"""Exact acceleration of the existing finite-bit simulator.

NumPy supplies precisely the original PCG64 drives and Sobol digital shifts.
Compiled loops only execute FSM updates, categorical draws, and AND/popcount.
The 16-cycle TT emission chunks retain their original RNG ordering.
"""
import ctypes
import math
import subprocess
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'original'))
import stochastic_model as sm
from attention_core import FrozenHI, scale_power2
import packed_kernels as pk

HERE=Path(__file__).resolve().parent
LIB=HERE/'finite_kernels.so'
_native=None
_original_gated=sm.gated
_original_stream=FrozenHI.stream


def build():
    subprocess.run(['c++','-O3','-march=native','-std=c++17','-shared','-fPIC',
                    str(HERE/'finite_kernels.cpp'),'-o',str(LIB)],check=True)


def native():
    global _native
    if _native is None:
        if not LIB.exists():
            raise RuntimeError('Compile once before starting workers: fast_finite.build()')
        _native=ctypes.CDLL(str(LIB))
        p=ctypes.c_void_p;i64=ctypes.c_int64;i32=ctypes.c_int32
        signatures={
            'fsm_update':[p,p,p,i64,i64,i32],
            'tt_events':[p,p,p,p,p,i64,i64,i32,i32,i32],
            'scalar_gates':[p]*11+[i64,i64,i64,i32],
            'bit_dot':[p]*5+[i64]*5,
        }
        for key,args in signatures.items():
            f=getattr(_native,key);f.argtypes=args;f.restype=None
    return _native


def ptr(a):
    return a.ctypes.data if a is not None else None


def thresholds(p):
    return np.ascontiguousarray(np.ceil(np.asarray(p,dtype=np.float64)*2.**32).clip(0,2.**32),dtype=np.uint64)


def gated(x,prob_input,table,L,seed,source='sobol',burn=64):
    if source!='sobol':
        return _original_gated(x,prob_input,table,L,seed,source,burn)
    shape=x.shape;scale=scale_power2(x,-1);norm=x/scale
    nx=x.size;np_=prob_input.size;n=len(table)
    state=np.full(np_,(n-1)//2,dtype=np.int16)
    rng=np.random.default_rng(seed+701)
    probability=np.asarray(prob_input).reshape(1,np_)
    drives=np.ascontiguousarray(rng.random((burn,np_))<probability,dtype=np.uint8)
    native().fsm_update(ptr(state),ptr(drives),None,np_,burn,n)
    mag=sm.UniformStream(shape,L,seed+1301,source,0)
    em=sm.UniformStream(prob_input.shape,L,seed+1709,source,1)
    mag_shift=np.ascontiguousarray(mag.shift.reshape(-1))
    em_shift=np.ascontiguousarray(em.shift.reshape(-1))
    mt=thresholds(np.abs(norm).reshape(-1));tt=thresholds(table)
    mapping=np.ascontiguousarray(np.broadcast_to(np.arange(np_,dtype=np.int32).reshape(prob_input.shape),shape).reshape(-1))
    signs=np.ascontiguousarray(np.sign(norm).reshape(-1),dtype=np.int32)
    counts=np.zeros(nx,dtype=np.int32)
    for start in range(0,L,16):
        width=min(16,L-start)
        drives=np.ascontiguousarray(rng.random((width,np_))<probability,dtype=np.uint8)
        mn=np.ascontiguousarray(mag.net[start:start+width]);en=np.ascontiguousarray(em.net[start:start+width])
        native().scalar_gates(ptr(state),ptr(drives),ptr(mag_shift),ptr(em_shift),
            ptr(mn),ptr(en),ptr(mt),ptr(tt),ptr(mapping),ptr(signs),ptr(counts),nx,np_,width,n)
    return counts.reshape(shape).astype(np.float64)/L*scale


def tt_stream(self,z,L,seed,gain=1.,burn=256,return_diagnostic=False):
    if return_diagnostic:
        return _original_stream(self,z,L,seed,gain,burn,True)
    shape=z.shape;x,order=self.encode(z.reshape(-1,self.n),gain)
    rows=len(x);rng=np.random.default_rng(seed);emit=np.random.default_rng(seed+900019)
    state=np.full(x.shape,(self.states-1)//2,dtype=np.int16)
    for start in range(0,burn,16):
        width=min(16,burn-start)
        drives=np.ascontiguousarray(rng.random((width,*x.shape))<x[None],dtype=np.uint8)
        native().fsm_update(ptr(state),ptr(drives),None,x.size,width,self.states)
    pad=max(max(c.shape[-2:]) for c in self.cdfs)
    cdfs=np.zeros((self.n,self.states,pad,pad),dtype=np.float64)
    sizes=np.asarray([c.shape[-1] for c in self.cdfs],dtype=np.int32)
    for j,c in enumerate(self.cdfs):cdfs[j,:,:c.shape[1],:c.shape[2]]=c
    counts=np.zeros(x.shape,dtype=np.int32)
    for start in range(0,L,16):
        width=min(16,L-start)
        drives=np.ascontiguousarray(rng.random((width,*x.shape))<x[None],dtype=np.uint8)
        trace=np.empty((width,*x.shape),dtype=np.int16)
        native().fsm_update(ptr(state),ptr(drives),ptr(trace),x.size,width,self.states)
        draws=np.ascontiguousarray(emit.random((self.n,width*rows)))
        native().tt_events(ptr(trace),ptr(draws),ptr(cdfs),ptr(sizes),ptr(counts),
                           rows,width,self.states,self.n,pad)
    a=np.empty_like(x);np.put_along_axis(a,order,counts/L,-1)
    return a.reshape(shape)


def integer_dot(xb,wb,xs,ws,batches,rows,outputs,features,words):
    xb=np.ascontiguousarray(xb);wb=np.ascontiguousarray(wb)
    xs=np.ascontiguousarray(xs,dtype=np.int32);ws=np.ascontiguousarray(ws,dtype=np.int32)
    counts=np.empty((batches,rows,outputs),dtype=np.int32)
    native().bit_dot(ptr(xb),ptr(wb),ptr(xs),ptr(ws),ptr(counts),batches,rows,outputs,features,words)
    return counts


def linear(x,w,bias,L,seed,source='sobol',encoding='magnitude'):
    if encoding!='magnitude':raise ValueError(encoding)
    shape=x.shape;x=x.reshape(-1,x.shape[-1]);sx=scale_power2(x,-1);sw=scale_power2(w,-1)
    xn=x/sx;wn=w/sw
    xb=pk.packed(np.abs(xn),L,seed+17,source,0);wb=pk.packed(np.abs(wn),L,seed+59,source,1)
    counts=integer_dot(xb,wb,np.sign(xn),np.sign(wn),1,len(x),len(w),x.shape[-1],L//64)[0]
    y=counts.astype(np.float64)/L*sx*sw.T+bias
    return y.reshape(*shape[:-1],len(w))


def qk(q,k,L,seed,source='sobol',encoding='magnitude'):
    if encoding!='magnitude':raise ValueError(encoding)
    b,n,d=q.shape;sq=scale_power2(q,-1);sk=scale_power2(k,-1);qn=q/sq;kn=k/sk
    qb=pk.packed(np.abs(qn),L,seed+101,source,0);kb=pk.packed(np.abs(kn),L,seed+211,source,1)
    counts=integer_dot(qb,kb,np.sign(qn),np.sign(kn),b,n,n,d,L//64)
    return counts.astype(np.float64)/L*(sq*sk.swapaxes(-1,-2))/math.sqrt(d)


def av(a,v,L,seed,source='sobol',encoding='magnitude',center=False):
    if encoding!='magnitude':raise ValueError(encoding)
    b,n,d=v.shape;mu=v.mean(1,keepdims=True) if center else np.zeros((b,1,d));vc=v-mu
    sv=scale_power2(vc,1);vn=vc/sv
    ab=pk.packed(a,L,seed+307,source,0);vb=pk.packed(np.abs(vn),L,seed+401,source,1)
    counts=integer_dot(ab,vb.transpose(0,2,1,3),np.ones(a.shape,dtype=np.int32),
                       np.sign(vn).transpose(0,2,1),b,n,d,n,L//64)
    return counts.astype(np.float64)/L*sv+mu


def install():
    import restoration_model as rm
    native()
    sm.gated=gated;sm.linear=linear;FrozenHI.stream=tt_stream
    rm.stochastic_qk=qk;rm.stochastic_av=av


if __name__=='__main__':
    build()
    print('Compiled exact finite-stream kernels')
