"""Literature-derived scalar SC exponentiation with decoded count normalization.

These are matched-network hybrid SC controls, not reproductions of an entire
published accelerator. Actual random bits drive SExp FSMs or Bernstein gates.
Counts are normalized numerically before the existing SC AV implementation.
Hardware costs of this normalization have NOT been measured in this stage.
"""
from pathlib import Path
import ctypes
import json
import subprocess
import numpy as np
from scipy.optimize import lsq_linear
from scipy.special import comb, softmax

HERE=Path(__file__).resolve().parent
LIB=HERE/'sc_exp_kernels.so'
_lib=None

def build():
    subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',
                    str(HERE/'sc_exp_kernels.cpp'),'-o',str(LIB)],check=True)

def native():
    global _lib
    if _lib is None:
        _lib=ctypes.CDLL(str(LIB));p=ctypes.c_void_p;i=ctypes.c_int;i64=ctypes.c_int64
        _lib.sexp_counts.argtypes=[p,p,p,p,i64,i,i,i,i]
        _lib.bernstein_counts.argtypes=[p,p,p,p,i64,i,i]
        _lib.sexp_counts.restype=None;_lib.bernstein_counts.restype=None
    return _lib

def thresholds(p):
    return np.ascontiguousarray(np.ceil(np.asarray(p,dtype=np.float64)*2.**32).clip(0,2.**32),dtype=np.uint64)

def seeds_for(size,seed):
    # Offline simulation initialization; every xorshift state is nonzero.
    return np.random.default_rng(seed).integers(1,2**32,size=size,dtype=np.uint32)

def xorshift(s):
    s=int(s);s^=(s<<13)&0xffffffff;s^=s>>17;s^=(s<<5)&0xffffffff
    return s&0xffffffff

def bernstein_basis(x,degree):
    k=np.arange(degree+1)
    return comb(degree,k)*np.asarray(x)[...,None]**k*(1-np.asarray(x)[...,None])**(degree-k)

def fit_bernstein(degree,bound=8.):
    x=np.linspace(0,1,4097);b=bernstein_basis(x,degree);y=np.exp(-bound*x)
    # Endpoint exactness is specified before data evaluation.
    fixed=np.array([1.,np.exp(-bound)])
    rhs=y-b[:,0]*fixed[0]-b[:,-1]*fixed[1]
    w=lsq_linear(b[:,1:-1],rhs,bounds=(0,1),tol=1e-12,max_iter=1000).x
    w=np.r_[fixed[0],w,fixed[1]]
    return w

class ExpCountNorm:
    def __init__(self,kind='sexp',states=32,gain=4,burn=256,degree=16,bound=8.,coefficients=None):
        if kind not in ['sexp','bernstein']:raise ValueError(kind)
        self.kind=kind;self.states=int(states);self.gain=int(gain);self.burn=int(burn)
        self.degree=int(degree);self.bound=float(bound)
        self.coefficients=np.array(coefficients,dtype=np.float64) if coefficients is not None else (
            fit_bernstein(self.degree,self.bound) if kind=='bernstein' else None)
        if self.kind=='sexp' and not (0<self.gain<self.states):raise ValueError('invalid FSM')
        if self.kind=='bernstein' and len(self.coefficients)!=self.degree+1:raise ValueError('coefficient length')
        self.diagnostics=[]

    def config(self):
        return dict(kind=self.kind,states=self.states,gain=self.gain,burn=self.burn,
                    degree=self.degree,bound=self.bound,
                    coefficients=None if self.coefficients is None else self.coefficients.tolist(),
                    normalization='decoded integer counts divided by their row sum; numerical',
                    rng='xorshift32(13,17,5), one nonzero state per scalar input',
                    exact_maximum=True)

    @classmethod
    def from_config(cls,c):
        return cls(**{k:v for k,v in c.items() if k in ['kind','states','gain','burn','degree','bound','coefficients']})

    def count(self,delta,L,seed):
        if not isinstance(L,int) or L<1:raise ValueError('positive integer L required')
        d=np.clip(np.asarray(delta,dtype=np.float64),0,self.bound)
        shape=d.shape;out=np.zeros(d.size,dtype=np.int32);s=seeds_for(d.size,seed)
        if self.kind=='sexp':
            p=np.clip(.5+d/(4*self.gain),.5,1)
            th=thresholds(p.ravel());one=np.ascontiguousarray((d.ravel()==0).astype(np.uint8))
            native().sexp_counts(th.ctypes.data,s.ctypes.data,one.ctypes.data,out.ctypes.data,d.size,L,self.burn,self.states,self.gain)
        else:
            th=thresholds(d.ravel()/self.bound);coeff=thresholds(self.coefficients)
            native().bernstein_counts(th.ctypes.data,s.ctypes.data,coeff.ctypes.data,out.ctypes.data,d.size,L,self.degree)
        return out.reshape(shape)

    def expected_exp(self,delta):
        d=np.clip(np.asarray(delta,dtype=np.float64),0,self.bound)
        if self.kind=='bernstein':return bernstein_basis(d/self.bound,self.degree)@self.coefficients
        p=np.clip(.5+d/(4*self.gain),.5,1)
        # Exact stationary occupancy of the finite FSM, not exp(-delta).
        t=np.divide(1-p,p)
        powers=t[...,None]**np.arange(self.states-1,-1,-1)
        y=powers[...,:self.states-self.gain].sum(-1)/powers.sum(-1)
        return np.where(d==0,1.,y)

    def analytic(self,z):
        z=np.asarray(z,dtype=np.float64);d=z.max(-1,keepdims=True)-z
        e=self.expected_exp(d);return e/e.sum(-1,keepdims=True)

    def stream(self,z,L,seed,*args,**kwargs):
        z=np.asarray(z,dtype=np.float64)
        if not np.isfinite(z).all():raise ValueError('finite scores required')
        delta=z.max(-1,keepdims=True)-z
        c=self.count(delta,L,seed);den=c.sum(-1,keepdims=True,dtype=np.int64)
        # A max-score component is exactly exp(0)=1, so denominator >= L.
        if not np.all(den>=L):raise AssertionError('zero/invalid denominator')
        a=c/den
        self.diagnostics.append(dict(rows=int(z.size/z.shape[-1]),
            clipped_components=int((delta>self.bound).sum()),components=int(z.size),
            maximum_row_mass_error=float(np.abs(a.sum(-1)-1).max()),min_denominator=int(den.min())))
        return a

def iterative_softmax_numeric(z,iterations=64):
    """ASCEND Algorithm-1 mathematical diagnostic ONLY: NOT an SC circuit.

    No thermometer encoding, gate/BSN processing or published training recipe
    is reproduced here. Never include this as measured ASCEND hardware/SC.
    """
    x=np.asarray(z,dtype=np.float64);x=x-x.mean(-1,keepdims=True)
    y=np.full_like(x,1/x.shape[-1])
    for _ in range(iterations):
        q=x*y;y=y+(q-y*q.sum(-1,keepdims=True))/iterations
    return y
