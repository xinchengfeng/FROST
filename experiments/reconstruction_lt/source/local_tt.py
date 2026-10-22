"""Experimental local-address TT, explicitly distinct from the original FSM encoding.
State law: adjacent grid indices with fractional Bernoulli selection, not an
N-state birth/death chain. Every retained sample follows the learned TT latent
CDFs and increments exactly one category. No softmax evaluation at inference.
Sobol dimensions are unique for every input-state and core-emission draw.
"""
from pathlib import Path
import sys,ctypes,subprocess,math
import numpy as np
import torch
from scipy.stats import qmc
from train_candidates import GridTT
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'source'))
from sc_baselines import ExpCountNorm,thresholds,seeds_for
_lib=None

def build():subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(HERE/'source/local_tt.cpp'),'-o',str(HERE/'source/local_tt_kernels.so')],check=True)
def native():
 global _lib
 if _lib is None:
  _lib=ctypes.CDLL(str(HERE/'source/local_tt_kernels.so'));p=ctypes.c_void_p;i=ctypes.c_int
  _lib.local_tt.argtypes=[p]*7+[i]*4;_lib.local_tt.restype=None
  _lib.bern_qmc.argtypes=[p]*5+[i]*3;_lib.bern_qmc.restype=None
  _lib.strat_tt.argtypes=[p]*6+[i]*4;_lib.strat_tt.restype=None
  _lib.strat_bern.argtypes=[p]*4+[i]*3;_lib.strat_bern.restype=None
  _lib.antithetic_tt.argtypes=[p]*6+[i]*4;_lib.antithetic_tt.restype=None
 return _lib

def net_shifts(L,dimensions,rows,seed):
 if L<1 or L&(L-1):raise ValueError('power-of-two L required')
 # Fixed scrambled net plus independent per-row digital shifts.
 net=(qmc.Sobol(dimensions,scramble=True,seed=2026101705,bits=32).random_base2(int(math.log2(L)))*2**32).astype(np.uint32)
 shifts=np.random.default_rng(seed).integers(0,2**32,size=(rows,dimensions),dtype=np.uint32)
 return np.ascontiguousarray(net),np.ascontiguousarray(shifts)

class LocalTT:
 def __init__(self,path):
  ck=torch.load(path,weights_only=True);c=ck['config'];self.config=c;self.states=c['states'];self.bound=c['bound'];self.rank=c['rank'];self.encoding=c.get('encoding','center')
  self.model=GridTT(self.states,self.rank,self.bound,'local',encoding=self.encoding);self.model.load_state_dict(ck['state_dict']);self.model.eval()
  self.cores=[torch.softmax(ck['state_dict'][f'cs.{j}'].double(),-1).numpy().transpose(1,0,2).copy() for j in range(16)]
  self.pad=max(16,self.rank);self.sizes=np.array([c.shape[-1] for c in self.cores],dtype=np.int32)
  self.cdf=np.zeros((16,self.states,self.pad,self.pad),dtype=np.uint64)
  for j,c in enumerate(self.cores):
   cdf=thresholds(np.cumsum(c,-1));cdf[...,-1]=2**32
   self.cdf[j,:,:c.shape[1],:c.shape[2]]=cdf
 def prep(self,z):
  z=np.asarray(z,dtype=np.float64);shape=z.shape;z=z.reshape(-1,16);order=np.argsort(-z,axis=-1,kind='stable');s=np.take_along_axis(z,order,-1);s=s-s[:,:1] if self.encoding=='max' else s-s.mean(-1,keepdims=True)
  t=np.clip((s+self.bound)/(self.bound if self.encoding=='max' else 2*self.bound),0,1)*(self.states-1);lo=np.minimum(np.floor(t).astype(np.int32),self.states-2);p=t-lo
  return shape,order,np.ascontiguousarray(lo),thresholds(p)
 def analytic(self,z):
  shape,order,lo,p=self.prep(z);f=p.astype(np.float64)/2**32;h=np.ones((len(lo),1))
  for j,c in enumerate(self.cores):h=np.einsum('bi,bij->bj',h,c[lo[:,j]]*(1-f[:,j,None,None])+c[lo[:,j]+1]*f[:,j,None,None])
  a=np.empty_like(h);np.put_along_axis(a,order,h,-1);return a.reshape(shape)
 def stream(self,z,L,seed):
  shape,order,lo,p=self.prep(z);rows=len(lo);net,shifts=net_shifts(L,32,rows,seed);counts=np.zeros((rows,16),dtype=np.int32)
  native().local_tt(*[a.ctypes.data for a in [lo,p,self.cdf,self.sizes,net,shifts,counts]],rows,L,self.states,self.pad)
  assert np.all(counts.sum(-1)==L)
  a=np.empty_like(counts,dtype=np.float64);np.put_along_axis(a,order,counts/L,-1);return a.reshape(shape)

class QMCBernstein(ExpCountNorm):
 def count(self,delta,L,seed):
  d=np.clip(np.asarray(delta,dtype=np.float64),0,self.bound);shape=d.shape;size=d.size
  threshold=thresholds(d.ravel()/self.bound);coef=thresholds(self.coefficients)
  net,shift=net_shifts(L,self.degree+1,size,seed);out=np.zeros(size,dtype=np.int32)
  native().bern_qmc(*[a.ctypes.data for a in [threshold,coef,net,shift,out]],size,L,self.degree)
  return out.reshape(shape)


class StratifiedTT(LocalTT):
 def stream(self,z,L,seed):
  shape,order,lo,p=self.prep(z);rows=len(lo);seeds=seeds_for(rows,seed);counts=np.zeros((rows,16),dtype=np.int32)
  native().strat_tt(*[a.ctypes.data for a in [lo,p,self.cdf,self.sizes,seeds,counts]],rows,L,self.states,self.pad)
  assert np.all(counts.sum(-1)==L)
  a=np.empty_like(counts,dtype=np.float64);np.put_along_axis(a,order,counts/L,-1);return a.reshape(shape)
class StratifiedBernstein(ExpCountNorm):
 def count(self,delta,L,seed):
  d=np.clip(np.asarray(delta,dtype=np.float64),0,self.bound);threshold=thresholds(d.ravel()/self.bound);coef=thresholds(self.coefficients);seeds=seeds_for(d.size,seed);out=np.zeros(d.size,dtype=np.int32)
  native().strat_bern(*[a.ctypes.data for a in [threshold,coef,seeds,out]],d.size,L,self.degree)
  return out.reshape(d.shape)


class AntitheticTT(LocalTT):
 def stream(self,z,L,seed):
  if L%2:raise ValueError('even total particle count required')
  shape,order,lo,p=self.prep(z);rows=len(lo)
  phases=np.random.default_rng(seed).integers(0,2**32,size=(rows,16,self.pad,3),dtype=np.uint32)
  counts=np.zeros((rows,16),dtype=np.int32)
  native().antithetic_tt(*[a.ctypes.data for a in [lo,p,self.cdf,self.sizes,phases,counts]],rows,L,self.states,self.pad)
  assert np.all(counts.sum(-1)==L)
  a=np.empty_like(counts,dtype=np.float64);np.put_along_axis(a,order,counts/L,-1);return a.reshape(shape)
if __name__=='__main__':build()
