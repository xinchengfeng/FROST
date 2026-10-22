"""Literature-derived NAND-Horner SC exponentiation, NOT full-accelerator reproduction.

For gap delta in [0,8], x=delta/8, T_m(x)=sum_{k=0}^m (-x)^k/k!.
Each output cycle computes eight independent stochastic copies of T_m(x)
and ANDs their outputs. Each copy uses h=1; h=NAND(x_bit, coefficient_bit, h)
for k=m,...,1, with P(coefficient_bit)=1/k. Hence E(output)=T_m(x)^8.
The max gap is zero -> all output bits one; count normalization is numerical.
This is an adapted application of the NAND/Horner series-expansion principle
of Parhi & Liu (IEEE TETC 2019, DOI 10.1109/TETC.2016.2618750), with explicitly
chosen range factorization; no claim to reproduce a particular published circuit.
"""
from pathlib import Path
import ctypes,subprocess,math
import numpy as np
from sc_baselines import thresholds,seeds_for
HERE=Path(__file__).resolve().parents[1]
_lib=None
def build():
 subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(HERE/'source/horner.cpp'),'-o',str(HERE/'source/horner_kernels.so')],check=True)
def native():
 global _lib
 if _lib is None:
  _lib=ctypes.CDLL(str(HERE/'source/horner_kernels.so'))
  _lib.horner_counts.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*4
  _lib.horner_counts.restype=None
 return _lib
class HornerCountNorm:
 def __init__(self,degree=5,factors=8,bound=8):
  assert factors>=bound>0 and degree>=1
  self.degree=int(degree);self.factors=int(factors);self.bound=float(bound)
  self.c=thresholds(1/np.arange(1,degree+1))
 def expected_exp(self,delta,quantized=True):
  x=np.clip(np.asarray(delta,dtype=np.float64),0,self.bound)/self.factors
  if quantized:x=thresholds(x).astype(np.float64)/2**32
  h=np.ones_like(x)
  for k in range(self.degree,0,-1):
   ck=float(self.c[k-1])/2**32 if quantized else 1/k
   h=1-x*ck*h
  return h**self.factors
 def count(self,delta,L,seed):
  d=np.asarray(delta,dtype=np.float64);x=thresholds(np.clip(d,0,self.bound).ravel()/self.factors)
  s=seeds_for(d.size,seed);out=np.zeros(d.size,dtype=np.int32)
  native().horner_counts(*[a.ctypes.data for a in [x,self.c,s,out]],d.size,int(L),self.degree,self.factors)
  return out.reshape(d.shape)
 def analytic(self,z):
  z=np.asarray(z,dtype=np.float64);e=self.expected_exp(z.max(-1,keepdims=True)-z);return e/e.sum(-1,keepdims=True)
 def stream(self,z,L,seed):
  z=np.asarray(z,dtype=np.float64)
  if not np.isfinite(z).all():raise ValueError('finite scores required')
  c=self.count(z.max(-1,keepdims=True)-z,L,seed);den=c.sum(-1,keepdims=True,dtype=np.int64)
  assert np.all(den>=L)
  return c/den
if __name__=='__main__':build()
