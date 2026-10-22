"""Small CPU implementation checks without training, PyTorch or archived weights.

Synthetic normalized CDFs exercise the actual compiled categorical sampler.
They are test fixtures, not additional paper experiments or trained operators.
"""
from pathlib import Path
import ctypes
import json
import shutil
import subprocess
import tempfile
import numpy as np

HERE=Path(__file__).resolve().parent
MASK=(1<<32)-1

def step(state):
    state^=(state<<13)&MASK;state^=state>>17;state^=(state<<5)&MASK
    return state&MASK

def literal_strata(low,frac,cdf,sizes,seeds,L):
    result=np.zeros((len(low),16),dtype=np.int32)
    for row in range(len(low)):
        state=int(seeds[row]);current=[L]+[0]*15
        for j in range(16):
            nxt=[0]*16
            for bond in range(1 if j==0 else int(sizes[j-1])):
                m=current[bond];high=0
                if m:
                    state=step(state)
                    high=sum((((t<<32)+state)//m)<int(frac[row,j]) for t in range(m))
                for side,num in enumerate([m-high,high]):
                    if not num:continue
                    state=step(state);v=cdf[j,low[row,j]+side,bond,:sizes[j]]
                    for t in range(num):
                        u=((t<<32)+state)//num
                        k=min(int(np.searchsorted(v,u,side='right')),int(sizes[j])-1)
                        nxt[k]+=1
            assert sum(nxt)==L
            current=nxt
        result[row]=current
    return result

def literal_iid(low,frac,cdf,sizes,net,shifts,L):
    counts=np.zeros((len(low),16),dtype=np.int32)
    for row in range(len(low)):
        for t in range(L):
            bond=0
            for j in range(16):
                s=int(low[row,j])+(int(net[t,j]^shifts[row,j])<int(frac[row,j]))
                u=int(net[t,16+j]^shifts[row,16+j])
                bond=min(int(np.searchsorted(cdf[j,s,bond,:sizes[j]],u,side='right')),int(sizes[j])-1)
            counts[row,bond]+=1
    return counts

def main():
    compiler=shutil.which('g++') or shutil.which('c++')
    if compiler is None:raise RuntimeError('A C++17 compiler (g++ or c++) is required.')
    rng=np.random.default_rng(20261022)
    rows,states,pad=4,33,16
    low=rng.integers(0,states-1,size=(rows,16),dtype=np.int32)
    frac=rng.integers(0,1<<32,size=(rows,16),dtype=np.uint64)
    p=rng.random((16,states,pad,pad));p/=p.sum(-1,keepdims=True)
    cdf=np.ceil(np.cumsum(p,-1)*(1<<32)).astype(np.uint64);cdf[...,-1]=1<<32
    sizes=np.full(16,16,dtype=np.int32)
    seeds=rng.integers(1,1<<32,size=rows,dtype=np.uint32)
    checks=[]
    with tempfile.TemporaryDirectory(prefix='frost-native-check-') as temp:
        temp=Path(temp)
        for name in ['local_tt','sc_exp_kernels','horner']:
            subprocess.run([compiler,'-O2','-std=c++17','-shared','-fPIC',str(HERE/'source'/f'{name}.cpp'),'-o',str(temp/f'{name}.so')],check=True)
        lib=ctypes.CDLL(str(temp/'local_tt.so'));ptr=ctypes.c_void_p;integer=ctypes.c_int
        lib.strat_tt.argtypes=[ptr]*6+[integer]*4;lib.strat_tt.restype=None
        lib.local_tt.argtypes=[ptr]*7+[integer]*4;lib.local_tt.restype=None
        for L in [64,256]:
            actual=np.zeros((rows,16),dtype=np.int32)
            lib.strat_tt(*[a.ctypes.data for a in [low,frac,cdf,sizes,seeds,actual]],rows,L,states,pad)
            expected=literal_strata(low,frac,cdf,sizes,seeds,L)
            assert np.array_equal(actual,expected) and np.all(actual.sum(-1)==L)
            checks.append({'kernel':'strat_tt','L':L,'literal_count_match':True,'exact_mass':True})
            net=rng.integers(0,1<<32,size=(L,32),dtype=np.uint32)
            shifts=rng.integers(0,1<<32,size=(rows,32),dtype=np.uint32)
            actual=np.zeros((rows,16),dtype=np.int32)
            lib.local_tt(*[a.ctypes.data for a in [low,frac,cdf,sizes,net,shifts,actual]],rows,L,states,pad)
            expected=literal_iid(low,frac,cdf,sizes,net,shifts,L)
            assert np.array_equal(actual,expected) and np.all(actual.sum(-1)==L)
            checks.append({'kernel':'local_tt','L':L,'literal_count_match':True,'exact_mass':True})
    print(json.dumps({'passed':True,'scope':'Synthetic implementation fixtures; no training or reconstruction experiment','compiled_kernels':3,'checks':checks},indent=2))

if __name__=='__main__':main()
