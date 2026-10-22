from pathlib import Path
import sys,json
import numpy as np
from local_tt import StratifiedTT,StratifiedBernstein,seeds_for,HERE
from sc_baselines import xorshift,thresholds

def literal(op,z,L,seed):
 shape,order,lo,p=op.prep(z);states=seeds_for(len(lo),seed);result=[]
 for row in range(len(lo)):
  state=int(states[row])
  def split(m,threshold):
   nonlocal state
   if not m:return 0
   state=xorshift(state);return sum((((t<<32)+state)//m)<int(threshold) for t in range(m))
  counts=[L]+[0]*(op.pad-1)
  for j in range(16):
   nxt=[0]*op.pad
   for bond in range(1 if j==0 else op.sizes[j-1]):
    m=counts[bond];high=split(m,p[row,j])
    for side,num in enumerate([m-high,high]):
     if not num:continue
     state=xorshift(state);c=op.cdf[j,lo[row,j]+side,bond,:op.sizes[j]]
     for t in range(num):
      u=((t<<32)+state)//num;k=int(np.searchsorted(c,u,side='right'));nxt[min(k,int(op.sizes[j])-1)]+=1
   counts=nxt;assert sum(counts)==L
  result.append(counts[:16])
 a=np.empty((len(lo),16));np.put_along_axis(a,order,np.array(result)/L,-1);return a.reshape(shape)
def main():
 path=Path(sys.argv[1]);op=StratifiedTT(path);z=np.random.default_rng(2026101707).normal(size=(4,16))*2;records=[]
 for L in [64,256]:
  for seed in [111,222,333]:
   a=op.stream(z,L,seed);b=literal(op,z,L,seed);assert np.array_equal(a,b)
   records.append({'check':'conditional stratum gates Python/C++ and per-stage count conservation','L':L,'seed':seed,'maxabs':0.})
 a=op.stream(z,4096,444);assert np.array_equal(a,op.stream(z+32,4096,444))
 records.append({'check':'translation invariance','maxabs':0.})
 a=op.stream(np.repeat(z,256,axis=0),4096,555).reshape(4,256,16);target=op.analytic(z);delta=np.abs(a.mean(1)-target);se=a.std(1,ddof=1)/16
 assert np.all(delta<np.maximum(6*se,.0005)),(delta.max(),se.max())
 records.append({'check':'256 independent strata offsets against TT expectation','max_bias':float(delta.max()),'max_SE':float(se.max())})
 cfg=json.loads((HERE/'inputs/stage2_protocol.json').read_text())['bernstein'];bern=StratifiedBernstein.from_config(cfg)
 d=np.array([0.,.2,2.,7.,8.]);L=256;seed=666;th=thresholds(d/8);ct=thresholds(bern.coefficients);states=seeds_for(5,seed);actual=bern.count(d,L,seed);expected=[]
 for i in range(5):
  state=int(states[i])
  def split(m,p):
   nonlocal state
   if not m:return 0
   state=xorshift(state);return sum((((t<<32)+state)//m)<int(p) for t in range(m))
  cs=[L]+[0]*8
  for j in range(8):
   nxt=[0]*9
   for k in range(j+1):hi=split(cs[k],th[i]);nxt[k]+=cs[k]-hi;nxt[k+1]+=hi
   cs=nxt
  expected.append(sum(split(cs[k],ct[k]) for k in range(9)))
 assert np.array_equal(actual,expected);records.append({'check':'stratified Bernstein literal gate comparison','maxabs':0.})
 out={'candidate':path.name,'passed':True,'checks':records};(HERE/'results'/f'verification_stratified_{path.stem}.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
