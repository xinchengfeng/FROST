import sys,json
from pathlib import Path
import numpy as np
from local_tt import LocalTT,QMCBernstein,net_shifts,HERE

def literal(op,z,L,seed):
 shape,order,lo,p=op.prep(z);net,shift=net_shifts(L,32,len(lo),seed);counts=np.zeros((len(lo),16),dtype=int)
 for row in range(len(lo)):
  for t in range(L):
   bond=0
   for j in range(16):
    state=lo[row,j]+(int(net[t,j]^shift[row,j])<p[row,j])
    u=int(net[t,16+j]^shift[row,16+j]);c=op.cdf[j,state,bond,:op.sizes[j]]
    bond=int(np.searchsorted(c,u,side='right'));bond=min(bond,int(op.sizes[j])-1)
   counts[row,bond]+=1
   assert counts[row].sum()==t+1
 a=np.empty_like(counts,dtype=float);np.put_along_axis(a,order,counts/L,-1)
 return a.reshape(shape)
def main():
 p=Path(sys.argv[1]);op=LocalTT(p);rng=np.random.default_rng(2026101706);z=rng.normal(size=(4,16))*3
 rows=[]
 for L in [64,256]:
  for seed in [111,222,333]:
   a=op.stream(z,L,seed);b=literal(op,z,L,seed);assert np.array_equal(a,b)
   rows.append({'check':'literal C++/Python and every prefix mass','L':L,'seed':seed,'max_difference':float(np.max(np.abs(a-b)))})
 a=op.stream(z,4096,444);b=op.stream(z+32,4096,444);assert np.array_equal(a,b)
 rows.append({'check':'translation invariance','max_difference':0.})
 for test in [np.zeros((4,16)),np.tile(np.r_[100.,np.full(15,-100.)],(4,1))]:
  a=op.stream(test,4096,555);assert np.all(a>=0) and np.all(a.sum(-1)==1)
 rows.append({'check':'uniform/extreme probabilities and exact mass','passed':True})
 zs=np.repeat(z,128,axis=0);a=op.stream(zs,4096,666).reshape(4,128,16);expected=op.analytic(z)
 error=np.abs(a.mean(1)-expected);se=a.std(1,ddof=1)/np.sqrt(128)
 assert np.all(error<np.maximum(6*se,.002))
 rows.append({'check':'128 digital-shift replicas against TT expectation','max_bias':float(error.max()),'max_SE':float(se.max())})
 # Independently verify strengthened Bernstein primitive, same Sobol utility.
 config=json.loads((HERE/'inputs/stage2_protocol.json').read_text())['bernstein'];bern=QMCBernstein.from_config(config)
 d=np.array([0,.2,2,7,8.]);L=256;seed=777;c=bern.count(d,L,seed);net,sh=net_shifts(L,9,len(d),seed)
 th=np.ceil(d/8*2**32).astype(np.uint64);ct=np.ceil(bern.coefficients*2**32).astype(np.uint64);ref=[]
 for i in range(len(d)):
  count=0
  for t in range(L):
   k=sum(int(net[t,j]^sh[i,j])<th[i] for j in range(8));count+=int(net[t,8]^sh[i,8])<ct[k]
  ref.append(count)
 assert np.array_equal(c,ref);rows.append({'check':'Bernstein QMC literal bit comparison','passed':True})
 out={'candidate':p.name,'checks':rows,'passed':True};(HERE/'results'/f'verification_{p.stem}.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
