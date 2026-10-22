import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,math,time
import numpy as np
from scipy.special import softmax
from horner import HornerCountNorm, HERE, build
from sc_baselines import xorshift,seeds_for,thresholds
def literal(op,d,L,seed):
 th=thresholds(np.clip(d,0,op.bound).ravel()/op.factors);ss=seeds_for(len(th),seed);out=[]
 for threshold,s in zip(th,ss):
  count=0
  for t in range(L):
   product=1
   for f in range(op.factors):
    h=1
    for k in range(op.degree,0,-1):
     s=xorshift(s);x=int(s<int(threshold));s=xorshift(s);c=int(s<int(op.c[k-1]));h=1-(x&c&h)
    product &= h
   count+=product
  out.append(count)
 return np.array(out).reshape(d.shape)
def main():
 build();checks=[];d=np.array([0.,.1,.5,1.,2.,4.,8.,9.])
 for m in [3,4,5,6,8]:
  op=HornerCountNorm(m);a=op.count(d,64,31415);b=literal(op,d,64,31415)
  assert np.array_equal(a,b);checks.append({'check':'literal Python vs C++ gate sequence','degree':m,'max_difference':0})
  x=np.linspace(0,1,101);poly=sum((-x)**k/math.factorial(k) for k in range(m+1))
  assert np.max(np.abs(op.expected_exp(x*8,False)-poly**8))<2e-15
  checks.append({'check':'Horner law matches Maclaurin polynomial algebraically','degree':m})
 op=HornerCountNorm(5);L=4096;a=op.count(np.repeat(d[:7],128),L,112233).reshape(7,128)/L
 expected=op.expected_exp(d[:7]);se=a.std(1,ddof=1)/np.sqrt(128);err=np.abs(a.mean(1)-expected)
 assert np.all(err<np.maximum(6*se,.0002))
 checks.append({'check':'128 replicas agree with polynomial expectation','max_bias':float(err.max()),'max_SE':float(se.max())})
 z=np.random.default_rng(9).normal(size=(5,16));a=op.stream(z,256,123);b=op.stream(z+16,256,123)
 assert np.array_equal(a,b) and np.allclose(a.sum(-1),1) and np.all(a>=0)
 checks.append({'check':'shift invariance, row normalization and positivity','passed':True})
 (HERE/'results/horner_verification.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2))
 protocol=json.loads((HERE/'results/horner_protocol.json').read_text())
 indices=json.loads((HERE/'results/finite_selection.json').read_text())['validation_row_indices']
 z=np.load(HERE/'results/calibration_rows.npz')['valid'][indices];target=softmax(z,axis=-1);records=[]
 for m in protocol['candidate_degrees']:
  op=HornerCountNorm(m);mses=[];t=time.perf_counter()
  for s in protocol['validation_seeds']:mses.append(float(np.mean((op.stream(z,4096,s)-target)**2)))
  row={'degree':m,'factors':8,'mean_MSE':float(np.mean(mses)),'RMSE':float(np.sqrt(np.mean(mses))),'seed_MSE':mses,'seconds':time.perf_counter()-t}
  records.append(row);print(json.dumps(row),flush=True)
 selected=min(records,key=lambda r:r['mean_MSE'])
 out={'selection_criterion':protocol['selection'],'rows':len(z),'candidate_results':records,'selected':selected}
 (HERE/'results/horner_selection.json').write_text(json.dumps(out,indent=2));print('SELECTED',json.dumps(selected),flush=True)
if __name__=='__main__':main()
