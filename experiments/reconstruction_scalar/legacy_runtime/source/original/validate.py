"""Structural, data-separation and stochastic-law checks independent of task testing."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import json,sys,time
import numpy as np,torch
from data import ROOT,DATASETS,load_data,patchify
from vision import VisionTransformer,load
from attention_core import FrozenHI,uniforms,scale_power2
from stochastic_model import SCTransformer,linear

def main():
 checks={};rng=np.random.default_rng(460001)
 for dataset in DATASETS:
  if not (ROOT/'data'/f'{dataset}_metadata.npz').exists():continue
  d=load_data(dataset);names=['train_indices','validation_indices','calibration_indices','calibration_validation_indices']
  allids=np.concatenate([d[k] for k in names]);assert len(allids)==len(np.unique(allids))==len(d['images'])
  assert len(d['test_images'])==10000;sub=d['test_subset_indices'];assert len(sub)==len(np.unique(sub))==1000
  assert (np.bincount(d['test_labels'][sub])==100).all()
  x=patchify(d['images'][d['calibration_validation_indices'][:3]],d['mean'],d['std'])
  torch.manual_seed(17);m=VisionTransformer(x.shape[-1]).eval();sc=SCTransformer(m)
  with torch.no_grad():ref=m(torch.tensor(x)).numpy()
  pred=sc.run(x,mode='exact')[0];diff=float(np.abs(pred-ref).max());assert diff<1e-5
  checks[dataset]=dict(split_disjoint=True,train_size=len(d['train_indices']),test_size=10000,diagnostic_counts=np.bincount(d['test_labels'][sub]).tolist(),exact_graph_max_error=diff)
 x=rng.normal(size=(11,7));w=rng.normal(size=(13,7));bias=rng.normal(size=13);L=256;seed=55
 sx=scale_power2(x,-1);sw=scale_power2(w,-1)
 xb=(uniforms(x.shape,L,seed+17,'sobol',0)<abs((x/sx)[...,None]))*np.sign(x[...,None])
 wb=(uniforms(w.shape,L,seed+59,'sobol',1)<abs((w/sw)[...,None]))*np.sign(w[...,None])
 explicit=xb.reshape(11,-1)@wb.reshape(13,-1).T/L*sx*sw.T+bias
 packed=linear(x,w,bias,L,seed);assert np.array_equal(explicit,packed)
 checks['packed_bits_exactly_match_explicit_products']=True
 # Conditional output law: contraction and ancestral latent-path sampling agree.
 hi=FrozenHI();state=rng.integers(16,size=(8,16));law=hi.selected(state);samples=100000;observed=[]
 for s in state:
  bond=np.zeros(samples,dtype=np.int64)
  for j,cdf in enumerate(hi.cdfs):
   c=cdf[s[j],bond];bond=np.minimum((c<rng.random((samples,1))).sum(-1),c.shape[-1]-1)
  observed.append(np.bincount(bond,minlength=16)/samples)
 observed=np.array(observed);se=np.sqrt(np.maximum(law*(1-law)/samples,1e-10));z=abs(observed-law)/se
 assert z.max()<6
 checks['ancestral_law_max_standardized_deviation']=float(z.max());checks['ancestral_samples_per_state']=samples
 a=hi.stream(rng.normal(size=(2,16,16)),64,470001);assert np.isfinite(a).all() and np.array_equal(a.sum(-1),np.ones((2,16)))
 checks['categorical_rows_sum_exactly_to_one']=True
 (ROOT/'provenance'/'checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
