import json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from local_tt import LocalTT,StratifiedTT,QMCBernstein,StratifiedBernstein,HERE
from sc_baselines import ExpCountNorm

def main():
 data=np.load(HERE/'results/calibration_rows.npz')['valid'];rng=np.random.default_rng(2026101709);idx=rng.choice(len(data),192,replace=False);z=data[idx];target=softmax(z.astype(float),axis=-1)
 records=[];configs=[];seeds=[1123,2246,3369]
 for p in sorted((HERE/'checkpoints').glob('*.pt')):
  ck=torch.load(p,weights_only=True);configs.append({'checkpoint':p.name,'config':ck['config'],'parameters':ck['parameters'],'expected_validation':ck['validation']})
  if ck['config']['basis']!='local':continue
  for sampler,cls in [('QMC',LocalTT),('stratified',StratifiedTT)]:
   op=cls(p);errors=[];tvs=[]
   for seed in seeds:
    a=op.stream(z,4096,seed);errors.append(float(np.mean((a-target)**2)));tvs.append(float(np.mean(np.abs(a-target).sum(-1)/2)))
   rec={'checkpoint':p.name,'checkpoint_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'sampler':sampler,'parameters':ck['parameters'],'RMSE':float(np.sqrt(np.mean(errors))),'TV':float(np.mean(tvs)),'seed_MSE':errors};records.append(rec);print(json.dumps(rec),flush=True)
 best=min(records,key=lambda x:x['RMSE']);best['scope']='selected exclusively using independent calibration-validation activation rows; exploratory development'
 (HERE/'results/selected.json').write_text(json.dumps(best,indent=2))
 (HERE/'results/candidate_configs.json').write_text(json.dumps(configs,indent=2))
 config=json.loads((HERE/'inputs/stage2_protocol.json').read_text());controls=[]
 for name,op in [('SExp',ExpCountNorm.from_config(config['sexp'])),('Bernstein',ExpCountNorm.from_config(config['bernstein'])),('Bernstein_QMC',QMCBernstein.from_config(config['bernstein'])),('Bernstein_stratified',StratifiedBernstein.from_config(config['bernstein']))]:
  es=[];vs=[]
  for seed in seeds:
   a=op.stream(z,4096,seed);es.append(float(np.mean((a-target)**2)));vs.append(float(np.mean(np.abs(a-target).sum(-1)/2)))
  controls.append({'method':name,'RMSE':float(np.sqrt(np.mean(es))),'TV':float(np.mean(vs))})
 out={'validation_row_indices':idx.tolist(),'selection_stream_seeds':seeds,'L':4096,'candidates':records,'controls':controls,'selected':best,'note':'An intermediate N=33 r=16 stratified image ablation was evaluated before this final finite-validation selection; it was not the selection objective.'}
 (HERE/'results/finite_selection.json').write_text(json.dumps(out,indent=2));print('SELECTED',json.dumps(best),flush=True);print('CONTROLS',json.dumps(controls),flush=True)
if __name__=='__main__':main()
