import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import sys,json,hashlib,argparse,time,concurrent.futures
import numpy as np
from local_tt import LocalTT,QMCBernstein,StratifiedTT,StratifiedBernstein,HERE
from sc_baselines import ExpCountNorm
ROOT=Path(os.environ.get('FROST_RECON_ROOT',str(HERE/'legacy_runtime')))
sys.path.insert(0,str(ROOT/'source/scripts'))
from restoration_model import load_teacher,SCWindowRestorer
from run_quality30 import coords
from restoration_metrics import quality
import fast_finite
from scipy.special import softmax
class Capture:
 def __init__(self,op,expectation=False):self.op=op;self.records=[];self.expectation=expectation
 def stream(self,z,L,seed):
  a=self.op.analytic(z) if self.expectation else self.op.stream(z,L,seed);ref=softmax(z,axis=-1)
  self.records.append({'rows':z.size//16,'TV':float(np.abs(a-ref).sum(-1).mean()/2),'RMSE':float(np.sqrt(np.mean((a-ref)**2)))})
  return a

def run(job):
 iid,seed,method,ckpath=job;name=f'{iid}_{method}_S{seed}';output=HERE/'results'/f'{name}.npz'
 if output.exists():return json.loads(output.with_suffix('.json').read_text())
 fast_finite.install();teacher,_=load_teacher(ROOT/'checkpoints'/f'render_{iid}.pt');model=SCWindowRestorer(teacher,ROOT/'checkpoints'/f'operator_render_{iid}.pt')
 config=json.loads((HERE/'inputs/stage2_protocol.json').read_text());archived=np.load(ROOT/'runs'/f'image_{iid}.npz');truth=archived['truth']
 if method=='SExp':op=ExpCountNorm.from_config(config['sexp'])
 elif method=='Bernstein':op=ExpCountNorm.from_config(config['bernstein'])
 elif method=='Bernstein_QMC':op=QMCBernstein.from_config(config['bernstein'])
 elif method=='Bernstein_stratified':op=StratifiedBernstein.from_config(config['bernstein'])
 elif method=='Horner':
  from horner import HornerCountNorm
  hc=json.loads((HERE/'results/horner_selection.json').read_text())['selected']
  op=HornerCountNorm(hc['degree'],hc['factors'])
 elif method in ['LocalTT_stratified','LocalTT33_stratified']:op=StratifiedTT(ckpath)
 elif method in ['LocalTT','LocalTT_analytic_DIAGNOSTIC']:op=LocalTT(ckpath)
 elif method=='GeometricTT':
  # Load same FrozenHI with new row-stochastic geometric FSM cores.
  from attention_core import FrozenHI
  op=FrozenHI(ckpath)
 else:raise ValueError(method)
 cap=Capture(op,'DIAGNOSTIC' in method);model.hi=cap;actualseed=seed+iid*9001+11*17011
 start=time.perf_counter();pred=np.clip(model.run(coords(),L=4096,seed=actualseed,mode='full')[0],0,1).astype(np.float32)
 row={'image_id':iid,'method':method,'seed':seed,'L':4096,**quality(pred,truth,'reconstruction'),'elapsed_CPU_seconds':time.perf_counter()-start,'operator_diagnostics':cap.records,'checkpoint':str(ckpath) if ckpath else None,'checkpoint_SHA256':hashlib.sha256(Path(ckpath).read_bytes()).hexdigest() if ckpath else None,'scope':'fitted-image reconstruction, common frozen teacher and non-attention kernels'}
 if seed==101 and method in ['SExp','Bernstein']:
  old=np.load(HERE/'inputs'/f'stage2_pilot_{iid}.npz')[{'SExp':'SExp_CountNorm','Bernstein':'Bernstein_CountNorm'}[method]]
  assert np.array_equal(pred,old);row['stage2_replay_maxabs']=0.
 np.savez_compressed(output,prediction=pred,truth=truth);output.with_suffix('.json').write_text(json.dumps(row,indent=2));return row
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--method',required=True);ap.add_argument('--checkpoint',default='');ap.add_argument('--seeds',default='101,202,303');ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
 jobs=[(i,s,args.method,args.checkpoint) for i in [2018,43033,235098] for s in map(int,args.seeds.split(','))]
 with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
  for row in pool.map(run,jobs):print(json.dumps(row),flush=True)
