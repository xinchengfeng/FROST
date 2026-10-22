import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json
import numpy as np
import torch
from scipy.special import softmax
ROOT=Path(os.environ.get('FROST_RECON_ROOT',str(Path(__file__).resolve().parents[1]/'legacy_runtime')))
OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source/scripts'))
from restoration_model import load_teacher,SCWindowRestorer
from run_quality30 import coords
from restoration_metrics import quality
import fast_finite
fast_finite.install()
class Diagnostic:
 def __init__(self,orig,kind):self.orig=orig;self.kind=kind;self.log=[]
 def stream(self,z,L,seed):
  exact=softmax(z,axis=-1);ana=self.orig.analytic(z)
  self.log.append({'rows':z.size//16,'analytic_tv':float(np.abs(ana-exact).sum(-1).mean()/2),'analytic_rmse':float(np.sqrt(np.mean((ana-exact)**2))),'score_percentiles':np.percentile(z-z.mean(-1,keepdims=True),[0,1,5,50,95,99,100]).tolist()})
  return exact if self.kind=='exact_softmax_DIAGNOSTIC' else ana
records=[]
for iid in [2018,43033,235098]:
 teacher,_=load_teacher(ROOT/'checkpoints'/f'render_{iid}.pt');model=SCWindowRestorer(teacher,ROOT/'checkpoints'/f'operator_render_{iid}.pt');orig=model.hi
 truth=np.load(ROOT/'runs'/f'image_{iid}.npz')['truth']
 for kind in ['expected_TT_DIAGNOSTIC','exact_softmax_DIAGNOSTIC']:
  model.hi=Diagnostic(orig,kind)
  pred=np.clip(model.run(coords(),L=4096,seed=101+iid*9001+11*17011,mode='full')[0],0,1).astype(np.float32)
  row={'id':iid,'kind':kind,**quality(pred,truth,'reconstruction'),'operator':model.hi.log};records.append(row)
  print(json.dumps(row),flush=True)
(OUT/'results/diagnosis.json').write_text(json.dumps(records,indent=2))
# Calibration activation rows from disjoint image IDs, no target pixels loaded.
ids=sorted(int(p.stem.split('_')[1]) for p in (ROOT/'checkpoints').glob('render_*.pt') if 'resume' not in p.stem and int(p.stem.split('_')[1]) not in [2018,43033,235098])
trainids=ids[:12];validids=ids[12:16];parts={}
for split,used in [('train',trainids),('valid',validids)]:
 arr=[]
 for iid in used:
  teacher,_=load_teacher(ROOT/'checkpoints'/f'render_{iid}.pt')
  with torch.no_grad():_,scores=teacher(torch.tensor(coords()),trace=True)
  arr.extend(z.numpy().reshape(-1,16) for z in scores)
 parts[split]=np.concatenate(arr)
np.savez_compressed(OUT/'results/calibration_rows.npz',**parts)
(OUT/'results/calibration_protocol.json').write_text(json.dumps({'training_image_ids':trainids,'validation_image_ids':validids,'pilot_image_ids':[2018,43033,235098],'training_rows':len(parts['train']),'validation_rows':len(parts['valid']),'selection':'independent-image activation probability error; pilot image PSNR is diagnostic only','frozen_weights':True,'L':4096,'seeds':[101,202,303]},indent=2))
print('CALIBRATION',trainids,validids,[(k,v.shape) for k,v in parts.items()],flush=True)
