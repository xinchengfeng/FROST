from pathlib import Path
import sys,json
import numpy as np
from scipy.special import softmax
from local_tt import AntitheticTT,HERE
p=HERE
v=json.loads((p/'results/finite_selection.json').read_text());z=np.load(p/'results/calibration_rows.npz')['valid'][v['validation_row_indices']];target=softmax(z,axis=-1);rows=[]
for name in ['local33_r16_max8_aug.pt','local33_r32_max8_aug.pt']:
 op=AntitheticTT(p/'checkpoints'/name);errs=[];tvs=[]
 for seed in v['selection_stream_seeds']:
  a=op.stream(z,4096,seed);errs.append(np.mean((a-target)**2));tvs.append(np.abs(a-target).sum(-1).mean()/2)
 row={'checkpoint':name,'RMSE':float(np.sqrt(np.mean(errs))),'TV':float(np.mean(tvs)),'total_particles':4096,'replicas':2,'particles_per_replica':2048};rows.append(row);print(json.dumps(row),flush=True)
(p/'results/antithetic_validation.json').write_text(json.dumps(rows,indent=2))
