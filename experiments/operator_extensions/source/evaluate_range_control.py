"""Secondary frozen-weight control separating a bound change from refitting.

Chosen after the primary dimension study, with no additional fitting, model
selection or test-driven hyperparameter choice. Every base fit is evaluated
at the already prespecified calibration bound of 8.
"""
from pathlib import Path
import csv, json
import numpy as np
import torch
from run_dimension_study import Operator, predict

ROOT=Path(__file__).resolve().parents[1]
rows=[]
for n in [16,32,64]:
    for seed in [11,22,33]:
        ck=torch.load(ROOT/'checkpoints'/f'dimension_n{n}_seed{seed}_base.pt',weights_only=True)
        model=Operator([ck['state_dict'][f'cs.{j}'] for j in range(n)],8)
        for family in ['gaussian','diffuse','sharp','correlated']:
            a=np.load(ROOT/'source_data'/f'dimension_n{n}_seed{seed}_base_{family}.npz')
            z=torch.tensor(a['scores']);p=torch.tensor(a['target']);y=predict(model,z)
            tv=.5*(y-p).abs().sum(-1)
            rows.append({'n':n,'seed':seed,'family':family,'mean_row_tv':tv.mean().item(),
                         'probability_rmse':(y-p).square().mean().sqrt().item()})
            np.savez_compressed(ROOT/'source_data'/f'dimension_n{n}_seed{seed}_range_only_{family}.npz',
                prediction=y.numpy(),row_tv=tv.numpy())
with (ROOT/'source_data/dimension_range_control.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
groups=[]
for n in [16,32,64]:
    for family in ['gaussian','diffuse','sharp','correlated']:
        r=[v for v in rows if v['n']==n and v['family']==family];g={'n':n,'family':family}
        for k in ['mean_row_tv','probability_rmse']:
            g[k+'_mean']=float(np.mean([v[k] for v in r]));g[k+'_fit_sd']=float(np.std([v[k] for v in r],ddof=1))
        groups.append(g)
(ROOT/'source_data/dimension_range_summary.json').write_text(json.dumps({'groups':groups},indent=2))
print('Range-only controls finished.',flush=True)
