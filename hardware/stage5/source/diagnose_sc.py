"""Record fidelity diagnostics without using them to select the hardware designs."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
data=np.load(ROOT/'inputs/evaluate_rows.npz')
z=data['score_words'].astype(np.int16).astype(np.float64)/1024
v=data['value_words'].astype(np.int16).astype(np.float64)/1024
assert np.array_equal(data['matrix_index'],np.arange(len(z))//16)
e=np.exp(z-z.max(-1,keepdims=True));a=e/e.sum(-1,keepdims=True)
reference=np.einsum('rk,rkd->rd',a,v[data['matrix_index']])
records=[]
for design in ['sexp16','sexp256','bern8','horner5','FROST_L64']:
    rows=[]
    for seed in [4004,5005,6006]:
        path=(ROOT/f'prior_frost/frost_L64_seed{seed}.csv' if design=='FROST_L64' else
              ROOT/f'results/{design}_evaluation_seed{seed}_rtl.csv')
        q=np.genfromtxt(path,delimiter=',',names=True)
        out=np.stack([q[f'word{k}'] for k in range(16)],-1).astype(np.uint32).view(np.int32)/1024
        err=out-reference
        rows.append(dict(seed=seed,RMSE=float(np.sqrt(np.mean(err**2))),MAE=float(np.mean(abs(err))),max_abs_error=float(abs(err).max())))
    records.append(dict(design=design,per_seed=rows,mean_RMSE=float(np.mean([r['RMSE'] for r in rows]))))
result={'scope':'Descriptive score-to-output error on the frozen 3200-row kernel corpus, relative to unbounded numerical softmax of the quantized input scores. No hardware accuracy matching or whole-network accuracy assertion.',
        'rows':len(z),'normalization': 'Inputs and outputs use Q10 units.', 'methods':records}
(ROOT/'results/fidelity_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
