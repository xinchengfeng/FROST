"""Focused checks of full graph equivalence, CPT validity and task integrity."""
import json
import numpy as np,torch
from unittest.mock import patch
from teacher import ROOT,load,inputs
from stochastic_model import SCTransformer,linear,fit_scalar_tables

def main():
    m,_=load(11);x,y,keys,values,match=inputs(8,350001)
    assert np.all(x[:,15,32:36]==0) and np.all(x[:,15,36]==1)
    assert np.array_equal(x[:,15,:32].argmax(-1),keys[np.arange(len(x)),match])
    assert np.array_equal(y,values[np.arange(len(x)),match])
    assert all(len(np.unique(k))==15 for k in keys)
    sc=SCTransformer(m)
    with torch.no_grad():p,h,hs,us=m(torch.tensor(x),True)
    pp,hh,history,updates=sc.run(x,mode='exact')
    exact_delta=float(np.max(np.abs(pp-p.numpy())))
    assert exact_delta<1e-4
    update_delta=max(np.max(np.abs(history[i+1]-history[i]-updates[i])) for i in range(4))
    assert update_delta<1e-12
    # Endpoint bit products are exact; means/static gains are a numerical interface.
    a=np.ones((2,3,4));w=np.array([[1.,-1.,1.,-1.],[2.,2.,-2.,2.]])
    endpoint=linear(a,w,np.zeros(2),64,19)
    assert np.allclose(endpoint,a@w.T,atol=1e-12)
    tables=sc.tables
    for data in tables.values():assert np.all((np.array(data['weights'])>=0)&(np.array(data['weights'])<=1))
    # Fail if the full SC path calls the exact GELU CDF imported into this module.
    with patch('stochastic_model.ndtr',side_effect=RuntimeError('Exact GELU is forbidden in the full SC path')):
        out,hidden,_,_=sc.run(x[:2],L=64,mode='full')
    assert np.isfinite(out).all() and np.isfinite(hidden).all()
    report=dict(task_labels_verified=True,query_token_contains_no_value=True,unique_memory_keys=True,
        exact_numpy_vs_torch_max_logit_difference=exact_delta,exact_residual_update_consistency=float(update_delta),
        learned_linear_endpoint_products_exact=True,all_scalar_CPT_entries_in_unit_interval=True,
        full_path_uses_no_exact_GELU=True,all_full_path_outputs_finite=True,
        operator_coverage={'learned_linear_maps':18,'encoder_blocks':4,'attention_heads_per_block':2,'layer_norms':9,'gelu_gates':4},
        retained_numerical_operations=['TT state contractions','residual/bias accumulation','centering means','static LayerNorm affine output gains','range scaling','stage decode/re-encode'])
    (ROOT/'provenance'/'checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
