"""Verify the joint CPT construction and run actual FSM/categorical emission."""
import json,time
import numpy as np,pandas as pd,torch
from unittest.mock import patch
from direct_study import ROOT,data,basis,batch,measures,tag
from categorical_study import CategoricalTT

def main():
    metrics=pd.read_csv(ROOT/'results'/'categorical_metrics.csv')
    chosen=metrics.loc[metrics.validation_rmse.idxmin()]
    c={k:chosen[k] for k in ['n','states','rank','heads','encoding','bound','seed','steps']}
    for k in ['n','states','rank','seed','steps']:c[k]=int(c[k])
    name='categorical_'+tag(c)
    ck=torch.load(ROOT/'checkpoints'/f'{name}.pt',weights_only=True)
    m=CategoricalTT(**{k:c[k] for k in ['n','states','rank','heads','encoding','bound']})
    m.load_state_dict(ck['state_dict']);m.prepare()
    z=data(16,128,140016);p=z.softmax(-1);analytic=batch(m,z)
    x,order=m.encode(z)
    rows=[];lengths=[64,256,1024];burn=256
    start=time.time()
    for seed in [101,202,303]:
        gen=torch.Generator().manual_seed(seed)
        state=torch.full(x.shape,(m.states-1)//2,dtype=torch.long)
        counts=torch.zeros_like(z);weighted=torch.zeros_like(z)
        for t in range(burn+max(lengths)):
            bits=torch.rand(x.shape,generator=gen)<x
            state=(state+2*bits.long()-1).clamp(0,m.states-1)
            if t<burn:continue
            with torch.no_grad():w=m.selected(state)
            # Floating rounding affects only the final cumulative boundary;
            # categorical emission produces exactly one event each cycle.
            cdf=w.cumsum(-1);u=torch.rand((len(z),1),generator=gen)
            event=(cdf<u).sum(-1).clamp_max(m.n-1)
            counts.scatter_add_(1,event[:,None],torch.ones(len(z),1));weighted+=w
            length=t-burn+1
            if length in lengths:
                for mode,q in [('categorical_bits',counts/length),('weighted_diagnostic',weighted/length)]:
                    q=torch.zeros_like(q).scatter(1,order,q)
                    rows.append({'model':name,'seed':seed,'L':length,'burn_in':burn,'rows':len(z),'mode':mode,
                        'sampling_rmse':(q-analytic).square().mean().sqrt().item(),**measures(p,q)})
        print('simulated',seed,flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'results'/'direct_stream_metrics.csv',index=False)
    (ROOT/'results'/'stream_protocol.json').write_text(json.dumps(dict(model=name,selection='smallest validation RMSE among 3 categorical configurations',
        test_rows=128,logits_seed=140016,seeds=[101,202,303],input_source='IID independent Bernoulli per variable and cycle',
        initialization='central state, followed by 256 actual FSM cycles of burn-in',
        reported_L='output sampling cycles only; total cycles = 256 + L',
        analytic_subset=measures(p,analytic),seconds=time.time()-start,
        scope='Actual FSM trajectories and categorical output events, floating compiled TT contractions. Not gate-level TT arithmetic or a timing estimate.'),indent=2))
    # Exact small-tensor enumeration validates the expected-value formula.
    torch.manual_seed(7103);small=CategoricalTT(n=3,states=3,rank=2,encoding='center').prepare()
    zz=data(3,7,7991);xx,_=small.encode(zz);pi=basis(xx,3)
    states=torch.cartesian_prod(torch.arange(3),torch.arange(3),torch.arange(3))
    joint=torch.ones(len(xx),len(states))
    for j in range(3):joint*=pi[:,j,states[:,j]]
    with torch.no_grad():ww=small.selected(states);explicit=joint@ww;contracted=small(zz)
    err=(explicit-contracted).abs().max().item()
    # Parameter transforms happen only before this block, during compilation.
    with patch.object(torch.Tensor,'softmax',side_effect=RuntimeError('runtime Softmax forbidden')),patch.object(torch.Tensor,'sigmoid',side_effect=RuntimeError('runtime sigmoid forbidden')),patch('torch.exp',side_effect=RuntimeError('runtime exp forbidden')):
        m(z[:8]);m.selected(torch.zeros(8,16,dtype=torch.long))
    checks=dict(enumerated_expectation_max_difference=err,
        enumerated_state_min=ww.min().item(),enumerated_state_max=ww.max().item(),
        enumerated_state_sum_max_error=(ww.sum(-1)-1).abs().max().item(),
        compiled_forward_uses_no_softmax_sigmoid_exp=True,
        categorical_finite_rows_sum_exactly_one=all(r['sum_error']==0 for r in rows if r['mode']=='categorical_bits'))
    assert err<3e-7 and checks['categorical_finite_rows_sum_exactly_one']
    (ROOT/'results'/'checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks),flush=True)

if __name__=='__main__':main()
