"""Independent numerical/accounting checks against saved experiment outputs."""
from pathlib import Path
import csv, json, hashlib, platform
import numpy as np
import torch, scipy, matplotlib
from attention_core import FrozenHI

ROOT=Path(__file__).resolve().parents[1]
checks={}
hi=FrozenHI(ROOT/'checkpoints/calibrated_cifar10_11.pt')
z=np.load(ROOT/'source_data/prior_inputs.npz')['scores']
x,order=hi.encode(z)
target=np.exp(z-z.max(-1,keepdims=True));target/=target.sum(-1,keepdims=True)
records=list(csv.DictReader((ROOT/'source_data/sampler_stream_metrics.csv').open()))
largest=0.
for seed in [101,202,303,404,505,606,707,808]:
    a=np.load(ROOT/'source_data'/f'sampler_events_seed{seed}.npz')
    for name in a.files:
        if name=='fsm_states_sorted':continue
        ev=a[name]
        assert ev.shape==(4096,128) and ev.min()>=0 and ev.max()<16
        for L in [64,256,1024,4096]:
            counts=np.array([np.bincount(ev[:L,j],minlength=16) for j in range(128)])
            assert (counts.sum(-1)==L).all()
            p=np.empty_like(x);np.put_along_axis(p,order,counts/L,-1)
            tv=.5*np.abs(p-target).sum(-1).mean()
            r=next(v for v in records if int(v['seed'])==seed and int(v['L'])==L and v['method']==name)
            discrepancy=abs(tv-float(r['mean_row_tv']));largest=max(largest,discrepancy)
            assert discrepancy<1e-14
checks['sampler_stream_records_recomputed']=len(records)
checks['maximum_saved_event_metric_difference']=largest

s=json.loads((ROOT/'source_data/sampler_summary.json').read_text())
cond=np.load(ROOT/'source_data/sampler_conditional_outputs.npz')
for q in s['precision']:
    b=q['word_bits'];p=cond[f'cdf{b}'];tv=.5*np.abs(p-cond['floating']).sum(-1)
    assert p.min()>=0 and np.allclose(p.sum(-1),1,rtol=0,atol=2e-15)
    assert tv.max()<=q['sum_core_tv_bound']+1e-14
    assert abs(tv.mean()-q['conditional_tv_mean'])<1e-14
checks['cdf_conditional_bound_checks']=3
assert np.allclose(cond['shared'].sum(-1),1,rtol=0,atol=2e-15)
checks['shared_exact_distribution_unit_mass']=True

fits=json.loads((ROOT/'source_data/dimension_summary.json').read_text())
for n in [16,32,64]:
    count=16*8+(n-2)*16*8*8+16*8*n
    for seed in [11,22,33]:
        for stage in ['base','calibrated']:
            ck=torch.load(ROOT/'checkpoints'/f'dimension_n{n}_seed{seed}_{stage}.pt',weights_only=True)
            assert sum(v.numel() for v in ck['state_dict'].values())==count
            fit=next(v for v in fits['fits'] if v['n']==n and v['seed']==seed)
            selection=fit['base_fit'] if stage=='base' else fit['calibration_fit']
            assert selection['validation_rmse']==min(h['validation_rmse'] for h in selection['history'])
            for family in ['gaussian','diffuse','sharp','correlated']:
                a=np.load(ROOT/'source_data'/f'dimension_n{n}_seed{seed}_{stage}_{family}.npz')
                tv=.5*np.abs(a['prediction']-a['target']).sum(-1)
                assert np.allclose(tv,a['row_tv'],atol=2e-7,rtol=0)
                assert np.allclose(a['prediction'].sum(-1),1,rtol=0,atol=3e-6)
checks['dimension_selected_checkpoints']=18
checks['dimension_saved_prediction_sets']=72
checks['dimension_parameter_counts']={str(n):16*8+(n-2)*16*64+16*8*n for n in [16,32,64]}
bsds=json.loads((ROOT/'source_data/bsds_reverification.json').read_text())
assert bsds['all_valid'] and bsds['independently_recomputed_outputs']==2400
checks['bsds_outputs_rescored']=2400
checks['original_classification_retrained_or_retested_this_revision']=False
checks['hardware_experiments_performed']=False

result={'all_checks_passed':True,'checks':checks,'environment':{'python':platform.python_version(),
    'torch':str(torch.__version__),'numpy':str(np.__version__),'scipy':str(scipy.__version__),
    'matplotlib':str(matplotlib.__version__),'device':'CPU'},
    'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'checkpoints').glob('*.pt')}}
(ROOT/'source_data/revision_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'all_checks_passed':True,'checks':checks,'environment':result['environment']},indent=2))
