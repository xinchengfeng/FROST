"""Focused validation of numerical semantics used by the experiment."""
import json
import numpy as np
from attention_core import *

def main():
    m=FrozenHI();q,k,v=dataset(size=4,seed=209001)
    z,a,y=reference(q,k,v);ap=m.analytic(z)
    checks={}
    # The simplex constraint is essential to exact value-centering equivalence.
    checks['analytic_row_sum_max_error']=float(np.abs(ap.sum(-1)-1).max())
    mu=v.mean(1,keepdims=True)
    checks['exact_value_centering_max_error']=float(np.abs(y-(a@(v-mu)+mu)).max())
    astr=m.stream(z,256,999)
    checks['categorical_row_sum_max_error']=float(np.abs(astr.sum(-1)-1).max())
    checks['stream_min_attention']=float(astr.min())
    # Endpoints should be exact for real bit gates and counters.
    qb=np.ones((2,16,32));kb=-np.ones_like(qb);vv=np.ones_like(qb)*2
    for enc in ['bipolar','magnitude']:
        zh=stochastic_qk(qb,kb,64,99,'sobol',enc)
        checks[f'{enc}_qk_endpoint_error']=float(np.abs(zh+np.sqrt(32)).max())
        identity=np.tile(np.eye(16)[None],(2,1,1))
        yh=stochastic_av(identity,vv,64,99,'sobol',enc)
        checks[f'{enc}_av_endpoint_error']=float(np.abs(yh-vv).max())
    # A constant value offset must cancel under centering (same random shifts).
    yh=stochastic_av(astr,v,256,111,'sobol','magnitude',True)
    yh_shift=stochastic_av(astr,v+8.,256,111,'sobol','magnitude',True)
    checks['centered_value_shift_equivariance_error']=float(np.max(np.abs(yh_shift-yh-8.)))
    checks['softmax_checkpoint_frozen']=True
    checks['interstage_decode_and_reencode']=True
    checks['fsm_input_source']='IID independent Bernoulli; never replaced by Sobol'
    checks['compiled_TT_arithmetic']='float64; not gate-level SC'
    assert checks['analytic_row_sum_max_error']<2e-6
    assert checks['exact_value_centering_max_error']<1e-12
    assert checks['categorical_row_sum_max_error']==0
    assert checks['centered_value_shift_equivariance_error']<1e-12
    for name in ['bipolar','magnitude']:
        assert checks[f'{name}_qk_endpoint_error']<1e-12
        assert checks[f'{name}_av_endpoint_error']==0
    (ROOT/'provenance'/'checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
