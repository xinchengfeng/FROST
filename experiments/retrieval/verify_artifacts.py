"""Verify all recorded predictions, nested summaries, depth errors and figures."""
import json
import numpy as np,pandas as pd
from PIL import Image
from teacher import ROOT,inputs
from run_experiments import key,normerror

def main():
    data=ROOT/'source_data';df=pd.read_csv(data/'all_metrics.csv');depth=pd.read_csv(data/'depth_metrics.csv')
    jobs=json.loads((ROOT/'jobs.json').read_text());assert len(df)==len(jobs)==87
    assert not df.duplicated(['config','model_seed','L','stream_seed']).any()
    with np.load(data/'test_inputs.npz') as z:
        expected=inputs(128,340001)
        for name,arr in zip(['x','labels','keys','values','query_match'],expected):assert np.array_equal(z[name],arr)
    maximum=0.;residual=0.
    for _,r in df.iterrows():
        identifier=key(r)
        with np.load(data/'runs'/f'{identifier}.npz') as a:
            for name in a.files:assert np.isfinite(a[name]).all()
            pred,ref,y=a['prediction'],a['reference'],a['labels']
            assert pred.shape==ref.shape==(128,4)
            values=dict(accuracy=np.mean(pred.argmax(-1)==y),agreement=np.mean(pred.argmax(-1)==ref.argmax(-1)),
                logit_rmse=np.sqrt(np.mean((pred-ref)**2)),
                centered_logit_relative_error=normerror(ref-ref.mean(-1,keepdims=True),pred-pred.mean(-1,keepdims=True)),
                query_hidden_relative_error=normerror(a['reference_hidden'],a['query_hidden']))
            for name,v in values.items():maximum=max(maximum,abs(v-r[name]));assert abs(v-r[name])<1e-12
            rr=np.max(np.abs(np.diff(a['query_history'],axis=1)-a['query_updates']));residual=max(residual,float(rr));assert rr<1e-11
            d=depth[(depth.config==r['config'])&(depth.model_seed==r.model_seed)&(depth.L==r.L)&(depth.stream_seed==r.stream_seed)].sort_values('layer')
            assert len(d)==4
            for k,(_,dr) in enumerate(d.iterrows()):
                assert abs(normerror(a['reference_history'][:,k+1],a['query_history'][:,k+1])-dr.query_relative_error)<1e-12
                assert abs(normerror(a['reference_updates'][:,k],a['query_updates'][:,k])-dr.update_relative_error)<1e-12
    pm=pd.read_csv(data/'per_model_metrics.csv');summary=pd.read_csv(data/'summary_metrics.csv').set_index(['config','L'])
    metric='centered_logit_relative_error'
    for (c,L),g in df.groupby(['config','L']):
        assert set(g.model_seed)=={11,22,33}
        counts=g.groupby('model_seed').size();assert (counts==(1 if 'analytic' in c else 3)).all()
        vals=g.groupby('model_seed')[metric].mean()
        assert abs(vals.mean()-summary.loc[(c,L),metric+'_mean'])<1e-12
        assert abs(vals.std()-summary.loc[(c,L),metric+'_std'])<1e-12
    figures={}
    for p in sorted((ROOT/'figures').glob('*.png')):
        with Image.open(p) as im:im.load();figures[p.name]={'pixels':list(im.size),'dpi':im.info.get('dpi')}
    report=dict(verified_runs=len(df),test_arrays_match_generator=True,max_metric_recalculation_difference=maximum,
        max_residual_update_consistency_error=residual,nested_mean_and_SD_verified=True,all_depth_records_verified=True,figures=figures)
    (ROOT/'provenance'/'artifact_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
