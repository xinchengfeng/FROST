"""Check consistency of all scientific source data and exported figures."""
import io,json,math,zipfile
import numpy as np,pandas as pd
from PIL import Image
from attention_core import *
from run_experiments import case_definitions,key

def main():
    df=pd.read_csv(ROOT/'source_data'/'all_metrics.csv');jobs=json.loads((ROOT/'jobs.json').read_text())
    assert len(df)==len(jobs)==170
    assert not df.duplicated(['case','method','stage','L','seed']).any()
    assert (df.groupby(['case','method','stage','L']).size()==5).all()
    assert np.isfinite(df.select_dtypes('number')).all().all()
    references={};checked=0
    for name,cfg in case_definitions().items():
        with np.load(ROOT/'source_data'/f'inputs_{name}.npz') as d:
            expected=dataset(**cfg)
            for name2,arr in zip(['q','k','v'],expected):assert np.array_equal(d[name2],arr)
        references[name]=reference(*expected)
    max_rmse_delta=0.;max_rel_delta=0.
    for _,r in df.iterrows():
        ident=key(r);p=ROOT/'source_data'/'runs'/f'{ident}_per_matrix.npz'
        with np.load(p) as d:
            for k in d.files:assert d[k].shape==(128,) and np.isfinite(d[k]).all()
            rmse=float(np.sqrt(np.mean(d['rmse']**2)))
            y=references[r['case']][2]
            rel=rmse/float(np.sqrt(np.mean(y**2)))
            max_rmse_delta=max(max_rmse_delta,abs(rmse-r.rmse))
            max_rel_delta=max(max_rel_delta,abs(rel-r.rel_frob))
            assert abs(d['cosine'].mean()-r.cosine)<1e-12
            assert abs(rmse-r.rmse)<1e-12 and abs(rel-r.rel_frob)<1e-12
            checked+=1
    for method in ['bipolar_iid','improved']:
        with np.load(ROOT/'source_data'/f'fidelity_{method}.npz') as a:
            assert np.array_equal(a['y'],references['normal_d32'][2])
            calc,_=metrics(a['y'],a['prediction'])
            row=df[(df.case=='normal_d32')&(df.method==method)&(df.L==256)&(df.stage=='full')&(df.seed==101)].iloc[0]
            for metric in ['rmse','rel_frob','cosine']:assert abs(calc[metric]-row[metric])<1e-12
            assert np.array_equal((a['attention']*256).round(),a['attention']*256)
            assert np.all(a['attention'].sum(-1)==1.)
    # Five stochastic-source seeds give paired translation-equivariant errors.
    center=df[(df.method=='improved')&(df.stage=='full')&(df.L==256)]
    delta=0.
    for seed in [101,202,303,404,505]:
        b=center[(center.case=='normal_d32')&(center.seed==seed)].rmse.iloc[0]
        for off in [2,4,8]:
            c=center[(center.case==f'offset_{off}_d32')&(center.seed==seed)].rmse.iloc[0]
            delta=max(delta,abs(b-c))
    assert delta<1e-12
    figchecks={}
    for p in sorted((ROOT/'figures').glob('*.png')):
        with Image.open(p) as im:im.load();figchecks[p.name]={'pixels':list(im.size),'dpi':im.info.get('dpi')}
    report=dict(run_records_verified=checked,inputs_match_recorded_generators=True,
        max_aggregate_rmse_difference=max_rmse_delta,max_aggregate_relative_error_difference=max_rel_delta,
        value_offset_paired_rmse_max_difference=delta,all_source_rows_have_five_seeds=True,
        finite_attention_weights_are_actual_integer_counts=True,figures=figchecks)
    (ROOT/'provenance'/'artifact_checks.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
