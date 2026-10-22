"""Frozen protocol: calibrations and main/stress test data are disjoint."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import io,json,time,hashlib,platform,concurrent.futures
import numpy as np,pandas as pd,torch,scipy
from attention_core import *

CONFIGS={
    'bipolar_iid':dict(source='iid',encoding='bipolar',center=False,gain=1.),
    'magnitude_iid':dict(source='iid',encoding='magnitude',center=False,gain=1.),
    'magnitude_sobol':dict(source='sobol',encoding='magnitude',center=False,gain=1.),
    'improved':dict(source='sobol',encoding='magnitude',center=True,gain=1.)}
SEEDS=[101,202,303,404,505]
LENGTHS=[64,128,256,512,1024]

def save_npz(path,**arrays):
    buf=io.BytesIO();np.savez_compressed(buf,**arrays);path.write_bytes(buf.getvalue())

def case_definitions():
    cases={'normal_d32':dict(size=128,d=32,seed=220001,family='normal')}
    for dim in [16,64]:cases[f'normal_d{dim}']=dict(size=128,d=dim,seed=220000+dim,family='normal')
    for j,f in enumerate(['diffuse','sharp','correlated']):cases[f'{f}_d32']=dict(size=128,d=32,seed=230001+j,family=f)
    for offset in [2,4,8]:cases[f'offset_{offset}_d32']=dict(size=128,d=32,seed=220001,family=f'offset_{offset}')
    return cases

def key(j):return f"{j['case']}_{j['method']}_{j['stage']}_L{j['L']}_seed{j['seed']}"

def task(j):
    start=time.time();model=FrozenHI()
    # Deterministic regeneration avoids concurrent reads through remote mounts.
    q,k,v=dataset(**case_definitions()[j['case']])
    agg,per,arrays=run_core(q,k,v,model,j['L'],j['seed'],stage=j['stage'],**CONFIGS[j['method']])
    row={**j,**agg,'seconds':time.time()-start}
    ident=key(j)
    (ROOT/'source_data'/'runs'/f'{ident}.json').write_text(json.dumps(row,indent=2))
    save_npz(ROOT/'source_data'/'runs'/f'{ident}_per_matrix.npz',**per)
    if j['case']=='normal_d32' and j['L']==256 and j['stage']=='full' and j['seed']==101 and j['method'] in ['bipolar_iid','improved']:
        save_npz(ROOT/'source_data'/f"fidelity_{j['method']}.npz",**arrays)
    return row

def main():
    (ROOT/'source_data'/'runs').mkdir(exist_ok=True)
    cases=case_definitions()
    jobs=[]
    for case,c in cases.items():
        q,k,v=dataset(**c)
        save_npz(ROOT/'source_data'/f'inputs_{case}.npz',q=q,k=k,v=v)
        methods=['bipolar_iid','improved'] if not case.startswith('offset') else ['magnitude_sobol','improved']
        lengths=LENGTHS if case=='normal_d32' else [256]
        for method in methods:
            for L in lengths:
                for seed in SEEDS:jobs.append(dict(case=case,method=method,L=L,seed=seed,stage='full',matrices=c['size'],d=c['d']))
    for method in ['magnitude_iid','magnitude_sobol']:
        for seed in SEEDS:jobs.append(dict(case='normal_d32',method=method,L=256,seed=seed,stage='full',matrices=128,d=32))
    for stage in ['qk_only','hi_only','av_only']:
        for method in ['bipolar_iid','improved']:
            for seed in SEEDS:jobs.append(dict(case='normal_d32',method=method,L=256,seed=seed,stage=stage,matrices=128,d=32))
    assert len({key(j) for j in jobs})==len(jobs)
    protocol=dict(scope='Unmasked, single-head scaled dot-product attention core from supplied Q,K,V to Y. Input/output projections, multi-head concatenation, residuals, LayerNorm, training and task accuracy are outside scope.',
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        n=16,cases=cases,configs=CONFIGS,seeds=SEEDS,lengths=LENGTHS,jobs=len(jobs),
        burn_in=256,total_sequential_bit_windows='QK L + FSM (256+L) + AV L = 256+3L; not wall-clock latency',
        baseline='Bipolar IID QK; IID unipolar-attention/bipolar-value AV. Same frozen HI-SMURF and power-of-two scaling as all variants.',
        softmax='Frozen joint categorical HI-SMURF from Section 3.1, 16 states, rank 8, 16512 parameters, one training seed 11. No additional fitting.',
        preprocessing='Per-Q/K-vector and per-V-feature power-of-two max-absolute scaling; no Q/K/V clipping. Softmax sorted centered scores encoded within [-4,4].',
        calibration='Gain grid 1.0 through 1.25 on 128 separate matrices selected 1.0, so no gain correction. Value centering checked on separate development matrices at offsets 0,4,8.',
        rng='Two distinct coordinates of one scrambled 2-D Sobol net, independent per-stream 32-bit digital shifts. IID FSM input drives and IID categorical emissions unchanged.',
        stage_controls='Replace only QK, HI-SMURF or AV and keep remaining operations exact; controls isolate errors but are NOT additive decompositions.',
        uncertainty='Mean and sample SD across 5 stream/digital-shift seeds, fixed checkpoint and fixed input matrices. Matrix entries are not independent replicates.',
        exclusions='No real-task data, gate-level TT arithmetic, synthesis, PPA, hardware timing, causal masks, or variable token counts.')
    if not (ROOT/'protocol.json').exists():
        (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (ROOT/'jobs.json').write_text(json.dumps(jobs,indent=2))
    (ROOT/'provenance'/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,scipy=scipy.__version__,platform=platform.platform(),gpu=False,workers=3),indent=2))
    # The protocol is on disk before any held-out result is evaluated.
    analytic=[];model=FrozenHI()
    for case in cases:
        q,k,v=dataset(**cases[case]);z,a,y=reference(q,k,v)
        ah=model.analytic(z);agg,per=metrics(y,ah@v)
        centered=z-z.mean(-1,keepdims=True)
        analytic.append(dict(case=case,**agg,attention_tv=float(np.mean(.5*np.abs(ah-a).sum(-1))),score_clip_fraction=float(np.mean(np.abs(centered)>4)),mean_max_attention=float(a.max(-1).mean())))
        if case=='normal_d32':
            save_npz(ROOT/'source_data'/'analytic_main.npz',y=y,prediction=ah@v,a=a,attention=ah,**per)
    pd.DataFrame(analytic).to_csv(ROOT/'source_data'/'analytic_metrics.csv',index=False)
    rows=[];pending=[];start=time.time()
    for j in jobs:
        path=ROOT/'source_data'/'runs'/f'{key(j)}.json'
        if path.exists():
            perpath=ROOT/'source_data'/'runs'/f'{key(j)}_per_matrix.npz'
            try:
                with np.load(perpath) as archive:
                    for name in archive.files:assert np.isfinite(archive[name]).all()
                rows.append(json.loads(path.read_text()));continue
            except (ValueError,EOFError,OSError,AssertionError):pass
        pending.append(j)
    print(f'Resuming {len(rows)} verified results; {len(pending)} remaining',flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as ex:
        futures={ex.submit(task,j):j for j in pending}
        for f in concurrent.futures.as_completed(futures):
            row=f.result();rows.append(row)
            if len(rows)%10==0 or len(rows)==len(jobs):print(f"Completed {len(rows)}/{len(jobs)}; elapsed {time.time()-start:.1f}s",flush=True)
    df=pd.DataFrame(rows).sort_values(['case','stage','method','L','seed']);df.to_csv(ROOT/'source_data'/'all_metrics.csv',index=False)
    grouped=df.groupby(['case','stage','method','L'])[['rmse','rel_frob','cosine','score_rmse','attention_tv']].agg(['mean','std'])
    grouped.columns=['_'.join(t) for t in grouped.columns];grouped.reset_index().to_csv(ROOT/'source_data'/'summary_metrics.csv',index=False)
    print(grouped.loc['normal_d32'].to_string(),flush=True)
    (ROOT/'provenance'/'run_complete.json').write_text(json.dumps(dict(jobs_completed=len(rows),seconds=time.time()-start,all_rows_sum_exactly_one=bool((df.attention_sum_error<1e-12).all())),indent=2))

if __name__=='__main__':main()
