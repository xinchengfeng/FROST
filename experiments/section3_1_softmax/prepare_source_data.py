"""Extract measured data for Section 3.1. No model fitting or data smoothing."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np,pandas as pd,torch

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'frost_direct_hi_smurf'
DATA=ROOT/'source_data'

def main():
    model='categorical_shared_sorted_center_n16_s16_r8_b4_seed11'
    path=OLD/'results'/f'{model}_predictions.npz'
    v=np.load(path);p=v['target'].astype(float);q=v['prediction'].astype(float)
    row,entry=np.indices(p.shape)
    pd.DataFrame(dict(row=row.ravel(),entry=entry.ravel(),reference=p.ravel(),prediction=q.ravel())).to_csv(DATA/'Fig3a_probabilities.csv',index=False)
    tv=.5*np.abs(q-p).sum(-1)
    rows=pd.DataFrame(dict(row=np.arange(len(p)),tv=tv,rmse=np.sqrt(np.mean((q-p)**2,-1)),reference_max=p.max(-1)))
    rows.to_csv(DATA/'Fig3b_row_errors.csv',index=False)
    configs=pd.read_csv(OLD/'results'/'categorical_metrics.csv')
    configs['label']=[f'N={n}, r={r}' for n,r in zip(configs.states,configs['rank'])]
    configs.to_csv(DATA/'Fig3d_model_comparison.csv',index=False)
    history=[];inputs=[path,OLD/'results'/'categorical_metrics.csv']
    for _,c in configs.iterrows():
        name=f"categorical_shared_sorted_center_n16_s{int(c.states)}_r{int(c['rank'])}_b4_seed11"
        file=OLD/'checkpoints'/f'{name}.pt';inputs.append(file)
        ck=torch.load(file,weights_only=True)
        for h in ck['history']:
            history.append(dict(N=int(c.states),rank=int(c['rank']),seed=11,step=h['step'],validation_rmse=np.sqrt(h['validation_mse'])))
    pd.DataFrame(history).to_csv(DATA/'Fig3c_validation.csv',index=False)
    stream=pd.read_csv(OLD/'results'/'direct_stream_metrics.csv')
    stream[stream['mode']=='categorical_bits'].to_csv(DATA/'Fig3e_output_error.csv',index=False)
    stream.to_csv(DATA/'Fig3f_sampling_error.csv',index=False)
    protocol=json.loads((OLD/'results'/'stream_protocol.json').read_text())
    summary=stream.groupby(['mode','L'])[['rmse','half_row_l1','sampling_rmse']].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.to_csv(DATA/'stream_summary.csv')
    main=configs[(configs.states==16)&(configs['rank']==8)].iloc[0]
    low=configs[(configs.states==16)&(configs['rank']==4)].iloc[0]
    # Fixed threshold diagnostic, not used to select the model or test rows.
    mask=p>.4
    numbers=dict(selected_model=model,test_rows=len(p),entries=int(p.size),dimension=p.shape[1],
        rmse=float(np.sqrt(np.mean((q-p)**2))),mean_tv=float(tv.mean()),median_tv=float(np.median(tv)),
        tv_p95=float(np.quantile(tv,.95)),tv_max=float(tv.max()),
        top1=float(main.top1),sum_error=float(main.sum_error),
        parameters=int(main.parameters),rank_parameter_factor=float(main.parameters/low.parameters),
        rank_rmse_reduction_percent=float(100*(1-main.rmse/low.rmse)),
        high_probability_threshold=.4,high_probability_entry_count=int(mask.sum()),
        high_probability_mean_signed_error=float((q-p)[mask].mean()),
        stream_analytic_rmse=float(protocol['analytic_subset']['rmse']),
        stream_analytic_tv=float(protocol['analytic_subset']['half_row_l1']),
        training_initializations=1,stream_randomizations=3,stream_rows=128,burn_in=256)
    (DATA/'numerical_summary.json').write_text(json.dumps(numbers,indent=2))
    for src in [OLD/'protocol.json',OLD/'environment.json',OLD/'results'/'checks.json',OLD/'results'/'stream_protocol.json']:
        shutil.copyfile(src,ROOT/'provenance'/src.name)
    inputs+=[OLD/'results'/'direct_stream_metrics.csv',OLD/'results'/'stream_protocol.json']
    (ROOT/'provenance'/'input_sha256.json').write_text(json.dumps({str(p.relative_to(OLD)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},indent=2))
    print(json.dumps(numbers,indent=2))

if __name__=='__main__':main()
