"""Frozen complete-graph experiment with nested model and stream repetitions."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import io,json,time,platform,concurrent.futures
import numpy as np,pandas as pd,torch,scipy
from teacher import ROOT,load,inputs
from stochastic_model import SCTransformer

MODELS=[11,22,33];STREAMS=[101,202,303];LENGTHS=[64,256,1024]
CONFIGS={
 'generic_full':dict(mode='full',calibrated=False,source='sobol',encoding='magnitude'),
 'calibrated_full':dict(mode='full',calibrated=True,source='sobol',encoding='magnitude'),
 'calibrated_iid':dict(mode='full',calibrated=True,source='iid',encoding='magnitude'),
 'attention_only':dict(mode='attention_only',calibrated=True,source='sobol',encoding='magnitude'),
 'linear_attention':dict(mode='linear_attention',calibrated=True,source='sobol',encoding='magnitude'),
 'generic_analytic':dict(mode='analytic',calibrated=False,source='sobol',encoding='magnitude'),
 'calibrated_analytic':dict(mode='analytic',calibrated=True,source='sobol',encoding='magnitude')}

def save_npz(path,**arrays):
 buf=io.BytesIO();np.savez_compressed(buf,**arrays);path.write_bytes(buf.getvalue())

def key(j):return f"{j['config']}_model{j['model_seed']}_L{j['L']}_stream{j['stream_seed']}"

def normerror(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30))

def task(j):
 start=time.time();m,_=load(j['model_seed']);conf=CONFIGS[j['config']].copy();cal=conf.pop('calibrated')
 selection=json.loads((ROOT/'calibration_selection.json').read_text())
 hi_path=ROOT/'checkpoints'/selection[str(j['model_seed'])] if cal else None
 sc=SCTransformer(m,hi_path);x,y,_,_,_=inputs(128,340001)
 preds=[];hidden=[];histories=[];updates=[]
 for i in range(0,len(x),8):
  p,h,hs,us=sc.run(x[i:i+8],j['L'],j['stream_seed']+i*9001,**conf)
  preds.append(p);hidden.append(h[:,-1]);histories.append(np.stack(hs,1)[:,:,-1]);updates.append(np.stack(us,1)[:,:,-1])
 pred=np.concatenate(preds);hidden=np.concatenate(hidden);hist=np.concatenate(histories);upd=np.concatenate(updates)
 with torch.no_grad():p,h,hs,us=m(torch.tensor(x),True)
 ref=p.numpy().astype(np.float64);rh=h.numpy()[:,-1];rhs=torch.stack(hs,1).numpy()[:,:,-1]
 rus=torch.stack([u['attn_update']+u['ffn_update'] for u in us],1).numpy()[:,:,-1]
 cr=ref-ref.mean(-1,keepdims=True);cp=pred-pred.mean(-1,keepdims=True)
 cos=(hidden*rh).sum(-1)/(np.linalg.norm(hidden,axis=-1)*np.linalg.norm(rh,axis=-1))
 row={**j,'examples':len(x),'accuracy':float(np.mean(pred.argmax(-1)==y)),
      'agreement':float(np.mean(pred.argmax(-1)==ref.argmax(-1))),
      'reference_accuracy':float(np.mean(ref.argmax(-1)==y)),
      'logit_rmse':float(np.sqrt(np.mean((pred-ref)**2))),
      'centered_logit_relative_error':normerror(cr,cp),'query_hidden_relative_error':normerror(rh,hidden),
      'query_hidden_cosine':float(cos.mean()),'seconds':time.time()-start}
 identifier=key(j)
 (ROOT/'source_data'/'runs'/f'{identifier}.json').write_text(json.dumps(row,indent=2))
 save_npz(ROOT/'source_data'/'runs'/f'{identifier}.npz',prediction=pred,reference=ref,labels=y,query_hidden=hidden,reference_hidden=rh,
          query_history=hist,reference_history=rhs,query_updates=upd,reference_updates=rus)
 depth=[]
 for k in range(1,5):depth.append({**j,'layer':k,'query_relative_error':normerror(rhs[:,k],hist[:,k]),'update_relative_error':normerror(rus[:,k-1],upd[:,k-1])})
 (ROOT/'source_data'/'runs'/f'{identifier}_depth.json').write_text(json.dumps(depth,indent=2))
 return row,depth

def main():
 (ROOT/'source_data'/'runs').mkdir(exist_ok=True)
 jobs=[]
 for c in CONFIGS:
  lens=LENGTHS if c in ['generic_full','calibrated_full'] else [256]
  seeds=[0] if 'analytic' in c else STREAMS
  for model in MODELS:
   for L in lens:
    for stream in seeds:jobs.append(dict(config=c,model_seed=model,L=L,stream_seed=stream))
 assert len(jobs)==87
 protocol=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
   architecture=dict(type='encoder-only pre-LayerNorm Transformer',tokens=16,input_width=37,d_model=32,heads=2,head_width=16,ff_width=64,blocks=4,classes=4,teacher_parameters=36100,
      subgraphs='learned input projection + position embedding + four [LN, QKV projections, multi-head attention, output projection, residual, LN, FFN(GELU), residual] blocks + final LN + query-token classifier'),
   task='Synthetic associative retrieval: 15 unique keys sampled from 32, independent 4-class values, final token queries one stored key. Output that key value. Chance 25%. No query value is exposed.',
   train='Three independent teacher initializations 11,22,33; 1800 AdamW steps on fresh batches of 64. Validation seed 310001, 512 sequences. Select accuracy then cross-entropy.',
   calibration='Reference activations on 128 sequences seed 330001; validation 128 seed 330002. Same-size shared HI-SMURF, 16 states/rank8/16512 parameters; 1200 Adam steps for bounds 4 and 8; select smallest validation probability RMSE. Teacher weights frozen. No task labels used.',
   development_seeds=[320001,320003],test_seed=340001,test_sequences=128,model_seeds=MODELS,stream_seeds=STREAMS,Ls=LENGTHS,configs=CONFIGS,jobs=jobs,
   nonlinear='Softmax: actual 16-state FSM trajectories and categorical events, 256 burn-in. GELU: x times an 8-state fitted CDF gate. LayerNorm: SC variance then a 4-state reciprocal-square-root gate. Scalar FSM burn-in 64. Fitted CPTs bounded in [0,1].',
   arithmetic='Sign-magnitude bit products for all 18 learned linear maps, QK and AV. IID or independent digital shifts of distinct 2D Sobol coordinates; FSM inputs remain IID Bernoulli. Weights share streams within a batch of eight sequences.',
   retained_numerical='Residual and bias additions, centering means, static LN affine gains, power-of-two range scaling, TT contractions, and stage decode/re-encode. No claim of all-bitstream or gate-level hardware.',
   uncertainty='First average three stream runs within each teacher, then report mean and sample SD of the three teacher means. Stream runs are nested, not nine independent trained models. Analytic controls have one run per teacher.',
   exclusions='No decoder/cross-attention, causal mask, pretrained large model, real vision/language task, hardware PPA, or execution-time speedup claim.')
 (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
 (ROOT/'jobs.json').write_text(json.dumps(jobs,indent=2))
 (ROOT/'provenance'/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),cpu_only=True,workers=3),indent=2))
 x,y,k,v,ix=inputs(128,340001);save_npz(ROOT/'source_data'/'test_inputs.npz',x=x,labels=y,keys=k,values=v,query_match=ix)
 teachers=[]
 for seed in MODELS:
  m,ck=load(seed)
  with torch.no_grad():pred=m(torch.tensor(x)).numpy()
  teachers.append(dict(model_seed=seed,test_accuracy=float(np.mean(pred.argmax(-1)==y)),parameters=sum(a.numel() for a in m.parameters()),best_step=ck['best_step']))
 pd.DataFrame(teachers).to_csv(ROOT/'source_data'/'reference_metrics.csv',index=False)
 rows=[];depth=[];pending=[]
 for j in jobs:
  p=ROOT/'source_data'/'runs'/f'{key(j)}.json'
  if p.exists():
   with np.load(p.with_suffix('.npz')) as z:
    for name in z.files:assert np.isfinite(z[name]).all()
   rows.append(json.loads(p.read_text()));depth.extend(json.loads((p.parent/(p.stem+'_depth.json')).read_text()))
  else:pending.append(j)
 start=time.time();print('Running',len(pending),'jobs; already have',len(rows),flush=True)
 with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
  fs=[pool.submit(task,j) for j in pending]
  for f in concurrent.futures.as_completed(fs):
   r,d=f.result();rows.append(r);depth.extend(d)
   if len(rows)%3==0:print(f'Completed {len(rows)}/{len(jobs)}; elapsed {time.time()-start:.1f}s',flush=True)
 df=pd.DataFrame(rows).sort_values(['config','model_seed','L','stream_seed']);df.to_csv(ROOT/'source_data'/'all_metrics.csv',index=False)
 dd=pd.DataFrame(depth).sort_values(['config','model_seed','L','stream_seed','layer']);dd.to_csv(ROOT/'source_data'/'depth_metrics.csv',index=False)
 cols=['accuracy','agreement','logit_rmse','centered_logit_relative_error','query_hidden_relative_error','query_hidden_cosine']
 bymodel=df.groupby(['config','model_seed','L'])[cols].mean().reset_index();bymodel.to_csv(ROOT/'source_data'/'per_model_metrics.csv',index=False)
 summary=bymodel.groupby(['config','L'])[cols].agg(['mean','std']);summary.columns=['_'.join(x) for x in summary.columns]
 summary.reset_index().to_csv(ROOT/'source_data'/'summary_metrics.csv',index=False)
 (ROOT/'provenance'/'run_complete.json').write_text(json.dumps(dict(runs=len(rows),seconds=time.time()-start),indent=2))
 print(summary.to_string(),flush=True)

if __name__=='__main__':main()
