"""Full official-test evaluation plus prespecified paired diagnostic subsets."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import io,json,time,hashlib,concurrent.futures,argparse
import numpy as np,pandas as pd,torch
from scipy.special import logsumexp
from sklearn.metrics import f1_score
from data import ROOT,DATASETS,load_data,patchify,save_npz
from vision import MODEL_SEEDS,load
from stochastic_model import SCTransformer

def metrics(pred,ref,y):
 a=pred.argmax(-1);r=ref.argmax(-1);cp=pred-pred.mean(-1,keepdims=True);cr=ref-ref.mean(-1,keepdims=True)
 return dict(examples=len(y),accuracy=float(np.mean(a==y)),reference_accuracy=float(np.mean(r==y)),agreement=float(np.mean(a==r)),macro_f1=float(f1_score(y,a,average='macro')),centered_logit_relative_error=float(np.linalg.norm(cp-cr)/np.linalg.norm(cr)),nll=float(np.mean(logsumexp(pred,axis=-1)-pred[np.arange(len(y)),y])),accuracy_delta_pp=float(100*np.mean((a==y).astype(float)-(r==y).astype(float))))

def job_key(j):return f"{j['dataset']}_model{j['model_seed']}_{j['config']}_L{j['L']}_stream{j['stream_seed']}_{j['split']}_part{j['part']}"

def make_jobs(datasets=DATASETS):
 jobs=[]
 for d in datasets:
  for model in MODEL_SEEDS:
   # Headline: same complete official test set, one prespecified stream seed.
   for config,mode in [('reference','exact'),('w8a8','w8a8'),('calibrated_analytic','analytic'),('calibrated_full','full')]:
    for part in range(20):jobs.append(dict(dataset=d,model_seed=model,config=config,mode=mode,L=256,stream_seed=101,split='full_test',part=part))
   # Diagnostics: a fixed class-balanced 1000-image subset, chosen before testing.
   for config,L,stream in [('generic_full',256,101),('calibrated_full',64,101),('calibrated_full',1024,101),('calibrated_full',256,202),('calibrated_full',256,303)]:
    for part in range(2):jobs.append(dict(dataset=d,model_seed=model,config=config,mode='full',L=L,stream_seed=stream,split='diagnostic',part=part))
 return jobs

_cache={}
def task(j):
 path=ROOT/'source_data'/'runs'/(job_key(j)+'.npz')
 if path.exists():return dict(job=j,cached=True)
 start=time.time();dname=j['dataset'];model=j['model_seed'];key=(dname,model,j['config'].startswith('calibrated'))
 if key not in _cache:
  teacher,_=load(dname,model);hi=ROOT/'checkpoints'/f'calibrated_{dname}_{model}.pt' if key[2] else None
  _cache[key]=(teacher,SCTransformer(teacher,hi),load_data(dname))
 teacher,sc,data=_cache[key];indices=np.arange(10000) if j['split']=='full_test' else data['test_subset_indices']
 indices=indices[j['part']*500:(j['part']+1)*500];x=patchify(data['test_images'][indices],data['mean'],data['std']);y=data['test_labels'][indices]
 with torch.no_grad():ref=teacher(torch.tensor(x)).numpy().astype(np.float64)
 if j['config']=='reference':pred=ref.copy()
 else:
  preds=[]
  bs=16 if j['mode']=='full' else 128
  for i in range(0,len(x),bs):
   seed=j['stream_seed']+(j['part']*500+i)*9001
   p=sc.run(x[i:i+bs],L=j['L'],seed=seed,mode=j['mode'],source='sobol',encoding='magnitude')[0];preds.append(p)
  pred=np.concatenate(preds)
 assert np.isfinite(pred).all()
 save_npz(path,indices=indices,labels=y,reference=ref,prediction=pred)
 row={**j,**metrics(pred,ref,y),'seconds':time.time()-start};path.with_suffix('.json').write_text(json.dumps(row,indent=2));return row

def aggregate():
 groups={}
 for path in (ROOT/'source_data'/'runs').glob('*.json'):
  j=json.loads(path.read_text());key=tuple(j[k] for k in ['dataset','model_seed','config','L','stream_seed','split']);groups.setdefault(key,[]).append((j,path))
 rows=[]
 for key,parts in groups.items():
  if len(parts)!=(20 if key[-1]=='full_test' else 2):continue
  parts.sort(key=lambda a:a[0]['part']);arrays=[dict(np.load(p.with_suffix('.npz'))) for _,p in parts]
  z={k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]};j=parts[0][0];base={k:j[k] for k in ['dataset','model_seed','config','L','stream_seed','split']}
  rows.append({**base,**metrics(z['prediction'],z['reference'],z['labels'])})
  if j['split']=='full_test':
   d=load_data(j['dataset']);idx=d['test_subset_indices'];order=np.argsort(z['indices']);sel=order[idx]
   rows.append({**base,'split':'diagnostic',**metrics(z['prediction'][sel],z['reference'][sel],z['labels'][sel])})
 df=pd.DataFrame(rows)
 if len(df):df.to_csv(ROOT/'source_data'/'all_metrics.csv',index=False)
 return df

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--datasets',nargs='+',default=DATASETS);parser.add_argument('--stage',choices=['fast','streams','all'],default='all');parser.add_argument('--workers',type=int,default=8);a=parser.parse_args()
 jobs=[j for j in make_jobs(a.datasets) if a.stage=='all' or (j['mode']=='full')==(a.stage=='streams')]
 pending=[j for j in jobs if not (ROOT/'source_data'/'runs'/(job_key(j)+'.npz')).exists()]
 start=time.time();print('evaluate',a.datasets,a.stage,'pending',len(pending),'jobs',flush=True)
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
  futures=[ex.submit(task,j) for j in pending]
  for i,f in enumerate(concurrent.futures.as_completed(futures),1):
   r=f.result()
   if i%8==0 or i==len(pending):print('evaluation',i,'/',len(pending),'elapsed',round(time.time()-start,1),'last',r.get('dataset'),r.get('config'),r.get('L'),flush=True)
 df=aggregate();print(df[df.split=='full_test'][['dataset','model_seed','config','accuracy','agreement','centered_logit_relative_error']].to_string(index=False),flush=True)

if __name__=='__main__':main()
