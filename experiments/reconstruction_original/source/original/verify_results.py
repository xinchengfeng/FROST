"""Verify recorded predictions, metrics, complete coverage and all figure sources."""
import hashlib,json
import numpy as np,pandas as pd
from PIL import Image
from data import ROOT,DATASETS,load_data
from evaluate import make_jobs,job_key,metrics,aggregate

def main():
 jobs=make_jobs();expected={job_key(j):j for j in jobs};files=list((ROOT/'source_data'/'runs').glob('*.json'))
 assert len(files)==len(jobs)==810,(len(files),len(jobs))
 assert {p.stem for p in files}==set(expected)
 data={d:load_data(d) for d in DATASETS};maxdiff=0.
 for p in files:
  row=json.loads(p.read_text());j=expected[p.stem];z=dict(np.load(p.with_suffix('.npz')))
  for name in ['prediction','reference']:assert np.isfinite(z[name]).all()
  ids=np.arange(10000) if j['split']=='full_test' else data[j['dataset']]['test_subset_indices'];ids=ids[j['part']*500:(j['part']+1)*500]
  assert np.array_equal(z['indices'],ids)
  assert np.array_equal(z['labels'],data[j['dataset']]['test_labels'][ids])
  for k,v in metrics(z['prediction'],z['reference'],z['labels']).items():maxdiff=max(maxdiff,abs(v-row[k]))
 assert maxdiff<1e-12
 d=aggregate();assert len(d[d.split=='full_test'])==36 and len(d[d.split=='diagnostic'])==81,len(d)
 assert (d[d.split=='full_test'].examples==10000).all();assert (d[d.split=='diagnostic'].examples==1000).all()
 for dataset in DATASETS:
  for seed in [11,22,33]:
   s=d[(d.dataset==dataset)&(d.model_seed==seed)&(d.config=='calibrated_full')&(d.L==256)&(d.split=='diagnostic')]
   assert set(s.stream_seed)=={101,202,303}
 figures={}
 for p in (ROOT/'figures').glob('*.png'):
  im=Image.open(p);im.load();figures[p.name]=dict(pixels=list(im.size),dpi=list(im.info.get('dpi',[])))
 assert len(figures)==3
 report=dict(verified_shards=len(jobs),full_test_evaluations=36,diagnostic_evaluations=81,max_recomputed_metric_difference=maxdiff,all_test_indices_and_labels_verified=True,full_and_subset_counts_verified=True,figures=figures)
 (ROOT/'provenance'/'result_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
