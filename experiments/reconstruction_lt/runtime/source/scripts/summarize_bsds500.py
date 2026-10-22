"""Recompute all outputs and aggregate on the image, not image-seed, unit."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import scipy
import torch
import PIL
from restoration_metrics import quality
from restoration_data import center_image


GROUPS=[('Float',0),('W8A8',0),('SC_base',4096),('FROST',1024),('FROST',4096),('FROST_analytic',0)]


def csv_write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def aggregate(root):
    protocol=json.loads((root/'source_data/protocol.json').read_text())
    ids=[row['image_id'] for row in protocol['images']]
    assert len(ids)==len(set(ids))==200
    all_rows=[];verification=[]
    for target in protocol['images']:
        image_id=target['image_id'];photo=root/'data/test'/target['file']
        assert hashlib.sha256(photo.read_bytes()).hexdigest()==target['sha256']
        npz=root/'runs'/f'image_{image_id}.npz';z=np.load(npz)
        truth=center_image(photo);assert np.array_equal(truth,z['truth'])
        rows=json.loads(npz.with_suffix('.json').read_text());assert len(rows)==12
        checked=[]
        for row in rows:
            key=row['method'] if row['stream_base_seed']==0 else f"{row['method']}_L{row['L']}_S{row['stream_base_seed']}"
            measured=quality(z[key],truth,'reconstruction')
            for metric in ['psnr','ssim','mse']:
                assert np.isclose(measured[metric],row[metric],rtol=0,atol=1e-12),(image_id,key,metric)
            assert np.isfinite(list(measured.values())).all()
            all_rows.append(row);checked.append(key)
        assert len(set(checked))==12
        if image_id in [2018,43033,235098]:
            old=np.load(root/'runs'/f'reference_previous_{image_id}.npz')
            for old_key,new_key in [('float','Float'),('w8a8','W8A8'),('frost','FROST_L4096_S101'),('SC_base','SC_base_L4096_S101')]:
                assert np.array_equal(old[old_key],z[new_key]),(image_id,old_key)
        verification.append(dict(image_id=image_id,outputs_recomputed=12,source_hash_verified=True,
            exact_reference_crop=True,metrics_match=True))
    assert len(all_rows)==2400
    csv_write(root/'source_data/all_metrics.csv',all_rows)
    per_image=[]
    for image_id in ids:
        rows=[r for r in all_rows if r['image_id']==image_id]
        floating=next(r for r in rows if r['method']=='Float')['psnr']
        for method,L in GROUPS:
            subset=[r for r in rows if r['method']==method and r['L']==L]
            assert len(subset)==(3 if method in ['FROST','SC_base'] else 1)
            rec=dict(image_id=image_id,method=method,L=L,stream_seed_count=len(subset))
            for metric in ['psnr','ssim','mse']:
                rec[metric+'_stream_mean']=float(np.mean([r[metric] for r in subset]))
                rec[metric+'_stream_sd']=float(np.std([r[metric] for r in subset],ddof=1)) if len(subset)>1 else 0.
            rec['psnr_loss_vs_float']=floating-rec['psnr_stream_mean']
            per_image.append(rec)
    csv_write(root/'source_data/per_image_metrics.csv',per_image)
    summaries=[]
    for method,L in GROUPS:
        subset=[r for r in per_image if r['method']==method and r['L']==L]
        raw=[r for r in all_rows if r['method']==method and r['L']==L]
        assert len(subset)==200
        rec=dict(method=method,L=L,images=200,streams_per_image=subset[0]['stream_seed_count'])
        for metric in ['psnr','ssim','mse']:
            v=np.array([r[metric+'_stream_mean'] for r in subset])
            rec[metric+'_image_mean']=float(v.mean());rec[metric+'_between_image_sd']=float(v.std(ddof=1))
            rec[metric+'_median_image_mean']=float(np.median(v));rec[metric+'_min_image_mean']=float(v.min())
            rec[metric+'_q05_image_mean']=float(np.quantile(v,.05));rec[metric+'_max_image_mean']=float(v.max())
            rec[metric+'_mean_within_image_stream_sd']=float(np.mean([r[metric+'_stream_sd'] for r in subset]))
        rec['psnr_loss_vs_float_mean']=float(np.mean([r['psnr_loss_vs_float'] for r in subset]))
        rec['images_with_mean_psnr_ge30']=sum(r['psnr_stream_mean']>=30 for r in subset)
        rec['raw_runs_psnr_ge30']=sum(r['psnr']>=30 for r in raw);rec['raw_runs_total']=len(raw)
        rec['min_run_psnr']=min(r['psnr'] for r in raw)
        summaries.append(rec)
    csv_write(root/'source_data/dataset_summary.csv',summaries)
    def values(method,L):
        return np.array([next(r for r in per_image if r['image_id']==i and r['method']==method and r['L']==L)['psnr_stream_mean'] for i in ids])
    rng=np.random.default_rng(20261009);bootstrap_indices=rng.integers(0,200,size=(10000,200))
    paired=[]
    for label,left,right in [('FROST4096_minus_SC4096',('FROST',4096),('SC_base',4096)),
                              ('FROST4096_minus_FROST1024',('FROST',4096),('FROST',1024))]:
        d=values(*left)-values(*right);bs=d[bootstrap_indices].mean(1)
        paired.append(dict(comparison=label,mean_gain_db=float(d.mean()),between_image_sd_db=float(d.std(ddof=1)),
            median_gain_db=float(np.median(d)),images_improved=int(np.sum(d>0)),images=200,
            image_bootstrap_ci95=[float(x) for x in np.quantile(bs,[.025,.975])],
            bootstrap_resamples=10000,bootstrap_seed=20261009,unit='image-specific seed mean'))
    summary=dict(dataset='BSDS500',image_pool='complete official 200-image test partition',
                 target_pixels_used_for_fitting=True,groups=summaries,paired_comparisons=paired,
                 standard_deviation_unit='200 image-level values, after averaging three streams where stochastic',
                 stream_seeds=[101,202,303],no_quality_based_exclusions=True)
    (root/'source_data/dataset_summary.json').write_text(json.dumps(summary,indent=2))
    (root/'source_data/results_verification.json').write_text(json.dumps(dict(
        images=200,independently_recomputed_outputs=2400,all_valid=True,checks=verification),indent=2))
    (root/'source_data/environment.json').write_text(json.dumps(dict(python=platform.python_version(),
        torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,pillow=PIL.__version__,
        device='CPU',threads_per_worker=1),indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    aggregate(p.parse_args().root.resolve())
