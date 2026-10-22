"""Recalculate image-level statistics from saved predictions, never from rounded scores."""
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runtime/source/scripts'))
from restoration_metrics import quality

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def bootstrap(values,seed=20261017):
    values=np.asarray(values,float)
    rng=np.random.default_rng(seed)
    sampled=values[rng.integers(len(values),size=(10000,len(values)))].mean(1)
    return np.percentile(sampled,[2.5,97.5]).tolist()

def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    ids=protocol['image_ids'];seeds=protocol['base_stream_seeds']
    old=[('Float',None),('W8A8',None),('SC_base',4096),('FROST_original',1024),('FROST_original',4096)]
    new=[('FROST-LT',1024),('FROST-LT',4096)]
    single=[];pooled=[];anchor=[];hashes={}
    nondev=set(protocol['nondevelopment_image_ids'])
    for iid in ids:
        with np.load(ROOT/'inputs'/f'image_{iid}.npz') as src:
            truth=src['truth']
            for method,L in old+new:
                per=[]
                for seed in ([None] if L is None else seeds):
                    if method=='FROST-LT':
                        path=ROOT/'results'/f'image_{iid}_FROST_LT_L{L}_S{seed}.npz'
                        with np.load(path) as z: pred=z['prediction']
                        recorded=json.loads(path.with_suffix('.json').read_text())
                        q=quality(pred,truth,'reconstruction')
                        assert all(abs(q[k]-recorded[k])<1e-12 for k in q), 'Metric changed'
                        assert hashlib.sha256(pred.tobytes()).hexdigest()==recorded['prediction_sha256']
                        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
                        if 'Fig10_anchor_max_absolute_difference' in recorded:
                            anchor.append(recorded['Fig10_anchor_max_absolute_difference'])
                    else:
                        key=method if L is None else f'{"FROST" if method=="FROST_original" else method}_L{L}_S{seed}'
                        pred=src[key];q=quality(pred,truth,'reconstruction')
                    row={'image_id':iid,'method':method,'L':L or '', 'seed':seed or '',
                         'operator_development_image':iid not in nondev,**q}
                    single.append(row);per.append(q)
                pooled.append({'image_id':iid,'method':method,'L':L or '',
                               'operator_development_image':iid not in nondev,
                               'psnr':float(np.mean([x['psnr'] for x in per])),
                               'ssim':float(np.mean([x['ssim'] for x in per])),
                               'stream_psnr_sd':float(np.std([x['psnr'] for x in per],ddof=1)) if L else 0.,
                               'stream_ssim_sd':float(np.std([x['ssim'] for x in per],ddof=1)) if L else 0.})
    out=ROOT/'summary';out.mkdir(exist_ok=True)
    write_csv(out/'all_metrics.csv',single);write_csv(out/'per_image_metrics.csv',pooled)
    statistics={}
    for split,subset in [('all_200',set(ids)),('nondevelopment_184',nondev),
                         ('new_nondevelopment_181',nondev-set(protocol['pilot_ids']))]:
        lookup={(m,L or ''):[r for r in pooled if r['method']==m and r['L']==(L or '') and r['image_id'] in subset] for m,L in old+new}
        ref=np.array([r['psnr'] for r in lookup['Float','']])
        stats=[]
        for method,L in old+new:
            group=lookup[method,L or ''];ps=np.array([r['psnr'] for r in group]);ss=np.array([r['ssim'] for r in group])
            individual=[r['psnr'] for r in single if r['method']==method and r['L']==(L or '') and r['image_id'] in subset]
            stats.append({'method':method,'L':L,'n_images':len(group),
                          'psnr_mean':float(ps.mean()),'psnr_sd':float(ps.std(ddof=1)),
                          'ssim_mean':float(ss.mean()),'ssim_sd':float(ss.std(ddof=1)),
                          'psnr_loss_vs_Float':float((ref-ps).mean()),
                          'images_ge_30':int((ps>=30).sum()),'images_ge_40':int((ps>=40).sum()),
                          'min_image_psnr':float(ps.min()),'min_run_psnr':float(min(individual)),
                          'runs_ge_30':int(sum(x>=30 for x in individual)),'n_runs':len(individual)})
        pairs=[]
        for a,b in [(('FROST-LT',4096),('FROST_original',4096)),
                    (('FROST-LT',4096),('SC_base',4096)),
                    (('FROST-LT',1024),('FROST_original',4096)),
                    (('FROST-LT',4096),('FROST-LT',1024))]:
            aa=np.array([r['psnr'] for r in lookup[a]]);bb=np.array([r['psnr'] for r in lookup[b]])
            diff=aa-bb
            pairs.append({'method_a':a,'method_b':b,'psnr_gain':float(diff.mean()),
                          'image_bootstrap_95CI':bootstrap(diff),'n_images_improved':int((diff>0).sum()),
                          'min_gain':float(diff.min()),'max_gain':float(diff.max())})
        statistics[split]={'statistics':stats,'paired_comparisons':pairs}
    assert len(anchor)==9 and max(anchor)==0
    validation={'new_predictions_recomputed':1200,'archived_predictions_recomputed':2200,
                'new_output_count':len(list((ROOT/'results').glob('*.npz'))),
                'pilot_replay_count':len(anchor),'pilot_max_abs_difference':max(anchor),
                'operator_checkpoint_sha256':hashlib.sha256((ROOT/'checkpoints/local33_r16_max8_aug.pt').read_bytes()).hexdigest(),
                'units':'PSNR dB; SSIM unit range; image means over three per-run scores, sample SD over images',
                'bootstrap':'10000 paired image resamples, seed 20261017; percentile 2.5 and 97.5'}
    assert validation['new_output_count']==1200
    assert validation['operator_checkpoint_sha256']==protocol['checkpoint_sha256']
    (out/'validation.json').write_text(json.dumps(validation,indent=2))
    (out/'dataset_summary.json').write_text(json.dumps(statistics,indent=2))
    (out/'prediction_sha256.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(statistics,indent=2))

if __name__=='__main__':main()
