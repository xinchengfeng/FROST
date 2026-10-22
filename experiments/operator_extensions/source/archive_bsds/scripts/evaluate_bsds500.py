"""Measured finite-stream reconstructions of all 200 fixed evaluation targets."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse
import concurrent.futures
import json
import time
from pathlib import Path
import numpy as np
import torch
from run_quality30 import coords
from restoration_model import load_teacher, SCWindowRestorer
from restoration_data import center_image
from restoration_metrics import quality
import fast_finite


def evaluate_one(job):
    root_s,image_id=job;root=Path(root_s);start=time.perf_counter()
    final=root/'runs'/f'image_{image_id}.npz'
    if final.exists():
        rows=json.loads(final.with_suffix('.json').read_text())
        assert len(rows)==12
        return dict(image_id=image_id,seconds=0,reused=True)
    fast_finite.install()
    model,ck=load_teacher(root/'checkpoints'/f'render_{image_id}.pt')
    assert ck['training_steps']==1500
    x=coords();truth=center_image(root/'data/test'/f'{image_id}.jpg')
    calibrated=SCWindowRestorer(model,root/'checkpoints'/f'operator_render_{image_id}.pt')
    base=SCWindowRestorer(model)
    with torch.no_grad():fl=model(torch.tensor(x)).numpy()[0]
    exact=calibrated.run(x,mode='exact')[0]
    difference=float(np.max(np.abs(fl-exact)));assert difference<2e-5,difference
    outputs={'truth':truth,'Float':np.clip(fl,0,1).astype(np.float32),
             'W8A8':np.clip(calibrated.run(x,mode='w8a8')[0],0,1).astype(np.float32),
             'FROST_analytic':np.clip(calibrated.run(x,mode='analytic')[0],0,1).astype(np.float32)}
    rows=[]
    def add(method,stream,L,pred):
        pred=np.clip(pred,0,1).astype(np.float32)
        key=method if stream==0 else f'{method}_L{L}_S{stream}'
        outputs[key]=pred
        rows.append(dict(image_id=image_id,method=method,L=L,stream_base_seed=stream,
            inference_seed=stream+image_id*9001+11*17011 if stream else 0,
            training_steps=1500,best_step=ck['best_step'],model_seed=ck['seed'],
            target_used_for_fitting=True,exact_graph_max_abs_difference=difference,
            **quality(pred,truth,'reconstruction')))
    for method in ['Float','W8A8','FROST_analytic']:add(method,0,0,outputs[method])
    for stream in [101,202,303]:
        seed=stream+image_id*9001+11*17011
        add('FROST',stream,1024,calibrated.run(x,L=1024,seed=seed,mode='full')[0])
        add('FROST',stream,4096,calibrated.run(x,L=4096,seed=seed,mode='full')[0])
        add('SC_base',stream,4096,base.run(x,L=4096,seed=seed,mode='full')[0])
        (root/'logs'/f'eval_{image_id}.json').write_text(json.dumps(dict(
            image_id=image_id,completed_seed=stream,elapsed_seconds=time.perf_counter()-start),indent=2))
    # Complete records and outputs are written under a new path; old evidence stays intact.
    assert len(rows)==12
    tmp=final.with_suffix('.tmp.npz');np.savez_compressed(tmp,**outputs);tmp.replace(final)
    final.with_suffix('.json').write_text(json.dumps(rows,indent=2))
    return dict(image_id=image_id,seconds=time.perf_counter()-start,reused=False,
                Float_psnr=rows[0]['psnr'],FROST4096_psnr=float(np.mean(
                    [r['psnr'] for r in rows if r['method']=='FROST' and r['L']==4096])))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);args=p.parse_args();root=args.root.resolve()
    verification=json.loads((root/'source_data/exact_kernel_verification.json').read_text())
    assert len(verification)==21 and all(r['max_absolute_difference']==0 for r in verification)
    manifest=json.loads((root/'source_data/protocol.json').read_text())
    ids=[row['image_id'] for row in manifest['images']];assert len(set(ids))==200
    for image_id in ids:
        assert (root/'checkpoints'/f'operator_render_{image_id}.pt').exists(),image_id
    completed=[];start=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(evaluate_one,(str(root),image_id)) for image_id in ids]
        for future in concurrent.futures.as_completed(futures):
            row=future.result();completed.append(row)
            print(json.dumps(dict(stage='evaluate',completed=len(completed),total=200,
                elapsed_seconds=round(time.perf_counter()-start,1),**row)),flush=True)
            (root/'source_data/evaluation_progress.json').write_text(json.dumps(completed,indent=2))
    print('COMPLETE all 200 images, three stream seeds and both FROST lengths',flush=True)


if __name__=='__main__':main()
