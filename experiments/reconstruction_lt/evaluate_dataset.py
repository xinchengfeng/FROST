"""Fixed-checkpoint extension of Fig.10 FROST-LT to 200 archived teachers."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import sys,json,time,hashlib,argparse,concurrent.futures,platform
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
sys.path.insert(0,str(ROOT/'runtime/source/scripts'))
from local_tt import StratifiedTT
from restoration_model import load_teacher,SCWindowRestorer
from run_quality30 import coords
from restoration_metrics import quality
import fast_finite

_op=None
def initialize():
    global _op
    fast_finite.install()
    _op=StratifiedTT(ROOT/'checkpoints/local33_r16_max8_aug.pt')

def run(job):
    iid,L,seed=job
    name=f'image_{iid}_FROST_LT_L{L}_S{seed}'
    out=ROOT/'results'/f'{name}.npz'
    record=out.with_suffix('.json')
    if record.exists() and out.exists():
        return json.loads(record.read_text())
    teacher,_=load_teacher(ROOT/'runtime/checkpoints'/f'render_{iid}.pt')
    model=SCWindowRestorer(teacher,ROOT/'runtime/checkpoints'/f'operator_render_{iid}.pt')
    model.hi=_op
    with np.load(ROOT/'inputs'/f'image_{iid}.npz') as arch:
        truth=arch['truth']
    start=time.perf_counter()
    pred=np.clip(model.run(coords(),L=L,seed=seed+iid*9001+11*17011,mode='full')[0],0,1).astype(np.float32)
    elapsed=time.perf_counter()-start
    assert pred.shape==(64,64,3) and np.isfinite(pred).all()
    row={'image_id':iid,'method':'FROST-LT','L':L,'seed':seed,
         **quality(pred,truth,'reconstruction'),'elapsed_wall_seconds':elapsed,
         'prediction_sha256':hashlib.sha256(pred.tobytes()).hexdigest()}
    anchor=ROOT/'provenance'/f'{iid}_LocalTT33_stratified_S{seed}.npz'
    if L==4096 and anchor.exists():
        with np.load(anchor) as old: expected=old['prediction'];ref=old['truth']
        assert np.array_equal(truth,ref),'Reference crop mismatch'
        error=float(np.max(np.abs(pred-expected)))
        row['Fig10_anchor_max_absolute_difference']=error
        if error!=0:
            raise AssertionError(f'Fig10 replay mismatch for {iid}/{seed}: {error}')
    temporary=out.with_name(out.stem+'.tmp.npz')
    np.savez_compressed(temporary,prediction=pred)
    temporary.replace(out)
    tmp=record.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2));tmp.replace(record)
    return row

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--pilot',action='store_true')
    ap.add_argument('--workers',type=int,default=7)
    args=ap.parse_args()
    config=json.loads((ROOT/'protocol.json').read_text())
    ids=config['pilot_ids'] if args.pilot else config['image_ids']
    lengths=[4096] if args.pilot else config['L']
    jobs=[(i,L,s) for i in ids for L in lengths for s in config['base_stream_seeds']]
    start=time.perf_counter();done=0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as pool:
        futures={pool.submit(run,job):job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            row=future.result();done+=1
            print(json.dumps(row),flush=True)
            if done%20==0 or done==len(jobs):
                status={'completed':done,'total':len(jobs),'wall_seconds':time.perf_counter()-start,'last_image':row['image_id']}
                tmp=ROOT/'progress.tmp';tmp.write_text(json.dumps(status));tmp.replace(ROOT/'progress.json')
    print(json.dumps({'complete':True,'jobs':len(jobs),'wall_seconds':time.perf_counter()-start}),flush=True)
