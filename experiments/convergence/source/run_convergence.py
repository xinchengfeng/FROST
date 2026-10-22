"""Real, nested finite-stream prefixes on frozen CIFAR-10 attention scores.

The native loop changes no RNG ordering, FSM transition, or categorical law.
All checkpoint frequencies are prefixes of ONE stream per method and seed.
The test-image and matrix selection rules depend on no stochastic result.
"""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse, concurrent.futures, csv, ctypes, hashlib, json, subprocess, time
from pathlib import Path
import numpy as np
import torch
from attention_core import FrozenHI
from vision import VisionTransformer

ROOT=Path(__file__).resolve().parent.parent
SEEDS=[101*i for i in range(1,9)]
LENGTHS=np.array([2**j for j in range(4,13)],dtype=np.int32)
torch.set_num_threads(1)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare(data_dir):
    data_dir=Path(data_dir)
    meta=dict(np.load(data_dir/'cifar10_metadata.npz'))
    index=meta['test_subset_indices'][:8]
    pixels=np.load(data_dir/'cifar10_test_images.npy',mmap_mode='r')[index].copy()
    # Original normalization and non-overlapping 4 x 4 patchification.
    x=(pixels.astype(np.float32)/255-meta['mean'][None,:,None,None])/meta['std'][None,:,None,None]
    x=x.reshape(8,3,4,8,4,8).transpose(0,2,4,1,3,5).reshape(8,16,192).copy()
    ck=torch.load(ROOT/'checkpoints/teacher_cifar10_seed11.pt',weights_only=True,map_location='cpu')
    m=VisionTransformer(**ck['architecture']);m.load_state_dict(ck['state_dict']);m.eval()
    with torch.no_grad():
        logit,_,_,tr=m(torch.tensor(x),True)
    # Matrix order: test image, block, head, query, key.
    full=np.stack([t['scores'].numpy() for t in tr],axis=1)
    matrices=full.reshape(64,16,16)
    selected=np.sort(np.random.default_rng(20261009).choice(64,8,replace=False))
    z=matrices[selected].reshape(128,16).astype(np.float64)
    p=np.exp(z-z.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
    entropy=-(p*np.log(np.maximum(p,1e-300))).sum(-1)
    entropy_order=np.argsort(entropy,kind='stable')
    representative=int(entropy_order[len(entropy_order)//2])
    np.savez_compressed(ROOT/'source_data/inputs.npz',scores=z,reference=p,
        pixels=pixels,tokens=x,all_scores=full,logits=logit.numpy(),
        test_indices=index,selected_matrices=selected,entropy=entropy,
        entropy_order=entropy_order,representative_row=np.array(representative),
        mean=meta['mean'],std=meta['std'])
    row_meta=[]
    for mi in selected:
        image_slot=int(mi)//8;block=int(mi)%8//2;head=int(mi)%2
        for q in range(16):
            row_meta.append(dict(row=len(row_meta),official_test_index=int(index[image_slot]),
                image_slot=image_slot,block=block+1,head=head+1,query=q+1,matrix_index=int(mi)))
    protocol=dict(dataset='CIFAR-10',model_seed=11,frozen_transformer=True,
        score_source='exact floating-point attention scores; QK/AV excluded from this diagnostic',
        images=8,test_indices=index.tolist(),
        image_selection='first eight entries of the pre-existing stratified 1000-image diagnostic subset',
        selected_matrices=selected.tolist(),matrix_selection_seed=20261009,
        matrix_selection='eight of 64 image/block/head matrices sampled without replacement; all 16 rows retained',
        rows=128,states=16,rank=8,parameters=16512,
        generic_bound=4.0,calibrated_bound=4.0,
        calibration='existing checkpoint; no additional fitting for this figure',
        lengths=LENGTHS.tolist(),stream_seeds=SEEDS,burn_in=256,initial_state=7,
        rng='NumPy default_rng PCG64; independent input and emission RNGs; emission seed = seed + 900019',
        prefixes='nested prefixes of one 4096-cycle trajectory for every method/seed; 16-cycle emission chunks',
        representative_selection='upper median row of exact reference Shannon entropy; chosen before running streams',
        representative_row=representative,representative_metadata=row_meta[representative],
        simplex_groups='largest exact reference probability, second largest, and sum of the other 14; no conditional renormalization',
        atlas='mean per-row TV over eight stream seeds; rows ordered by exact reference entropy',
        curve_uncertainty='sample standard deviation of eight seed-level mean-row TV values at fixed scores and weights',
        checkpoints_sha256={p.name:sha(p) for p in sorted((ROOT/'checkpoints').glob('*.pt'))},
        input_pixels_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
        official_metadata_sha256=sha(data_dir/'cifar10_metadata.npz'))
    slots=sorted(set(int(mi)//8 for mi in selected))
    protocol['contributing_image_slots']=slots
    protocol['contributing_test_indices']=[int(index[i]) for i in slots]
    (ROOT/'source_data/protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    with (ROOT/'source_data/attention_rows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row_meta[0]));w.writeheader();w.writerows(row_meta)
    print('Input preparation:',protocol['representative_metadata'],'selected matrices',selected.tolist(),flush=True)

def lib():
    dll=ctypes.CDLL(str(ROOT/'source/trace_kernels.so'))
    p=ctypes.c_void_p;i64=ctypes.c_int64;i32=ctypes.c_int32
    dll.fsm_trace.argtypes=[p,p,p,i64,i32,i32];dll.fsm_trace.restype=None
    dll.categorical_trace.argtypes=[p,p,p,p,p,p,i64,i32,i32,i32,i32]
    dll.categorical_trace.restype=None
    return dll

def ptr(a):return a.ctypes.data if a is not None else None

def run(job):
    method,seed=job;start=time.time()
    ck='calibrated_cifar10_11.pt' if method=='FROST' else 'softmax_N16_r8.pt'
    model=FrozenHI(ROOT/'checkpoints'/ck)
    inp=np.load(ROOT/'source_data/inputs.npz');z=inp['scores'];ref=inp['reference']
    rep=int(inp['representative_row']);analytic=model.analytic(z)
    x,order=model.encode(z);rows,n=x.shape;states=model.states
    state=np.full(x.shape,(states-1)//2,dtype=np.int16)
    rng=np.random.default_rng(seed);emission=np.random.default_rng(seed+900019);native=lib()
    for st in range(0,256,16):
        drive=np.ascontiguousarray(rng.random((16,*x.shape))<x[None],dtype=np.uint8)
        native.fsm_trace(ptr(state),ptr(drive),None,x.size,16,states)
    pad=max(max(c.shape[-2:]) for c in model.cdfs)
    cdfs=np.zeros((n,states,pad,pad),dtype=np.float64)
    sizes=np.asarray([c.shape[-1] for c in model.cdfs],dtype=np.int32)
    for j,c in enumerate(model.cdfs):cdfs[j,:,:c.shape[1],:c.shape[2]]=c
    events=np.empty((int(LENGTHS[-1]),rows),dtype=np.uint8)
    micro=np.empty((256,n),dtype=np.int16)
    counts=np.zeros(x.shape,dtype=np.int32);pred=[];stats=[]
    trace=np.empty((16,*x.shape),dtype=np.int16);ev=np.empty((16,rows),dtype=np.uint8)
    for st in range(0,int(LENGTHS[-1]),16):
        drive=np.ascontiguousarray(rng.random((16,*x.shape))<x[None],dtype=np.uint8)
        native.fsm_trace(ptr(state),ptr(drive),ptr(trace),x.size,16,states)
        draws=np.ascontiguousarray(emission.random((n,16*rows)))
        native.categorical_trace(ptr(trace),ptr(draws),ptr(cdfs),ptr(sizes),ptr(ev),ptr(counts),rows,16,states,n,pad)
        events[st:st+16]=ev
        if st<256:micro[st:st+16]=trace[:,rep,:]
        L=st+16
        if L in LENGTHS:
            a=np.empty_like(x);np.put_along_axis(a,order,counts/L,-1);pred.append(a.copy())
            tv=.5*np.abs(a-ref).sum(-1);sampling=.5*np.abs(a-analytic).sum(-1)
            stats.append(dict(method=method,stream_seed=seed,L=L,
                mean_row_tv=float(tv.mean()),probability_rmse=float(np.sqrt(np.mean((a-ref)**2))),
                sampling_mean_row_tv=float(sampling.mean()),sampling_rmse=float(np.sqrt(np.mean((a-analytic)**2))),
                analytic_mean_row_tv=float((.5*np.abs(analytic-ref).sum(-1)).mean()),
                analytic_rmse=float(np.sqrt(np.mean((analytic-ref)**2))),
                row_sum_error=float(np.max(np.abs(a.sum(-1)-1))),min_probability=float(a.min())))
    pred=np.stack(pred);tv=.5*np.abs(pred-ref[None]).sum(-1)
    sample_tv=.5*np.abs(pred-analytic[None]).sum(-1)
    np.savez_compressed(ROOT/f'source_data/trace_{method}_seed{seed}.npz',
        probabilities=pred,row_tv=tv,sampling_row_tv=sample_tv,
        events_sorted=events,representative_states=micro,input_probabilities=x,
        order=order,analytic=analytic)
    elapsed=time.time()-start
    print(method,'seed',seed,'TV 64 / 4096',round(stats[2]['mean_row_tv'],5),
        round(stats[-1]['mean_row_tv'],5),'seconds',round(elapsed,2),flush=True)
    return stats

def verify():
    inp=np.load(ROOT/'source_data/inputs.npz');z=inp['scores'];ref=inp['reference'];checks=[]
    for method in ['SC_base','FROST']:
        ck='calibrated_cifar10_11.pt' if method=='FROST' else 'softmax_N16_r8.pt'
        model=FrozenHI(ROOT/'checkpoints'/ck)
        for seed in [101,303,808]:
            a=np.load(ROOT/f'source_data/trace_{method}_seed{seed}.npz')
            for L in [64,256,1024]:
                expected=model.stream(z,L,seed)
                actual=a['probabilities'][list(LENGTHS).index(L)]
                delta=float(np.max(np.abs(actual-expected)));assert delta==0,(method,seed,L,delta)
                checks.append(dict(method=method,seed=seed,L=L,max_original_difference=delta))
        # Also verify the complete horizon against the original Python sampler.
        a=np.load(ROOT/f'source_data/trace_{method}_seed101.npz')
        expected=model.stream(z,int(LENGTHS[-1]),101)
        delta=float(np.max(np.abs(a['probabilities'][-1]-expected)));assert delta==0
        checks.append(dict(method=method,seed=101,L=int(LENGTHS[-1]),max_original_difference=delta))
    max_mass_error=0.;max_event_checkpoint_difference=0.
    for method in ['SC_base','FROST']:
        for seed in SEEDS:
            a=np.load(ROOT/f'source_data/trace_{method}_seed{seed}.npz')
            ev=a['events_sorted'];order=a['order']
            cum=np.zeros((len(ev),16),dtype=np.int32)
            for j in range(16):cum[:,j]=np.cumsum(ev[:,int(inp['representative_row'])]==j)
            assert np.array_equal(cum.sum(-1),np.arange(1,len(ev)+1))
            allprefix=cum/np.arange(1,len(ev)+1)[:,None]
            max_mass_error=max(max_mass_error,float(np.abs(allprefix.sum(-1)-1).max()))
            for li,L in enumerate(LENGTHS):
                sorted_counts=np.stack([(ev[:L]==j).sum(0) for j in range(16)],axis=-1)
                assert np.array_equal(sorted_counts.sum(-1),np.full(len(z),L))
                out=np.empty_like(z);np.put_along_axis(out,order,sorted_counts/L,-1)
                delta=float(np.max(np.abs(out-a['probabilities'][li])))
                assert delta==0;max_event_checkpoint_difference=max(max_event_checkpoint_difference,delta)
            assert a['probabilities'].min()>=0
    result=dict(original_sampler_checks=checks,
        original_sampler_exact_matches=len(checks),
        independently_recounted_probability_sets=2*len(SEEDS)*len(LENGTHS),
        max_recount_difference=max_event_checkpoint_difference,
        representative_every_prefix_max_mass_error=max_mass_error,
        all_checkpoint_integer_count_totals_exact=True,
        all_checkpoint_probabilities_nonnegative=True)
    (ROOT/'source_data/verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Verified',len(checks),'original-sampler equivalence checks and',result['independently_recounted_probability_sets'],'event recounts',flush=True)

def summarize(rows):
    with (ROOT/'source_data/stream_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={}
    for method in ['SC_base','FROST']:
        series=[]
        for L in LENGTHS:
            rr=[r for r in rows if r['method']==method and r['L']==L]
            r=dict(L=int(L))
            for k in ['mean_row_tv','probability_rmse','sampling_mean_row_tv','sampling_rmse']:
                vals=np.array([x[k] for x in rr]);r[k+'_mean']=float(vals.mean());r[k+'_stream_sd']=float(vals.std(ddof=1))
            series.append(r)
        summary[method]=dict(analytic_mean_row_tv=rr[0]['analytic_mean_row_tv'],
            analytic_rmse=rr[0]['analytic_rmse'],series=series)
    summary['paired_long_stream_gain_percent']=100*(1-summary['FROST']['series'][-1]['mean_row_tv_mean']/summary['SC_base']['series'][-1]['mean_row_tv_mean'])
    summary['analytic_gain_percent']=100*(1-summary['FROST']['analytic_mean_row_tv']/summary['SC_base']['analytic_mean_row_tv'])
    summary['FROST_sampling_reduction_factor']=summary['FROST']['series'][2]['sampling_mean_row_tv_mean']/summary['FROST']['series'][-1]['sampling_mean_row_tv_mean']
    (ROOT/'source_data/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Analytic mean row TV:',summary['SC_base']['analytic_mean_row_tv'],summary['FROST']['analytic_mean_row_tv'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir');ap.add_argument('--verify-only',action='store_true');args=ap.parse_args()
    if args.verify_only:verify()
    else:
        if args.data_dir:prepare(args.data_dir)
        assert (ROOT/'source_data/inputs.npz').exists(),'provide --data-dir for initial extraction'
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(ROOT/'source/trace_kernels.cpp'),'-o',str(ROOT/'source/trace_kernels.so')],check=True)
        jobs=[(m,s) for m in ['SC_base','FROST'] for s in SEEDS]
        with concurrent.futures.ProcessPoolExecutor(max_workers=7) as pool:
            lists=list(pool.map(run,jobs))
        rows=[r for rr in lists for r in rr];summarize(rows);verify()
