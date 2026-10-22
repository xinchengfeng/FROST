"""Stage 3: recompute frozen software evidence without training any model.

Each audit writes its own checkpoint before the next audit begins. Saved
outputs are inputs, never overwritten. BSDS SSIM is independently implemented
with a two-dimensional valid Gaussian convolution rather than the archived
separable implementation. Hardware and full classification are not evaluated.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse, csv, hashlib, json, platform, sys, time
import numpy as np
import torch
import scipy
from scipy.signal import correlate2d
from PIL import Image

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parents[1]
WORKSPACE = HERE.parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--revision', type=Path, default=WORKSPACE/'output/FROST_NC_revision')
parser.add_argument('--bsds', type=Path, default=WORKSPACE/'build_nc_revision/bsds_assets')
parser.add_argument('--convergence', type=Path, default=WORKSPACE/'build_nc_revision/base_assets')
parser.add_argument('--out', type=Path, default=HERE/'source_data')
parser.add_argument('--groups', nargs='+', default=['dimensions','sampler','convergence','bsds'])
args = parser.parse_args()
REV, BSDS, CONV, OUT = [p.resolve() for p in (args.revision,args.bsds,args.convergence,args.out)]
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REV/'source'))
from run_dimension_study import Operator, predict
from attention_core import FrozenHI
from run_sampler_study import contract, stationary, shared_exact
DATA = REV/'source_data'
SEEDS = [101,202,303,404,505,606,707,808]
LS = [64,256,1024,4096]
start = time.perf_counter()

def jsonread(p): return json.loads(p.read_text())
def csvread(p): return list(csv.DictReader(p.open()))
def dump(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2)+'\n')
def writecsv(name, records):
    with (OUT/name).open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(records[0]));w.writeheader();w.writerows(records)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def tv(p,q): return float(.5*np.abs(p-q).sum(-1).mean())
def rmse(p,q): return float(np.sqrt(np.mean((p-q)**2)))
def counts_probability(events, L, order):
    n = order.shape[1]
    count = np.stack([np.bincount(events[:L,j],minlength=n) for j in range(len(order))])
    assert np.all(count.sum(-1)==L)
    p=np.empty((len(order),n));np.put_along_axis(p,order,count/L,-1)
    assert np.max(np.abs(p.sum(-1)-1)) == 0
    return p
def softmax(z):
    e=np.exp(z-z.max(-1,keepdims=True));return e/e.sum(-1,keepdims=True)
def compiled(hi):
    cores=[c/c.sum(-1,keepdims=True) for c in hi.cores]
    cdfs=[c.cumsum(-1) for c in cores]
    for c in cdfs: c[...,-1]=1
    return cores,cdfs
def finite_states(x, seed, L=4096):
    rng=np.random.default_rng(seed); state=np.full(x.shape,7,dtype=np.int16)
    for _ in range(256): state=np.clip(state+2*(rng.random(x.shape)<x)-1,0,15)
    for _ in range(L//16):
        block=[]
        for _ in range(16):
            state=np.clip(state+2*(rng.random(x.shape)<x)-1,0,15)
            block.append(state.copy())
        yield np.asarray(block)

def dimensions():
    original={ (int(r['n']),int(r['seed']),r['stage'],r['family']):r for r in csvread(DATA/'dimension_metrics.csv') }
    range_original={ (int(r['n']),int(r['seed']),r['family']):r for r in csvread(DATA/'dimension_range_control.csv') }
    records=[];maxpred=0.;maxmetric=0.;maxmass=0.;hashes={}
    for n in [16,32,64]:
      for seed in [11,22,33]:
        fit=jsonread(DATA/f'dimension_n{n}_seed{seed}_fit.json')
        for stage in ['base','calibrated','range_only']:
          ckpath=REV/'checkpoints'/f'dimension_n{n}_seed{seed}_{"base" if stage=="range_only" else stage}.pt'
          ck=torch.load(ckpath,weights_only=True)
          hashes[ckpath.name]=sha(ckpath)
          count=sum(t.numel() for t in ck['state_dict'].values())
          assert count == 16*8+(n-2)*16*64+16*8*n
          if stage!='range_only':
            selection=fit['base_fit' if stage=='base' else 'calibration_fit']
            best=min(selection['history'],key=lambda h:h['validation_rmse'])
            assert selection['validation_rmse']==best['validation_rmse']
            assert ck['config']['selected_step']==best['step']
          bound=8 if stage=='range_only' else ck['config']['bound']
          model=Operator([ck['state_dict'][f'cs.{j}'] for j in range(n)],bound)
          for family in ['gaussian','diffuse','sharp','correlated']:
            with np.load(DATA/f'dimension_n{n}_seed{seed}_{"base" if stage=="range_only" else stage}_{family}.npz') as a:
              z=a['scores'];target=a['target'];saved=a['prediction'].copy()
            if stage=='range_only':
              with np.load(DATA/f'dimension_n{n}_seed{seed}_range_only_{family}.npz') as a: saved=a['prediction'].copy()
            out=predict(model,torch.from_numpy(z)).numpy()
            pred_error=float(np.max(np.abs(out-saved)));assert pred_error<2e-7,(n,seed,stage,family,pred_error)
            maxpred=max(maxpred,pred_error)
            y=torch.from_numpy(out);p=torch.from_numpy(target)
            row_tv=.5*(y-p).abs().sum(-1)
            center=torch.from_numpy(z);center=center-center.mean(-1,keepdim=True)
            r=dict(n=n,seed=seed,stage=stage,family=family,mean_row_tv=row_tv.mean().item(),
                probability_rmse=(y-p).square().mean().sqrt().item(),maximum_mass_error=(y.sum(-1)-1).abs().max().item(),
                clipped_score_fraction=(center.abs()>bound).float().mean().item(),parameters=count,bound=float(bound))
            old=range_original[(n,seed,family)] if stage=='range_only' else original[(n,seed,stage,family)]
            for key in ['mean_row_tv','probability_rmse']:
              err=abs(r[key]-float(old[key]));assert err<2e-7;maxmetric=max(maxmetric,err)
            # Double-precision contraction on eight rows is an additional
            # implementation check; full archived predictions use float32.
            hi=FrozenHI(ckpath);hi.bound=float(bound)
            independent=hi.analytic(z[:8].astype(np.float64))
            assert np.max(np.abs(independent-out[:8]))<3e-6
            maxmass=max(maxmass,r['maximum_mass_error']);records.append(r)
        print(f'dimension n={n}, fit seed={seed}: checkpoint predictions verified',flush=True)
    writecsv('dimension_verified_metrics.csv',records)
    groups=[]
    for n in [16,32,64]:
      for stage in ['base','calibrated','range_only']:
       for family in ['gaussian','diffuse','sharp','correlated']:
        g=[r for r in records if (r['n'],r['stage'],r['family'])==(n,stage,family)]
        row=dict(n=n,stage=stage,family=family,parameters=g[0]['parameters'],fits=3)
        for k in ['mean_row_tv','probability_rmse','clipped_score_fraction']:
          row[k+'_mean']=float(np.mean([r[k] for r in g]));row[k+'_fit_sd']=float(np.std([r[k] for r in g],ddof=1))
        groups.append(row)
    stream_records=[];stream_original=csvread(DATA/'dimension_stream_metrics.csv');maxstream=0.
    for n in [16,32,64]:
      ckpath=REV/'checkpoints'/f'dimension_n{n}_seed11_base.pt'
      hi=FrozenHI(ckpath);cores,cdfs=compiled(hi)
      with np.load(DATA/f'dimension_n{n}_seed11_base_gaussian.npz') as a:z=a['scores'][:128].astype(np.float64)
      x,order=hi.encode(z);own=stationary(cores,x,order);target=softmax(z)
      for seed in SEEDS:
       with np.load(DATA/f'dimension_n{n}_events_seed{seed}.npz') as a:events=a['events']
       assert events.shape==(4096,128) and events.max()<n
       for L in LS:
        p=counts_probability(events,L,order)
        r=dict(n=n,seed=seed,L=L,mean_row_tv=tv(p,target),probability_rmse=rmse(p,target),sampling_tv=tv(p,own),mass_error=0.)
        old=next(v for v in stream_original if (int(v['n']),int(v['seed']),int(v['L']))==(n,seed,L))
        for k in ['mean_row_tv','probability_rmse','sampling_tv']:
          e=abs(r[k]-float(old[k]));assert e<2e-12;maxstream=max(maxstream,e)
        stream_records.append(r)
      print(f'dimension n={n}: 8 categorical event streams verified',flush=True)
    writecsv('dimension_stream_verified.csv',stream_records)
    sg=[]
    for n in [16,32,64]:
     for L in LS:
      g=[r for r in stream_records if (r['n'],r['L'])==(n,L)];row=dict(n=n,L=L,streams=8)
      for k in ['mean_row_tv','probability_rmse','sampling_tv']:
       row[k+'_mean']=float(np.mean([r[k] for r in g]));row[k+'_stream_sd']=float(np.std([r[k] for r in g],ddof=1))
      sg.append(row)
    dump('dimension_verified_summary.json',dict(groups=groups,stream_groups=sg,statistical_unit='three separately optimized fits with a common resized warm start; eight event streams for fixed fit seed 11'))
    dump('dimension_audit.json',dict(all_passed=True,checkpoint_count=18,prediction_sets=108,
       held_out_rows_per_prediction_set=4096,range_only_sets=36,stream_records=96,
       max_checkpoint_prediction_difference=maxpred,max_aggregate_metric_difference=maxmetric,
       max_event_metric_difference=maxstream,max_float32_row_mass_error=maxmass,checkpoint_sha256=hashes))

def sampler():
    hi=FrozenHI(REV/'checkpoints/calibrated_cifar10_11.pt')
    with np.load(DATA/'prior_inputs.npz') as a:z=a['scores'].copy()
    x,order=hi.encode(z);target=softmax(z);cores,cdfs=compiled(hi);floating=stationary(cores,x,order)
    states=np.random.default_rng(620999).integers(0,16,(512,16))
    conditional=contract(cores,states);shared=np.asarray([shared_exact(cdfs,s) for s in states])
    methods={'float_independent':(cdfs,cores,None),'float_shared':(cdfs,cores,None)}
    precision=[];payload=dict(states=states,floating=conditional,shared=shared,shared_row_tv=.5*np.abs(shared-conditional).sum(-1))
    with np.load(DATA/'sampler_conditional_outputs.npz') as old:
      assert np.array_equal(states,old['states'])
      assert np.max(np.abs(shared-old['shared']))<1e-14
      assert np.max(np.abs(conditional-old['floating']))<1e-14
    for b in [8,12,16]:
      M=2**(b-1);q=[np.rint(c*M).astype(np.int64) for c in cdfs]
      for c in q:
        c[...,-1]=M;assert np.all(np.diff(c,axis=-1)>=0)
      qc=[np.diff(np.concatenate([np.zeros((*c.shape[:-1],1),dtype=np.int64),c],-1),axis=-1)/M for c in q]
      cond=contract(qc,states);own=stationary(qc,x,order);deltas=[float(.5*np.abs(c-a).sum(-1).max()) for c,a in zip(cores,qc)]
      dist=.5*np.abs(cond-conditional).sum(-1);assert dist.max()<=sum(deltas)+1e-14
      precision.append(dict(word_bits=b,fraction_bits=b-1,conditional_tv_mean=float(dist.mean()),conditional_tv_max=float(dist.max()),
        sum_core_tv_bound=float(sum(deltas)),core_maximum_tv=deltas,stationary_tv_to_float=tv(own,floating),stationary_tv_to_softmax=tv(own,target),
        parameters=sum(c.size for c in cores),packed_threshold_payload_kib=sum(c.size for c in cores)*b/(8*1024)))
      methods[f'cdf{b}_independent']=(q,qc,b);payload[f'cdf{b}']=cond
      assert np.max(np.abs(cond.sum(-1)-1))<2e-15
    original=csvread(DATA/'sampler_stream_metrics.csv');records=[];maxmetric=0.;event_mismatch=0
    for seed in SEEDS:
     with np.load(DATA/f'sampler_events_seed{seed}.npz') as a:
      saved={k:a[k].copy() for k in methods}
      state_archive=a['fsm_states_sorted'].copy() if seed==101 else None
     emitter=np.random.default_rng(seed+900019)
     for ci, trace in enumerate(finite_states(x,seed)):
      if seed==101: assert np.array_equal(trace,state_archive[ci*16:(ci+1)*16])
      ss=trace.reshape(-1,16);uniform=emitter.random((16,len(ss)))
      for name,(cdf,_,bits) in methods.items():
       bond=np.zeros(len(ss),dtype=np.int64)
       for j,c in enumerate(cdf):
        u=uniform[0] if name=='float_shared' else uniform[j]
        if bits is not None: u=np.floor(u*2**(bits-1)).astype(np.int64)
        bond=(c[ss[:,j],bond]<=u[:,None]).sum(-1)
       mismatch=int(np.count_nonzero(bond.reshape(16,128)!=saved[name][ci*16:(ci+1)*16]))
       event_mismatch+=mismatch;assert mismatch==0,(seed,name,ci,mismatch)
     for name,(_,cs,_) in methods.items():
      own=stationary(cs,x,order)
      for L in LS:
       p=counts_probability(saved[name],L,order)
       r=dict(seed=seed,L=L,method=name,mean_row_tv=tv(p,target),probability_rmse=rmse(p,target),
         tv_to_float_stationary=tv(p,floating),tv_to_own_stationary=tv(p,own),maximum_mass_error=0.)
       old=next(a for a in original if (int(a['seed']),int(a['L']),a['method'])==(seed,L,name))
       for k in ['mean_row_tv','probability_rmse','tv_to_float_stationary','tv_to_own_stationary']:
        d=abs(r[k]-float(old[k]));assert d<1e-13;maxmetric=max(maxmetric,d)
       records.append(r)
     print(f'CDF sampler seed={seed}: all 5 × 4096 categorical event schedules exactly reproduced',flush=True)
    groups=[]
    for method in methods:
     for L in LS:
      rr=[r for r in records if (r['method'],r['L'])==(method,L)];g=dict(method=method,L=L,streams=8)
      for k in ['mean_row_tv','probability_rmse','tv_to_float_stationary','tv_to_own_stationary']:
       g[k+'_mean']=float(np.mean([r[k] for r in rr]));g[k+'_stream_sd']=float(np.std([r[k] for r in rr],ddof=1))
      groups.append(g)
    np.savez_compressed(OUT/'sampler_verified_conditional.npz',**payload)
    writecsv('sampler_verified_streams.csv',records)
    dump('sampler_verified_summary.json',dict(precision=precision,groups=groups,
      floating_stationary=dict(mean_row_tv=tv(floating,target),probability_rmse=rmse(floating,target)),
      shared_conditional_tv_mean=float(payload['shared_row_tv'].mean()),shared_conditional_tv_max=float(payload['shared_row_tv'].max()),
      maximum_stream_mass_error=0.,burn_in_cycles=256,recorded_L_max=4096,
      shared_control_own_stationary_note='The reported own law is the independent contracted reference; it is a bias diagnostic for shared U.'))
    dump('sampler_audit.json',dict(all_passed=True,conditional_state_tuples=512,precision_bounds_checked=3,
      stream_records=160,fully_replayed_method_seed_streams=40,categorical_events_replayed=5*8*4096*128,
      event_mismatches=event_mismatch,max_event_metric_difference=maxmetric,hardware_rng_evaluated=False))

def convergence():
    protocol=jsonread(DATA/'prior_protocol.json');lengths=protocol['lengths'];original=csvread(DATA/'prior_stream_metrics.csv')
    assert max(lengths)==4096
    with np.load(DATA/'prior_inputs.npz') as a:
      z=a['scores'].copy();ref=a['reference'].copy();entropy=a['entropy'].copy();rep=int(a['representative_row']);row_order=a['entropy_order'].copy()
    assert np.max(np.abs(ref-softmax(z)))<1e-14
    assert rep==np.argsort(entropy)[len(entropy)//2]
    records=[];atlas={};maxprob=0.;maxmetric=0.
    for method,ckname in [('SC_base','softmax_N16_r8.pt'),('FROST','calibrated_cifar10_11.pt')]:
      hi=FrozenHI(REV/'checkpoints'/ckname);analytic=hi.analytic(z);rr=[]
      for seed in SEEDS:
       with np.load(CONV/'source_data'/f'trace_{method}_seed{seed}.npz') as a:
        ev=a['events_sorted'].copy();order=a['order'].copy();prob=a['probabilities'].copy();rows=a['row_tv'].copy();savedanalytic=a['analytic'].copy()
       assert ev.shape==(4096,128) and ev.max()<16
       assert np.max(np.abs(analytic-savedanalytic))<1e-14
       rows_new=[]
       for li,L in enumerate(lengths):
        p=counts_probability(ev,L,order);d=float(np.max(np.abs(p-prob[li])));assert d==0;maxprob=max(maxprob,d)
        rowtv=.5*np.abs(p-ref).sum(-1);assert np.max(np.abs(rowtv-rows[li]))<1e-14;rows_new.append(rowtv)
        r=dict(method=method,stream_seed=seed,L=L,mean_row_tv=tv(p,ref),probability_rmse=rmse(p,ref),
         sampling_mean_row_tv=tv(p,analytic),sampling_rmse=rmse(p,analytic),analytic_mean_row_tv=tv(analytic,ref),analytic_rmse=rmse(analytic,ref),row_sum_error=0.,min_probability=float(p.min()))
        old=next(a for a in original if (a['method'],int(a['stream_seed']),int(a['L']))==(method,seed,L))
        for k in ['mean_row_tv','probability_rmse','sampling_mean_row_tv','sampling_rmse','analytic_mean_row_tv']:
          d=abs(r[k]-float(old[k]));assert d<1e-13;maxmetric=max(maxmetric,d)
        records.append(r)
       rr.append(rows_new)
      atlas[method]=np.mean(rr,axis=0).T[row_order]
      print(f'{method}: 8 stored event traces and all 9 nested prefixes verified',flush=True)
    summary={}
    for method in ['SC_base','FROST']:
      series=[]
      for L in lengths:
       rr=[r for r in records if (r['method'],r['L'])==(method,L)];g=dict(L=L)
       for k in ['mean_row_tv','probability_rmse','sampling_mean_row_tv','sampling_rmse']:
        g[k+'_mean']=float(np.mean([r[k] for r in rr]));g[k+'_stream_sd']=float(np.std([r[k] for r in rr],ddof=1))
       series.append(g)
      summary[method]=dict(series=series,analytic_mean_row_tv=rr[0]['analytic_mean_row_tv'],analytic_rmse=rr[0]['analytic_rmse'])
    summary['paired_long_stream_gain_percent']=100*(1-summary['FROST']['series'][-1]['mean_row_tv_mean']/summary['SC_base']['series'][-1]['mean_row_tv_mean'])
    writecsv('convergence_verified_streams.csv',records);dump('convergence_verified_summary.json',summary)
    np.savez_compressed(OUT/'convergence_verified_atlas.npz',**atlas,row_entropy_order=row_order,lengths=lengths)
    dump('convergence_audit.json',dict(all_passed=True,streams=16,prefix_records=144,representative_row=rep,
      max_prefix_probability_difference=maxprob,max_metric_difference=maxmetric,recorded_L_max=4096,
      statistical_unit='eight seed means over the same 128 fixed attention rows, not independent images'))

def independent_quality(output, truth):
    p=np.clip(output.astype(np.float64),0,1);q=truth.astype(np.float64)
    mse=float(np.mean((p-q)**2));psnr=float(-10*np.log10(max(mse,1e-15)))
    x=np.arange(-5,6,dtype=np.float64);v=np.exp(-.5*(x/1.5)**2);v/=v.sum();window=np.outer(v,v)
    scores=[]
    for channel in range(3):
      a=p[:,:,channel];b=q[:,:,channel]
      ma=correlate2d(a,window,mode='valid');mb=correlate2d(b,window,mode='valid')
      va=np.maximum(correlate2d(a*a,window,mode='valid')-ma*ma,0)
      vb=np.maximum(correlate2d(b*b,window,mode='valid')-mb*mb,0)
      cov=correlate2d(a*b,window,mode='valid')-ma*mb
      scores.append(np.mean((2*ma*mb+.01**2)*(2*cov+.03**2)/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))))
    return dict(psnr=psnr,ssim=float(np.mean(scores)),mse=mse)

def bsds():
    protocol=jsonread(DATA/'bsds_protocol.json');images=protocol['images'];ids=sorted(i['image_id'] for i in images)
    assert len(ids)==len(set(ids))==200
    jpegfiles={int(p.stem):p for p in (BSDS/'data/test').glob('*.jpg')}
    runs={int(p.stem.split('_')[-1]):p for p in (BSDS/'runs').glob('image_*.npz')}
    assert set(ids)==set(jpegfiles)==set(runs)
    old={(int(r['image_id']),r['method'],int(r['L']),int(r['stream_base_seed'])):r for r in csvread(DATA/'bsds_all_metrics.csv')}
    rows=[];hashes={};maxscore=dict(psnr=0.,ssim=0.,mse=0.);maxcrop=0.;beststeps={};checkpoint_hashes={}
    for index,image_id in enumerate(ids):
      info=next(v for v in images if v['image_id']==image_id);jpeg=jpegfiles[image_id]
      assert sha(jpeg)==info['sha256'];hashes[jpeg.name]=info['sha256']
      im=Image.open(jpeg).convert('RGB');w,h=im.size;im=im.resize((round(w*96/min(w,h)),round(h*96/min(w,h))),Image.Resampling.BICUBIC)
      clean=np.asarray(im);h,w=clean.shape[:2];clean=clean[(h-64)//2:(h-64)//2+64,(w-64)//2:(w-64)//2+64].astype(np.float32)/255
      with np.load(runs[image_id]) as saved:
       truth=saved['truth'];diff=float(np.max(np.abs(truth-clean)));maxcrop=max(maxcrop,diff);assert diff==0
       keys=[k for k in saved if k!='truth'];assert len(keys)==12
       for key in keys:
        if '_L' in key:
          method,tail=key.rsplit('_L',1);sL,ss=tail.split('_S');L=int(sL);seed=int(ss)
        else:method=key;L=seed=0
        r=dict(image_id=image_id,method=method,L=L,stream_base_seed=seed,**independent_quality(saved[key],truth))
        archived=old[(image_id,method,L,seed)]
        for metric,tolerance in [('psnr',1e-10),('ssim',1e-10),('mse',1e-14)]:
          discrepancy=abs(r[metric]-float(archived[metric]));maxscore[metric]=max(maxscore[metric],discrepancy);assert discrepancy<tolerance,(image_id,key,metric,discrepancy)
        rows.append(r)
      path=BSDS/'checkpoints'/f'render_{image_id}.pt';ck=torch.load(path,weights_only=True)
      assert ck['training_steps']==1500;assert 0<ck['best_step']<=1500 and ck['best_step']%250==0
      assert ck['seed']==image_id+11;beststeps[str(image_id)]=ck['best_step']
      for name in [f'render_{image_id}.pt',f'operator_render_{image_id}.pt']:
        p=BSDS/'checkpoints'/name;assert p.exists();checkpoint_hashes[name]=sha(p)
      if (index+1)%25==0: print(f'BSDS500: {index+1}/200 targets, {(index+1)*12} outputs rescored',flush=True)
    assert len(rows)==2400;writecsv('bsds_verified_all_metrics.csv',rows)
    per_image=[]
    methods=[('Float',0),('W8A8',0),('SC_base',4096),('FROST',1024),('FROST',4096),('FROST_analytic',0)]
    for image_id in ids:
     for method,L in methods:
      g=[r for r in rows if (r['image_id'],r['method'],r['L'])==(image_id,method,L)]
      assert len(g)==(3 if L else 1);row=dict(image_id=image_id,method=method,L=L,stream_seed_count=len(g))
      for metric in ['psnr','ssim','mse']:
       a=np.asarray([r[metric] for r in g]);row[metric+'_stream_mean']=float(a.mean());row[metric+'_stream_sd']=float(a.std(ddof=1)) if len(a)>1 else 0.
      per_image.append(row)
    groups=[]
    for method,L in methods:
      rr=[r for r in per_image if (r['method'],r['L'])==(method,L)];g=dict(method=method,L=L,images=200,streams_per_image=3 if L else 1)
      for metric in ['psnr','ssim','mse']:
        vals=np.array([r[metric+'_stream_mean'] for r in rr]);g[metric+'_image_mean']=float(vals.mean());g[metric+'_between_image_sd']=float(vals.std(ddof=1));g[metric+'_min_image_mean']=float(vals.min());g[metric+'_q05_image_mean']=float(np.quantile(vals,.05))
      g['images_with_mean_psnr_ge30']=sum(r['psnr_stream_mean']>=30 for r in rr)
      raw=[r for r in rows if (r['method'],r['L'])==(method,L)];g['raw_runs_total']=len(raw);g['raw_runs_psnr_ge30']=sum(r['psnr']>=30 for r in raw);g['min_run_psnr']=min(r['psnr'] for r in raw)
      groups.append(g)
    pair=[]
    def vals(method,L):return np.asarray([next(r['psnr_stream_mean'] for r in per_image if (r['image_id'],r['method'],r['L'])==(i,method,L)) for i in ids])
    for name,method,L in [('FROST4096_minus_SC4096','SC_base',4096),('FROST4096_minus_FROST1024','FROST',1024)]:
      delta=vals('FROST',4096)-vals(method,L);idx=np.random.default_rng(20261009).integers(0,200,size=(10000,200))
      ci=np.quantile(delta[idx].mean(-1),[.025,.975])
      pair.append(dict(comparison=name,mean_gain_db=float(delta.mean()),between_image_sd_db=float(delta.std(ddof=1)),median_gain_db=float(np.median(delta)),images_improved=int((delta>0).sum()),images=200,image_bootstrap_ci95=ci.tolist(),bootstrap_resamples=10000,bootstrap_seed=20261009,unit='image-specific seed mean'))
    frost=[r for r in per_image if (r['method'],r['L'])==('FROST',4096)]
    failures=[dict(image_id=r['image_id'],psnr_stream_mean=r['psnr_stream_mean'],ssim_stream_mean=r['ssim_stream_mean']) for r in frost if r['psnr_stream_mean']<30]
    writecsv('bsds_verified_per_image.csv',per_image)
    dump('bsds_verified_summary.json',dict(groups=groups,paired_comparisons=pair,failure_tail_below30=failures,standard_deviation_unit='200 image means after averaging stochastic streams',target_specific_fitting=True,no_quality_based_exclusions=True))
    dump('bsds_audit.json',dict(all_passed=True,official_test_images=200,independently_rescored_outputs=2400,matching_reference_jpeg_hashes=200,
      checkpoint_files_checked=400,max_crop_difference=maxcrop,max_metric_differences=maxscore,reference_sha256=hashes,checkpoint_sha256=checkpoint_hashes,
      target_pixels_used_for_fitting=True,training_steps=1500,best_steps=beststeps,models_retrained_this_stage=False))

for group in args.groups:
    globals()[group]()
audit_files={p.stem:jsonread(p) for p in OUT.glob('*_audit.json') if p.stem!='stage3_audit'}
dump('stage3_audit.json',dict(all_checks_passed=all(a['all_passed'] for a in audit_files.values()),groups_completed=list(audit_files),
    elapsed_seconds=time.perf_counter()-start,environment=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=str(np.__version__),scipy=str(scipy.__version__),device='CPU'),
    hardware_evaluation=False,full_classification_or_retrieval_rerun=False,models_retrained=False,verification='frozen inference replay and independent rescoring, not newly trained models'))
print('Stage 3 audits complete:',', '.join(args.groups),flush=True)
