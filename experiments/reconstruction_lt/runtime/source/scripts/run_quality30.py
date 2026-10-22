"""Matched restoration retraining plus separately labelled image-specific reconstruction.

Reconstruction inputs are coordinates only. It is training-set reconstruction,
not a held-out denoising score. No target pixels enter the inference graph.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,copy,csv,json,math,time
from pathlib import Path
import numpy as np
import torch
from restoration_model import WindowRestorer,SCWindowRestorer,load_teacher,ORIGINAL
from restoration_data import TrainSampler,center_image,make_input,patchify,TASKS
from restoration_metrics import quality
from calibrate import CalTT

IDS=[2018,43033,235098]
def save(path,obj):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)
def write(path,obj):path.write_text(json.dumps(obj,indent=2))
def coords():
    y,x=np.meshgrid(np.arange(16)/15*2-1,np.arange(16)/15*2-1,indexing='ij')
    features=[x,y]
    for frequency in np.linspace(1,8,12):
        for a in [x,y]:features.extend([np.sin(np.pi*frequency*a),np.cos(np.pi*frequency*a)])
    return np.stack(features,-1).reshape(1,256,50).astype(np.float32)
def examples(data,ids,task,seedbase):
    clean=np.stack([center_image(data/'val'/f'{id}.jpg') for id in ids])
    z=[make_input(y,task,seedbase+id*97+TASKS.index(task)*131) for y,id in zip(clean,ids)]
    return patchify(np.stack([a for a,b in z]),np.stack([b for a,b in z])),clean

def train_restoration(data,out,previous,newsteps):
    seed=11;torch.manual_seed(seed);sampler=TrainSampler(data,seed);model,old=load_teacher(previous/'checkpoints/restorer_seed11.pt')
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
    total=6000;step0=0;history=[];best=-math.inf;beststep=0
    resume=out/'checkpoints/restoration_resume.pt'
    if resume.exists():
        z=torch.load(resume,weights_only=False);model.load_state_dict(z['state_dict']);opt.load_state_dict(z['optimizer'])
        step0=z['step'];history=z['history'];best=z['best'];beststep=z['beststep'];beststate=z['beststate']
        sampler.rng.bit_generator.state=z['rng'];torch.set_rng_state(z['torch_rng'])
    ids=[int(p.stem) for p in sorted((data/'val').glob('*.jpg'),key=lambda p:int(p.stem))[:12]]
    validation={t:examples(data,ids,t,500001) for t in TASKS}
    for step in range(step0,min(step0+newsteps,total)):
        model.train();x,y=sampler.batch(12);loss=(model(torch.tensor(x))-torch.tensor(y)).square().mean()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        opt.param_groups[0]['lr']=2e-4*(.1+.9*.5*(1+math.cos(math.pi*(step+1)/total)))
        if (step+1)%250==0:
            model.eval();values={}
            with torch.no_grad():
                for task,(vx,vy) in validation.items():
                    pred=torch.cat([model(z) for z in torch.tensor(vx).split(4)]).numpy()
                    values[task]=float(np.mean([quality(p,y,task)['psnr'] for p,y in zip(pred,vy)]))
            score=float(np.mean(list(values.values())));history.append(dict(step=step+1,validation_psnr=values,training_mse=float(loss.detach())))
            if score>best:best=score;beststep=step+1;beststate=copy.deepcopy(model.state_dict())
            save(out/'checkpoints/restoration_teacher.pt',dict(architecture=model.config,state_dict=beststate,steps=step+1,best_step=beststep,model_seed=seed,train_images=200,validation_ids=ids))
            save(resume,dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),step=step+1,history=history,best=best,beststep=beststep,beststate=beststate,rng=sampler.rng.bit_generator.state,torch_rng=torch.get_rng_state()))
            write(out/'source_data/restoration_training.json',history)
            print('RESTORATION',step+1,values,flush=True)

def train_render(data,out,id,newsteps):
    seed=11+id;torch.manual_seed(seed);total=3000
    model=WindowRestorer(d=64,depth=2,heads=4,ff=128,reconstruct=True,jitter=.004)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=8e-4,weight_decay=1e-4)
    x=torch.tensor(coords());truth=center_image(data/'test'/f'{id}.jpg');target=torch.tensor(truth[None]);step0=0;history=[];best=-math.inf;beststep=0
    path=out/'checkpoints'/f'render_resume_{id}.pt'
    if path.exists():
        z=torch.load(path,weights_only=False);model.load_state_dict(z['state_dict']);opt.load_state_dict(z['optimizer']);torch.set_rng_state(z['rng'])
        step0=z['step'];history=z['history'];best=z['best'];beststate=z['beststate']
        beststep=z.get('beststep',torch.load(out/'checkpoints'/f'render_{id}.pt',weights_only=True)['best_step'])
    for step in range(step0,min(step0+newsteps,total)):
        model.train();prediction=model(x);loss=(prediction-target).square().mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        opt.param_groups[0]['lr']=8e-4*(.1+.9*.5*(1+math.cos(math.pi*(step+1)/total)))
        if (step+1)%250==0:
            model.eval()
            with torch.no_grad():p=model(x).numpy()[0]
            q=quality(p,truth,'reconstruction');history.append(dict(step=step+1,psnr=q['psnr'],ssim=q['ssim'],mse=q['mse']))
            if q['psnr']>best:best=q['psnr'];beststate=copy.deepcopy(model.state_dict());beststep=step+1
            save(out/'checkpoints'/f'render_{id}.pt',dict(architecture=model.config,state_dict=beststate,image_id=id,seed=seed,training_steps=step+1,best_step=beststep,
                 trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),target_training=True,coordinate_inputs=True))
            save(path,dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),step=step+1,history=history,best=best,beststep=beststep,beststate=beststate))
            write(out/'source_data'/f'render_training_{id}.json',history)
            print('RECONSTRUCTION',id,step+1,round(q['psnr'],3),round(q['ssim'],5),flush=True)

def calibrate(data,out,kind,id):
    path=out/'checkpoints'/('restoration_teacher.pt' if kind=='restoration' else f'render_{id}.pt');teacher,_=load_teacher(path)
    rows=[]
    with torch.no_grad():
        if kind=='restoration':
            ids=[int(p.stem) for p in sorted((data/'val').glob('*.jpg'),key=lambda p:int(p.stem))[12:20]]
            for task in TASKS:
                x,_=examples(data,ids,task,505001);_,scores=teacher(torch.tensor(x),trace=True);rows.extend(s.reshape(-1,16) for s in scores)
        else:
            _,scores=teacher(torch.tensor(coords()),trace=True);rows.extend(s.reshape(-1,16) for s in scores)
    allrows=torch.cat(rows);rng=torch.Generator().manual_seed(450011+id);indices=torch.randperm(len(allrows),generator=rng)
    k=int(len(indices)*.8);tr=allrows[indices[:k]][:8192];va=allrows[indices[k:]][:4096];target=tr.softmax(-1);vt=va.softmax(-1)
    base=torch.load(ORIGINAL/'checkpoints/softmax_N16_r8.pt',weights_only=True);bestoverall=math.inf;records=[]
    for bound in [2,4,8]:
        m=CalTT(base,bound);opt=torch.optim.Adam(m.parameters(),lr=.01);best=math.inf
        for step in range(600):
            idx=torch.randint(len(tr),(512,),generator=rng);loss=256*(m(tr[idx])-target[idx]).square().mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
            opt.param_groups[0]['lr']=.01*(.1+.9*.5*(1+math.cos(math.pi*(step+1)/600)))
            if (step+1)%200==0:
                with torch.no_grad():e=float(torch.cat([m(z) for z in va.split(512)]).sub(vt).square().mean().sqrt())
                if e<best:best=e;state=copy.deepcopy(m.state_dict())
        records.append(dict(bound=bound,validation_probability_rmse=best))
        if best<bestoverall:bestoverall=best;selected=dict(config={**base['config'],'bound':float(bound)},state_dict=state,validation_rmse=best)
    name='restoration' if kind=='restoration' else f'render_{id}'
    save(out/'checkpoints'/f'operator_{name}.pt',selected);write(out/'source_data'/f'calibration_{name}.json',dict(candidates=records,selected_bound=selected['config']['bound'],training_rows=len(tr),selection_rows=len(va),split='Random activation-row split within calibration images; not separate-image validation.'))
    print('CALIBRATION',name,records,flush=True)

def evaluate(data,out,previous,kind,id,task,L,stream):
    name=f'{kind}_{id}_{task}_L{L}'+(f'_S{stream}' if stream!=101 else '');path=out/'runs'/f'{name}.npz'
    if path.exists():return
    ck=out/'checkpoints'/('restoration_teacher.pt' if kind=='restoration' else f'render_{id}.pt');model,_=load_teacher(ck)
    if kind=='restoration':
        truth=center_image(data/'test'/f'{id}.jpg');inp,condition=make_input(truth,task,510001+id*97+TASKS.index(task)*131);x=patchify(inp[None],condition[None])
    else:truth=center_image(data/'test'/f'{id}.jpg');x=coords();inp=None
    sc=SCWindowRestorer(model,out/'checkpoints'/('operator_restoration.pt' if kind=='restoration' else f'operator_render_{id}.pt'))
    with torch.no_grad():fl=model(torch.tensor(x)).numpy()[0]
    exact=sc.run(x,mode='exact')[0];diff=float(np.max(np.abs(fl-exact)));assert diff<2e-5,diff
    outputs={'truth':truth,'float':np.clip(fl,0,1).astype(np.float32),
             'w8a8':np.clip(sc.run(x,mode='w8a8')[0],0,1).astype(np.float32),
             'analytic':np.clip(sc.run(x,mode='analytic')[0],0,1).astype(np.float32)}
    if inp is not None:outputs['input']=np.clip(inp,0,1).astype(np.float32)
    seed=stream+id*9001+11*17011;start=time.time();outputs['frost']=np.clip(sc.run(x,L=L,seed=seed,mode='full')[0],0,1).astype(np.float32)
    if L in [1024,4096]:
        base=SCWindowRestorer(model);outputs['SC_base']=np.clip(base.run(x,L=L,seed=seed,mode='full')[0],0,1).astype(np.float32)
    rows=[]
    for method,pred in outputs.items():
        if method=='truth':continue
        rows.append(dict(experiment=kind,image_id=id,task=task,method=method,L=L if method in ['frost','SC_base'] else 0,
                  **quality(pred,truth,task),target_used_for_training=kind=='render',exact_graph_max_abs_difference=diff,
                  inference_seed=seed if method in ['frost','SC_base'] else 0,stream_base_seed=stream))
    np.savez_compressed(path,**outputs);write(path.with_suffix('.json'),rows)
    print('MEASURED',name,'seconds',round(time.time()-start,1),[(r['method'],round(r['psnr'],3),round(r['ssim'],4)) for r in rows],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--previous',type=Path,required=True)
    ap.add_argument('--stage',choices=['train_restoration','train_render','calibrate','evaluate'],required=True);ap.add_argument('--steps',type=int,default=500)
    ap.add_argument('--kind',choices=['restoration','render'],default='restoration');ap.add_argument('--id',type=int,default=2018);ap.add_argument('--task',default='denoise25');ap.add_argument('--L',type=int,default=1024);ap.add_argument('--stream',type=int,default=101)
    a=ap.parse_args()
    for d in ['checkpoints','source_data','runs','figures']:(a.output/d).mkdir(parents=True,exist_ok=True)
    if a.stage=='train_restoration':train_restoration(a.data,a.output,a.previous,a.steps)
    if a.stage=='train_render':train_render(a.data,a.output,a.id,a.steps)
    if a.stage=='calibrate':calibrate(a.data,a.output,a.kind,a.id)
    if a.stage=='evaluate':evaluate(a.data,a.output,a.previous,a.kind,a.id,a.task,a.L,a.stream)
