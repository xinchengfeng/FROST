"""Activation calibration of the same-size HI-SMURF operator; no task-test use."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import io,json,copy,time,concurrent.futures
import numpy as np,pandas as pd,torch
from torch import nn
from teacher import ROOT,load,inputs

class CalTT(nn.Module):
    def __init__(self,base,bound):
        super().__init__();self.bound=bound
        self.cs=nn.ParameterList([nn.Parameter(base['state_dict'][f'cs.{j}'].clone()) for j in range(16)])
    def forward(self,z):
        z,order=z.sort(-1,descending=True);z=z-z.mean(-1,keepdim=True)
        x=((z+self.bound)/(2*self.bound)).clamp(0,1);s=torch.arange(16,dtype=x.dtype)
        pi=x[...,None]**s*(1-x[...,None])**(15-s);pi=pi/pi.sum(-1,keepdim=True)
        h=torch.ones(len(z),1)
        for j,c in enumerate(self.cs):
            v=torch.einsum('bs,asr->bar',pi[:,j],c.softmax(-1));h=torch.bmm(h[:,None],v).squeeze(1)
        return torch.zeros_like(h).scatter(1,order,h)

def scores(seed,data_seed):
    m,_=load(seed);x,_,*_=inputs(128,data_seed)
    with torch.no_grad():_,_,_,t=m(torch.tensor(x),True)
    return torch.cat([a['scores'].reshape(-1,16) for a in t]).detach()

def run(job):
    seed,bound=job;torch.manual_seed(seed+777);start=time.time()
    base=torch.load(ROOT/'checkpoints'/'softmax_N16_r8.pt',weights_only=True)
    m=CalTT(base,bound);train=scores(seed,330001);val=scores(seed,330002)
    target=train.softmax(-1);vtarget=val.softmax(-1)
    opt=torch.optim.Adam(m.parameters(),lr=.01);rng=torch.Generator().manual_seed(seed+19001)
    best=float('inf');hist=[]
    with torch.no_grad():initial=(m(val)-vtarget).square().mean().sqrt().item()
    for step in range(1200):
        idx=torch.randint(len(train),(512,),generator=rng);p=m(train[idx]);loss=256*(p-target[idx]).square().mean()
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
        opt.param_groups[0]['lr']=.01*(.1+.9*.5*(1+np.cos(np.pi*(step+1)/1200)))
        if (step+1)%200==0:
            with torch.no_grad():rmse=torch.cat([m(z) for z in val.split(512)]).sub(vtarget).square().mean().sqrt().item()
            hist.append(dict(step=step+1,validation_probability_rmse=rmse))
            if rmse<best:best=rmse;state=copy.deepcopy(m.state_dict());beststep=step+1
    conf={**base['config'],'bound':float(bound)}
    ck=dict(config=conf,state_dict=state,teacher_seed=seed,calibration_train_seed=330001,calibration_validation_seed=330002,steps=1200,best_step=beststep,history=hist,validation_rmse=best,initial_validation_rmse=initial)
    path=ROOT/'checkpoints'/f'calibrated_hi_teacher{seed}_bound{bound}.pt';buf=io.BytesIO();torch.save(ck,buf);path.write_bytes(buf.getvalue())
    row=dict(teacher_seed=seed,bound=bound,initial_validation_rmse=initial,validation_rmse=best,best_step=beststep,seconds=time.time()-start,path=str(path.name))
    (ROOT/'source_data'/f'calibration_teacher{seed}_bound{bound}.json').write_text(json.dumps(row,indent=2))
    print(row,flush=True);return row

if __name__=='__main__':
    jobs=[(seed,bound) for seed in [11,22,33] for bound in [4,8]]
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as ex:rows=list(ex.map(run,jobs))
    df=pd.DataFrame(rows);df.to_csv(ROOT/'source_data'/'softmax_calibration.csv',index=False)
    best=df.loc[df.groupby('teacher_seed').validation_rmse.idxmin()]
    (ROOT/'calibration_selection.json').write_text(json.dumps({str(int(r.teacher_seed)):r.path for _,r in best.iterrows()},indent=2))
