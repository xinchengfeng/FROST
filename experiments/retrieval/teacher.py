"""A complete encoder-only Transformer on synthetic associative retrieval."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import io,copy,json,math,time
import numpy as np,pandas as pd,torch
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parent

def inputs(size,seed):
    rng=np.random.default_rng(seed)
    keys=np.argsort(rng.random((size,32)),axis=1)[:,:15]
    values=rng.integers(0,4,(size,15))
    match=rng.integers(0,15,size)
    x=np.zeros((size,16,37),dtype=np.float32);b=np.arange(size)
    for j in range(15):x[b,j,keys[:,j]]=1.;x[b,j,32+values[:,j]]=1.
    x[b,15,keys[b,match]]=1.;x[:,15,36]=1.
    y=values[b,match].astype(np.int64)
    return x,y,keys,values,match

class Block(nn.Module):
    def __init__(self,d=32,h=2,ff=64):
        super().__init__();self.d=d;self.h=h
        self.ln1=nn.LayerNorm(d);self.qkv=nn.Linear(d,3*d);self.out=nn.Linear(d,d)
        self.ln2=nn.LayerNorm(d);self.fc1=nn.Linear(d,ff);self.fc2=nn.Linear(ff,d)
    def forward(self,x,trace=False):
        u=self.ln1(x);q,k,v=self.qkv(u).chunk(3,-1);b,n,d=q.shape
        split=lambda a:a.reshape(b,n,self.h,d//self.h).transpose(1,2)
        q,k,v=map(split,[q,k,v]);z=q@k.transpose(-2,-1)/math.sqrt(d//self.h)
        a=z.softmax(-1);att=(a@v).transpose(1,2).reshape(b,n,d)
        r1=self.out(att);h=x+r1;pre=self.fc1(self.ln2(h));g=F.gelu(pre);r2=self.fc2(g);out=h+r2
        if trace:return out,{'scores':z,'gelu_input':pre,'ln1_input':x,'ln2_input':h,'attn_update':r1,'ffn_update':r2}
        return out

class Transformer(nn.Module):
    def __init__(self,depth=4):
        super().__init__();self.embed=nn.Linear(37,32);self.pos=nn.Parameter(torch.zeros(16,32))
        nn.init.normal_(self.pos,std=.02)
        self.blocks=nn.ModuleList([Block() for _ in range(depth)])
        self.norm=nn.LayerNorm(32);self.head=nn.Linear(32,4)
    def forward(self,x,trace=False):
        x=self.embed(x)+self.pos
        histories=[x];details=[]
        for layer in self.blocks:
            if trace:x,t=layer(x,True);details.append(t)
            else:x=layer(x)
            histories.append(x)
        hidden=self.norm(x);logits=self.head(hidden[:,-1])
        if trace:return logits,hidden,histories,details
        return logits

def load(seed):
    ck=torch.load(ROOT/'checkpoints'/f'teacher_seed{seed}.pt',weights_only=True,map_location='cpu')
    m=Transformer();m.load_state_dict(ck['state_dict']);m.eval();return m,ck

def train(seed,steps=1800):
    torch.manual_seed(seed);m=Transformer();opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.01)
    xv,yv,*_=inputs(512,310001);xv=torch.tensor(xv);yv=torch.tensor(yv)
    best=-1.;bestloss=float('inf');beststate=None;hist=[];start=time.time()
    for step in range(steps):
        x,y,*_=inputs(64,seed*100000+step);x=torch.tensor(x);y=torch.tensor(y)
        m.train();pred=m(x);loss=F.cross_entropy(pred,y);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
        opt.param_groups[0]['lr']=.002*(.1+.9*.5*(1+math.cos(math.pi*(step+1)/steps)))
        if (step+1)%100==0 or step+1==steps:
            m.eval()
            with torch.no_grad():p=m(xv);acc=(p.argmax(-1)==yv).float().mean().item();vl=F.cross_entropy(p,yv).item()
            hist.append(dict(step=step+1,train_loss=float(loss),validation_accuracy=acc,validation_loss=vl))
            if acc>best or (acc==best and vl<bestloss):best=acc;bestloss=vl;beststate=copy.deepcopy(m.state_dict());beststep=step+1
            if (step+1)%300==0:print('teacher',seed,step+1,acc,flush=True)
    ck=dict(seed=seed,state_dict=beststate,history=hist,best_step=beststep,validation_accuracy=best,validation_loss=bestloss,steps=steps,seconds=time.time()-start)
    buf=io.BytesIO();torch.save(ck,buf);(ROOT/'checkpoints'/f'teacher_seed{seed}.pt').write_bytes(buf.getvalue())
    pd.DataFrame(hist).to_csv(ROOT/'source_data'/f'teacher_history_seed{seed}.csv',index=False)
    print('finished',seed,best,'seconds',time.time()-start,flush=True)
    return {k:v for k,v in ck.items() if k not in ['state_dict','history']}

if __name__=='__main__':
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:rows=list(pool.map(train,[11,22,33]))
    (ROOT/'source_data'/'teacher_training_summary.json').write_text(json.dumps(rows,indent=2))
