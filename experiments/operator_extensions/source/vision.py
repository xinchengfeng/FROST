"""Common four-block patch Transformer for three image datasets."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import copy, io, json, math, time, concurrent.futures
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from data import ROOT, DATASETS, load_data, patchify
torch.set_num_threads(1)
STEPS={'mnist':6000,'fashion_mnist':8000,'cifar10':12000}
MODEL_SEEDS=[11,22,33]

class Block(nn.Module):
 def __init__(self,d=32,h=2,ff=64):
  super().__init__();self.h=h
  self.ln1=nn.LayerNorm(d);self.qkv=nn.Linear(d,3*d);self.out=nn.Linear(d,d)
  self.ln2=nn.LayerNorm(d);self.fc1=nn.Linear(d,ff);self.fc2=nn.Linear(ff,d)
 def forward(self,x,trace=False):
  q,k,v=self.qkv(self.ln1(x)).chunk(3,-1);b,n,d=q.shape
  split=lambda a:a.reshape(b,n,self.h,d//self.h).transpose(1,2)
  q,k,v=map(split,[q,k,v]);z=q@k.transpose(-2,-1)/math.sqrt(d//self.h)
  a=z.softmax(-1);att=(a@v).transpose(1,2).reshape(b,n,d)
  r1=self.out(att);h=x+r1;f=self.fc1(self.ln2(h));r2=self.fc2(F.gelu(f));out=h+r2
  if trace:return out,dict(scores=z,values=v,gelu_input=f,attn_update=r1,ffn_update=r2)
  return out

class VisionTransformer(nn.Module):
 def __init__(self,input_width,d=32,depth=4,heads=2,ff=64):
  super().__init__();self.config=dict(input_width=input_width,d=d,depth=depth,heads=heads,ff=ff)
  self.heads=heads;self.pool='mean';self.embed=nn.Linear(input_width,d)
  self.pos=nn.Parameter(torch.empty(16,d));nn.init.normal_(self.pos,std=.02)
  self.blocks=nn.ModuleList([Block(d,heads,ff) for _ in range(depth)])
  self.norm=nn.LayerNorm(d);self.head=nn.Linear(d,10)
 def forward(self,x,trace=False):
  x=self.embed(x)+self.pos;hs=[x];ts=[]
  for block in self.blocks:
   if trace:x,t=block(x,True);ts.append(t)
   else:x=block(x)
   hs.append(x)
  h=self.norm(x);p=self.head(h.mean(1))
  return (p,h,hs,ts) if trace else p

def load(dataset,seed):
 ck=torch.load(ROOT/'checkpoints'/f'teacher_{dataset}_seed{seed}.pt',weights_only=True,map_location='cpu')
 m=VisionTransformer(**ck['architecture']);m.load_state_dict(ck['state_dict']);m.eval();return m,ck

def augmentation(images,dataset,rng):
 pad=4 if dataset=='cifar10' else 2
 b,c,h,w=images.shape;z=np.pad(images,((0,0),(0,0),(pad,pad),(pad,pad)),mode='reflect' if dataset=='cifar10' else 'constant')
 yy=rng.integers(0,2*pad+1,b);xx=rng.integers(0,2*pad+1,b)
 z=np.stack([z[i,:,y:y+h,x:x+w] for i,(y,x) in enumerate(zip(yy,xx))])
 if dataset!='mnist':
  flip=rng.random(b)<.5;z[flip]=z[flip,:,:,::-1]
 return z

def evaluate(m,x,y,batch=256):
 with torch.no_grad():pred=torch.cat([m(a) for a in x.split(batch)])
 return float((pred.argmax(-1)==y).float().mean()),float(F.cross_entropy(pred,y))

def train(job):
 dataset,seed=job;steps=STEPS[dataset];path=ROOT/'checkpoints'/f'teacher_{dataset}_seed{seed}.pt'
 if path.exists():
  _,ck=load(dataset,seed);return {k:v for k,v in ck.items() if k not in ['state_dict','history']}
 torch.manual_seed(seed);rng=np.random.default_rng(430000+seed);data=load_data(dataset)
 vi=data['validation_indices'];tr=data['train_indices'];xv=torch.tensor(patchify(data['images'][vi],data['mean'],data['std']));yv=torch.tensor(data['train_labels'][vi])
 m=VisionTransformer(xv.shape[-1]);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
 best=(-1.,float('inf'));beststate=None;hist=[];start=time.time()
 for step in range(steps):
  idx=rng.choice(tr,128,replace=False);im=augmentation(data['images'][idx],dataset,rng)
  x=torch.tensor(patchify(im,data['mean'],data['std']));y=torch.tensor(data['train_labels'][idx])
  m.train();logits=m(x);loss=F.cross_entropy(logits,y,label_smoothing=.05 if dataset=='cifar10' else 0.)
  opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  warm=min(1.,(step+1)/200);cos=.05+.95*.5*(1+math.cos(math.pi*(step+1)/steps));opt.param_groups[0]['lr']=.001*warm*cos
  if (step+1)%500==0 or step+1==steps:
   m.eval();acc,vl=evaluate(m,xv,yv);hist.append(dict(step=step+1,train_loss=float(loss.detach()),validation_accuracy=acc,validation_loss=vl,seconds=time.time()-start))
   if acc>best[0] or (acc==best[0] and vl<best[1]):best=(acc,vl);beststate=copy.deepcopy(m.state_dict());beststep=step+1
   print(dataset,seed,'step',step+1,'val',round(acc,4),'elapsed',round(time.time()-start,1),flush=True)
 ck=dict(dataset=dataset,seed=seed,architecture=m.config,parameters=sum(p.numel() for p in m.parameters()),state_dict=beststate,steps=steps,best_step=beststep,validation_accuracy=best[0],validation_loss=best[1],history=hist,seconds=time.time()-start)
 b=io.BytesIO();torch.save(ck,b);path.write_bytes(b.getvalue());(ROOT/'source_data'/f'training_{dataset}_{seed}.json').write_text(json.dumps(hist,indent=2))
 return {k:v for k,v in ck.items() if k not in ['state_dict','history']}

if __name__=='__main__':
 jobs=[(d,s) for s in MODEL_SEEDS for d in DATASETS]
 with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
  rows=list(pool.map(train,jobs))
 (ROOT/'source_data'/'training_summary.json').write_text(json.dumps(rows,indent=2))
