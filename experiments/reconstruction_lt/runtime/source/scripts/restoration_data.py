import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image

TASKS=['denoise25','denoise50','sr2']
MODEL_SEEDS=[11,22,33]

def scaled_image(path,short=96):
    im=Image.open(path).convert('RGB');w,h=im.size
    im=im.resize((round(w*short/min(w,h)),round(h*short/min(w,h))),Image.Resampling.BICUBIC)
    return np.asarray(im).copy()

def center_image(path):
    im=scaled_image(path,96);h,w=im.shape[:2]
    return im[(h-64)//2:(h-64)//2+64,(w-64)//2:(w-64)//2+64].astype(np.float32)/255

def make_input(clean,task,seed):
    if task=='sr2':
        im=Image.fromarray(np.rint(clean*255).clip(0,255).astype(np.uint8))
        low=im.resize((32,32),Image.Resampling.BICUBIC)
        degraded=np.asarray(low.resize((64,64),Image.Resampling.BICUBIC)).astype(np.float32)/255
        condition=[0.,1.]
    else:
        sigma=int(task.replace('denoise',''))
        rng=seed if isinstance(seed,np.random.Generator) else np.random.default_rng(seed)
        degraded=clean+rng.normal(0,sigma/255,clean.shape).astype(np.float32)
        condition=[sigma/50.,0.]
    return degraded,np.asarray(condition,np.float32)

def patchify(images,conditions):
    images=np.asarray(images,dtype=np.float32)
    b,h,w,c=images.shape;assert (h,w,c)==(64,64,3)
    base=((images-.5)*2).reshape(b,16,4,16,4,3).transpose(0,1,3,5,2,4).reshape(b,256,48)
    cond=np.broadcast_to(np.asarray(conditions,np.float32)[:,None,:],(b,256,2))
    return np.concatenate([base,cond],axis=-1).copy()

def unpatchify(tokens):
    b=len(tokens)
    return tokens.reshape(b,16,16,3,4,4).transpose(0,1,4,2,5,3).reshape(b,64,64,3)

def fixed_split(root,split,task,seedbase):
    files=sorted((root/split).glob('*.jpg'),key=lambda p:int(p.stem))
    ids=np.asarray([int(p.stem) for p in files])
    clean=np.stack([center_image(p) for p in files])
    examples=[make_input(im,task,seedbase+int(i)*97+TASKS.index(task)*131) for im,i in zip(clean,ids)]
    inputs=np.stack([e[0] for e in examples]);condition=np.stack([e[1] for e in examples])
    return dict(ids=ids,clean=clean,input=inputs,condition=condition,x=patchify(inputs,condition))

class TrainSampler:
    def __init__(self,root,seed):
        self.rng=np.random.default_rng(430000+seed)
        paths=sorted((root/'train').glob('*.jpg'),key=lambda p:int(p.stem))
        assert len(paths)==200
        self.images=[[scaled_image(p,s) for s in [80,96,128]] for p in paths]
    def batch(self,batch_size):
        clean=[];inputs=[];conds=[]
        for _ in range(batch_size):
            im=self.images[self.rng.integers(len(self.images))][self.rng.integers(3)]
            h,w=im.shape[:2];y=self.rng.integers(h-63);x=self.rng.integers(w-63)
            target=im[y:y+64,x:x+64]
            if self.rng.random()<.5:target=target[:,::-1]
            target=np.rot90(target,int(self.rng.integers(4))).copy().astype(np.float32)/255
            task=TASKS[self.rng.integers(3)]
            degraded,condition=make_input(target,task,self.rng)
            clean.append(target);inputs.append(degraded);conds.append(condition)
        return patchify(np.stack(inputs),np.stack(conds)),np.stack(clean)
