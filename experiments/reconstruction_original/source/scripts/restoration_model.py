"""Windowed Transformer plus the actual original FROST bit kernels.

All learned products and nonlinear operators are simulated with actual bits
in full mode. Reflection, partition, residual addition and clipping are numeric.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import math,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(1)
ORIGINAL=Path(__file__).resolve().parents[1]/'original'
sys.path.insert(0,str(ORIGINAL))
import stochastic_model as sm
from attention_core import stochastic_qk,stochastic_av
from scipy.special import ndtr
from restoration_data import unpatchify
import packed_kernels as pk
sm.linear=pk.linear
sm.layer_norm=pk.layer_norm
stochastic_qk=pk.stochastic_qk
stochastic_av=pk.stochastic_av

def partition_torch(x,shift):
    b,n,d=x.shape;z=x.reshape(b,16,16,d)
    if shift:z=F.pad(z.permute(0,3,1,2),(shift,shift,shift,shift),mode='reflect').permute(0,2,3,1)
    side=16+2*shift
    return z.reshape(b,side//4,4,side//4,4,d).permute(0,1,3,2,4,5).reshape(-1,16,d)

def reverse_torch(w,b,shift):
    side=16+2*shift;d=w.shape[-1]
    z=w.reshape(b,side//4,side//4,4,4,d).permute(0,1,3,2,4,5).reshape(b,side,side,d)
    if shift:z=z[:,shift:-shift,shift:-shift]
    return z.reshape(b,256,d)

def partition_numpy(x,shift):
    b,n,d=x.shape;z=x.reshape(b,16,16,d)
    if shift:z=np.pad(z,((0,0),(shift,shift),(shift,shift),(0,0)),mode='reflect')
    side=16+2*shift
    return z.reshape(b,side//4,4,side//4,4,d).transpose(0,1,3,2,4,5).reshape(-1,16,d)

def reverse_numpy(w,b,shift):
    side=16+2*shift;d=w.shape[-1]
    z=w.reshape(b,side//4,side//4,4,4,d).transpose(0,1,3,2,4,5).reshape(b,side,side,d)
    if shift:z=z[:,shift:-shift,shift:-shift]
    return z.reshape(b,256,d)

class WindowBlock(nn.Module):
    def __init__(self,d=48,heads=3,ff=96,shift=0):
        super().__init__();self.h=heads;self.shift=shift
        self.ln1=nn.LayerNorm(d);self.qkv=nn.Linear(d,3*d);self.out=nn.Linear(d,d)
        self.ln2=nn.LayerNorm(d);self.fc1=nn.Linear(d,ff);self.fc2=nn.Linear(ff,d)
    def forward(self,x,trace=False):
        b,n,d=x.shape;q,k,v=self.qkv(self.ln1(x)).chunk(3,-1)
        def split(z):return partition_torch(z,self.shift).reshape(-1,16,self.h,d//self.h).permute(0,2,1,3)
        q,k,v=map(split,[q,k,v]);scores=q@k.transpose(-2,-1)/math.sqrt(d//self.h)
        weights=scores.softmax(-1);att=(weights@v).permute(0,2,1,3).reshape(-1,16,d)
        att=reverse_torch(att,b,self.shift);h=x+self.out(att)
        y=h+self.fc2(F.gelu(self.fc1(self.ln2(h))))
        return (y,scores) if trace else y

class WindowRestorer(nn.Module):
    def __init__(self,d=48,depth=4,heads=3,ff=96,reconstruct=False,jitter=0.):
        super().__init__();self.config=dict(d=d,depth=depth,heads=heads,ff=ff,reconstruct=reconstruct,jitter=jitter)
        self.reconstruct=reconstruct;self.jitter=jitter
        self.heads=heads;self.pool='patch';self.embed=nn.Linear(50,d)
        self.pos=nn.Parameter(torch.empty(256,d),requires_grad=not reconstruct)
        if reconstruct:nn.init.zeros_(self.pos)
        else:nn.init.normal_(self.pos,std=.01)
        self.blocks=nn.ModuleList([WindowBlock(d,heads,ff,0 if i%2==0 else 2) for i in range(depth)])
        self.norm=nn.LayerNorm(d);self.head=nn.Linear(d,48)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,tokens,trace=False):
        base=torch.full_like(tokens[...,:48],.5) if self.reconstruct else tokens[...,:48]/2+.5
        x=self.embed(tokens)+self.pos;scores=[]
        for block in self.blocks:
            if trace:x,z=block(x,True);scores.append(z)
            else:x=block(x)
            if self.training and self.jitter:x=x+torch.randn_like(x)*self.jitter
        pixels=base+self.head(self.norm(x))/2
        b=len(pixels);y=pixels.reshape(b,16,16,3,4,4).permute(0,1,4,2,5,3).reshape(b,64,64,3)
        return (y,scores) if trace else y

class SCWindowRestorer(sm.SCTransformer):
    def __init__(self,teacher,hi_path=None):
        super().__init__(teacher,hi_path);self.reconstruct=teacher.reconstruct
    def run(self,tokens,L=256,seed=101,mode='full',source='sobol',encoding='magnitude'):
        w=self.w;counter=0;all_linear=mode=='full';all_nonlinear=mode in ['full','analytic'];analytic=mode=='analytic'
        def take():
            nonlocal counter
            counter+=1;return seed+counter*10009
        def lin(a,name):
            s=take();weight=w[name+'.weight'];bias=w[name+'.bias']
            if mode=='w8a8':return sm.quant_linear(a,weight,bias)
            if all_linear:return sm.linear(a,weight,bias,L,s,source,encoding)
            return a@weight.T+bias
        def norm(a,name):
            s=take()
            if all_nonlinear:return sm.layer_norm(a,w[name+'.weight'],w[name+'.bias'],self.tables,L,s,source,analytic)
            c=a-a.mean(-1,keepdims=True)
            return c/np.sqrt((c*c).mean(-1,keepdims=True)+1e-5)*w[name+'.weight']+w[name+'.bias']
        base=np.full_like(tokens[...,:48],.5) if self.reconstruct else tokens[...,:48]/2+.5
        x=lin(tokens,'embed')+w['pos']
        for layer in range(self.depth):
            name=f'blocks.{layer}';shift=0 if layer%2==0 else 2
            q,k,v=np.split(lin(norm(x,name+'.ln1'),name+'.qkv'),3,axis=-1);b,n,d=q.shape
            def split(z):return partition_numpy(z,shift).reshape(-1,16,self.heads,d//self.heads).transpose(0,2,1,3).reshape(-1,16,d//self.heads)
            q,k,v=map(split,[q,k,v]);s=take()
            if mode in ['full','analytic']:
                z=q@k.swapaxes(-1,-2)/math.sqrt(d//self.heads) if analytic else stochastic_qk(q,k,L,s,source,encoding)
                a=self.hi.analytic(z) if analytic else self.hi.stream(z,L,s+3901)
                att=a@v if analytic else stochastic_av(a,v,L,s+5903,source,encoding,center=True)
            elif mode=='w8a8':
                qi,qs=sm.q8(q);ki,ks=sm.q8(k)
                z=(qi@ki.swapaxes(-1,-2)).astype(np.float64)*qs*ks.swapaxes(-1,-2)/math.sqrt(d//self.heads)
                a=np.exp(z-z.max(-1,keepdims=True));a/=a.sum(-1,keepdims=True)
                ai=np.rint(a*255).clip(0,255).astype(np.int32);vi,vs=sm.q8(v,axis=1)
                att=(ai@vi).astype(np.float64)*vs/255
            else:
                z=q@k.swapaxes(-1,-2)/math.sqrt(d//self.heads)
                a=np.exp(z-z.max(-1,keepdims=True));a/=a.sum(-1,keepdims=True);att=a@v
            att=att.reshape(-1,self.heads,16,d//self.heads).transpose(0,2,1,3).reshape(-1,16,d)
            h=x+lin(reverse_numpy(att,b,shift),name+'.out')
            f=lin(norm(h,name+'.ln2'),name+'.fc1');s=take()
            f=sm.gelu(f,self.tables,L,s,source,analytic) if all_nonlinear else f*ndtr(f)
            x=h+lin(f,name+'.fc2')
        return unpatchify(base+lin(norm(x,'norm'),'head')/2)

def load_teacher(path):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    model=WindowRestorer(**ck['architecture']);model.load_state_dict(ck['state_dict']);model.eval()
    return model,ck
