"""Joint categorical TT with canonical input ordering: valid simplex CPT.

Sorted scores remain coupled through a single multivariate FSM/TT operator.
The output permutation is inverted. This is a constrained nonnegative-TT
variant, not an assertion about every possible HI-SMURF parameterization.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import copy,json,time,concurrent.futures
import numpy as np,pandas as pd,torch
from direct_study import ROOT,DirectTT,basis,data,batch,measures,tag

class CategoricalTT(DirectTT):
    def __init__(self,n=16,states=8,rank=4,heads='shared',encoding='sorted_center',bound=4.):
        super().__init__(n,states,rank,'shared',encoding,bound)
        with torch.no_grad():
            self.cs[0].normal_(0,.1)
            for c in self.cs[1:-1]:c.copy_(5.*torch.eye(rank)[:,None,:]+.1*torch.randn_like(c))
            self.cs[-1].normal_(0,.2)
        self._compiled=None
    def cores(self):return self._compiled if self._compiled is not None else [c.softmax(-1) for c in self.cs]
    def prepare(self):self.eval();self._compiled=[c.detach() for c in self.cores()];return self
    def raw_from_basis(self,p):
        v=torch.ones(len(p),1)
        for m,c in enumerate(self.cores()):
            a=torch.einsum('bs,asr->bar',p[:,m],c)
            v=torch.bmm(v[:,None],a).squeeze(1)
        return v
    def simplex_sum(self,raw):return raw

def run(c):
    torch.manual_seed(c['seed']);start=time.time()
    m=CategoricalTT(**{k:c[k] for k in ['n','states','rank','heads','encoding','bound']})
    z=data(c['n'],16384,110000+c['n']);p=z.softmax(-1)
    zv=data(c['n'],2048,120000+c['n']);pv=zv.softmax(-1)
    opt=torch.optim.Adam(m.parameters(),lr=.03);gen=torch.Generator().manual_seed(c['seed']+900000)
    best=float('inf');history=[]
    for step in range(c['steps']):
        idx=torch.randint(len(z),(256,),generator=gen);q=m(z[idx]);loss=c['n']**2*(q-p[idx]).square().mean()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
        if step>=.6*c['steps']:
            frac=(step-.6*c['steps'])/(.4*c['steps']);opt.param_groups[0]['lr']=.03*(.1+.9*.5*(1+np.cos(np.pi*frac)))
        if (step+1)%200==0 or step+1==c['steps']:
            vm=(batch(m,zv)-pv).square().mean().item();history.append({'step':step+1,'validation_mse':vm})
            if vm<best:best=vm;beststate=copy.deepcopy(m.state_dict());beststep=step+1
    m.load_state_dict(beststate);m.prepare()
    zt=data(c['n'],4096,130000+c['n']);pt=zt.softmax(-1);qt=batch(m,zt)
    x,_=m.encode(zt);pi=basis(x,m.states);g=torch.Generator().manual_seed(96321)
    codes=torch.multinomial(pi.reshape(-1,m.states),8,replacement=True,generator=g).reshape(len(zt),m.n,8).permute(0,2,1).reshape(-1,m.n)
    with torch.no_grad():w=torch.cat([m.selected(v) for v in codes.split(256)])
    row={**c,'kind':'categorical','parameters':sum(v.numel() for v in m.parameters()),'best_step':beststep,
        'validation_rmse':best**.5,**measures(pt,qt),'mean_sum_error':(qt.sum(-1)-1).abs().mean().item(),
        'state_negative_fraction':(w<0).float().mean().item(),'state_above_one_fraction':(w>1).float().mean().item(),
        'state_min':w.min().item(),'state_max':w.max().item(),'state_max_sum_error':(w.sum(-1)-1).abs().max().item(),
        'seconds':time.time()-start}
    name='categorical_'+tag(c)
    torch.save({'config':c,'state_dict':beststate,'history':history,'metrics':row},ROOT/'checkpoints'/f'{name}.pt')
    (ROOT/'results'/f'{name}.json').write_text(json.dumps(row,indent=2))
    np.savez_compressed(ROOT/'results'/f'{name}_predictions.npz',logits=zt.numpy(),target=pt.numpy(),prediction=qt.numpy())
    print(json.dumps(row),flush=True);return row

if __name__=='__main__':
    jobs=json.loads((ROOT/'categorical_jobs.json').read_text())
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as ex:rows=list(ex.map(run,jobs))
    pd.DataFrame(rows).to_csv(ROOT/'results'/'categorical_metrics.csv',index=False)
