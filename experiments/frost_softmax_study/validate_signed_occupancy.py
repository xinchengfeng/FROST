import pandas as pd,torch
from study import ROOT,logits_data,encode,basis
from signed_diagnostic import SignedTT

if __name__=='__main__':
 rows=[];z=logits_data(16,8192,30016);p=basis(encode(z),4)
 gen=torch.Generator().manual_seed(47191)
 codes=torch.multinomial(p.reshape(-1,4),8,replacement=True,generator=gen).reshape(8192,16,8).permute(2,0,1).reshape(-1,16)
 for r in [16,32]:
  for seed in [11,22,33,44,55]:
   ck=torch.load(ROOT/'checkpoints'/f'signed_diagnostic_r{r}_seed{seed}.pt',weights_only=True)
   m=SignedTT(rank=r);m.load_state_dict(ck['state_dict']);m.eval()
   with torch.no_grad():w=torch.cat([m.selected(s) for s in codes.split(512)])
   rows.append({'rank':r,'seed':seed,'state_codes':len(codes),
       'occupancy_negative_fraction':(w<0).float().mean().item(),
       'occupancy_above_one_fraction':(w>1).float().mean().item(),
       'occupancy_mean_abs_sum_error':(w.sum(-1)-1).abs().mean().item()})
 pd.DataFrame(rows).to_csv(ROOT/'results'/'signed_occupancy_diagnostic.csv',index=False)
 print(pd.DataFrame(rows).groupby('rank').mean().to_string())
