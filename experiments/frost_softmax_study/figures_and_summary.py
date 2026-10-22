from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study import ROOT

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
 'axes.labelsize':10,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,
 'pdf.fonttype':42,'ps.fonttype':42,'axes.grid':True,'grid.alpha':.15,'axes.axisbelow':True})
C={'strict':'#176B87','rank16':'#0E9F83','uniform':'#777777','noise':'#DD8452','rb':'#9370A7','oracle':'#A33438'}
R=ROOT/'results';F=ROOT/'figures'

def agg(df,by,metric):
 return df.groupby(by)[metric].agg(['mean','std']).fillna(0).reset_index()

def err(ax,d,x,y,label,color,**kw):
 a=agg(d,x,y);ax.errorbar(a[x],a['mean'],yerr=a['std'],marker='o',ms=4,capsize=3,label=label,color=color,**kw)

def save(fig,name):
 fig.savefig(F/f'{name}.png',dpi=190,bbox_inches='tight',facecolor='white')
 fig.savefig(F/f'{name}.pdf',bbox_inches='tight',facecolor='white');plt.close(fig)

def panel(ax,letter,title):ax.set_title(f'{letter}  {title}',loc='left',fontweight='bold',pad=10)

def main():
 m=pd.read_csv(R/'main_metrics.csv');b=pd.read_csv(R/'baseline_metrics.csv')
 s=pd.read_csv(R/'stress_metrics.csv');sb=pd.read_csv(R/'stress_baselines.csv')
 st=pd.read_csv(R/'stream_metrics.csv');o=pd.read_csv(R/'ideal_categorical_oracle.csv')
 inv=pd.read_csv(R/'invariance_metrics.csv');qt=pd.read_csv(R/'core_quantization_metrics.csv')
 # Each point is first averaged over stream repetitions WITHIN a trained model.
 stream=st.groupby(['rank','seed','init','burn','rng_share_probability','group','length','estimator']).mean(numeric_only=True).reset_index()
 m.groupby(['kind','n','states','rank']).agg({k:['mean','std'] for k in ['rmse','tv','js','kl','top1','parameters']}).to_csv(R/'main_summary.csv')
 s.groupby(['rank','family']).agg({k:['mean','std'] for k in ['rmse','tv','top1','clip_row_fraction']}).to_csv(R/'stress_summary.csv')
 stream.to_csv(R/'stream_seed_means.csv',index=False)
 primary=m[(m.kind=='strict')&(m.states==4)&(m['rank']==8)]
 fig,axs=plt.subplots(2,3,figsize=(14,8.3),layout='constrained')
 ax=axs[0,0];panel(ax,'a','Input dimensionality')
 err(ax,primary,'n','tv','Probability TT: r=8, N=4',C['strict'])
 ub=b[b.method=='uniform'];ax.plot(ub.n,ub.tv,'--',color=C['uniform'],label='Uniform output')
 dense=m[m.kind=='dense'];ax.errorbar([4],[dense.tv.mean()],yerr=[dense.tv.std()],fmt='s',color=C['noise'],label='Dense CPT: n=4')
 ax.set(xscale='log',xlabel='Softmax row length n',ylabel='Mean row total variation',xticks=[4,8,16,32,64]);ax.set_xticklabels([4,8,16,32,64]);ax.legend(loc='upper left')
 ax=axs[0,1];panel(ax,'b','Tensor-train rank')
 ds=m[(m.kind=='strict')&(m.n==16)&(m.states==4)]
 err(ax,ds,'rank','tv','Probability TT',C['strict'])
 legacy=m[(m.kind=='legacy')&(m.n==16)];err(ax,legacy,'rank','tv','TT + exact Softmax (audit)',C['noise'],linestyle='--')
 ax.axhline(float(b[(b.n==16)&(b.method=='exp_lut_16_float')].tv.iloc[0]),ls=':',color=C['oracle'],label='16-segment exp LUT (float)')
 ax.set(xlabel='TT rank r',ylabel='Mean row total variation',xticks=[2,4,8,16,32]);ax.legend()
 ax=axs[0,2];panel(ax,'c','FSM state resolution')
 err(ax,m[(m.kind=='strict')&(m.n==16)&(m['rank']==8)],'states','tv','n=16, r=8',C['strict'])
 ax.set(xlabel='States per input N',ylabel='Mean row total variation',xticks=[2,4,8]);ax.legend()
 ax=axs[1,0];panel(ax,'d','Finite-bitstream accuracy')
 d=stream[(stream.group=='main')&(stream['rank']==8)&(stream.init=='center')&(stream.burn==64)]
 for estimator,label,color in [('bits','FSM + categorical output',C['noise']),('rb','FSM + probability averaging',C['rb'])]:
  err(ax,d[d.estimator==estimator],'length','rmse',label,color)
 baseline=np.sqrt(d[d.estimator=='bits'].groupby('seed').first().mse_approx.mean())
 ax.axhline(baseline,color=C['strict'],ls='--',label='Analytical approximation floor')
 ax.plot(o.groupby('length').theoretical_rmse.mean(),color=C['oracle'],ls=':',label='Exact-probability IID sampling')
 ax.set(xscale='log',yscale='log',xlabel='Measured output cycles L (+64 burn-in)',ylabel='RMSE to FP32 Softmax');ax.legend(fontsize=7)
 ax=axs[1,1];panel(ax,'e','Distribution stress tests')
 families=['sigma_0.25','normal','sigma_2','sigma_4','student_t3','one_peak','near_tie','qk_d64']
 labels=['Flat','ID','Sharp*','Wide*','t(3)*','Peak*','Tie','QK']
 x=np.arange(len(families))
 for offset,r,col in [(-.2,8,C['strict']),(.04,16,C['rank16'])]:
  ds=s[s['rank']==r].groupby('family').tv.agg(['mean','std']).reindex(families)
  ax.bar(x+offset,ds['mean'],.23,yerr=ds['std'],capsize=2,label=f'Probability TT: r={r}',color=col)
 clip=sb[sb.method=='encoding_clipped_oracle'].set_index('family').reindex(families)
 ax.plot(x,clip.tv,'d--',ms=4,color=C['noise'],label='Input-clipping oracle')
 ax.set(xticks=x,xticklabels=labels,ylabel='Mean row total variation',xlabel='* Non-negligible input clipping');ax.legend(fontsize=7)
 ax=axs[1,2];panel(ax,'f','Shift invariance within encoding range')
 for r,col in [(8,C['strict']),(16,C['rank16']),(32,C['rb'])]:
  err(ax,inv[(inv.test=='shift')&(inv['rank']==r)],'shift','invariance_tv',f'r={r}',col)
 ax.axhline(0,color=C['uniform'],ls=':');ax.set(xlabel='Common shift added to every logit',ylabel='TV between shifted and original outputs');ax.legend()
 fig.suptitle('Softmax fitting: probability-preserving finite-state TT reference',fontsize=15,fontweight='bold')
 save(fig,'softmax_systematic_results')
 fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 ax=axs[0,0];panel(ax,'a','Separate state and emission noise')
 d=stream[(stream.group=='main')&(stream['rank']==8)&(stream.init=='stationary')]
 for est,label,col in [('rb','FSM state sampling only',C['rb']),('bits','FSM state + output sampling',C['noise'])]:
  err(ax,d[d.estimator==est],'length','rmse_to_analytic',label,col)
 ax.set(xscale='log',yscale='log',xlabel='Output cycles L',ylabel='RMSE to analytical TT output');ax.legend()
 ax=axs[0,1];panel(ax,'b','Initialization and burn-in')
 for init,burn,label,col in [('center',0,'Center reset, no burn-in',C['noise']),('center',64,'Center reset, 64-cycle burn-in',C['strict']),('stationary',0,'Stationary initialization (ideal)',C['rb'])]:
  d=stream[(stream.group=='main')&(stream['rank']==8)&(stream.init==init)&(stream.burn==burn)&(stream.estimator=='rb')]
  err(ax,d,'length','rmse_to_analytic',label,col)
 ax.set(xscale='log',yscale='log',xlabel='Measured cycles L',ylabel='RMSE to analytical TT output');ax.legend()
 ax=axs[1,0];panel(ax,'c','Dependence induced by shared RNG')
 for L,col in [(256,C['noise']),(1024,C['rb']),(4096,C['strict'])]:
  d=stream[(stream.group=='correlation')&(stream.estimator=='rb')&(stream.length==L)]
  err(ax,d,'rng_share_probability','tv_to_analytic',f'L={L}',col)
 ax.set(xlabel='Probability of sharing one uniform draw',ylabel='TV to independent-input analytical output');ax.legend()
 ax=axs[1,1];panel(ax,'d','Offline core-probability quantization')
 for r,col in [(8,C['strict']),(16,C['rank16']),(32,C['rb'])]:
  err(ax,qt[qt['rank']==r],'probability_resolution_bits','tv',f'r={r}',col)
  ds=m[(m.kind=='strict')&(m.n==16)&(m.states==4)&(m['rank']==r)]
  ax.axhline(ds.tv.mean(),ls=':',color=col,alpha=.65)
 ax.set(xlabel='Probability resolution B: grid step 2$^{-B}$',ylabel='Mean row total variation',xticks=[4,8,12]);ax.legend(title='Dotted lines: FP32 cores',fontsize=8)
 fig.suptitle('Sampling diagnostics and probability constraints',fontsize=15,fontweight='bold')
 save(fig,'softmax_sampling_diagnostics')
 # All seed convergence traces: makes optimization budget and variability visible.
 fig,axs=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
 for ax,r in zip(axs,[8,16,32]):
  for seed in [11,22,33,44,55]:
   h=pd.read_json(R/f'strict_n16_s4_r{r}_seed{seed}_history.json')
   ax.plot(h.step,h.val_kl,lw=1,label=f'Seed {seed}')
  ax.set(title=f'n=16, N=4, r={r}',xlabel='Training steps',ylabel='Validation KL (nats)',ylim=(0,.45))
 axs[-1].legend();save(fig,'softmax_training_convergence')
 # Machine-readable headline values for the final report.
 summary={}
 for r in [2,4,8,16,32]:
  d=m[(m.kind=='strict')&(m.n==16)&(m.states==4)&(m['rank']==r)]
  summary[f'rank_{r}']={k:{'mean':d[k].mean(),'sd':d[k].std()} for k in ['rmse','tv','js','kl','top1','parameters']}
 summary['baseline_n16']=b[b.n==16].to_dict('records')
 summary['streams_r8_center64']=stream[(stream.group=='main')&(stream['rank']==8)&(stream.init=='center')&(stream.burn==64)].groupby(['length','estimator'])[['rmse','tv','rmse_to_analytic','zero_mass']].mean().reset_index().to_dict('records')
 (R/'headline_results.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
