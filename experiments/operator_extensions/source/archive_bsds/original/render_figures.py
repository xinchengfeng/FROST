"""Publication figures from completed recorded runs; no generated metric values."""
import io,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator,FuncFormatter,FixedLocator
from evaluate import aggregate
from data import ROOT,DATASETS
DATA=ROOT/'source_data';FIG=ROOT/'figures'
BLUE='#0072B2';TEAL='#009E73';ORANGE='#D55E00';GREY='#87919A';INK='#252A2E'
LABELS={'mnist':'MNIST','fashion_mnist':'Fashion-MNIST','cifar10':'CIFAR-10'}
COLORS=dict(zip(DATASETS,[BLUE,ORANGE,TEAL]))
plt.rcParams.update({'font.family':'Nimbus Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':8,'axes.labelcolor':INK,'text.color':INK,'xtick.labelsize':7.2,'ytick.labelsize':7.2,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1,'xtick.major.width':1,'ytick.major.width':1,'legend.fontsize':6.8,'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white','mathtext.fontset':'dejavusans','lines.linewidth':1.2})

def pair(height=82):
 f,axs=plt.subplots(1,2,figsize=(180/25.4,height/25.4));f.subplots_adjust(left=.09,right=.985,bottom=.22,top=.84,wspace=.43);return f,axs
def tag(ax,letter,title):
 ax.text(-.16,1.10,letter,transform=ax.transAxes,fontweight='bold',fontsize=9,ha='left');ax.set_title(title,loc='left',pad=10)
def point(ax,x,vals,color,label=None,marker='o'):
 v=np.asarray(vals);ax.scatter(x+np.linspace(-.025,.025,len(v)),v,s=12,facecolors='none',edgecolors=color,lw=.8,alpha=.65,zorder=3)
 ax.errorbar(x,v.mean(),yerr=v.std(ddof=1),color=color,marker=marker,ms=4,capsize=2.5,elinewidth=1,label=label,zorder=5)
def save(f,name):
 for ext in ['png','pdf','svg']:
  b=io.BytesIO();f.savefig(b,format=ext,dpi=400);(FIG/f'{name}.{ext}').write_bytes(b.getvalue())
 plt.close(f)

def main():
 df=aggregate();full=df[df.split=='full_test'];diag=df[(df.split=='diagnostic')&(df.stream_seed==101)]
 assert len(full)==36 and len(diag)==63,(len(full),len(diag))
 # Figure 1: complete official test sets.
 f,axs=pair();configs=[('reference',GREY,'Floating point','s'),('w8a8',TEAL,'W8A8','D'),('calibrated_analytic',ORANGE,'FROST, analytic','^'),('calibrated_full',BLUE,'FROST, $L=256$','o')]
 full.to_csv(DATA/'Fig_dataset_accuracy.csv',index=False)
 for i,d in enumerate(DATASETS):
  for k,(config,color,label,marker) in enumerate(configs):
   v=full[(full.dataset==d)&(full.config==config)]
   point(axs[0],i+(k-1.5)*.13,100*v.accuracy,color,label if i==0 else None,marker)
  for k,(config,color,label,marker) in enumerate([configs[1],configs[3]]):
   v=full[(full.dataset==d)&(full.config==config)]
   point(axs[1],i+(k-.5)*.18,v.accuracy_delta_pp,color,label if i==0 else None,marker)
 axs[0].set(ylim=(0,104),ylabel='Test accuracy (%)');axs[0].legend(loc='lower left',labelspacing=.45)
 axs[1].axhline(0,color=GREY,ls='--',lw=1);axs[1].set_ylabel('Accuracy change from reference (pp)');axs[1].margins(y=.25)
 for ax in axs:ax.set(xticks=range(3),xticklabels=[LABELS[d] for d in DATASETS],xlim=(-.45,2.45))
 axs[1].legend(loc='best');tag(axs[0],'a','Classification on complete test sets');tag(axs[1],'b','Paired accuracy differences');save(f,'Fig_dataset_accuracy')
 # Figure 2: same prespecified diagnostic images at each stream length.
 f,axs=pair();sweep=diag[diag.config=='calibrated_full'];sweep.to_csv(DATA/'Fig_dataset_streams.csv',index=False)
 for d in DATASETS:
  z=sweep[sweep.dataset==d]
  for ax,metric,mult in [(axs[0],'accuracy_delta_pp',1),(axs[1],'centered_logit_relative_error',1)]:
   stats=z.groupby('L')[metric].agg(['mean','std']);color=COLORS[d]
   ax.scatter(z.L,z[metric]*mult,s=12,facecolors='none',edgecolors=color,lw=.8,alpha=.6)
   ax.errorbar(stats.index,stats['mean']*mult,yerr=stats['std']*mult,color=color,marker='o',ms=4,capsize=2.5,label=LABELS[d])
  baseline=diag[(diag.dataset==d)&(diag.config=='calibrated_analytic')].centered_logit_relative_error.mean()
  axs[1].axhline(baseline,color=COLORS[d],ls=':',lw=.95,alpha=.8)
 for ax in axs:
  ax.set_xscale('log',base=2);ax.set(xticks=[64,256,1024],xticklabels=['64','256','1,024'],xlim=(48,1400),xlabel='Bitstream length, $L$');ax.xaxis.set_minor_locator(NullLocator())
 axs[0].axhline(0,color=GREY,ls='--',lw=1);axs[0].set_ylabel('Accuracy change from reference (pp)');axs[0].margins(y=.25)
 axs[1].set(yscale='log',ylabel='Relative centred-logit error');axs[1].yaxis.set_major_locator(FixedLocator([.02,.05,.1,.2,.5,1]));axs[1].yaxis.set_major_formatter(FuncFormatter(lambda y,p:f'{y:g}'));axs[1].yaxis.set_minor_locator(NullLocator());axs[1].legend(loc='best')
 tag(axs[0],'a','Task accuracy versus stream length');tag(axs[1],'b','Numerical fidelity and analytic controls');save(f,'Fig_dataset_streams')
 # Figure 3: direct transfer versus same-size activation calibration.
 f,axs=pair();dd=diag[(diag.L==256)&diag.config.isin(['generic_full','calibrated_full'])];dd.to_csv(DATA/'Fig_dataset_calibration.csv',index=False)
 for ax,metric,mult,letter,title,ylabel in [(axs[0],'centered_logit_relative_error',1,'a','Calibration of numerical outputs','Relative centred-logit error'),(axs[1],'agreement',100,'b','Agreement with floating-point predictions','Prediction disagreement (%)')]:
  for i,d in enumerate(DATASETS):
   for seed in [11,22,33]:
    vals=[dd[(dd.dataset==d)&(dd.model_seed==seed)&(dd.config==c)][metric].iloc[0] for c in ['generic_full','calibrated_full']]
    vals=np.array(vals)*mult
    if metric=='agreement':vals=100-vals
    ax.plot([i-.12,i+.12],vals,color='#D6DBDE',lw=.8,zorder=1)
   for k,(config,color,label) in enumerate([('generic_full',GREY,'Direct transfer'),('calibrated_full',BLUE,'Calibrated')]):
    vals=dd[(dd.dataset==d)&(dd.config==config)][metric].values*mult
    if metric=='agreement':vals=100-vals
    point(ax,i+(-.12 if k==0 else .12),vals,color,label if i==0 else None)
  ax.set(xticks=range(3),xticklabels=[LABELS[d] for d in DATASETS],xlim=(-.45,2.45),ylabel=ylabel);ax.set_ylim(bottom=0);ax.margins(y=.2);ax.legend(loc='best');tag(ax,letter,title)
 save(f,'Fig_dataset_calibration')
 # Main table and nested stream diagnostic table have deliberately different denominators.
 cols=['accuracy','accuracy_delta_pp','agreement','macro_f1','centered_logit_relative_error','nll']
 summary=full.groupby(['dataset','config'])[cols].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(DATA/'headline_results.csv',index=False)
 repeated=df[(df.split=='diagnostic')&(df.config=='calibrated_full')&(df.L==256)]
 nested=repeated.groupby(['dataset','model_seed'])[cols].mean().reset_index();nested.to_csv(DATA/'stream_repeats_per_model.csv',index=False)
 sr=nested.groupby('dataset')[cols].agg(['mean','std']);sr.columns=['_'.join(c) for c in sr.columns];sr.reset_index().to_csv(DATA/'stream_repeats_summary.csv',index=False)
 print(summary.to_string())

if __name__=='__main__':main()
