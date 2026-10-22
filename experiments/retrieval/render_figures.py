"""NC-style vector figures from recorded nested Transformer experiments."""
from pathlib import Path
import io,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator,FixedLocator,FuncFormatter
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'source_data';FIG=ROOT/'figures'
BLUE='#0072B2';TEAL='#009E73';GREY='#7E8790';ORANGE='#D55E00';INK='#22272C'
plt.rcParams.update({'font.family':'Nimbus Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':8,
 'axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK,'xtick.labelsize':7.1,
 'ytick.labelsize':7.1,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.,
 'xtick.major.width':1.,'ytick.major.width':1.,'xtick.major.size':3,'ytick.major.size':3,
 'legend.fontsize':6.8,'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42,
 'svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white','mathtext.fontset':'dejavusans','lines.linewidth':1.3})

def pair(height=80):
 f,ax=plt.subplots(1,2,figsize=(180/25.4,height/25.4));f.subplots_adjust(left=.105,right=.98,bottom=.235,top=.86,wspace=.42)
 return f,ax
def tag(ax,letter,title):
 ax.text(-.17,1.08,letter,transform=ax.transAxes,fontweight='bold',fontsize=9,ha='left',va='bottom')
 ax.set_title(title,loc='left',pad=9)
def save(f,name):
 for ext in ['png','pdf','svg']:
  b=io.BytesIO();f.savefig(b,format=ext,dpi=400);(FIG/f'{name}.{ext}').write_bytes(b.getvalue())
 plt.close(f)
def line(ax,df,x,metric,color,label,marker='o',factor=1.):
 a=df.groupby(x)[metric].agg(['mean','std'])
 ax.scatter(df[x],df[metric]*factor,s=13,facecolors='none',edgecolors=color,lw=.9,alpha=.5,zorder=3)
 ax.errorbar(a.index,a['mean']*factor,yerr=a['std']*factor,color=color,marker=marker,ms=4,capsize=2.5,elinewidth=1,label=label)
def points(ax,x,vals,color):
 vals=np.asarray(vals);ax.scatter(x+np.linspace(-.045,.045,len(vals)),vals,s=18,facecolors='none',edgecolors=color,linewidth=.9,alpha=.7,zorder=3)
 ax.errorbar(x,vals.mean(),yerr=vals.std(ddof=1),color=color,marker='o',ms=4,capsize=3,elinewidth=1,zorder=5)

def main():
 pm=pd.read_csv(DATA/'per_model_metrics.csv');raw=pd.read_csv(DATA/'all_metrics.csv');d=pd.read_csv(DATA/'depth_metrics.csv')
 d=d.groupby(['config','model_seed','L','layer'])[['query_relative_error','update_relative_error']].mean().reset_index()
 d.to_csv(DATA/'depth_per_model.csv',index=False)
 configs=[('generic_full',GREY,'Direct transfer','s'),('calibrated_full',BLUE,'Calibrated','o'),('calibrated_analytic',ORANGE,'Calibrated, analytic','D')]
 f,axs=pair(79)
 for ax,metric,letter,title,ylabel in zip(axs,['query_relative_error','update_relative_error'],['a','b'],['Query-feature error across blocks','Residual-update error across blocks'],['Relative query-feature error','Relative residual-update error']):
  b=d[(d.L==256)&d.config.isin([c[0] for c in configs])];b.to_csv(DATA/f'Fig_transformer_depth_{letter}.csv',index=False)
  for c,color,label,marker in configs:line(ax,b[b.config==c],'layer',metric,color,label,marker)
  upper=max(1.,b[metric].max()*1.18)
  ax.set(xticks=[1,2,3,4],xlim=(.8,4.2),ylim=(0,upper),xlabel='Encoder block',ylabel=ylabel)
  ax.legend(loc='center left',bbox_to_anchor=(.015,.46),labelspacing=.3,handlelength=1.5)
  tag(ax,letter,title)
 save(f,'Fig_transformer_depth')

 f,axs=pair(79);dd=pm[pm.config.isin(['generic_full','calibrated_full'])]
 dd.to_csv(DATA/'Fig_transformer_performance.csv',index=False)
 for ax in axs:
  ax.set_xscale('log');ax.set_xlim(48,1400);ax.set_xticks([64,256,1024],['64','256','1,024']);ax.xaxis.set_minor_locator(NullLocator());ax.set_xlabel('Bitstream length, $L$')
 for c,color,label,marker in configs[:2]:
  line(axs[0],dd[dd.config==c],'L','accuracy',color,label,marker,100)
  line(axs[1],dd[dd.config==c],'L','centered_logit_relative_error',color,label,marker)
 axs[0].axhline(100,color=ORANGE,ls='--',lw=1,label='Floating-point reference')
 axs[0].axhline(25,color='#B6BDC2',ls=':',lw=1,label='Chance')
 axs[0].set(ylim=(0,106),yticks=[0,25,50,75,100],ylabel='Retrieval accuracy (%)');axs[0].legend(loc='center right',bbox_to_anchor=(.99,.44),labelspacing=.3,handlelength=1.7)
 tag(axs[0],'a','End-to-end prediction accuracy')
 analytic=pm[pm.config=='calibrated_analytic'].centered_logit_relative_error.mean()
 axs[1].axhline(analytic,color=ORANGE,ls='--',lw=1,label='Calibrated, analytic')
 axs[1].set(yscale='log',ylim=(.015,2.),ylabel='Relative centred-logit error')
 axs[1].yaxis.set_major_locator(FixedLocator([.02,.05,.1,.2,.5,1.]));axs[1].yaxis.set_major_formatter(FuncFormatter(lambda y,pos:f'{y:g}'));axs[1].yaxis.set_minor_locator(NullLocator())
 axs[1].legend(loc='center right',bbox_to_anchor=(.99,.60),labelspacing=.3,handlelength=1.7);tag(axs[1],'b','Prediction fidelity beyond accuracy')
 save(f,'Fig_transformer_performance')

 f,axs=pair(80)
 series=[(['attention_only','linear_attention','calibrated_full'],['Attention\ncore only','+ All linear\nmaps','+ LN and GELU\n(full graph)'],[GREY,TEAL,BLUE],
          'a','Coverage of stochastic operators'),
         (['generic_full','calibrated_iid','calibrated_full'],['Direct transfer\nSobol','Calibrated\nIID','Calibrated\nSobol'],[GREY,ORANGE,BLUE],
          'b','Calibration and stream generation')]
 for ax,(names,labels,colors,letter,title) in zip(axs,series):
  sub=pm[(pm.L==256)&pm.config.isin(names)].copy();sub.to_csv(DATA/f'Fig_transformer_ablation_{letter}.csv',index=False)
  for seed in sorted(sub.model_seed.unique()):
   vals=[sub[(sub.model_seed==seed)&(sub.config==c)].centered_logit_relative_error.iloc[0] for c in names]
   ax.plot(range(3),vals,color='#DCE0E3',lw=.8,zorder=1)
  for i,(name,color) in enumerate(zip(names,colors)):points(ax,i,sub[sub.config==name].centered_logit_relative_error,color)
  ax.set(xticks=range(3),xticklabels=labels,xlim=(-.35,2.35),ylim=(0,max(.15,sub.centered_logit_relative_error.max()*1.18)),ylabel='Relative centred-logit error')
  tag(ax,letter,title)
 save(f,'Fig_transformer_ablation')
 # Values supporting main manuscript claims, one record per setting.
 s=pd.read_csv(DATA/'summary_metrics.csv');(DATA/'headline_results.json').write_text(json.dumps(s.to_dict(orient='records'),indent=2))
 print('Generated three core figures in PNG, PDF and SVG; all values from recorded runs.')

if __name__=='__main__':main()
