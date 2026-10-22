"""Publication figures generated solely from recorded attention results."""
from pathlib import Path
import io,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,LinearSegmentedColormap
from matplotlib.ticker import NullLocator

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'source_data';FIG=ROOT/'figures'
BLUE='#0072B2';TEAL='#009E73';GREY='#7E8790';ORANGE='#D55E00';INK='#22272C'
plt.rcParams.update({'font.family':'Nimbus Sans','font.size':8,'axes.labelsize':8,
 'axes.titlesize':8,'axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK,
 'xtick.labelsize':7.1,'ytick.labelsize':7.1,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':1.,'xtick.major.width':1.,'ytick.major.width':1.,'xtick.major.size':3,
 'ytick.major.size':3,'legend.fontsize':7,'legend.frameon':False,'pdf.fonttype':42,
 'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white',
 'mathtext.fontset':'dejavusans','lines.linewidth':1.3})

def tag(ax,letter,title):
 ax.text(-.17,1.08,letter,transform=ax.transAxes,fontweight='bold',fontsize=9,ha='left',va='bottom')
 ax.set_title(title,loc='left',pad=9);ax.tick_params(direction='out')

def save(fig,name):
 for ext in ['png','pdf','svg']:
  buf=io.BytesIO();fig.savefig(buf,format=ext,dpi=400);(FIG/f'{name}.{ext}').write_bytes(buf.getvalue())
 plt.close(fig)

def axes_pair(height=76):
 fig,axs=plt.subplots(1,2,figsize=(180/25.4,height/25.4))
 fig.subplots_adjust(left=.10,right=.982,bottom=.235,top=.86,wspace=.45)
 return fig,axs

def points(ax,x,values,color,marker='o'):
 values=np.asarray(values);jitter=np.linspace(-.027,.027,len(values))
 ax.scatter(x+jitter,values,s=13,facecolors='none',edgecolors=color,alpha=.6,lw=.9,zorder=3)
 ax.errorbar(x,values.mean(),yerr=values.std(ddof=1),fmt=marker,color=color,ms=4,capsize=2.5,elinewidth=1,zorder=5)

def main():
 df=pd.read_csv(DATA/'all_metrics.csv');ana=pd.read_csv(DATA/'analytic_metrics.csv').set_index('case')
 main=df[df.case=='normal_d32'];full=main[main.stage=='full'];base=full[(full.method=='bipolar_iid')&(full.L==256)]
 opt=full[(full.method=='improved')&(full.L==256)]
 headline={}
 for method in ['bipolar_iid','improved']:
  for L in [64,128,256,512,1024]:
   b=full[(full.method==method)&(full.L==L)]
   headline[f'{method}_L{L}']={k:{'mean':float(b[k].mean()),'sd':float(b[k].std())} for k in ['rmse','rel_frob','cosine','score_rmse','attention_tv']}
 headline['rmse_reduction_percent_L256']=float(100*(1-opt.rmse.mean()/base.rmse.mean()))
 headline['analytic_main']=ana.loc['normal_d32'].to_dict()
 (DATA/'headline_results.json').write_text(json.dumps(headline,indent=2))

 # Figure 1: output fidelity, then mechanism-level replacement controls.
 fig,axs=axes_pair(77);ax=axs[0]
 arr=np.load(DATA/'fidelity_improved.npz');y=arr['y'].ravel();p=arr['prediction'].ravel()
 pd.DataFrame({'reference':y,'prediction':p}).to_csv(DATA/'Fig_attention_fidelity_a.csv',index=False)
 cmap=LinearSegmentedColormap.from_list('counts',['#EFF6FA','#B2D6E8','#4A9BC6','#005184','#002B49'])
 lim=float(np.ceil(max(np.abs(y).max(),np.abs(p).max())*2)/2)
 h=ax.hexbin(y,p,gridsize=61,extent=(-lim,lim,-lim,lim),mincnt=1,cmap=cmap,norm=LogNorm(),linewidths=0)
 ax.plot([-lim,lim],[-lim,lim],ls='--',lw=1,color=GREY)
 ax.set(xlim=(-lim,lim),ylim=(-lim,lim),xlabel='Reference attention output',ylabel='SC attention output')
 row=opt[opt.seed==101].iloc[0]
 ax.text(.045,.95,f'RMSE = {row.rmse:.4f}\nRelative error = {row.rel_frob:.3f}',transform=ax.transAxes,va='top',fontsize=7.2,linespacing=1.45)
 ax.text(.97,.04,'128 matrices; $L=256$',transform=ax.transAxes,ha='right',fontsize=7)
 cb=fig.colorbar(h,ax=ax,fraction=.047,pad=.025);cb.set_label('Entries per bin',fontsize=7.1);cb.ax.tick_params(labelsize=6.5,width=1,length=2.5)
 tag(ax,'a','Attention output fidelity')
 ax=axs[1];stages=['qk_only','hi_only','av_only','full'];labels=['QK only','HI-SMURF\nonly','AV only','Full core']
 dd=main[(main.L==256)&main.method.isin(['bipolar_iid','improved'])];dd.to_csv(DATA/'Fig_attention_fidelity_b.csv',index=False)
 for method,color,offset,label in [('bipolar_iid',GREY,-.13,'Bipolar IID'),('improved',BLUE,.13,'Improved')]:
  means=[];sds=[]
  for i,stage in enumerate(stages):
   vals=dd[(dd.method==method)&(dd.stage==stage)].rmse
   points(ax,i+offset,vals,color);means.append(vals.mean());sds.append(vals.std())
  ax.plot([],[],color=color,marker='o',ls='none',label=label)
 ax.set(xticks=np.arange(4),xticklabels=labels,xlim=(-.5,3.5),ylim=(0,.36),ylabel='Attention output RMSE')
 ax.legend(loc='upper left');tag(ax,'b','Replacing individual stages')
 save(fig,'Fig_attention_fidelity')

 # Figure 2: each simple change and L scaling, with same-model analytic floor.
 fig,axs=axes_pair(77);ax=axs[0]
 methods=['bipolar_iid','magnitude_iid','magnitude_sobol','improved'];colors=[GREY,ORANGE,BLUE,TEAL]
 ab=full[full.L==256];ab.to_csv(DATA/'Fig_attention_efficiency_a.csv',index=False)
 for seed in sorted(ab.seed.unique()):
  vals=[ab[(ab.method==m)&(ab.seed==seed)].rmse.iloc[0] for m in methods]
  ax.plot(range(4),vals,color='#DCE0E3',lw=.7,zorder=1)
 for i,(m,color) in enumerate(zip(methods,colors)):points(ax,i,ab[ab.method==m].rmse,color)
 ax.set(xticks=range(4),xticklabels=['Bipolar\nIID','Magnitude\nIID','Magnitude\nSobol','+ Value\ncentering'],xlim=(-.45,3.45),ylim=(0,.36),ylabel='Attention output RMSE')
 ax.text(.98,.94,f"{headline['rmse_reduction_percent_L256']:.1f}% lower RMSE",transform=ax.transAxes,ha='right',va='top',fontsize=7.4)
 ax.text(.98,.83,'Fixed HI-SMURF; $L=256$',transform=ax.transAxes,ha='right',va='top',fontsize=7)
 tag(ax,'a','Ablation of simple improvements')
 ax=axs[1];lens=full[full.method.isin(['bipolar_iid','improved'])];lens.to_csv(DATA/'Fig_attention_efficiency_b.csv',index=False)
 for method,color,marker,label in [('bipolar_iid',GREY,'s','Bipolar IID'),('improved',BLUE,'o','Improved')]:
  b=lens[lens.method==method];s=b.groupby('L').rel_frob.agg(['mean','std'])
  ax.scatter(b.L,b.rel_frob,s=12,facecolors='none',edgecolors=color,lw=.9,alpha=.55)
  ax.errorbar(s.index,s['mean'],yerr=s['std'],color=color,marker=marker,ms=3.8,capsize=2.5,elinewidth=1,label=label)
 ax.axhline(ana.loc['normal_d32','rel_frob'],color=ORANGE,lw=1,ls='--',label='Analytic HI-SMURF')
 ax.set(xscale='log',xlim=(49,1350),ylim=(0,1.55),xlabel='Bitstream length, $L$',ylabel='Relative Frobenius error')
 ax.set_xticks([64,128,256,512,1024],['64','128','256','512','1,024']);ax.xaxis.set_minor_locator(NullLocator())
 ax.legend(loc='upper right',labelspacing=.35,handlelength=1.8)
 tag(ax,'b','Accuracy versus stream length')
 save(fig,'Fig_attention_efficiency')

 # Figure 3: distribution shift and an exact attention-specific invariant.
 fig,axs=axes_pair(79);ax=axs[0]
 cases=['diffuse_d32','normal_d32','sharp_d32','correlated_d32'];labels=['Diffuse\n($0.5Q$)','Standard','Sharp\n($2Q$)','Correlated\nQ, K, V']
 stress=df[(df.L==256)&(df.stage=='full')&df.case.isin(cases)&df.method.isin(['bipolar_iid','improved'])]
 stress.to_csv(DATA/'Fig_attention_robustness_a.csv',index=False)
 for method,color,offset,label in [('bipolar_iid',GREY,-.13,'Bipolar IID'),('improved',BLUE,.13,'Improved')]:
  for i,case in enumerate(cases):points(ax,i+offset,stress[(stress.case==case)&(stress.method==method)].rel_frob,color)
  ax.plot([],[],color=color,marker='o',ls='none',label=label)
 ax.scatter(range(4),ana.loc[cases,'rel_frob'],color=ORANGE,marker='_',s=80,linewidths=1.4,label='Analytic HI-SMURF',zorder=6)
 ax.set(xticks=range(4),xticklabels=labels,xlim=(-.5,3.5),ylim=(0,max(1.5,stress.rel_frob.max()*1.2)),ylabel='Relative Frobenius error')
 ax.legend(loc='upper left',labelspacing=.3,handlelength=1.3)
 tag(ax,'a','Sensitivity to input structure')
 ax=axs[1];cases=['normal_d32','offset_2_d32','offset_4_d32','offset_8_d32'];offsets=[0,2,4,8]
 off=df[(df.L==256)&(df.stage=='full')&df.case.isin(cases)&df.method.isin(['magnitude_sobol','improved'])].copy()
 off['offset']=off.case.map(dict(zip(cases,offsets)));off.to_csv(DATA/'Fig_attention_robustness_b.csv',index=False)
 for method,color,marker,label in [('magnitude_sobol',ORANGE,'s','Without value centering'),('improved',BLUE,'o','With value centering')]:
  b=off[off.method==method];s=b.groupby('offset').rmse.agg(['mean','std'])
  ax.scatter(b.offset,b.rmse,s=12,facecolors='none',edgecolors=color,lw=.9,alpha=.55)
  ax.errorbar(s.index,s['mean'],yerr=s['std'],color=color,marker=marker,ms=3.8,capsize=2.5,elinewidth=1,label=label)
 ax.set(xlim=(-.4,8.4),ylim=(0,max(.21,off.rmse.max()*1.2)),xticks=offsets,xlabel='Uniform offset added to V',ylabel='Attention output RMSE')
 ax.legend(loc='upper left',labelspacing=.35,handlelength=1.7)
 ax.text(.97,.08,'Same Q, K and centred V',transform=ax.transAxes,ha='right',fontsize=7)
 tag(ax,'b','Preserving value-shift equivariance')
 save(fig,'Fig_attention_robustness')

 # A compact supplemental dimensionality test; the token count stays 16.
 fig,ax=plt.subplots(figsize=(90/25.4,73/25.4));fig.subplots_adjust(left=.2,right=.96,bottom=.2,top=.85)
 ds=df[(df.stage=='full')&(df.L==256)&df.case.isin(['normal_d16','normal_d32','normal_d64'])&df.method.isin(['bipolar_iid','improved'])]
 ds.to_csv(DATA/'Fig_attention_dimension.csv',index=False)
 for method,color,marker,label in [('bipolar_iid',GREY,'s','Bipolar IID'),('improved',BLUE,'o','Improved')]:
  b=ds[ds.method==method];s=b.groupby('d').rel_frob.agg(['mean','std'])
  ax.errorbar(s.index,s['mean'],yerr=s['std'],color=color,marker=marker,ms=4,capsize=2.5,label=label)
 ax.set(xticks=[16,32,64],xlim=(12,68),ylim=(0,1.1),xlabel='Head dimension, $d_k=d_v$',ylabel='Relative Frobenius error')
 ax.legend(loc='upper right');ax.set_title('Head dimension at fixed token count',loc='left')
 save(fig,'Fig_attention_dimension_supplement')
 print(json.dumps(headline,indent=2))

if __name__=='__main__':main()
