"""Vector scientific artwork from archived measured source data only."""
from pathlib import Path
import io,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,LinearSegmentedColormap
from matplotlib.ticker import FixedLocator,FuncFormatter,NullLocator

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'source_data';FIG=ROOT/'figures'
BLUE='#0072B2';TEAL='#009E73';GREY='#7E8790';ORANGE='#D55E00';INK='#22272C'
CONFIGS=[(8,4,GREY,'o'),(16,4,TEAL,'s'),(16,8,BLUE,'D')]

plt.rcParams.update({'font.family':'Nimbus Sans','font.size':8,'axes.labelsize':8,
    'axes.titlesize':8,'axes.titleweight':'normal','axes.labelcolor':INK,'text.color':INK,
    'xtick.color':INK,'ytick.color':INK,'xtick.labelsize':7.2,'ytick.labelsize':7.2,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.,
    'xtick.major.width':1.,'ytick.major.width':1.,'xtick.major.size':3,'ytick.major.size':3,
    'legend.fontsize':7,'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42,
    'svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white',
    'mathtext.fontset':'dejavusans','lines.linewidth':1.3})

def tag(ax,letter,title):
    ax.text(-.16,1.08,letter,transform=ax.transAxes,fontweight='bold',fontsize=9,va='bottom',ha='left')
    ax.set_title(title,loc='left',pad=9)
    ax.tick_params(direction='out')

def panel_a(fig,ax):
    a=pd.read_csv(DATA/'Fig3a_probabilities.csv');s=json.loads((DATA/'numerical_summary.json').read_text())
    cmap=LinearSegmentedColormap.from_list('counts',['#EFF6FA','#B2D6E8','#4A9BC6','#005184','#002B49'])
    h=ax.hexbin(a.reference,a.prediction,gridsize=65,extent=(0,.8,0,.8),mincnt=1,
        cmap=cmap,norm=LogNorm(),linewidths=0,rasterized=False)
    ax.plot([0,.8],[0,.8],color=GREY,ls='--',lw=1,label='Identity')
    ax.set(xlim=(0,.8),ylim=(0,.8),xlabel='Reference softmax probability',ylabel='HI-SMURF probability')
    ax.set_xticks([0,.2,.4,.6,.8]);ax.set_yticks([0,.2,.4,.6,.8])
    ax.text(.045,.94,f"RMSE = {s['rmse']:.5f}\nMean TV = {s['mean_tv']:.5f}",transform=ax.transAxes,
        ha='left',va='top',fontsize=7.4,linespacing=1.5)
    ax.text(.96,.045,'4,096 test rows',transform=ax.transAxes,ha='right',va='bottom',fontsize=7.2)
    cbar=fig.colorbar(h,ax=ax,fraction=.05,pad=.025)
    cbar.set_label('Entries per bin',size=7.2,labelpad=5)
    cbar.ax.tick_params(labelsize=6.8,width=1,length=2.5)
    cbar.outline.set_linewidth(1)
    tag(ax,'a','Direct probability approximation')

def panel_b(fig,ax):
    a=pd.read_csv(DATA/'Fig3b_row_errors.csv');s=json.loads((DATA/'numerical_summary.json').read_text())
    x=np.sort(a.tv.to_numpy());y=np.arange(1,len(x)+1)/len(x)
    ax.step(x,y,where='post',color=BLUE,lw=1.4)
    for value,height in [(s['median_tv'],.5),(s['tv_p95'],.95)]:
        ax.plot([value,value],[0,height],color=GREY,ls=':',lw=1)
        ax.scatter([value],[height],color=BLUE,s=18,zorder=4)
    ax.text(.97,.32,f"Median  {s['median_tv']:.3f}\n95th percentile  {s['tv_p95']:.3f}",
        transform=ax.transAxes,ha='right',va='top',fontsize=7.4,linespacing=1.5)
    ax.set(xlim=(0,np.ceil(x.max()*20)/20),ylim=(0,1.035),xlabel='Row total variation distance',ylabel='Cumulative fraction of test rows')
    ax.set_yticks([0,.25,.5,.75,1]);ax.set_xticks(np.arange(0,np.ceil(x.max()*20)/20+.001,.05))
    tag(ax,'b','Distribution of row errors')

def panel_c(fig,ax):
    a=pd.read_csv(DATA/'Fig3c_validation.csv')
    for n,r,c,m in CONFIGS:
        block=a[(a.N==n)&(a['rank']==r)]
        ax.plot(block.step,block.validation_rmse,color=c,lw=1.5,marker=m,ms=3,
            markevery=[0,4,9,14],label=f'$N={n},\\ r={r}$')
    ax.set(xlim=(0,3100),ylim=(0,.021),xlabel='Optimization step',ylabel='Validation RMSE')
    ax.set_xticks([0,1000,2000,3000],['0','1,000','2,000','3,000']);ax.set_yticks([0,.005,.01,.015,.02])
    ax.legend(loc='lower left',handlelength=1.8,labelspacing=.35)
    tag(ax,'c','Learning the joint mapping')

def panel_d(fig,ax):
    a=pd.read_csv(DATA/'Fig3d_model_comparison.csv')
    for n,r,c,m in CONFIGS:
        row=a[(a.states==n)&(a['rank']==r)].iloc[0]
        x=row.parameters/1000;y=row.rmse
        ax.scatter(x,y,c=c,marker=m,s=34,zorder=4,edgecolors='white',linewidths=1)
        offset={(8,4):(3,14),(16,4):(5,-25),(16,8):(-5,14)}[(n,r)]
        ax.annotate(f'$N={n},\\ r={r}$\n{int(row.parameters):,} parameters',(x,y),xytext=offset,
            textcoords='offset points',ha='right' if n==16 and r==8 else 'left',fontsize=7,
            va='bottom' if offset[1]>0 else 'top',color=c,linespacing=1.3)
    ax.set(xlim=(0,18.5),ylim=(0,.0205),xlabel='Trainable parameters ($\\times10^3$)',ylabel='Test RMSE')
    ax.set_xticks([0,5,10,15]);ax.set_yticks([0,.005,.01,.015,.02])
    tag(ax,'d','Parameter count and approximation error')

def panel_e(fig,ax):
    a=pd.read_csv(DATA/'Fig3e_output_error.csv');s=json.loads((DATA/'numerical_summary.json').read_text())
    block=a.groupby('L').rmse.agg(['mean','std'])
    ax.scatter(a.L,a.rmse,s=14,facecolors='none',edgecolors=BLUE,linewidths=1,alpha=.5,zorder=3)
    ax.errorbar(block.index,block['mean'],yerr=block['std'],color=BLUE,lw=1.4,marker='o',ms=4,
        capsize=2.5,elinewidth=1,label='Categorical output')
    ax.axhline(s['stream_analytic_rmse'],color=GREY,lw=1,ls='--',label='Analytic limit, same rows')
    ax.set(xscale='log',xlim=(48,1400),ylim=(0,.055),xlabel='Sampling length, $L$ (cycles)',ylabel='RMSE relative to exact softmax')
    ax.set_xticks([64,256,1024],['64','256','1,024']);ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([0,.01,.02,.03,.04,.05]);ax.legend(loc='upper right',handlelength=1.8,labelspacing=.4)
    ax.text(.97,.045,'256 burn-in cycles excluded',ha='right',va='bottom',transform=ax.transAxes,fontsize=7)
    tag(ax,'e','Finite-stream approximation')

def panel_f(fig,ax):
    a=pd.read_csv(DATA/'Fig3f_sampling_error.csv')
    for mode,c,m,label in [('categorical_bits',BLUE,'o','Categorical output'),('weighted_diagnostic',ORANGE,'s','State-average diagnostic')]:
        raw=a[a['mode']==mode];block=raw.groupby('L').sampling_rmse.agg(['mean','std'])
        ax.scatter(raw.L,raw.sampling_rmse,s=14,facecolors='none',edgecolors=c,linewidths=1,alpha=.45,zorder=3)
        ax.errorbar(block.index,block['mean'],yerr=block['std'],color=c,lw=1.4,marker=m,ms=4,
            capsize=2.5,elinewidth=1,label=label)
    base=a[(a['mode']=='categorical_bits')&(a.L==64)].sampling_rmse.mean()
    ax.plot([64,256,1024],base*np.sqrt(64/np.array([64,256,1024])),color=GREY,ls=':',lw=1.1,label='$L^{-1/2}$ reference')
    ax.set(xscale='log',yscale='log',xlim=(48,1400),ylim=(.006,.09),xlabel='Sampling length, $L$ (cycles)',ylabel='RMSE relative to analytic output')
    ax.set_xticks([64,256,1024],['64','256','1,024']);ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator([.01,.02,.05]));ax.yaxis.set_major_formatter(FuncFormatter(lambda y,pos:f'{y:g}'));ax.yaxis.set_minor_locator(NullLocator())
    ax.legend(loc='upper right',handlelength=1.8,labelspacing=.4)
    tag(ax,'f','Error from finite-state sampling')

def save(fig,name):
    # A single write avoids truncated multi-chunk image writes in remote mounts.
    for ext in ['png','pdf','svg']:
        buf=io.BytesIO();fig.savefig(buf,format=ext,dpi=400,facecolor='white')
        (FIG/f'{name}.{ext}').write_bytes(buf.getvalue())
    plt.close(fig)

def main():
    panels=[panel_a,panel_b,panel_c,panel_d,panel_e,panel_f]
    for name,pair in [('Fig3ab_softmax_fidelity',panels[:2]),('Fig3cd_model_capacity',panels[2:4]),('Fig3ef_bitstream_length',panels[4:])]:
        fig,axs=plt.subplots(1,2,figsize=(180/25.4,70/25.4))
        fig.subplots_adjust(left=.105,right=.98,bottom=.205,top=.86,wspace=.51)
        for func,ax in zip(pair,axs):func(fig,ax)
        save(fig,name)
    fig,axs=plt.subplots(3,2,figsize=(180/25.4,190/25.4))
    fig.subplots_adjust(left=.105,right=.98,bottom=.075,top=.958,wspace=.51,hspace=.65)
    for func,ax in zip(panels,axs.flat):func(fig,ax)
    save(fig,'Fig3_softmax_combined')
    print('Rendered three paired figures and the combined Figure 3; PNG/PDF/SVG.')

if __name__=='__main__':main()
