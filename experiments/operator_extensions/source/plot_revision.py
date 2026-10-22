"""Editable vector figures from measured source data, with embedded fonts."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.colors import LinearSegmentedColormap

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT/'fonts').glob('*.ttf'):font_manager.fontManager.addfont(path)
plt.rcParams.update({'font.family':'FROST Nimbus Sans','font.size':8,'axes.titlesize':8.5,
    'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.8,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':.7,
    'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':3,'ytick.major.size':3})
TEAL='#0B8C86';BLUE='#08739E';ORANGE='#BC650D';INK='#25343D';GRAY='#73838D'
LIGHT='#E8F4F2';GRID='#E5EAED';PURPLE='#6C63A5'

def save(fig,name):
    for suffix in ['pdf','svg']:
        fig.savefig(ROOT/'Images'/f'{name}.{suffix}',facecolor='white',bbox_inches='tight',pad_inches=.025)
    fig.savefig(ROOT/'Images'/f'{name}.png',facecolor='white',dpi=180,bbox_inches='tight',pad_inches=.025)
    plt.close(fig)

def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color(GRAY)
    ax.tick_params(colors=INK)
    ax.set_axisbelow(True)

def panel(ax,letter):
    ax.text(-.13,1.11,letter,transform=ax.transAxes,fontweight='bold',fontsize=10,color=INK)

def sampler():
    d=json.loads((ROOT/'source_data/sampler_summary.json').read_text())
    cond=np.load(ROOT/'source_data/sampler_conditional_outputs.npz')
    fig,axs=plt.subplots(1,3,figsize=(7.08,2.55),gridspec_kw={'width_ratios':[1,1.2,1.1]})
    fig.subplots_adjust(left=.065,right=.99,bottom=.23,top=.77,wspace=.43)
    a=axs[0];clean(a);panel(a,'a')
    bits=np.array([p['word_bits'] for p in d['precision']])
    mean=np.array([p['conditional_tv_mean'] for p in d['precision']])
    maximum=np.array([p['conditional_tv_max'] for p in d['precision']])
    bound=np.array([p['sum_core_tv_bound'] for p in d['precision']])
    a.fill_between(bits,mean,maximum,color=TEAL,alpha=.12,lw=0)
    a.plot(bits,bound,'--',color=GRAY,lw=1.2,label='Core TV bound')
    a.plot(bits,maximum,'o-',color=TEAL,lw=.8,mfc='white',ms=3.8,label='Maximum observed')
    a.plot(bits,mean,'o-',color=TEAL,lw=1.3,ms=3.8,label='Mean over states')
    a.set_yscale('log');a.set_ylim(3e-5,.6);a.set_xticks(bits)
    a.set_xlabel('CDF word length (bit)');a.set_ylabel('Conditional TV to float')
    a.grid(axis='y',color=GRID,lw=.6)
    a.set_title('Quantization error is bounded',loc='left',pad=17,fontweight='bold')
    a.legend(loc='upper right',bbox_to_anchor=(1.03,1.025),frameon=False,fontsize=6.2,handlelength=1.5)
    a.text(.02,-.34,'512 fixed state tuples; N = 16, r = 8',transform=a.transAxes,fontsize=6.6,color=GRAY)
    b=axs[1];clean(b);panel(b,'b')
    choices=[('float_shared','Shared U',ORANGE,'s'),('cdf8_independent','8-bit, independent',PURPLE,'^'),
             ('cdf12_independent','12-bit, independent',TEAL,'o'),('float_independent','Float, independent',BLUE,'o')]
    for method,label,color,marker in choices:
        values=[g for g in d['groups'] if g['method']==method]
        L=np.array([g['L'] for g in values]);y=np.array([g['mean_row_tv_mean'] for g in values])
        sd=np.array([g['mean_row_tv_stream_sd'] for g in values])
        b.fill_between(L,y-sd,y+sd,color=color,alpha=.12,lw=0)
        b.plot(L,y,color=color,marker=marker,lw=1.25,ms=3.5,label=label,
            mfc='white' if method=='float_independent' else color,zorder=3 if method=='float_independent' else 2)
    b.axhline(d['floating_stationary']['mean_row_tv'],ls=':',color=GRAY,lw=1)
    b.set_xscale('log',base=2);b.set_xticks([64,256,1024,4096],['64','256','1024','4096'])
    b.set_ylim(.045,.255);b.set_xlabel('Recorded stream length L');b.set_ylabel('Mean row TV to softmax')
    b.set_title('Independence controls the error floor',loc='left',pad=17,fontweight='bold')
    b.legend(frameon=False,fontsize=6.2,loc='upper right',bbox_to_anchor=(1.05,1.035),handlelength=1.4)
    b.text(.02,-.34,'128 CIFAR-10 rows; mean ± stream s.d. (8)',transform=b.transAxes,fontsize=6.6,color=GRAY)
    c=axs[2];clean(c);panel(c,'c')
    ix=int(np.argmax(cond['shared_row_tv']));k=np.arange(1,17)
    c.bar(k-.25,cond['floating'][ix],width=.24,color=BLUE,label='Float contraction')
    c.bar(k,cond['cdf12'][ix],width=.24,color=TEAL,label='12-bit independent')
    c.bar(k+.25,cond['shared'][ix],width=.24,color=ORANGE,label='Shared U (exact)')
    c.set_xlim(.2,16.8);c.set_xticks([1,4,8,12,16]);c.set_xlabel('Sorted output category')
    c.set_ylabel('Conditional probability');c.set_ylim(0,.92)
    c.set_title('Unit mass can hide a biased law',loc='left',pad=17,fontweight='bold')
    c.legend(frameon=False,fontsize=6.2,loc='upper right',handlelength=1)
    c.text(.02,-.34,'Largest shared-U TV among the 512 tuples',transform=c.transAxes,fontsize=6.6,color=GRAY)
    c.text(.42,.48,f'TV = {cond["shared_row_tv"][ix]:.3f}\nMass = 1 in all cases',transform=c.transAxes,
        fontsize=7,color=ORANGE,ha='center')
    save(fig,'FROST_sampler_precision')

def dimensions():
    d=json.loads((ROOT/'source_data/dimension_summary.json').read_text())
    s=json.loads((ROOT/'source_data/dimension_stream_summary.json').read_text())
    families=['gaussian','diffuse','sharp','correlated'];ns=[16,32,64]
    def val(n,family,stage):return next(g for g in d['groups'] if (g['n'],g['family'],g['stage'])==(n,family,stage))['mean_row_tv_mean']
    base=np.array([[val(n,f,'base') for n in ns] for f in families])
    delta=np.array([[val(n,f,'calibrated')-val(n,f,'base') for n in ns] for f in families])
    fig,axs=plt.subplots(1,3,figsize=(7.08,2.8),gridspec_kw={'width_ratios':[1.0,1.0,1.18]})
    fig.subplots_adjust(left=.08,right=.99,bottom=.23,top=.8,wspace=.54)
    for a,letter in zip(axs,'abc'):panel(a,letter)
    a,b,c=axs
    cmap=LinearSegmentedColormap.from_list('error',['#F5F8F8','#88C5BF',TEAL,'#064F4B'])
    a.pcolormesh(np.arange(4)-.5,np.arange(5)-.5,base,cmap=cmap,vmin=0,vmax=.35,shading='flat',rasterized=False)
    a.set_xlim(-.5,2.5);a.set_ylim(3.5,-.5)
    a.set_title('Rank-8 operators at larger n',loc='left',pad=17,fontweight='bold')
    a.set_yticks(range(4),['Gaussian','Diffuse','Sharp','Correlated'])
    a.set_xticks(range(3),['16','32','64']);a.set_xlabel('Score dimension n')
    a.tick_params(length=0,pad=5)
    for (i,j),v in np.ndenumerate(base):a.text(j,i,f'{v:.3f}',ha='center',va='center',fontsize=8,color='white' if v>.16 else INK)
    for ax in [a,b]:
        for sp in ax.spines.values():sp.set_visible(False)
    a.text(0,-.41,'Mean row TV, lower is better\n16.5k / 34.9k / 71.8k parameters',transform=a.transAxes,fontsize=6.7,color=GRAY)
    dcmap=LinearSegmentedColormap.from_list('transfer',[TEAL,'#F7F8F8',ORANGE])
    b.pcolormesh(np.arange(4)-.5,np.arange(5)-.5,delta,cmap=dcmap,vmin=-.27,vmax=.27,shading='flat',rasterized=False)
    b.set_xlim(-.5,2.5);b.set_ylim(3.5,-.5)
    b.set_title('Calibration has a transfer cost',loc='left',pad=17,fontweight='bold')
    b.set_yticks(range(4),['Gaussian','Diffuse','Sharp','Correlated'])
    b.set_xticks(range(3),['16','32','64']);b.set_xlabel('Score dimension n')
    b.tick_params(length=0,pad=5)
    for (i,j),v in np.ndenumerate(delta):b.text(j,i,f'{v:+.3f}',ha='center',va='center',fontsize=8,color='white' if abs(v)>.16 else INK)
    b.text(0,-.41,'Sharp-calibrated TV minus base TV\nTeal: improvement; orange: increase',transform=b.transAxes,fontsize=6.7,color=GRAY)
    clean(c)
    for n,color,marker in zip(ns,[BLUE,TEAL,PURPLE],['o','s','^']):
        v=[g for g in s['groups'] if g['n']==n];L=np.array([g['L'] for g in v]);y=np.array([g['sampling_tv_mean'] for g in v]);sd=np.array([g['sampling_tv_stream_sd'] for g in v])
        c.fill_between(L,y-sd,y+sd,color=color,alpha=.13,lw=0)
        c.plot(L,y,color=color,marker=marker,lw=1.3,ms=3.5,label=f'n = {n}')
    c.set_xscale('log',base=2);c.set_yscale('log');c.set_xticks([64,256,1024,4096],['64','256','1024','4096'])
    c.set_ylim(.023,.43);c.set_yticks([.03,.1,.3],['0.03','0.1','0.3'])
    c.set_xlabel('Recorded stream length L');c.set_ylabel('TV to own stationary law')
    c.set_title('Sampling error grows with n',loc='left',pad=17,fontweight='bold')
    c.legend(frameon=False,loc='upper right',fontsize=7)
    c.text(0,-.41,'128 rows; 8 stream realizations\nUnit mass at every reported prefix',transform=c.transAxes,fontsize=6.7,color=GRAY)
    save(fig,'FROST_dimension_transfer')

def execution():
    fig=plt.figure(figsize=(7.08,4.75));ax=fig.add_axes([.015,.01,.97,.98]);ax.set_xlim(0,100);ax.set_ylim(0,67);ax.axis('off')
    def text(x,y,s,size=8,**kw):
        kw.setdefault('color',INK)
        ax.text(x,y,s,fontsize=size,ha='center',va='center',**kw)
    def box(x,y,w,h,label,kind='sc',size=8):
        colors={'sc':(TEAL,LIGHT),'fsm':(BLUE,'#EAF3F8'),'num':(GRAY,'#F0F3F5')}
        col,fill=colors[kind]
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.3,rounding_size=1',lw=.8,ec=col,fc=fill))
        text(x+w/2,y+h/2,label,size)
    def arrow(a,b,color=GRAY,rad=0):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,linewidth=.8,color=color,connectionstyle=f'arc3,rad={rad}'))
    text(1,64,'a',10,fontweight='bold');ax.text(4,64,'The evaluated Transformer uses pre-normalization',fontweight='bold',fontsize=9,color=INK,va='center')
    text(4,52,'X',10,fontweight='bold')
    box(11,47,15,10,'Scalar FSM\nnormalization','fsm')
    box(33,47,30,10,'Stochastic attention\nQKV → QK → HI-SMURF → AV','sc',7.7)
    box(71,47,14,10,'Output\nprojection','sc')
    ax.add_patch(Circle((93,52),2.2,fc='white',ec=GRAY,lw=.9));text(93,52,'+',11)
    arrow((6,52),(10.5,52));arrow((26.5,52),(32.5,52));arrow((63.5,52),(70.5,52));arrow((85.5,52),(90.5,52))
    ax.plot([4,4,93,93],[54,60,60,54.5],color=GRAY,lw=.8);arrow((93,55),(93,54.2));text(93,43,'H',9,fontweight='bold')
    arrow((93,49.5),(93,41));arrow((93,41),(86,41))
    box(69,36,16,10,'Scalar FSM\nnormalization','fsm')
    box(47,36,15,10,'Stochastic\nlinear W1','sc')
    box(26,36,14,10,'Scalar FSM\nGELU','fsm')
    box(5,36,14,10,'Stochastic\nlinear W2','sc')
    arrow((68.5,41),(62.5,41));arrow((46.5,41),(40.5,41));arrow((25.5,41),(19.5,41))
    ax.add_patch(Circle((2,30.5),1.8,fc='white',ec=GRAY,lw=.9));text(2,30.5,'+',10)
    ax.plot([12,12,2],[35.6,33,33],lw=.8,color=GRAY);arrow((2,33),(2,32.4))
    ax.plot([92,92,2,2],[41,32.7,32.7,32.4],lw=.8,color=GRAY)
    arrow((2,28.5),(10,28.5));text(13,28.5,'Y',9,fontweight='bold')
    text(60,29.0,'Numerical helpers: centering, range scaling, bias / residual sums, conversions',6.9,color=GRAY)
    ax.plot([0,100],[26,26],lw=.65,color=GRID)
    text(1,23,'b',10,fontweight='bold');ax.text(4,23,'A joint categorical output without a runtime output softmax',fontsize=9,color=INK,fontweight='bold',va='center')
    # Input-driven state traces feed selected core rows; independent U drives
    # latent draws through the TT chain. Intermediate ranks are illustrative.
    box(1,7,16,11,'Scores z\nSort + center\nEncode to [0, 1]','num',7.2)
    arrow((17.5,12.5),(21,12.5))
    starts=[23,42,63]
    labels=['G1(s1)','G2(s2)','Gn(sn)']
    for i,(xx,label) in enumerate(zip(starts,labels)):
        text(xx+6,19.5,'Bernoulli input → FSM state',6.4)
        box(xx,8,12,9,label,'fsm',8.4)
        # A small matrix glyph indicates a selected nonnegative row-stochastic core.
        for k in range(4):
            ax.add_patch(Rectangle((xx+1+k*2.4,8.8),2.0,1,fc=TEAL,ec='none',alpha=.22+.17*k))
        text(xx+6,4.8,f'Independent U{i+1}' if i<2 else 'Independent Un',7,color=BLUE)
        arrow((xx+6,6),(xx+6,7.8),BLUE)
    arrow((35.5,12.5),(41.5,12.5));text(38.5,15,'b1',7,color=GRAY)
    arrow((54.5,12.5),(62.5,12.5));text(58.5,15,'... b(n-1)',7,color=GRAY)
    box(81,8,18,9,'Category c\nOne-hot event','sc',7.6)
    arrow((75.5,12.5),(80.5,12.5))
    text(91,4.7,'Counts / L → attention',7,color=TEAL)
    text(49,.8,'Every event has unit mass. Independent latent draws preserve the contracted TT law.',7.2,fontweight='bold',color=TEAL)
    save(fig,'FROST_execution')

if __name__=='__main__':
    execution();sampler();dimensions()
