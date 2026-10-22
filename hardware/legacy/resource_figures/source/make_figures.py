from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
D = json.loads((ROOT / 'source' / 'stage3_summary.json').read_text())
FP = {d['design']: d for d in D['fpga']}
COL = {'fp32_1':'#75859A', 'fp32_4':'#64528A', 'fx16':'#CB8A39', 'frost':'#008C85'}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],
    'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'axes.linewidth':.65,
    'xtick.labelsize':7,'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42,
    'svg.fonttype':'none','savefig.facecolor':'white'})

def save(fig, name):
    fig.savefig(ROOT/'Images'/f'{name}.pdf')
    fig.savefig(ROOT/'Images'/f'{name}.svg')
    fig.savefig(ROOT/f'{name}.png', dpi=200)
    plt.close(fig)

def box(ax, x,y,w,h,label,face='#F4F5F7',edge='#8B94A1',size=8):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006,rounding_size=.012',
        linewidth=.75,edgecolor=edge,facecolor=face))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size,linespacing=1.35)

def arrow(ax, p,q,color='#71808D'):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=9,color=color,linewidth=.9))

fig=plt.figure(figsize=(7.20,4.65))
ax=fig.add_axes([.045,.55,.925,.40]); ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.text(0,1.05,'a',fontsize=12,fontweight='bold',va='top')
ax.text(.055,1.05,'From weighted products to categorical value selection',fontsize=9,va='top')
box(ax,.008,.26,.135,.48,'Common inputs\n16 scores\n16 × 16 values')
box(ax,.225,.62,.215,.20,'Exponential lookup\nand sum',edge=COL['fp32_4'])
box(ax,.515,.62,.24,.20,'Value multiply–accumulate\nand normalization',edge=COL['fp32_4'],size=7.5)
box(ax,.225,.16,.215,.20,'FSM + TT sampler\nCDFs + random states',face='#E7F3F1',edge=COL['frost'])
box(ax,.515,.16,.24,.20,'Select value row\naccumulate and scale',face='#E7F3F1',edge=COL['frost'])
box(ax,.845,.26,.145,.48,'Common output\n16 components')
for yy in [.72,.26]:
    cc=COL['fp32_4'] if yy>.5 else COL['frost']
    arrow(ax,(.145,.5),(.219,yy),cc);arrow(ax,(.445,yy),(.508,yy),cc);arrow(ax,(.76,yy),(.838,.5),cc)
ax.text(.225,.87,'FP32 / FX16',fontsize=8,color=COL['fp32_4'],fontweight='bold')
ax.text(.225,.415,'FROST',fontsize=8,color=COL['frost'],fontweight='bold')
ax.text(.50,.02,'Declared kernel includes control, conversion, buffering, sorting and random generation.',
        ha='center',fontsize=7,color='#515A65')
fig.text(.045,.515,'b',fontsize=12,fontweight='bold',va='top')
fig.text(.095,.515,'Post-route resource allocation',fontsize=9,va='top')
names=['fp32_1','fp32_4','fx16','frost']; yy=np.arange(4)
labels=['FP32 · 1 lane','FP32 · 4 lanes','FX16 · 4 lanes','FROST']
for i,(key,title) in enumerate([('comb','COMB'),('ff','Flip-flops'),('dsp','DSP blocks'),('bram18','Block RAMs')]):
    a=fig.add_axes([.17+i*.204,.105,.167,.32])
    vals=[FP[n][key] for n in names]; xmax=max(vals)*1.34
    for y,n,v in zip(yy,names,vals):
        a.barh(y,v,height=.53,color=COL[n],zorder=3)
        if v==0:a.plot(0,y,'o',color=COL[n],ms=4,clip_on=False,zorder=5)
        a.text(v+xmax*.035,y,f'{v:,}',va='center',fontsize=7.5,color=COL[n] if n=='frost' else '#303942',fontweight='bold' if n=='frost' else 'normal')
    a.set(xlim=(0,xmax),ylim=(3.7,-.65),yticks=yy)
    a.set_yticklabels(labels if i==0 else [])
    a.tick_params(axis='y',length=0,pad=7)
    a.set_title(title,pad=8,fontweight='bold',fontsize=8)
    a.spines[['top','right','left']].set_visible(False)
    a.grid(axis='x',color='#E8ECEF',linewidth=.55,zorder=0)
    if i==0:a.set_xticks([0,10000,20000],['0','10k','20k'])
    elif i==1:a.set_xticks([0,1000,2000],['0','1k','2k'])
    elif i==2:a.set_xticks([0,8,16])
    else:a.set_xticks([0,1,2,3])
fig.text(.17,.028,'One FROST circuit supports L = 64, 256 and 1024; no resource reallocation.',fontsize=7.3,color='#515A65')
save(fig,'FROST_hardware_stage3')

P={(x['design'],x['L']):x for x in D['performance']}
A={(x['design'],x['L']):x for x in D['asic']}
keys=[('fp32_1',0),('fp32_4',0),('fx16',0),('frost',64),('frost',256),('frost',1024)]
labels=['FP32 · 1 lane','FP32 · 4 lanes','FX16 · 4 lanes','FROST · L = 64','FROST · L = 256','FROST · L = 1024']
fig=plt.figure(figsize=(7.20,3.25))
ax=fig.add_axes([.09,.22,.37,.59])
base=A[('fp32_4',0)]; short=A[('frost',64)]
ratios=[short['area_um2']/base['area_um2'],short['total_mW']/base['total_mW'],short['tile_cycles']/base['tile_cycles'],short['tile_energy_uJ']/base['tile_energy_uJ']]
ax.axhline(1,color=COL['fp32_4'],lw=1.1,ls='--',zorder=2)
ax.bar(np.arange(4),ratios,color=COL['frost'],width=.57,zorder=3)
for i,v in enumerate(ratios):ax.text(i,v-.085 if i==2 else v+.035,f'−{100*(1-v):.1f}%',ha='center',fontsize=8,color='white' if i==2 else COL['frost'],fontweight='bold')
ax.text(1.5,1.055,'FP32, 4 lanes = 1',ha='center',fontsize=7.7,color=COL['fp32_4'])
ax.set(ylim=(0,1.21),xticks=np.arange(4),xticklabels=['Area','Power','Latency','Energy'],yticks=[0,.5,1])
ax.set_ylabel('Normalized to four-lane FP32')
ax.set_title('FROST at L = 64',fontsize=9,pad=12)
ax.grid(axis='y',color='#E8ECEF',lw=.6,zorder=0);ax.spines[['top','right']].set_visible(False)
ax=fig.add_axes([.615,.22,.35,.59])
for n,lab in [('fp32_1','FP32 · 1 lane'),('fp32_4','FP32 · 4 lanes'),('fx16','FX16 · 4 lanes')]:
    ax.axhline(A[(n,0)]['tile_energy_uJ'],color=COL[n],ls='--',lw=1,label=lab)
xx=[64,256,1024]; yy=[A[('frost',v)]['tile_energy_uJ'] for v in xx]
ee=[A[('frost',v)]['tile_energy_uJ_sd'] for v in xx]
ax.errorbar(xx,yy,yerr=ee,fmt='o-',color=COL['frost'],lw=1.3,ms=4,capsize=2,zorder=4)
for x,y in zip(xx,yy):ax.annotate(f'{y:.3f}',(x,y),xytext=(7,6 if x==64 else -2),textcoords='offset points',fontsize=7.3,color=COL['frost'],va='bottom')
ax.set_xscale('log',base=2);ax.set(xlim=(48,1750),ylim=(0,4.95),xticks=xx,xticklabels=['64','256','1024'])
ax.set_xlabel('Stream length, L');ax.set_ylabel('Estimated energy per tile (µJ)')
ax.set_title('Fixed FROST resource allocation',fontsize=9,pad=12)
ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#E8ECEF',lw=.6,zorder=0)
ax.legend(loc='upper left',frameon=False,fontsize=7,handlelength=1.9,borderpad=.2)
fig.text(.035,.925,'a',fontsize=12,fontweight='bold');fig.text(.55,.925,'b',fontsize=12,fontweight='bold')
fig.text(.52,.05,'Nangate45 · 15 MHz · pre-layout estimates · 16-query tiles',ha='center',fontsize=7.5,color='#515A65')
save(fig,'FROST_hardware_tradeoffs')
print('Created two vector PDF figures and editable SVGs from unchanged stage-3 data.')
