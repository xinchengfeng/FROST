"""Vector hardware figures from verified stage-4/5 reports; no fitted data."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parents[2]
OUT=WORK/'outputs/FROST_hw_stage6/Images'
S=json.loads((ROOT/'stage5_summary.json').read_text())
P={r['design']:r for r in S['standard_cell_power']}
R=json.loads((ROOT/'stage4_summary.json').read_text())
F={r['design']:r['fpga'] for r in R['SC_results']}
F['hot_reuse']=R['FROST_reference']['fpga']
ORDER=['hot_reuse','sexp16','sexp256','bern8','horner5']
LABEL=['FROST','SExp, burn-in 16','SExp, burn-in 256','Bernstein-8','Horner-5']
COLOR=dict(hot_reuse='#087E83',sexp16='#B77757',sexp256='#8F5A42',bern8='#717C95',horner5='#A68D50')
INK='#263540';MUTED='#65717C';GRID='#E7EBED';TEAL=COLOR['hot_reuse']
for f in (WORK/'tmp/fig10_nc_review/fonts').glob('*.TTF'):
    font_manager.fontManager.addfont(str(f))
font_manager.findfont(font_manager.FontProperties(family='Arial'),fallback_to_default=False)
plt.rcParams.update({'font.family':'Arial','font.size':7.2,'axes.labelsize':7.2,
    'axes.titlesize':7.5,'axes.linewidth':1,'xtick.labelsize':7,'ytick.labelsize':7.2,
    'xtick.major.width':1,'ytick.major.width':1,'pdf.fonttype':42,'ps.fonttype':42,
    'svg.fonttype':'none','text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,
    'ytick.color':INK,'savefig.facecolor':'white','axes.unicode_minus':False})

def clean(ax):
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(axis='y',length=0,pad=5)
    ax.tick_params(axis='x',length=3,pad=4)
    ax.set_axisbelow(True)
    ax.grid(axis='x',color=GRID,linewidth=1)

def rows(ax,values,xmax,ticks,number,first=False):
    clean(ax)
    ax.axhspan(-.44,.44,color='#EAF5F3',zorder=0)
    for y,(name,v) in enumerate(zip(ORDER,values)):
        ax.barh(y,v,height=.55,color=COLOR[name],zorder=3)
        if v==0:ax.plot(0,y,'o',ms=3.2,color=TEAL,clip_on=False,zorder=5)
        ax.text(v+xmax*(.10 if v==0 else .025),y,number(v),va='center',ha='left',fontsize=7,
                fontweight='bold' if y==0 else 'normal',color=TEAL if y==0 else INK)
    ax.set(xlim=(0,xmax),ylim=(4.65,-.65),xticks=ticks,yticks=range(5))
    ax.set_yticklabels(LABEL if first else [])
    if first:ax.get_yticklabels()[0].set(color=TEAL,fontweight='bold')

def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/f'{name}.pdf',metadata={'Title':name,'Creator':'Matplotlib; verified FROST hardware source data'})
    fig.savefig(ROOT/'qa'/f'{name}.png',dpi=220)
    plt.close(fig)

# Figure 11: one row of aligned resource panels, no hidden SExp trade-off.
fig=plt.figure(figsize=(180/25.4,77/25.4))
pos=[(.235,.20,.235,.56),(.545,.20,.235,.56),(.855,.20,.115,.56)]
axes=[fig.add_axes(p) for p in pos]
rows(axes[0],[P[n]['mapped_cell_area_um2']/1000 for n in ORDER],111,[0,25,50,75,100],lambda x:f'{x:.2f}',True)
rows(axes[1],[F[n]['TRELLIS_COMB'] for n in ORDER],35000,[0,10000,20000,30000],lambda x:f'{x:,}')
axes[1].set_xticklabels(['0','10,000','20,000','30,000'])
rows(axes[2],[F[n]['MULT18X18D'] for n in ORDER],11,[0,4,8],lambda x:f'{x}')
axes[0].set_xlabel('Cell area (10³ µm²)',labelpad=7)
axes[1].set_xlabel('Combinational resources',labelpad=7)
axes[2].set_xlabel('DSP blocks',labelpad=7)
for ax,letter in zip(axes,'abc'):
    ax.text(-.02,1.10,letter,transform=ax.transAxes,fontsize=8,fontweight='bold')
fig.text(.35,.925,'Standard-cell mapping',ha='center',fontsize=7.5)
fig.text(.75,.925,'FPGA, post-route',ha='center',fontsize=7.5)
fig.add_artist(plt.Line2D([.235,.470],[.885,.885],transform=fig.transFigure,color=GRID,lw=1))
fig.add_artist(plt.Line2D([.545,.970],[.885,.885],transform=fig.transFigure,color=GRID,lw=1))
fig.text(.235,.04,'L = 64  |  Complete score-to-output kernel boundary',fontsize=7,color=MUTED)
save(fig,'FROST_hardware_stage3')

# Figure 12: energy drives the main message; area/latency panel shows both costs.
fig=plt.figure(figsize=(180/25.4,82/25.4))
ax=fig.add_axes([.205,.22,.345,.57])
clean(ax)
ax.axhspan(-.44,.44,color='#EAF5F3',zorder=0)
for y,n in enumerate(ORDER):
    v=P[n]['energy_per_tile_uJ']['mean'];sd=P[n]['energy_per_tile_uJ']['sample_sd']
    ax.barh(y,v,height=.55,color=COLOR[n],zorder=3)
    ax.errorbar(v,y,xerr=sd,fmt='none',ecolor=INK,lw=1,capsize=2,zorder=4)
    ax.text(v+.06,y,f'{v:.3f}',va='center',fontsize=7,color=TEAL if y==0 else INK,fontweight='bold' if y==0 else 'normal')
ax.set(xlim=(0,3.35),ylim=(4.65,-.65),xticks=[0,1,2,3],yticks=range(5),yticklabels=LABEL)
ax.get_yticklabels()[0].set(color=TEAL,fontweight='bold')
ax.set_xlabel('Estimated energy per tile (µJ)',labelpad=7)
ax.text(-.02,1.13,'a',transform=ax.transAxes,fontsize=8,fontweight='bold')
ax.text(.06,1.13,'Energy per 16-query tile',transform=ax.transAxes,fontsize=7.5)
# A compact separate column attributes every reduction to its own baseline.
for y,n in enumerate(ORDER[1:],start=1):
    reduction=S['optimized_FROST_vs_SC'][n]['energy_reduction_percent']
    ax.text(1.10,y,f'{reduction:.1f}%',transform=ax.get_yaxis_transform(),ha='center',va='center',color=TEAL,fontsize=7.3,fontweight='bold')
ax.text(1.10,-.87,'FROST\nsaving',transform=ax.get_yaxis_transform(),ha='center',va='bottom',fontsize=6.8,color=TEAL,linespacing=1.2)

bx=fig.add_axes([.715,.22,.265,.57])
bx.spines[['top','right']].set_visible(False)
bx.spines[['left','bottom']].set_color(MUTED)
bx.grid(color=GRID,lw=1,zorder=0)
bx.set(xlim=(0,110),ylim=(0,1150),xticks=[0,50,100],yticks=[0,500,1000])
bx.set_xlabel('Cell area (10³ µm²)',labelpad=7)
bx.set_ylabel('Tile latency (µs)',labelpad=5)
bx.text(-.02,1.13,'b',transform=bx.transAxes,fontsize=8,fontweight='bold')
bx.text(.07,1.13,'Area and execution time',transform=bx.transAxes,fontsize=7.5)
scatter_label={'hot_reuse':('FROST',(-8,-15),'right'),
               'sexp16':('SExp (16)',(-7,10),'right'),
               'sexp256':('SExp (256)',(-7,9),'right'),
               'bern8':('Bernstein-8',(3,-13),'center'),
               'horner5':('Horner-5',(0,9),'center')}
for n in ORDER:
    x=P[n]['mapped_cell_area_um2']/1000;y=P[n]['tile_latency_us']
    bx.scatter(x,y,s=34 if n=='hot_reuse' else 24,c=COLOR[n],marker='D' if n=='hot_reuse' else 'o',zorder=4)
    lab,offset,ha=scatter_label[n]
    bx.annotate(lab,(x,y),xytext=offset,textcoords='offset points',ha=ha,va='center',fontsize=6.8,
                color=TEAL if n=='hot_reuse' else INK,fontweight='bold' if n=='hot_reuse' else 'normal')
fig.text(.205,.055,'Nangate45  |  1.10 V, 25 °C, 8 MHz  |  Pre-layout estimates',fontsize=7,color=MUTED)
save(fig,'FROST_hardware_tradeoffs')

manifest={'figure_width_mm':180,'font':'Arial, embedded TrueType; same family as Fig. 10',
          'data_sources':['stage4_summary.json','stage5_summary.json'],
          'first_figure_height_mm':77,'second_figure_height_mm':82,
          'format':'PDF, vector paths and embedded editable text; no image rasters',
          'scope':'Fig11 combines separate standard-cell and FPGA resource flows, clearly labelled. Fig12 uses 8-MHz standard-cell data only.',
          'error_bars':'sample SD across three stream seeds on one fixed tile; not population uncertainty',
          'minimum_stroke_pt':1,'all_bar_axes_begin_at_zero':True,'methods':ORDER,'colors':COLOR}
(ROOT/'figure_protocol.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Generated two vector PDFs from the complete verified SC comparisons.')
