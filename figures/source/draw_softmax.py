"""Redraw the four softmax panels from their archived measured CSVs.

No model fitting, simulations or inferred measurements are used. PDF objects
are native vector paths/text; even the density hexagons remain vector.
"""
from pathlib import Path
import hashlib, json, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT=Path(__import__('os').environ['FROST_FIGURE_WORK'])
OUT=ROOT/'outputs/frost_plot_refresh'
DATA=Path(__file__).parent/'fig5_data'
ARCHIVE=ROOT/'tmp/pdfs/frost_plot_refresh/fig5_data/section3_1_softmax/source_data'
DATA.mkdir(exist_ok=True)
FILES=['Fig3a_probabilities.csv','Fig3b_row_errors.csv','Fig3e_output_error.csv',
       'Fig3f_sampling_error.csv','numerical_summary.json']
for filename in FILES:
    if (ARCHIVE/filename).exists():shutil.copy2(ARCHIVE/filename,DATA/filename)

FONT=ROOT/'tmp/pdfs/frost_redesign/arial_fonts'
for f in ['Arial.TTF','Arialbd.TTF','Ariali.TTF','Arialbi.TTF']:
    font_manager.fontManager.addfont(FONT/f)

TEAL='#007F86';CORAL='#C56B52';GREY='#6B7280';INK='#25303A';LIGHT='#D9E0E4'
W=510.236;H=218.268
plt.rcParams.update({
    'font.family':(__import__('os').environ['FROST_PLOT_FONT_FAMILY']+''),'font.size':8.5,'axes.labelsize':8.5,
    'axes.titlesize':8.5,'axes.titleweight':'bold','axes.labelcolor':INK,
    'text.color':INK,'xtick.color':INK,'ytick.color':INK,
    'xtick.labelsize':7.5,'ytick.labelsize':7.5,'axes.spines.top':False,
    'axes.spines.right':False,'axes.linewidth':1.0,'xtick.major.width':1.0,
    'ytick.major.width':1.0,'xtick.major.size':3,'ytick.major.size':3,
    'legend.fontsize':7.4,'legend.frameon':False,'pdf.fonttype':42,
    'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white',
    'figure.facecolor':'white','mathtext.fontset':'custom',
    'mathtext.rm':(__import__('os').environ['FROST_PLOT_FONT_FAMILY']+''),'mathtext.it':(__import__('os').environ['FROST_PLOT_FONT_FAMILY']+':italic'),'mathtext.bf':(__import__('os').environ['FROST_PLOT_FONT_FAMILY']+':bold'),
    'lines.linewidth':1.5,'axes.unicode_minus':True,
})

summary=json.loads((DATA/'numerical_summary.json').read_text())
prob=pd.read_csv(DATA/'Fig3a_probabilities.csv')
rows=pd.read_csv(DATA/'Fig3b_row_errors.csv')
output=pd.read_csv(DATA/'Fig3e_output_error.csv')
sampling=pd.read_csv(DATA/'Fig3f_sampling_error.csv')
L=np.array([64,256,1024])
assert len(prob)==65536 and len(rows)==4096
assert len(output)==9 and len(sampling)==18
assert np.isclose(np.sqrt(np.mean((prob.prediction-prob.reference)**2)),summary['rmse'],atol=2e-10)
assert np.isclose(rows.tv.mean(),summary['mean_tv'],atol=2e-10)

def label(fig,letter,x):
    fig.text(x,.948,letter,fontweight='bold',fontsize=9.5,va='top',ha='left',color=INK)

def axes(fig,rect,title):
    ax=fig.add_axes(rect)
    ax.set_title(title,loc='left',pad=11)
    ax.tick_params(direction='out')
    return ax

def save(fig,name):
    dest=OUT/'Images'/f'{name}.pdf'
    fig.savefig(dest,format='pdf',metadata={'Title':name,'Subject':'Archived source-data replot; no new experimental measurements'})
    fig.savefig(ROOT/'tmp/pdfs/frost_plot_refresh'/f'{name}.png',dpi=200)
    plt.close(fig)

fig=plt.figure(figsize=(W/72,H/72))
ax=axes(fig,[.105,.21,.31,.64],'Joint probability fit')
cmap=LinearSegmentedColormap.from_list('FROST_density',
    ['#EEF7F7','#C3E2E1','#7FBDBE','#358F94','#07515B','#073441'])
hx=ax.hexbin(prob.reference,prob.prediction,gridsize=65,extent=(0,.8,0,.8),
    mincnt=1,cmap=cmap,norm=LogNorm(),linewidths=0,rasterized=False)
ax.plot([0,.8],[0,.8],color=GREY,lw=1.15,ls=(0,(4,2.7)),zorder=3)
ax.set(xlim=(0,.8),ylim=(0,.8),xlabel='Reference softmax probability',ylabel='HI-SMURF probability')
ax.set_xticks([0,.2,.4,.6,.8]);ax.set_yticks([0,.2,.4,.6,.8])
ax.text(.045,.945,f"RMSE  {summary['rmse']:.5f}\nMean TV  {summary['mean_tv']:.5f}",
    transform=ax.transAxes,ha='left',va='top',fontsize=7.7,linespacing=1.5)
ax.text(.96,.035,'4,096 test rows',transform=ax.transAxes,ha='right',va='bottom',fontsize=7.1)
cax=fig.add_axes([.429,.21,.012,.64])
cb=fig.colorbar(hx,cax=cax)
cb.solids.set_rasterized(False)
cb.set_label('Entries per bin',size=7.5,labelpad=5)
cb.ax.tick_params(labelsize=7,width=1,length=2.5);cb.outline.set_linewidth(1)
label(fig,'a',.025)

ax=axes(fig,[.645,.21,.334,.64],'Distribution of row errors')
x=np.sort(rows.tv.to_numpy());y=np.arange(1,len(x)+1)/len(x)
ax.step(x,y,where='post',color=TEAL,lw=1.5)
for value,height in [(summary['median_tv'],.5),(summary['tv_p95'],.95)]:
    ax.plot([value,value],[0,height],color=GREY,ls=(0,(1.2,2.5)),lw=1)
    ax.scatter([value],[height],s=21,facecolors=TEAL,edgecolors='white',linewidths=1,zorder=4)
ax.text(.97,.34,f"Median  {summary['median_tv']:.4f}\n95th percentile  {summary['tv_p95']:.4f}",
    transform=ax.transAxes,ha='right',va='top',fontsize=7.7,linespacing=1.5)
ax.set(xlim=(0,.25),ylim=(0,1.035),xlabel='Row total variation distance',ylabel='Cumulative fraction of test rows')
ax.set_yticks([0,.25,.5,.75,1]);ax.set_xticks([0,.05,.1,.15,.2,.25])
label(fig,'b',.568)
save(fig,'Softmax_approximation')

fig=plt.figure(figsize=(W/72,H/72))
ax=axes(fig,[.105,.21,.35,.64],'Finite-stream approximation')
g=output.groupby('L').rmse.agg(['mean','std']).reindex(L)
ax.axhline(summary['stream_analytic_rmse'],color=GREY,lw=1.15,ls=(0,(4,2.7)),
    label='Analytic limit, same rows',zorder=1)
ax.scatter(output.L,output.rmse,s=18,facecolors='none',edgecolors=TEAL,
    linewidths=1,alpha=.48,zorder=4)
ax.errorbar(L,g['mean'],yerr=g['std'],color=TEAL,lw=1.5,marker='o',ms=4.2,
    capsize=2.7,elinewidth=1.1,markeredgewidth=1,label='Categorical output',zorder=5)
ax.set(xscale='log',xlim=(48,1400),ylim=(0,.055),
       xlabel='Sampling length, $L$ (cycles)',ylabel='RMSE relative to exact softmax')
ax.set_xticks(L,['64','256','1,024']);ax.xaxis.set_minor_locator(NullLocator())
ax.set_yticks([0,.01,.02,.03,.04,.05]);ax.legend(loc='upper right',handlelength=1.8,labelspacing=.45,borderpad=.2)
ax.text(.97,.04,'256 burn-in cycles excluded',ha='right',va='bottom',transform=ax.transAxes,fontsize=7.1,color=GREY)
label(fig,'c',.025)

ax=axes(fig,[.645,.21,.334,.64],'Finite-state sampling error')
groups={}
for mode,color,marker,style,name in [
    ('categorical_bits',TEAL,'o','-','Categorical output'),
    ('weighted_diagnostic',CORAL,'s',(0,(4,2.5)),'State-average diagnostic')]:
    raw=sampling[sampling['mode']==mode]
    group=raw.groupby('L').sampling_rmse.agg(['mean','std']).reindex(L);groups[mode]=group
    ax.scatter(raw.L,raw.sampling_rmse,s=18,facecolors='none',edgecolors=color,
        linewidths=1,alpha=.48,zorder=4)
    ax.errorbar(L,group['mean'],yerr=group['std'],color=color,lw=1.5,ls=style,
        marker=marker,ms=4.2,capsize=2.7,elinewidth=1.1,markeredgewidth=1,label=name,zorder=5)
anchor=float(groups['categorical_bits'].loc[64,'mean'])
reference=anchor*np.sqrt(64/L)
ax.plot(L,reference,color=GREY,ls=(0,(1.2,2.4)),lw=1.2,label='$L^{-1/2}$ reference',zorder=2)
ax.set(xscale='log',yscale='log',xlim=(48,1400),ylim=(.006,.09),
       xlabel='Sampling length, $L$ (cycles)',ylabel='RMSE relative to analytic output')
ax.set_xticks(L,['64','256','1,024']);ax.xaxis.set_minor_locator(NullLocator())
ax.yaxis.set_major_locator(FixedLocator([.01,.02,.05]));ax.yaxis.set_major_formatter(FuncFormatter(lambda y,pos:f'{y:g}'))
ax.yaxis.set_minor_locator(NullLocator())
handles,labels=ax.get_legend_handles_labels()
legend_items=dict(zip(labels,handles))
legend_order=['Categorical output','State-average diagnostic','$L^{-1/2}$ reference']
ax.legend([legend_items[name] for name in legend_order],legend_order,
    loc='upper right',handlelength=1.7,labelspacing=.3,borderpad=.1,fontsize=6.9)

# The inset removes a known scaling trend, rather than estimating new data.
# Its denominator is fixed by the original three-seed mean at L=64;
# every seed and sample standard deviation is divided by that same value.
inset=ax.inset_axes([.10,.09,.39,.28])
inset.axhline(1,color=GREY,lw=1,ls=(0,(1.2,2.4)),zorder=1)
for mode,color,marker,style in [
    ('categorical_bits',TEAL,'o','-'),('weighted_diagnostic',CORAL,'s',(0,(4,2.5)))]:
    group=groups[mode];raw=sampling[sampling['mode']==mode]
    den=anchor*np.sqrt(64/raw.L.to_numpy())
    inset.scatter(raw.L,raw.sampling_rmse.to_numpy()/den,s=6,
                  facecolors='none',edgecolors=color,linewidths=.6,alpha=.45,zorder=3)
    inset.errorbar(L,group['mean'].to_numpy()/reference,yerr=group['std'].to_numpy()/reference,
        color=color,ls=style,lw=1.0,marker=marker,ms=2.3,capsize=1.5,elinewidth=1,zorder=4)
inset.set(xscale='log',xlim=(48,1400),ylim=(.72,1.18))
inset.set_xticks([64,1024],['64','1,024']);inset.xaxis.set_minor_locator(NullLocator())
inset.set_yticks([.8,1,1.2],['0.8','1.0','1.2']);inset.tick_params(labelsize=5.8,length=2,width=1,pad=1.5)
inset.text(.0,1.07,'Ratio to reference',transform=inset.transAxes,fontsize=6.2,ha='left',va='bottom',color=INK)
inset.spines['top'].set_visible(False);inset.spines['right'].set_visible(False)
label(fig,'d',.568)
save(fig,'Softmax_bitstream')

audit={
 'source_archive':'Section3_1_Softmax_Figures_and_Text.zip',
 'source_hashes':{name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in FILES},
 'probability_entries':len(prob),'independent_test_rows':len(rows),
 'stream_output_rows':len(output),'stream_sampling_rows':len(sampling),
 'stream_seeds':sorted(int(x) for x in output.seed.unique()),
 'retained_lengths':L.tolist(),'raw_measurements_unchanged':True,
 'output_rmse_mean':g['mean'].tolist(),'output_rmse_sample_sd':g['std'].tolist(),
 'sampling_rmse_mean':{k:v['mean'].tolist() for k,v in groups.items()},
 'sampling_rmse_sample_sd':{k:v['std'].tolist() for k,v in groups.items()},
 'analytic_output_rmse':summary['stream_analytic_rmse'],
 'ratio_inset_reference':{'anchor_L':64,'anchor_sampling_rmse':anchor,
                        'formula':'anchor * sqrt(64 / L)','new_measurements':False},
 'pdf_dimensions_pt':[W,H],'all_data_native_vector':True,
 'palette':{'FROST':TEAL,'state_average_diagnostic':CORAL,'reference':GREY}
}
(OUT/'Source/fig5_validation.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
