"""Clean mean/SD display with a nonintersecting, explicitly scaled zoom."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__import__('os').environ['FROST_FIGURE_WORK'])
OUT=ROOT/'outputs/frost_fig7_clean'
TMP=ROOT/'tmp/pdfs/frost_fig7_clean'
RAW=ROOT/'outputs/frost_fig7_overlap/Source/data/Fig_attention_efficiency_a.csv'
FONTS=ROOT/'tmp/pdfs/frost_redesign/arial_fonts'
for n in ['Arial.TTF','Arialbd.TTF','Ariali.TTF','Arialbi.TTF']:
    font_manager.fontManager.addfont(str(FONTS/n))
plt.rcParams.update({'font.family':(__import__('os').environ['FROST_PLOT_FONT_FAMILY']+''),'font.size':7.6,
    'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.85,
    'axes.labelcolor':'#26343D','text.color':'#26343D',
    'xtick.color':'#26343D','ytick.color':'#26343D',
    'axes.edgecolor':'#26343D','savefig.facecolor':'white'})
W,H=261.970455,218.267715
df=pd.read_csv(RAW)
methods=['bipolar_iid','magnitude_iid','magnitude_sobol','improved']
seeds=sorted(df.seed.unique().tolist())
assert seeds==[101,202,303,404,505]
ys=np.array([df[df.method==m].sort_values('seed').rmse.to_numpy() for m in methods])
assert ys.shape==(4,5)
means=ys.mean(axis=1);sd=ys.std(axis=1,ddof=1)
colours=['#C56B52','#96AFBF','#426A90','#007F86']
fig=plt.figure(figsize=(W/72,H/72))
ax=fig.add_axes([51.3/W,51.407715/H,186.7/W,136.44/H])
ax.set_xlim(-.5,3.6);ax.set_ylim(0,.36)
ax.set_yticks(np.arange(0,.351,.05))
ax.set_yticklabels([f'{v:.2f}' for v in np.arange(0,.351,.05)],fontsize=7.4)
ax.set_xticks(range(4))
ax.set_xticklabels(['Bipolar\nIID','Magnitude\nIID','Magnitude\nSobol','+ Value\ncentering'],fontsize=7.3)
ax.set_ylabel('Attention output RMSE',fontsize=8.2,labelpad=7)
ax.tick_params(axis='both',width=.85,length=3,pad=3)
ax.tick_params(axis='x',length=0,pad=7)
ax.spines[['top','right']].set_visible(False)
main_line,=ax.plot(np.arange(4),means,color='#A7B3BA',lw=.9,zorder=2)
for i,c in enumerate(colours):
    ax.errorbar(i,means[i],yerr=sd[i],fmt='o',ms=4.3,
        color=c,mec='white',mew=.45,capsize=2.2,
        elinewidth=1.05,capthick=.8,zorder=5)
fig.text(19.797165/W,1-17.675339/H,'a',fontsize=9.5,fontweight='bold')
fig.text(51.3/W,1-17.675339/H,'Ablation of simple improvements',fontsize=8.4,fontweight='bold')
fig.text(51.3/W,1-27/H,'Fixed HI-SMURF; L = 256',fontsize=6.7,color='#6B7280')
gain=(1-means[-1]/means[0])*100
fig.text(51.3/W,1-198/H,f'{gain:.1f}% lower RMSE vs bipolar IID',fontsize=7.0,color='#007F86')
fig.text(51.3/W,1-209/H,'Mean ± s.d.; 5 stream seeds',fontsize=6.6,color='#6B7280')

# Keep the inset, its title and tick labels in the unoccupied upper right.
# Unlike the prior version, none of its decorations touch the main trajectory.
zoom=ax.inset_axes([.49,.52,.49,.33],zorder=10)
zoom.set_facecolor('white')
zoom.set_xlim(1.5,3.5);zoom.set_ylim(.107,.112)
zoom.set_yticks([.108,.110,.112])
zoom.set_yticklabels(['0.108','0.110','0.112'],fontsize=6.0)
zoom.set_xticks([2,3]);zoom.set_xticklabels(['Sobol','+ Centering'],fontsize=6.0)
zoom.tick_params(axis='both',width=.65,length=2,pad=2)
zoom.spines[['top','right']].set_visible(False)
for spine in zoom.spines.values():
    spine.set_color('#9AA7B0');spine.set_linewidth(.65)
for i in [2,3]:
    zoom.errorbar(i,means[i],yerr=sd[i],fmt='o',ms=3.7,color=colours[i],
        mec='white',mew=.35,capsize=2,elinewidth=.85,zorder=5)
zoom.set_title('Zoom: final two conditions',fontsize=6.5,pad=4,loc='left')

# Quantify the minimal distance between all inset text and main trajectory.
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
verts=ax.transData.transform(np.c_[np.arange(4),means])
def segdist(a,b,r):
    ts=np.linspace(0,1,400)
    p=a[None,:]+ts[:,None]*(b-a)[None,:]
    dx=np.maximum(np.maximum(r.x0-p[:,0],0),p[:,0]-r.x1)
    dy=np.maximum(np.maximum(r.y0-p[:,1],0),p[:,1]-r.y1)
    return float(np.min(np.hypot(dx,dy)))*72/fig.dpi
texts=[zoom.title]+zoom.get_xticklabels()+zoom.get_yticklabels()
clearance=min(segdist(a,b,t.get_window_extent(renderer)) for a,b in zip(verts[:-1],verts[1:]) for t in texts if t.get_text())
assert clearance>4,clearance
fig.savefig(TMP/'panel_c.pdf');fig.savefig(TMP/'panel_c.png',dpi=300)
plt.close(fig)
record={'source_csv':RAW.name,'source_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),
    'methods':methods,'seeds':seeds,'observations':ys.tolist(),'means':means.tolist(),
    'sample_sd':sd.tolist(),'observations_count':20,'displayed_means':4,
    'individual_seed_markers':False,'main_y_limits':[0,.36],
    'zoom_y_limits':[.107,.112],'vertical_jitter':False,'new_experiments':False,
    'minimum_inset_text_clearance_from_main_line_pt':clearance,
    'derived_total_reduction_percent':float(gain)}
(OUT/'Source/panel_c_validation.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
