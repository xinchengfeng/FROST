from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

R=Path(__file__).resolve().parents[1]
D=json.loads((R/'source/L64_summary.json').read_text())
B={x['design']:x for x in D['baseline']}; V={x['design']:x for x in D['variants']}; F=V['l64c']
C={'fp32_1':'#7B899C','fp32_4':'#665187','fx16':'#BE873F','frost':'#9CA9AD','l64c':'#00877F'}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],
 'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'axes.linewidth':.6,'xtick.labelsize':7,'ytick.labelsize':7.7,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
def save(fig,name):
 fig.savefig(R/'Images'/f'{name}.pdf');fig.savefig(R/'Images'/f'{name}.svg');fig.savefig(R/f'{name}.png',dpi=200);plt.close(fig)

fig=plt.figure(figsize=(7.2,4.45))
fig.text(.04,.962,'a',fontsize=12,fontweight='bold');fig.text(.09,.962,'Same samples and outputs; less implementation overhead',fontsize=9)
ax=fig.add_axes([.06,.625,.90,.275]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
cards=[('480 → 15 bits','State-history storage','Delay update enables;\nkeep FSM states local.'),
       ('32 → 22 bits','Accumulation width','Exact range for 64\nsigned 16-bit values.'),
       ('Reuse valid signal','Completion detection','Reuse pipeline validity;\nremove a sample counter.')]
for i,(big,title,detail) in enumerate(cards):
 x=.005+i*.333
 ax.add_patch(FancyBboxPatch((x,.04),.317,.90,boxstyle='round,pad=.005,rounding_size=.022',facecolor='#F0F6F5',edgecolor='#B7D5D1',lw=.65))
 ax.text(x+.1585,.77,title,ha='center',fontsize=8,color='#42595A')
 ax.text(x+.1585,.50,big,ha='center',fontsize=12 if i<2 else 10,color=C['l64c'],fontweight='bold')
 ax.text(x+.1585,.21,detail,ha='center',va='center',fontsize=7.6,linespacing=1.35)
fig.text(.04,.555,'b',fontsize=12,fontweight='bold');fig.text(.09,.555,'Post-route FPGA resources at L = 64',fontsize=9)
items=[B['fp32_1'],B['fp32_4'],B['fx16'],B['frost'],F]
labels=['FP32 · 1 lane','FP32 · 4 lanes','FX16 · 4 lanes','FROST · before','FROST · optimized']
for i,(k,title) in enumerate([('comb','COMB'),('ff','Flip-flops'),('dsp','DSP blocks'),('bram18','Block RAMs')]):
 a=fig.add_axes([.20+i*.195,.13,.162,.35]);vals=[x[k] for x in items]; xmax=max(vals)*1.37
 for y,(x,v) in enumerate(zip(items,vals)):
  color=C[x['design']];a.barh(y,v,height=.58,color=color,zorder=3)
  if v==0:a.plot(0,y,'o',color=color,ms=3.5,clip_on=False)
  a.text(v+xmax*.035,y,f'{v:,}',va='center',fontsize=7.2,color=color if x['design']=='l64c' else '#354149',fontweight='bold' if x['design']=='l64c' else 'normal')
 a.set(xlim=(0,xmax),ylim=(4.7,-.65),yticks=np.arange(5));a.set_yticklabels(labels if i==0 else [])
 a.tick_params(axis='y',length=0,pad=5);a.set_title(title,fontsize=8,pad=8,fontweight='bold')
 a.spines[['top','right','left']].set_visible(False);a.grid(axis='x',color='#E9ECEF',lw=.5,zorder=0)
 if i==0:a.set_xticks([0,10000,20000],['0','10k','20k'])
 elif i==1:a.set_xticks([0,1000,2000],['0','1k','2k'])
 elif i==2:a.set_xticks([0,8,16])
 else:a.set_xticks([0,1,2,3])
fig.text(.20,.046,'ECP5 LFE5U-85F · three routing seeds · auxiliary logic and storage included',fontsize=7.2,color='#515A65')
save(fig,'FROST_hardware_stage3')

fig=plt.figure(figsize=(7.2,3.45))
fig.text(.035,.955,'a',fontsize=12,fontweight='bold');fig.text(.09,.955,'Same-flow reductions relative to each control',fontsize=9)
fig.text(.76,.955,'b',fontsize=12,fontweight='bold')
keys=['area_um2','power_mW','asic_latency_us','energy_uJ']
matrix=np.array([[100*(1-F[k]/B[n][k]) for k in keys] for n in ['fp32_1','fp32_4','fx16']])
assert (matrix>0).all()
ax=fig.add_axes([.18,.30,.49,.49]);cm=LinearSegmentedColormap.from_list('teal',['#F0F6F5','#00877F'])
im=ax.pcolormesh(np.arange(5)-.5,np.arange(4)-.5,matrix,vmin=0,vmax=80,cmap=cm,shading='flat',rasterized=False)
ax.set(xlim=(-.5,3.5),ylim=(2.5,-.5))
ax.set(xticks=np.arange(4),xticklabels=['Area','Power','Latency','Energy'],yticks=np.arange(3),yticklabels=['FP32 · 1 lane','FP32 · 4 lanes','FX16 · 4 lanes'])
ax.xaxis.tick_top();ax.tick_params(length=0,pad=8)
for i in range(3):
 for j in range(4):ax.text(j,i,f'{matrix[i,j]:.1f}%',ha='center',va='center',fontsize=9,fontweight='bold',color='white' if matrix[i,j]>=43 else '#006E69')
for x in np.arange(-.5,4,1):ax.axvline(x,color='white',lw=2)
for y in np.arange(-.5,3,1):ax.axhline(y,color='white',lw=2)
for s in ax.spines.values():s.set_visible(False)
cax=fig.add_axes([.20,.185,.45,.024]);cb=fig.colorbar(im,cax=cax,orientation='horizontal');cb.solids.set_rasterized(False);cb.solids.set_edgecolor("face");cb.set_ticks([0,20,40,60,80]);cb.outline.set_visible(False);cb.ax.tick_params(labelsize=6.7,length=2);cb.set_label('Reduction relative to each control (%)',fontsize=7,labelpad=3)
ax=fig.add_axes([.80,.30,.17,.49]);ys=[B['frost']['energy_uJ'],F['energy_uJ']];es=[B['frost']['energy_uJ_sd'],F['energy_uJ_sd']]
ax.bar([0,1],ys,color=[C['frost'],C['l64c']],width=.6,zorder=3)
ax.errorbar([0,1],ys,yerr=es,fmt='none',ecolor='#273E3D',capsize=2,lw=.75,zorder=4)
for i,y in enumerate(ys):ax.text(i,y+.020,f'{y:.4f}',ha='center',fontsize=7.5)
ax.set(ylim=(0,.59),xticks=[0,1],xticklabels=['Before','Optim.'],yticks=[0,.2,.4]);ax.tick_params(axis='x',length=0,pad=7)
ax.set_ylabel('Energy per tile (µJ)',labelpad=3);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#E9ECEF',lw=.5,zorder=0)
ax.set_title('−6.7% energy',fontsize=9,color=C['l64c'],fontweight='bold',pad=12)
fig.text(.535,.039,'L = 64 · Nangate45 · 15 MHz · pre-layout estimates · 16-query tiles',ha='center',fontsize=7.3,color='#515A65')
save(fig,'FROST_hardware_tradeoffs')
print('Generated two vector figures using newly verified L64 results.')
