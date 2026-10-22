"""Build Fig. 10 directly from actual prediction arrays and all three stream scores.
No generated imagery, denoising, resampling-based enhancement, or cherry-picked seeds.
The file is vector PDF for text/annotations; observed 64x64 RGB pixels are embedded.
"""
from pathlib import Path
import json,csv,sys,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import font_manager
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'legacy_runtime/source/scripts'))
from restoration_metrics import quality
IDS=[2018,43033,235098];SEEDS=[101,202,303]
METHODS=['SExp','Horner','Bernstein','Bernstein_QMC','Bernstein_stratified','LocalTT33_stratified']
LABELS=['Reference','SExp-FSM','Horner-SC','Bernstein-SC','Bernstein-LD','Bernstein-Strat','FROST-LT']
SUB=['Target crop','FSM exponent','NAND series','Polynomial','Low discrepancy','Stratified','Local-state TT']
COLORS=['#65717C','#B77757','#A68D50','#717C95','#8A7CA3','#4F79A0','#087E83']
# Prespecified from reference images alone: left/right chair detail, bird head, track edge.
ROIS={2018:(36,26,16,16),43033:(26,23,16,16),235098:(27,15,16,16)}
FIG=HERE/'figure';FIG.mkdir(exist_ok=True)
# Supply a legitimately installed Arial font directory via FROST_FIGURE_FONT_DIR
# if Arial is not already available to Matplotlib. Fonts are not redistributed.
if os.environ.get('FROST_FIGURE_FONT_DIR'):
 for font in Path(os.environ['FROST_FIGURE_FONT_DIR']).glob('*'):
  if font.suffix.lower()=='.ttf':font_manager.fontManager.addfont(str(font))
font_manager.findfont(font_manager.FontProperties(family='Arial'),fallback_to_default=False)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial'],
 'font.size':7,'pdf.fonttype':42,'ps.fonttype':42,'axes.unicode_minus':False,'svg.fonttype':'none'})
def metrics():
 rows=[]
 for i in IDS:
  for m in METHODS:
   for seed in SEEDS:
    p=HERE/'results'/f'{i}_{m}_S{seed}.npz';d=np.load(p);j=json.loads(p.with_suffix('.json').read_text())
    q=quality(d['prediction'],d['truth'],'reconstruction')
    for k in ['psnr','ssim','mse']:assert abs(q[k]-j[k])<1e-10
    rows.append({'image_id':i,'method':m,'seed':seed,'L':4096,**q})
 summary=[]
 for i in IDS:
  for m in METHODS:
   rr=[r for r in rows if r['image_id']==i and r['method']==m]
   s={'image_id':i,'method':m,'n_seeds':3}
   for k in ['psnr','ssim','mse']:
    s[k+'_mean']=float(np.mean([r[k] for r in rr]));s[k+'_SD']=float(np.std([r[k] for r in rr],ddof=1))
   summary.append(s)
 with (HERE/'results/fig10_source_data.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (HERE/'results/fig10_summary.json').write_text(json.dumps(summary,indent=2))
 paired=[]
 for i in IDS:
  for m in METHODS[:-1]:
   a=next(r for r in summary if r['image_id']==i and r['method']==METHODS[-1]);b=next(r for r in summary if r['image_id']==i and r['method']==m)
   paired.append({'image_id':i,'control':m,'delta_PSNR':a['psnr_mean']-b['psnr_mean'],'delta_SSIM':a['ssim_mean']-b['ssim_mean']})
 (HERE/'results/fig10_paired_comparisons.json').write_text(json.dumps(paired,indent=2))
 return summary,paired
def main():
 summary,paired=metrics();D={(r['image_id'],r['method']):r for r in summary}
 W,H=180,170
 fig=plt.figure(figsize=(W/25.4,H/25.4),facecolor='white')
 # Coordinates in millimetres, measured from the top for predictable publication sizing.
 def text(x,y,s,**kw):fig.text(x/W,1-y/H,s,va='top',**kw)
 def rect(x,y,w,h,**kw):
  p=Rectangle((x/W,1-(y+h)/H),w/W,h/H,transform=fig.transFigure,**kw);fig.patches.append(p);return p
 def axat(x,y,w,h):return fig.add_axes([x/W,1-(y+h)/H,w/W,h/H])
 ink='#263540';muted='#66737C';teal=COLORS[-1]
 # Context and title belong in the caption, leaving room for the actual data.
 # Exact integer pixel replication avoids intermediate 100-dpi resampling;
 # it does not increase the information content of the native 64x64 images.
 def pixels(a,k):return np.repeat(np.repeat(a,k,axis=0),k,axis=1)
 left=3;pitch=25.0;side=22.6
 rect(left+6*pitch-1.15,2.2,24.9,155.0,facecolor='#EAF5F3',edgecolor='none',zorder=-10)
 for c,(lab,sub,col) in enumerate(zip(LABELS,SUB,COLORS)):
  x=left+c*pitch
  text(x+side/2,3.4,lab,ha='center',fontsize=7.45,fontweight='bold',color=teal if c==6 else ink)
  text(x+side/2,7.3,sub,ha='center',fontsize=6.35,color=muted)
  rect(x,11.1,side,.45,facecolor=col,edgecolor='none')
 errorcmap=LinearSegmentedColormap.from_list('paper_error',['#FAFAF7','#E9CA8C','#D77C52','#9A3B55','#332548'])
 for r,i in enumerate(IDS):
  top=14.0+r*47.0;x0,y0,ww,hh=ROIS[i]
  text(left,top,chr(97+r),fontsize=8,fontweight='bold',color=ink)
  text(left+4.1,top+.3,f'BSDS500 · {i}',fontsize=7.3,color=ink)
  gain=next(p['delta_PSNR'] for p in paired if p['image_id']==i and p['control']=='SExp')
  text(177,top+.3,f'FROST-LT vs SExp-FSM: +{gain:.2f} dB',fontsize=7.3,color=teal,ha='right',fontweight='bold')
  data=np.load(HERE/'results'/f'{i}_Horner_S101.npz');truth=data['truth']
  for c in range(7):
   x=left+c*pitch;y=top+4.4;pred=truth if c==0 else np.load(HERE/'results'/f'{i}_{METHODS[c-1]}_S101.npz')['prediction']
   ax=axat(x,y,side,side);ax.imshow(pixels(pred,8),interpolation='none',vmin=0,vmax=1,extent=(-.5,63.5,63.5,-.5));ax.set_axis_off()
   ax.add_patch(Rectangle((x0-.5,y0-.5),ww,hh,facecolor='none',edgecolor='#FFFFFF',linewidth=1.6))
   ax.add_patch(Rectangle((x0-.5,y0-.5),ww,hh,facecolor='none',edgecolor='#267DB3',linewidth=.9))
   small=(side-.8)/2;yy=y+side+.7
   a=axat(x,yy,small,small);a.imshow(pixels(pred[y0:y0+hh,x0:x0+ww],16),interpolation='none',vmin=0,vmax=1);a.set_axis_off()
   if c:
    a=axat(x+small+.8,yy,small,small);err=np.mean(np.abs(pred-truth),axis=-1)
    a.imshow(pixels(err[y0:y0+hh,x0:x0+ww],16),cmap=errorcmap,vmin=0,vmax=.06,interpolation='none');a.set_axis_off()
   else:
    text(x+small+1.2,yy+1.5,'16 × 16\nROI / error',fontsize=6.4,color=muted,linespacing=1.4)
   ty=yy+small+1.0
   if c:
    d=D[i,METHODS[c-1]];color=ink
    text(x+side/2,ty,f"{d['psnr_mean']:.2f} ± {d['psnr_SD']:.2f} dB",ha='center',fontsize=7.0,color=color,fontweight='bold' if c==6 else 'normal')
    text(x+side/2,ty+3.05,f"{d['ssim_mean']:.4f} ± {d['ssim_SD']:.4f}",ha='center',fontsize=6.6,color=color)
   else:
    text(x,ty,'PSNR (dB)',fontsize=6.8,color=muted)
    text(x,ty+3.05,'RGB SSIM',fontsize=6.8,color=muted)
 # The common normalization is fixed across every error inset and every method.
 text(3,160.4,'Mean absolute RGB error',fontsize=7,color=ink)
 cbax=axat(38,160.7,40,2.3);cb=fig.colorbar(ScalarMappable(norm=Normalize(0,.06),cmap=errorcmap),cax=cbax,orientation='horizontal')
 cb.set_ticks([0,.03,.06]);cb.set_ticklabels(['0','0.03','≥0.06']);cb.ax.tick_params(labelsize=6.2,length=1.8,pad=1.1);cb.outline.set_linewidth(.6)
 text(84,160.4,'FROST-LT: experimental local-state TT variant',fontsize=7,color=teal)
 text(84,164.7,'Image-specific fits · numerical count normalization in controls',fontsize=6.35,color=muted)
 metadata={'Title':'Fig. 10 - Reconstruction with multiple stochastic-computing controls',
 'Author':'FROST reconstruction experiment','Subject':'Actual finite-bit simulations; L4096; three fixed images and three stream seeds; local-state TT variant',
 'Keywords':'SC;FROST;Horner;Bernstein;reconstruction;PSNR;SSIM'}
 fig.savefig(FIG/'FROST_reconstruction_SC.pdf',metadata=metadata,dpi=600)
 fig.savefig(FIG/'FROST_reconstruction_SC.svg')
 fig.savefig(FIG/'FROST_reconstruction_SC.png',dpi=300)
 plt.close(fig)
 (HERE/'results/figure_protocol.json').write_text(json.dumps({'figure_mm':[W,H],'font':'Arial; embedded TrueType','image_ids':IDS,'visual_seed':101,'metric_seeds':SEEDS,'ROI_xywh':ROIS,'error_scale':[0,.06],'image_interpolation':'no interpolation in vector outputs; exact integer pixel replication (8x full crop, 16x ROI), identical for all methods; no new detail','native_RGB_shape':[64,64,3],'embedded_full_crop_shape':[512,512,3],'embedded_ROI_shape':[256,256],'scores':'mean plus/minus sample standard deviation; n=3 streams at fixed weights','gain_rounding':'difference of unrounded means, then rounded to 2 decimal places','figure_vector_elements':['text','boxes','rules','colorbar labels'],'raster_elements':'RGB reference/prediction pixels and derived ROI errors; no denoising or detail generation'},indent=2))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
