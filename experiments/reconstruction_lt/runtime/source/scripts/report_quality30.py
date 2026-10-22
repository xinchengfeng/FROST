"""Plots and tables from actual saved outputs, with fitting/held-out scope explicit."""
import argparse,csv,json,hashlib,shutil,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

IDS=[2018,43033,235098];NAMES=['Snow scene','Bird and reeds','Roller coaster']
COLORS=['#08739E','#BC650D','#0B8C86'];INK='#181818';TEAL='#0B8C86';GRAY='#777777'

def collect(out):
    rows=[];seen=set()
    for path in sorted((out/'runs').glob('*.json')):
        for r in json.loads(path.read_text()):
            r.setdefault('stream_base_seed',101)
            key=tuple(r[k] for k in ['experiment','image_id','task','method','L','inference_seed'])
            if key not in seen:seen.add(key);rows.append(r)
    with (out/'source_data/all_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summaries=[]
    for id in IDS:
        for method,L in [('float',0),('w8a8',0),('analytic',0),('SC_base',4096),('frost',1024),('frost',4096)]:
            group=[r for r in rows if r['experiment']=='render' and r['image_id']==id and r['method']==method and r['L']==L]
            assert group,(id,method,L)
            summaries.append(dict(image_id=id,method=method,L=L,n_streams=len(group),
              psnr_mean=float(np.mean([r['psnr'] for r in group])),psnr_std=float(np.std([r['psnr'] for r in group],ddof=1)) if len(group)>1 else 0.,
              ssim_mean=float(np.mean([r['ssim'] for r in group])),ssim_std=float(np.std([r['ssim'] for r in group],ddof=1)) if len(group)>1 else 0.,
              psnr_min=min(r['psnr'] for r in group),psnr_max=max(r['psnr'] for r in group)))
    with (out/'source_data/reconstruction_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    (out/'source_data/reconstruction_summary.json').write_text(json.dumps(summaries,indent=2));return rows,summaries

def configure(out,previous):
    dest=out/'fonts';dest.mkdir(exist_ok=True)
    for name in ['NimbusSans-Regular.ttf','NimbusSans-Bold.ttf','URW-font-copyright.txt']:shutil.copy2(previous/'fonts'/name,dest/name)
    fm.fontManager.ttflist=[f for f in fm.fontManager.ttflist if f.name!='Nimbus Sans']
    for name in ['NimbusSans-Regular.ttf','NimbusSans-Bold.ttf']:fm.fontManager.addfont(str(dest/name))
    fm.fontManager._findfont_cached.cache_clear()
    plt.rcParams.update({'font.family':'Nimbus Sans','font.size':7,'axes.linewidth':.6,'axes.labelsize':7,'xtick.labelsize':6,
      'ytick.labelsize':6,'pdf.fonttype':42,'svg.fonttype':'none','text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':INK,
      'xtick.color':INK,'ytick.color':INK,'savefig.facecolor':'white'})

def axis_mm(fig,x,y,w,h,W,H):return fig.add_axes([x/W,1-(y+h)/H,w/W,h/H])
def savefig(fig,out,name):
    for ext in ['pdf','svg','png']:fig.savefig(out/'figures'/f'{name}.{ext}',dpi=600)
    plt.close(fig)

def gallery(out,previous):
    rois=json.loads((previous/'source_data/roi_coordinates.json').read_text())['coordinates']
    W,H=180,177;fig=plt.figure(figsize=(W/25.4,H/25.4));size=24;pitch=28;x0=6
    fig.text(5/W,1-4/H,'Image-specific coordinate reconstruction',fontsize=8,fontweight='bold',va='top')
    fig.text(5/W,1-8/H,'Weights fitted to these targets; coordinates only at inference. Fixed stream seed 101.',fontsize=6,color=GRAY,va='top')
    methods=['truth','float','w8a8','SC_base','frost1024','frost4096']
    headings=['Reference','Float','W8A8','SC (base TT)\nL = 4096','FROST\nL = 1024','FROST\nL = 4096']
    for j,s in enumerate(headings):fig.text((x0+j*pitch+size/2)/W,1-16/H,s,fontsize=6.5,ha='center',va='center',linespacing=1.05,fontweight='bold' if j>=4 else 'normal',color=TEAL if j>=4 else INK)
    for row,id in enumerate(IDS):
        long=np.load(out/'runs'/f'render_{id}_reconstruction_L4096.npz');short=np.load(out/'runs'/f'render_{id}_reconstruction_L1024.npz');truth=long['truth']
        assert np.array_equal(truth,np.load(previous/'runs'/f'denoise25_{id}.npz')['truth'])
        score={r['method']:r for r in json.loads((out/'runs'/f'render_{id}_reconstruction_L4096.json').read_text())}
        sshort=next(r for r in json.loads((out/'runs'/f'render_{id}_reconstruction_L1024.json').read_text()) if r['method']=='frost')
        top=24+row*46;rx,ry,rs=rois[str(id)]
        fig.text(5/W,1-(top-2.5)/H,chr(97+row),fontsize=8,fontweight='bold',va='bottom')
        fig.text(10/W,1-(top-2.5)/H,f'{NAMES[row]} | BSDS500 #{id} | fitting target',fontsize=6.5,va='bottom')
        for j,method in enumerate(methods):
            x=x0+j*pitch
            if method=='truth':img=truth;r=None
            elif method=='frost1024':img=short['frost'];r=sshort
            elif method=='frost4096':img=long['frost'];r=score['frost']
            else:img=long[method];r=score[method]
            ax=axis_mm(fig,x,top,size,size,W,H);ax.imshow(img,interpolation='nearest');ax.set_axis_off()
            ax.add_patch(Rectangle((rx-.5,ry-.5),rs,rs,fill=False,edgecolor=COLORS[0],linewidth=.75))
            if method.startswith('frost'):ax.add_patch(Rectangle((-.5,-.5),64,64,fill=False,edgecolor=TEAL,linewidth=.7))
            label='Reference' if r is None else f"{r['psnr']:.2f} / {r['ssim']:.4f}"
            colour=(TEAL if r['psnr']>=30 else '#BC650D') if method.startswith('frost') else INK
            fig.text((x+size/2)/W,1-(top+size+.7)/H,label,ha='center',va='top',fontsize=6.2,color=colour,fontweight='bold' if method=='frost4096' else 'normal')
            zoom=axis_mm(fig,x,top+size+4,11.5,11.5,W,H);zoom.imshow(img[ry:ry+rs,rx:rx+rs],interpolation='nearest');zoom.set_axis_off()
            zoom.add_patch(Rectangle((-.5,-.5),rs,rs,fill=False,edgecolor=COLORS[0],linewidth=.65))
            ea=axis_mm(fig,x+12.5,top+size+4,11.5,11.5,W,H)
            error=np.mean(np.abs(img-truth),axis=-1);im=ea.imshow(error[ry:ry+rs,rx:rx+rs],cmap='magma',vmin=0,vmax=.1,interpolation='nearest');ea.set_axis_off()
        if row<2:fig.add_artist(Line2D([5/W,175/W],[1-(top+41)/H]*2,transform=fig.transFigure,color='#D9DFE2',lw=.5))
    fig.text(5/W,1-164/H,'Numbers: PSNR (dB) / RGB SSIM. Teal FROST scores denote PSNR >= 30 dB.',fontsize=5.8,va='center')
    fig.text(5/W,1-167/H,'Blue boxes: same 16 x 16 ROI; below: enlarged ROI (left), RGB absolute error (right).',fontsize=5.8,va='center')
    fig.text(5/W,1-170/H,'Image-specific fitting results; no held-out restoration or unconditional generation claim.',fontsize=5.8,va='center')
    cax=axis_mm(fig,138,168,33,2,W,H);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,.05,.1]);cb.ax.tick_params(labelsize=5,length=2,pad=1)
    cb.outline.set_linewidth(.4);cb.ax.set_title('Mean RGB absolute error',fontsize=5.4,pad=3)
    cb.solids.set_rasterized(False);cb.solids.set_edgecolor('face')
    savefig(fig,out,'FROST_reconstruction_30dB')

def curves(out,rows):
    fig,axes=plt.subplots(1,2,figsize=(180/25.4,100/25.4));fig.subplots_adjust(left=.085,right=.98,bottom=.235,top=.81,wspace=.27)
    for col,metric in enumerate(['psnr','ssim']):
        ax=axes[col]
        for j,id in enumerate(IDS):
            means=[];sd=[]
            for L in [64,256,1024,4096]:
                r=[z for z in rows if z['experiment']=='render' and z['image_id']==id and z['method']=='frost' and z['L']==L]
                means.append(np.mean([z[metric] for z in r]));sd.append(np.std([z[metric] for z in r],ddof=1) if len(r)>1 else 0)
            ax.errorbar([64,256,1024,4096],means,yerr=sd,color=COLORS[j],lw=1.1,marker=['o','s','^'][j],markersize=4,
                        markerfacecolor='white',markeredgewidth=.7,capsize=2,elinewidth=.8,zorder=3)
            limit=next(r[metric] for r in rows if r['experiment']=='render' and r['image_id']==id and r['method']=='analytic')
            ax.axhline(limit,color=COLORS[j],lw=.7,ls=(0,(4,3)),alpha=.65)
        if metric=='psnr':ax.axhline(30,color=INK,lw=.9,ls=(0,(2,2)));ax.text(64,30.6,'30 dB',fontsize=6)
        ax.set_xscale('log',base=2);ax.set_xticks([64,256,1024,4096],['64','256','1024','4096']);ax.minorticks_off()
        ax.set_xlabel('Bitstream length, L');ax.set_ylabel('PSNR (dB)' if metric=='psnr' else 'RGB SSIM')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#E9ECEE',lw=.5);ax.set_axisbelow(True);ax.tick_params(width=.6,length=2.5,pad=2)
        ax.text(-.12,1.09,chr(97+col),transform=ax.transAxes,fontsize=8,fontweight='bold')
    fig.text(.085,.96,'Reconstruction quality under real finite bitstreams',fontsize=8,fontweight='bold',va='top')
    fig.text(.085,.91,'Fitted targets, not held-out images. At L = 4096: mean +/- s.d. over 3 fixed stream seeds.',fontsize=6,color=GRAY,va='top')
    legend=[Line2D([],[],color=c,marker=m,lw=1,markerfacecolor='white',markersize=4,label=f'{name} (#{id})') for c,m,name,id in zip(COLORS,['o','s','^'],NAMES,IDS)]
    legend.append(Line2D([],[],color=GRAY,ls=(0,(4,3)),lw=.8,label='Analytical FROST expectations'))
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.07),ncol=2,fontsize=6,frameon=False,columnspacing=2)
    fig.text(.085,.025,'Shorter lengths use seed 101 only. Error bars describe stream variability for fixed weights.',fontsize=5.8,color=GRAY)
    savefig(fig,out,'FROST_reconstruction_streams')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--previous',type=Path,required=True);ap.add_argument('--stage',choices=['collect','figures'],required=True)
    a=ap.parse_args();rows,summary=collect(a.output)
    if a.stage=='figures':configure(a.output,a.previous);gallery(a.output,a.previous);curves(a.output,rows)
    print(json.dumps([r for r in summary if r['method']=='frost' and r['L']==4096],indent=2))
