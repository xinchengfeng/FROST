"""Regenerate the retained pilot visual from the whole-partition measured outputs."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from restoration_data import center_image
IDS=[2018,43033,235098];NAMES=['Snow scene','Bird and reeds','Roller coaster']
COLORS=['#08739E','#BC650D','#0B8C86'];INK='#181818';TEAL='#0B8C86';GRAY='#777777'

def configure(out):
    for name in ['NimbusSans-Regular.ttf','NimbusSans-Bold.ttf']:fm.fontManager.addfont(str(out/'fonts'/name))
    plt.rcParams.update({'font.family':'Nimbus Sans','font.size':7,'axes.linewidth':.6,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'pdf.fonttype':42,'svg.fonttype':'none','text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':INK,'xtick.color':INK,'ytick.color':INK,'savefig.facecolor':'white'})

def axis_mm(fig,x,y,w,h,W,H):return fig.add_axes([x/W,1-(y+h)/H,w/W,h/H])

def savefig(fig,out,name):
    for ext in ['pdf','svg','png']:fig.savefig(out/'figures'/f'{name}.{ext}',dpi=600)
    plt.close(fig)

def gallery(out,previous):
    rois=json.loads((out/'source_data/roi_coordinates.json').read_text())['coordinates']
    W,H=180,177;fig=plt.figure(figsize=(W/25.4,H/25.4));size=24;pitch=28;x0=6
    fig.text(5/W,1-4/H,'Image-specific coordinate reconstruction',fontsize=8,fontweight='bold',va='top')
    fig.text(5/W,1-8/H,'Weights fitted to these targets; coordinates only at inference. Fixed stream seed 101.',fontsize=6,color=GRAY,va='top')
    methods=['truth','float','w8a8','SC_base','frost1024','frost4096']
    headings=['Reference','Float','W8A8','SC (base TT)\nL = 4096','FROST\nL = 1024','FROST\nL = 4096']
    for j,s in enumerate(headings):fig.text((x0+j*pitch+size/2)/W,1-16/H,s,fontsize=6.5,ha='center',va='center',linespacing=1.05,fontweight='bold' if j>=4 else 'normal',color=TEAL if j>=4 else INK)
    for row,id in enumerate(IDS):
        long=np.load(out/'runs'/f'image_{id}.npz');short=np.load(out/'runs'/f'image_{id}.npz');truth=long['truth']
        assert np.array_equal(truth,center_image(out/'data/test'/f'{id}.jpg'))
        raw=json.loads((out/'runs'/f'image_{id}.json').read_text())
        score={m:next(r for r in raw if r['method']==n and r['L']==length and r['stream_base_seed']==seed) for m,n,length,seed in [('float','Float',0,0),('w8a8','W8A8',0,0),('SC_base','SC_base',4096,101),('frost','FROST',4096,101)]}
        sshort=next(r for r in raw if r['method']=='FROST' and r['L']==1024 and r['stream_base_seed']==101)
        top=24+row*46;rx,ry,rs=rois[str(id)]
        fig.text(5/W,1-(top-2.5)/H,chr(97+row),fontsize=8,fontweight='bold',va='bottom')
        fig.text(10/W,1-(top-2.5)/H,f'{NAMES[row]} | BSDS500 #{id} | fitting target',fontsize=6.5,va='bottom')
        for j,method in enumerate(methods):
            x=x0+j*pitch
            if method=='truth':img=truth;r=None
            elif method=='frost1024':img=short['FROST_L1024_S101'];r=sshort
            elif method=='frost4096':img=long['FROST_L4096_S101'];r=score['frost']
            else:img=long[{'float':'Float','w8a8':'W8A8','SC_base':'SC_base_L4096_S101'}[method]];r=score[method]
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
    savefig(fig,out,'FROST_reconstruction')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    root=p.parse_args().root.resolve();configure(root);gallery(root,root)
