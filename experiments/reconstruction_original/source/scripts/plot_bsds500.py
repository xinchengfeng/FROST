"""Publication figure of measured image-level scores and paired conversion gains."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

TEAL='#0B8C86';BLUE='#08739E';ORANGE='#BC650D';INK='#181818';GRAY='#777777'


def plot(root):
    for name in ['NimbusSans-Regular.ttf','NimbusSans-Bold.ttf']:
        fm.fontManager.addfont(str(root/'fonts'/name))
    plt.rcParams.update({'font.family':'Nimbus Sans','font.size':7,
        'axes.linewidth':.6,'axes.labelsize':7,'xtick.labelsize':6.2,'ytick.labelsize':6.2,
        'pdf.fonttype':42,'svg.fonttype':'none','text.color':INK,
        'axes.labelcolor':INK,'axes.edgecolor':INK,'xtick.color':INK,'ytick.color':INK})
    rows=list(csv.DictReader((root/'source_data/per_image_metrics.csv').open()))
    summary=json.loads((root/'source_data/dataset_summary.json').read_text())
    ids=sorted({int(r['image_id']) for r in rows});assert len(ids)==200
    def v(method,L):
        return np.array([float(next(r for r in rows if int(r['image_id'])==i and r['method']==method and int(r['L'])==L)['psnr_stream_mean']) for i in ids])
    data=[v('Float',0),v('W8A8',0),v('SC_base',4096),v('FROST',1024),v('FROST',4096)]
    colors=[GRAY,BLUE,ORANGE,'#70B6B1',TEAL]
    fig=plt.figure(figsize=(180/25.4,83/25.4))
    ax=fig.add_axes([.078,.27,.415,.495]);ay=fig.add_axes([.625,.27,.335,.495])
    fig.text(.035,.96,'BSDS500 image-specific coordinate reconstruction',fontsize=8,fontweight='bold',va='top')
    fig.text(.035,.902,'All 200 evaluation targets | 64 x 64 RGB crops | target-specific fitted weights',fontsize=6.2,color=GRAY,va='top')
    rng=np.random.default_rng(20261009)
    boxes=ax.boxplot(data,positions=np.arange(1,6),widths=.48,whis=(0,100),showfliers=False,
        patch_artist=True,medianprops=dict(color=INK,linewidth=.9),
        whiskerprops=dict(color=GRAY,linewidth=.55),capprops=dict(color=GRAY,linewidth=.55))
    for j,(values,color,box) in enumerate(zip(data,colors,boxes['boxes']),1):
        box.set_facecolor(color);box.set_alpha(.22);box.set_edgecolor(color);box.set_linewidth(.7)
        ax.scatter(j+rng.uniform(-.13,.13,200),values,s=3.0,color=color,alpha=.40,linewidths=0,zorder=2)
    ax.axhline(30,color=ORANGE,ls=(0,(3,2)),lw=.7,zorder=1)
    ax.text(.52,30.45,'30 dB',fontsize=6,color=ORANGE)
    ax.set_xticks(np.arange(1,6),['Float','W8A8','SC\n4096','FROST\n1024','FROST\n4096'])
    ax.set_xlabel('Implementation and stream length L',labelpad=3)
    ax.set_ylabel('Image-level PSNR (dB)',labelpad=3)
    ax.set_ylim(min(x.min() for x in data)-1,max(x.max() for x in data)+1)
    ax.text(-.16,1.12,'a',transform=ax.transAxes,fontsize=8,fontweight='bold')
    ax.text(0,1.12,'Reconstruction fidelity',transform=ax.transAxes,fontsize=7,fontweight='bold')
    x=data[2];y=data[4]
    lo=np.floor(min(x.min(),y.min())-1);hi=np.ceil(max(x.max(),y.max())+1)
    ay.plot([lo,hi],[lo,hi],color=GRAY,lw=.75,ls=(0,(3,2)),zorder=1)
    ay.axhline(30,color=ORANGE,lw=.55,ls=(0,(3,2)),alpha=.7)
    ay.scatter(x,y,c=np.where(y>=30,TEAL,ORANGE),s=7.0,alpha=.70,linewidths=0,zorder=3)
    ay.set_xlim(lo,hi);ay.set_ylim(lo,hi)
    ay.set_xlabel('SC, L = 4096: PSNR (dB)',labelpad=3)
    ay.set_ylabel('FROST, L = 4096: PSNR (dB)',labelpad=3)
    ay.text(-.20,1.12,'b',transform=ay.transAxes,fontsize=8,fontweight='bold')
    ay.text(0,1.12,'Paired image comparison',transform=ay.transAxes,fontsize=7,fontweight='bold')
    g=next(r for r in summary['paired_comparisons'] if r['comparison']=='FROST4096_minus_SC4096')
    ci=g['image_bootstrap_ci95']
    text=f"Mean gain: {g['mean_gain_db']:.2f} dB\n95% image-bootstrap CI:\n{ci[0]:.2f} to {ci[1]:.2f} dB"
    ay.text(.96,.06,text,transform=ay.transAxes,ha='right',va='bottom',fontsize=6.0,
        bbox=dict(facecolor='white',edgecolor='none',alpha=.94,pad=1.2))
    j=int(np.argmin(y))
    ay.annotate(f'Lowest FROST mean\nBSDS500 #{ids[j]}',xy=(x[j],y[j]),xytext=(8,-19),
        textcoords='offset points',fontsize=5.7,color=INK,ha='left',va='top',
        arrowprops=dict(arrowstyle='-',lw=.55,color=GRAY,shrinkA=1,shrinkB=2),
        bbox=dict(facecolor='white',edgecolor='none',alpha=.96,pad=.7))
    for a in [ax,ay]:
        a.spines[['top','right']].set_visible(False);a.tick_params(width=.55,length=2.5,pad=2)
        a.grid(axis='y',color='#E9ECEE',lw=.45);a.set_axisbelow(True)
    fig.text(.035,.098,'Each point is one fitted image. Stochastic scores are averaged over base seeds 101, 202 and 303.',fontsize=6.0,color=GRAY)
    fig.text(.035,.05,'Boxes: median and interquartile range; whiskers: minimum to maximum. Dashed diagonal in b: equal fidelity.',fontsize=6.0,color=GRAY)
    stem=root/'figures/FROST_BSDS500_distribution'
    for ext in ['pdf','svg','png']:fig.savefig(stem.with_suffix('.'+ext),dpi=220)
    plt.close(fig)
    caption=r'''% Optional additional figure. Requires graphicx.
\begin{figure}[!htbp]
\centering
\includegraphics[width=\linewidth]{Images/FROST_BSDS500_distribution.pdf}
\caption{\textbf{Reconstruction fidelity across the complete BSDS500 evaluation image pool.}
\textbf{a}, PSNR distributions over all 200 target-specific $64\times64$ RGB
fits. Each point represents one image. SC and FROST points average the
scores of base stream seeds 101, 202 and 303; Float and W8A8 points are
deterministic. Boxes show the median and interquartile range, and whiskers
span the full observed range. \textbf{b}, Paired image-level PSNRs of
uncalibrated SC and calibrated FROST at $L=4096$. The diagonal denotes
equal fidelity. Orange points have FROST mean PSNR below 30 dB; teal
points reach or exceed 30 dB. The target with the lowest FROST mean PSNR
is identified. The reported 95\% confidence interval uses 10,000 paired
image-bootstrap resamples. Every image participates in fitting its own
model; the results measure fitted-image reconstruction.}
\label{fig:frost_bsds500_distribution}
\end{figure}
'''
    (root/'FROST_BSDS500_distribution_insert.tex').write_text(caption)
    print('Wrote vector distribution figure and optional LaTeX caption')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    plot(p.parse_args().root.resolve())
