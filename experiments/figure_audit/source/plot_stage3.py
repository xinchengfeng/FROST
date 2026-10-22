"""Seven journal figures drawn from verified software outputs.

PDF/SVG text, meshes, paths, boxes, axes and colour bars are vector objects.
Only the reference/reconstruction photographs and their pixel error maps are
embedded rasters. No curves are smoothed and no measurements are invented.
"""
from pathlib import Path
import csv, json, importlib.util, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, colors, ticker
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'source_data';INPUT=ROOT/'figure_inputs';IMAGES=ROOT/'Images'
IMAGES.mkdir(exist_ok=True)
REG=ROOT/'fonts/FROSTNimbusSans-Regular.ttf';BOLD=ROOT/'fonts/FROSTNimbusSans-Bold.ttf'
for p in [REG,BOLD]:fm.fontManager.addfont(str(p))
plt.rcParams.update({'font.family':'FROST Nimbus Sans','font.size':8.5,'axes.labelsize':8.5,
  'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
  'axes.linewidth':.7,'xtick.major.width':.65,'ytick.major.width':.65,'xtick.major.size':3,
  'ytick.major.size':3,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
  'mathtext.fontset':'stixsans','text.color':'#25343D','axes.labelcolor':'#25343D',
  'xtick.color':'#25343D','ytick.color':'#25343D','savefig.facecolor':'white'})
TEAL='#0B8C86';BLUE='#08739E';ORANGE='#BC650D';INK='#25343D';GRAY='#73838D'
GRID='#E5EAED';PURPLE='#6C63A5';LIGHT='#E8F4F2'
ERROR=colors.LinearSegmentedColormap.from_list('row_tv',['#F5FAFD','#CCE4F1','#83BBD0','#2A82A4','#153F5B'])
BASEMAP=colors.LinearSegmentedColormap.from_list('base_tv',['#F5F8F8','#88C5BF',TEAL,'#064F4B'])
DELTAMAP=colors.LinearSegmentedColormap.from_list('delta_tv',[TEAL,'#F7F8F8',ORANGE])
FIGMETA=[]

def read(name):return json.loads((DATA/name).read_text())
def records(name):return list(csv.DictReader((DATA/name).open()))
def panel(fig,x,y,letter,title):
    fig.text(x,y,letter,fontsize=12,fontweight='bold',va='baseline')
    fig.text(x+.035,y,title,fontsize=9.5,fontweight='bold',va='baseline')
def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color(GRAY)
    ax.set_axisbelow(True)
def mesh(ax, a, cmap, **kw):
    ny,nx=a.shape
    m=ax.pcolormesh(np.arange(nx+1)-.5,np.arange(ny+1)-.5,a,cmap=cmap,
      shading='flat',rasterized=False,edgecolors='none',antialiased=False,**kw)
    ax.set_ylim(ny-.5,-.5);ax.set_xlim(-.5,nx-.5);ax.tick_params(length=0,pad=4)
    for s in ax.spines.values():s.set_visible(False)
    return m
def cbar(fig,m,bounds,ticks,label,orientation='vertical'):
    cb=fig.colorbar(m,cax=fig.add_axes(bounds),ticks=ticks,orientation=orientation)
    cb.outline.set_visible(False);cb.ax.tick_params(length=0,labelsize=7.8,pad=3)
    cb.ax.xaxis.set_minor_locator(ticker.NullLocator());cb.ax.yaxis.set_minor_locator(ticker.NullLocator())
    cb.set_label(label,fontsize=8.2,labelpad=4)
    cb.solids.set_rasterized(False);cb.solids.set_edgecolor('face')
    return cb
def save(fig,stem,bars=None):
    fig.canvas.draw()
    for cb in bars or []:
        if cb.solids is not None:cb.solids.set_rasterized(False);cb.solids.set_edgecolor('face')
    for t in fig.findobj(Text):
        weight=t.get_fontweight();bold=weight in ['bold','semibold','heavy','black'] or isinstance(weight,(float,int)) and weight>=600
        t.set_fontproperties(fm.FontProperties(fname=str(BOLD if bold else REG),size=t.get_fontsize(),weight='bold' if bold else 'normal'))
    # Fixed canvas bounds preserve an intended physical figure size.
    fig.savefig(IMAGES/f'{stem}.pdf',metadata={'Title':stem,'Author':'','Subject':'Verified FROST software evaluation; no hardware data'})
    fig.savefig(IMAGES/f'{stem}.svg')
    fig.savefig(IMAGES/f'{stem}.png',dpi=240)
    fontsizes=[t.get_fontsize() for t in fig.findobj(Text) if t.get_text()]
    FIGMETA.append(dict(name=stem,width_mm=float(fig.get_figwidth()*25.4),height_mm=float(fig.get_figheight()*25.4),
      minimum_text_pt=float(min(fontsizes)),photographic_pixels=(stem=='FROST_reconstruction')))
    plt.close(fig);print('Saved vector figure',stem,flush=True)

def dimensions():
    d=read('dimension_verified_summary.json');ns=[16,32,64];families=['gaussian','diffuse','sharp','correlated']
    def val(n,f,stage):return next(g['mean_row_tv_mean'] for g in d['groups'] if (g['n'],g['family'],g['stage'])==(n,f,stage))
    base=np.array([[val(n,f,'base') for n in ns] for f in families])
    delta=np.array([[val(n,f,'calibrated')-val(n,f,'base') for n in ns] for f in families])
    fig=plt.figure(figsize=(180/25.4,117/25.4))
    panel(fig,.025,.956,'a','Base operators across distributions')
    panel(fig,.532,.956,'b','The cost of calibration transfer')
    a=fig.add_axes([.136,.605,.304,.282]);b=fig.add_axes([.647,.605,.304,.282])
    for ax,data,cmap,lo,hi in [(a,base,BASEMAP,0,.35),(b,delta,DELTAMAP,-.27,.27)]:
      m=mesh(ax,data,cmap,vmin=lo,vmax=hi)
      ax.set_yticks(range(4),['Gaussian','Diffuse','Sharp','Correlated']);ax.set_xticks(range(3),['16','32','64']);ax.set_xlabel('Score dimension, n',labelpad=5)
      for (i,j),v in np.ndenumerate(data):
        ax.text(j,i,f'{v:+.3f}' if ax==b else f'{v:.3f}',ha='center',va='center',fontsize=9.2,
          color='white' if (abs(v)>.15 if ax==b else v>.16) else INK,fontweight='bold')
      cb=cbar(fig,m,[.137 if ax==a else .648,.521,.303,.014], [0,.175,.35] if ax==a else [-.27,0,.27],
        'Mean row TV' if ax==a else 'Calibrated TV - base TV',orientation='horizontal')
    fig.text(.138,.424,'4,096 rows per family; three fits with a shared warm start',fontsize=8,color=GRAY)
    fig.text(.648,.424,'Teal: reduced error; orange: increased error',fontsize=8,color=GRAY)
    panel(fig,.025,.365,'c','More categories require longer streams')
    ax=fig.add_axes([.108,.136,.851,.177]);clean(ax)
    for n,col,mk in zip(ns,[BLUE,TEAL,PURPLE],['o','s','^']):
      g=[v for v in d['stream_groups'] if v['n']==n];L=np.array([v['L'] for v in g]);y=np.array([v['sampling_tv_mean'] for v in g]);sd=np.array([v['sampling_tv_stream_sd'] for v in g])
      ax.fill_between(L,y-sd,y+sd,color=col,alpha=.15,lw=0);ax.plot(L,y,color=col,lw=1.5,marker=mk,ms=4.1,label=f'n = {n}')
    ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xlim(50,5300);ax.set_ylim(.025,.42)
    ax.set_xticks([64,256,1024,4096],['64','256','1,024','4,096']);ax.set_yticks([.03,.1,.3],['0.03','0.1','0.3']);ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel('Recorded stream length, L',labelpad=4);ax.set_ylabel('TV to own\nstationary law',labelpad=6)
    ax.grid(axis='y',color=GRID,lw=.6);ax.legend(loc='upper right',frameon=False,ncol=3,columnspacing=1.2)
    fig.text(.108,.014,'128 fixed rows per dimension; mean ± stream s.d. (8). L excludes 256 burn-in cycles.',fontsize=8,color=GRAY)
    save(fig,'FROST_dimension_transfer')

def sampler():
    d=read('sampler_verified_summary.json')
    with np.load(DATA/'sampler_verified_conditional.npz') as a:cond={k:a[k] for k in a}
    fig=plt.figure(figsize=(180/25.4,117/25.4))
    panel(fig,.025,.956,'a','CDF precision bounds distortion')
    panel(fig,.532,.956,'b','Independent draws preserve fidelity')
    a=fig.add_axes([.105,.60,.354,.282]);clean(a)
    bits=np.array([q['word_bits'] for q in d['precision']]);mean=np.array([q['conditional_tv_mean'] for q in d['precision']]);mx=np.array([q['conditional_tv_max'] for q in d['precision']]);bound=np.array([q['sum_core_tv_bound'] for q in d['precision']])
    a.fill_between(bits,mean,mx,color=TEAL,alpha=.13,lw=0)
    a.plot(bits,bound,'--',color=GRAY,lw=1.2,label='Core-wise bound')
    a.plot(bits,mx,'o-',color=TEAL,lw=1,mfc='white',ms=4,label='Maximum of 512 tuples')
    a.plot(bits,mean,'o-',color=TEAL,lw=1.5,ms=4,label='Mean of 512 tuples')
    a.set_yscale('log');a.set_ylim(3e-5,.6);a.set_xticks(bits);a.set_xlabel('CDF word length (bit)');a.set_ylabel('Conditional TV to float')
    a.yaxis.set_minor_locator(ticker.NullLocator());a.grid(axis='y',color=GRID,lw=.6)
    a.legend(loc='upper right',frameon=False,fontsize=7.7,handlelength=1.5)
    b=fig.add_axes([.627,.60,.333,.282]);clean(b)
    choices=[('float_shared','Shared U',ORANGE,'s'),('cdf8_independent','8-bit independent',PURPLE,'^'),
      ('cdf12_independent','12-bit independent',TEAL,'o'),('float_independent','Float independent',BLUE,'o')]
    for method,label,col,mk in choices:
      g=[v for v in d['groups'] if v['method']==method];L=np.array([v['L'] for v in g]);y=np.array([v['mean_row_tv_mean'] for v in g]);sd=np.array([v['mean_row_tv_stream_sd'] for v in g])
      b.fill_between(L,y-sd,y+sd,color=col,alpha=.12,lw=0)
      b.plot(L,y,color=col,marker=mk,ms=4,lw=1.35,mfc='white' if method=='float_independent' else col,label=label,zorder=4 if method=='float_independent' else 3)
    b.axhline(d['floating_stationary']['mean_row_tv'],ls=':',color=GRAY,lw=1)
    b.set_xscale('log',base=2);b.set_xticks([64,256,1024,4096],['64','256','1,024','4,096']);b.set_ylim(.043,.263)
    b.set_xlabel('Recorded stream length, L');b.set_ylabel('Mean row TV to softmax');b.grid(axis='y',color=GRID,lw=.6)
    b.legend(loc='upper right',frameon=False,fontsize=7.7,handlelength=1.3)
    fig.text(.105,.468,'512 diagnostic state tuples; quantized kernels are row stochastic',fontsize=8,color=GRAY)
    fig.text(.627,.468,'128 fixed rows; mean ± stream s.d. (8)',fontsize=8,color=GRAY)
    panel(fig,.025,.390,'c','Unit mass can hide a different categorical law')
    ax=fig.add_axes([.105,.153,.855,.180]);clean(ax)
    ix=int(np.argmax(cond['shared_row_tv']));cat=np.arange(1,17)
    for off,key,col,label in [(-.26,'floating',BLUE,'Float contraction'),(0,'cdf12',TEAL,'12-bit independent'),(.26,'shared',ORANGE,'Shared U: exact integration')]:
      ax.bar(cat+off,cond[key][ix],width=.25,color=col,label=label,zorder=3)
    ax.set_xlim(.35,16.65);ax.set_ylim(0,.93);ax.set_xticks(cat);ax.set_xlabel('Sorted output category');ax.set_ylabel('Conditional\nprobability');ax.grid(axis='y',color=GRID,lw=.6)
    ax.legend(loc='upper right',frameon=False,ncol=1,fontsize=8,handlelength=1.1)
    ax.text(.32,.62,f'Shared-U TV = {cond["shared_row_tv"][ix]:.3f}\nMass = 1 for all three laws',transform=ax.transAxes,color=ORANGE,fontsize=9,linespacing=1.6)
    fig.text(.105,.017,'Largest shared-U discrepancy among 512 tuples. N = 16, r = 8; L excludes 256 burn-in cycles.',fontsize=8,color=GRAY)
    save(fig,'FROST_sampler_precision')

def convergence():
    d=read('convergence_verified_summary.json')
    with np.load(DATA/'convergence_verified_atlas.npz') as a:atlas={k:a[k].copy() for k in ['SC_base','FROST']};lengths=a['lengths'].copy()
    fig=plt.figure(figsize=(180/25.4,122/25.4))
    panel(fig,.025,.956,'a','The same score rows, progressively longer streams')
    for i,method in enumerate(['SC_base','FROST']):
      ax=fig.add_axes([.115+i*.385,.591,.339,.258]);m=mesh(ax,atlas[method],ERROR,norm=colors.LogNorm(vmin=.01,vmax=.8))
      ax.set_yticks([0,31,63,95,127],['1','32','64','96','128'] if i==0 else ['']*5)
      if i==0:ax.set_ylabel('Row rank by\nreference entropy',labelpad=6)
      ax.set_xticks([0,2,4,6,8],['16','64','256','1,024','4,096']);ax.set_xlabel('Recorded stream length, L',labelpad=4)
      ax.set_title('SC: base TT' if i==0 else 'FROST: calibrated TT',color=ORANGE if i==0 else TEAL,fontweight='bold',pad=6)
    cb=cbar(fig,m,[.873,.591,.012,.258],[.01,.02,.05,.1,.2,.5,.8],'Row TV')
    cb.ax.set_yticklabels(['0.01','0.02','0.05','0.1','0.2','0.5','0.8'])
    panel(fig,.025,.446,'b','Sampling declines; the approximation floor remains')
    ax=fig.add_axes([.115,.159,.845,.223]);clean(ax)
    for method,col in [('SC_base',ORANGE),('FROST',TEAL)]:
      g=d[method]['series'];y=np.array([q['mean_row_tv_mean'] for q in g]);sd=np.array([q['mean_row_tv_stream_sd'] for q in g])
      ax.fill_between(lengths,y-sd,y+sd,color=col,alpha=.16,lw=0)
      ax.plot(lengths,y,'o-',color=col,lw=1.55,ms=4,label='SC to softmax' if method=='SC_base' else 'FROST to softmax')
      ax.axhline(d[method]['analytic_mean_row_tv'],color=col,ls=(0,(4,2.4)),lw=1)
    g=d['FROST']['series'];y=np.array([q['sampling_mean_row_tv_mean'] for q in g]);sd=np.array([q['sampling_mean_row_tv_stream_sd'] for q in g])
    ax.fill_between(lengths,y-sd,y+sd,color=BLUE,alpha=.13,lw=0);ax.plot(lengths,y,'s:',color=BLUE,lw=1.35,ms=3.5,label='FROST to stationary TT')
    ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xlim(12,7500);ax.set_ylim(.022,.54)
    ax.set_xticks([16,32,64,128,256,512,1024,2048,4096],['16','32','64','128','256','512','1,024','2,048','4,096'])
    ax.set_yticks([.03,.06,.12,.24,.48],['0.03','0.06','0.12','0.24','0.48']);ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel('Recorded stream length, L');ax.set_ylabel('Mean row TV');ax.grid(axis='y',color=GRID,lw=.6)
    ax.legend(loc='upper right',frameon=False,fontsize=8.1)
    ax.text(22,.078,'Base TT limit',color=ORANGE,fontsize=8);ax.text(22,.046,'FROST limit',color=TEAL,fontsize=8)
    base=d['SC_base']['series'][-1]['mean_row_tv_mean'];frost=d['FROST']['series'][-1]['mean_row_tv_mean']
    ax.plot([4096,4850,4850,4096],[base,base,frost,frost],color=TEAL,lw=.9)
    ax.text(5200,.073,f'{d["paired_long_stream_gain_percent"]:.1f}%\nlower TV',color=TEAL,fontsize=8.2,va='center')
    fig.text(.115,.040,'128 fixed rows; eight matched stream seeds; mean ± stream s.d. Same row order and colour scale above.',fontsize=8,color=GRAY)
    fig.text(.115,.011,'N = 16, r = 8; all weights frozen. Recorded L excludes 256 burn-in cycles.',fontsize=8,color=GRAY)
    save(fig,'FROST_attention_convergence')

def formation():
    # The two-panel layout is preserved; verified event data are its only input.
    spec=importlib.util.spec_from_file_location('formation_layout',ROOT/'source/formation_layout.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.ROOT=ROOT;mod.save=save
    with np.load(INPUT/'convergence/inputs.npz') as a:inp={k:a[k].copy() for k in a}
    protocol=json.loads((INPUT/'convergence/protocol.json').read_text());traces={}
    for method in ['SC_base','FROST']:
      traces[method]=[]
      for seed in protocol['stream_seeds']:
        with np.load(INPUT/'convergence'/f'trace_{method}_seed{seed}.npz') as a:traces[method].append({k:a[k].copy() for k in a})
    # Override layout-module defaults after import for consistent typography.
    plt.rcParams.update({'font.family':'FROST Nimbus Sans','text.color':INK,'axes.labelcolor':INK,
      'xtick.color':INK,'ytick.color':INK,'font.size':8.5,'pdf.fonttype':42,'svg.fonttype':'none'})
    mod.formation(inp,np.array(protocol['lengths']),traces)

def bsds_distribution():
    rows=records('bsds_verified_per_image.csv');summary=read('bsds_verified_summary.json')
    def vals(method,L):return np.array([float(r['psnr_stream_mean']) for r in rows if (r['method'],int(r['L']))==(method,L)])
    methods=[('Float',0),('W8A8',0),('SC_base',4096),('FROST',1024),('FROST',4096)]
    values=[vals(*m) for m in methods];cols=[GRAY,BLUE,ORANGE,PURPLE,TEAL]
    fig=plt.figure(figsize=(180/25.4,101/25.4))
    panel(fig,.025,.950,'a','Every target contributes one point')
    panel(fig,.535,.950,'b','Paired fidelity at the same L')
    a=fig.add_axes([.100,.247,.358,.589]);clean(a)
    box=a.boxplot(values,positions=np.arange(1,6),widths=.49,whis=(0,100),showfliers=False,patch_artist=True,
      medianprops=dict(color=INK,lw=1.2),whiskerprops=dict(color=GRAY,lw=.9),capprops=dict(color=GRAY,lw=.9))
    jitter=np.random.default_rng(20261010).uniform(-.19,.19,(5,200))
    for i,(v,c) in enumerate(zip(values,cols)):
      box['boxes'][i].set(facecolor=c,alpha=.14,edgecolor=c,linewidth=1)
      a.scatter(i+1+jitter[i],v,s=5.5,color=c,alpha=.50,edgecolors='none',zorder=3)
    a.axhline(30,color=ORANGE,lw=.8,ls='--');a.text(.60,30.7,'30 dB',color=ORANGE,fontsize=8)
    a.set_ylim(22,54);a.set_xlim(.45,5.55);a.set_ylabel('PSNR (dB)')
    a.set_xticks(np.arange(1,6),['Float','W8A8','SC\n4,096','FROST\n1,024','FROST\n4,096']);a.tick_params(axis='x',pad=5)
    a.grid(axis='y',color=GRID,lw=.6)
    fig.text(.100,.101,'Stochastic points average three stream seeds.\nBoxes: median and IQR; whiskers: full range.',fontsize=8,color=GRAY,linespacing=1.4)
    b=fig.add_axes([.625,.247,.338,.589]);clean(b)
    x=values[2];y=values[4];ids=np.array([int(r['image_id']) for r in rows if (r['method'],int(r['L']))==('FROST',4096)])
    b.plot([22,43],[22,43],color=GRAY,lw=.9,ls='--');b.axhline(30,color=ORANGE,lw=.75,ls=':')
    passed=y>=30;b.scatter(x[passed],y[passed],s=15,c=TEAL,alpha=.7,edgecolors='white',lw=.35,zorder=3)
    b.scatter(x[~passed],y[~passed],s=22,c=ORANGE,edgecolors='white',lw=.35,zorder=4)
    ix=int(np.argmin(y));b.annotate(f'#{ids[ix]}: {y[ix]:.2f} dB',xy=(x[ix],y[ix]),xytext=(23.7,34),
      fontsize=8,color=ORANGE,arrowprops=dict(arrowstyle='-',color=ORANGE,lw=.8),bbox=dict(fc='white',ec='none',alpha=.92,pad=1))
    b.set_xlim(22,43);b.set_ylim(22,43);b.set_xticks([25,30,35,40]);b.set_yticks([25,30,35,40]);b.set_xlabel('SC, L = 4,096: PSNR (dB)');b.set_ylabel('FROST, L = 4,096: PSNR (dB)')
    b.grid(color=GRID,lw=.6)
    pair=summary['paired_comparisons'][0];ci=pair['image_bootstrap_ci95'];b.text(.035,.95,f'Mean paired gain: {pair["mean_gain_db"]:.2f} dB\n95% image CI: {ci[0]:.2f}-{ci[1]:.2f} dB',
      transform=b.transAxes,va='top',fontsize=8.4,linespacing=1.5,bbox=dict(fc='white',ec='none',alpha=.9,pad=1.6))
    fig.text(.625,.101,'196/200 image means ≥ 30 dB; all 200 shown.\n199/200 improve over SC at L = 4,096.',fontsize=8,color=GRAY,linespacing=1.4)
    fig.text(.100,.022,'All target pixels participated in fitting. This is fitted-image reconstruction, not held-out restoration.',fontsize=8,color=GRAY)
    save(fig,'FROST_BSDS500_distribution')

def reconstruction():
    W,H=180,200;fig=plt.figure(figsize=(W/25.4,H/25.4))
    def textmm(x,y,s,**kw):fig.text(x/W,1-y/H,s,**kw)
    def axis(x,y,w,h):return fig.add_axes([x/W,1-(y+h)/H,w/W,h/H])
    rois=json.loads((INPUT/'bsds/roi_coordinates.json').read_text())['coordinates']
    raw=records('bsds_verified_all_metrics.csv');ids=[2018,43033,235098];names=['Snow scene','Bird and reeds','Roller coaster']
    keys=['truth','Float','W8A8','SC_base_L4096_S101','FROST_L1024_S101','FROST_L4096_S101']
    heads=['Reference','Float','W8A8','SC: base TT\nL = 4,096','FROST\nL = 1,024','FROST\nL = 4,096']
    textmm(4,4,'The same fitted image, different stochastic implementations',fontsize=9.5,fontweight='bold',va='top')
    textmm(4,10,'Three retained pilot targets; encoded coordinates only at inference; stream seed 101.',fontsize=8,color=GRAY,va='top')
    size=25.5;pitch=29;x0=4
    for j,head in enumerate(heads):textmm(x0+j*pitch+size/2,23,head,fontsize=8,ha='center',va='center',linespacing=1.18,color=TEAL if j>=4 else INK,fontweight='bold' if j>=4 else 'normal')
    for row,(image_id,name) in enumerate(zip(ids,names)):
      top=34+row*52
      textmm(4,top-3,chr(97+row),fontsize=11,fontweight='bold',va='bottom')
      textmm(10,top-3,f'{name}  |  BSDS500 #{image_id}',fontsize=8.4,va='bottom')
      with np.load(INPUT/'bsds'/f'image_{image_id}.npz') as z:saved={k:z[k].copy() for k in z}
      truth=saved['truth'];rx,ry,rs=rois[str(image_id)];assert rs==16
      for j,key in enumerate(keys):
        img=saved[key];xx=x0+j*pitch
        ax=axis(xx,top,size,size);ax.imshow(img,interpolation='nearest');ax.axis('off')
        ax.add_patch(Rectangle((rx-.5,ry-.5),rs,rs,fill=False,ec=BLUE,lw=.85))
        if j>=4:ax.add_patch(Rectangle((-.5,-.5),64,64,fill=False,ec=TEAL,lw=.8))
        if j==0:label='PSNR / RGB SSIM';col=GRAY
        else:
          method=['','Float','W8A8','SC_base','FROST','FROST'][j];L=[0,0,0,4096,1024,4096][j];seed=0 if j<3 else 101
          score=next(r for r in raw if (int(r['image_id']),r['method'],int(r['L']),int(r['stream_base_seed']))==(image_id,method,L,seed))
          label=f'{float(score["psnr"]):.2f} / {float(score["ssim"]):.4f}';col=(TEAL if float(score['psnr'])>=30 else ORANGE) if j>=4 else INK
        textmm(xx+size/2,top+size+1.1,label,fontsize=7.8,ha='center',va='top',color=col,fontweight='bold' if j==5 else 'normal')
        zo=axis(xx,top+size+6,12,12);zo.imshow(img[ry:ry+rs,rx:rx+rs],interpolation='nearest');zo.axis('off');zo.add_patch(Rectangle((-.5,-.5),rs,rs,fill=False,ec=BLUE,lw=.75))
        er=axis(xx+13.5,top+size+6,12,12);error=np.mean(np.abs(img.astype(float)-truth),-1)
        artist=er.imshow(error[ry:ry+rs,rx:rx+rs],cmap='magma',vmin=0,vmax=.1,interpolation='nearest');er.axis('off')
      if row<2:fig.add_artist(Line2D([4/W,176/W],[1-(top+45.3)/H]*2,transform=fig.transFigure,color=GRID,lw=.65))
    textmm(4,185,'Blue boxes: the same 16 × 16 ROI.',fontsize=8,va='top')
    textmm(4,189.7,'Below each crop: enlarged ROI (left); RGB absolute error (right).',fontsize=8,va='top')
    textmm(4,194.4,'Target-specific fits; figure scores are individual seed values.',fontsize=8,color=GRAY,va='top')
    cb=fig.colorbar(artist,cax=axis(138,188,36,2.1),orientation='horizontal',ticks=[0,.05,.1]);cb.ax.tick_params(labelsize=7.5,length=2,pad=2);cb.outline.set_linewidth(.5)
    cb.ax.set_title('Mean RGB absolute error',fontsize=7.7,pad=4);cb.solids.set_rasterized(False);cb.solids.set_edgecolor('face')
    save(fig,'FROST_reconstruction',[cb])

def execution():
    fig=plt.figure(figsize=(180/25.4,122/25.4));ax=fig.add_axes([.025,.025,.95,.95]);ax.set_xlim(0,100);ax.set_ylim(0,100);ax.axis('off')
    def txt(x,y,s,size=8.5,**kw):ax.text(x,y,s,fontsize=size,ha='center',va='center',**kw)
    def box(x,y,w,h,label,kind='sc',size=8.4):
      col,fill={'sc':(TEAL,LIGHT),'fsm':(BLUE,'#EAF3F8'),'num':(GRAY,'#F1F4F6')}[kind]
      ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.25,rounding_size=.9',fc=fill,ec=col,lw=.85));txt(x+w/2,y+h/2,label,size)
    def arr(a,b,col=GRAY):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,color=col,lw=.9))
    def plus(x,y):ax.add_patch(Circle((x,y),1.8,fc='white',ec=GRAY,lw=.9));txt(x,y,'+',11)
    txt(1,97,'a',12,fontweight='bold');ax.text(4,97,'The evaluated block uses two distinct pre-normalization operators',fontsize=9.5,fontweight='bold',va='center')
    txt(3,82,'X',10,fontweight='bold',color=INK)
    box(10,75,22,14,'Norm 1\nSC variance + FSM rsqrt\nOwn affine parameters','fsm',8)
    box(41,75,38,14,'Stochastic multi-head attention\nQKV → QK → HI-SMURF → AV\nHead concatenation + projection','sc',8)
    plus(91,82);txt(98,82,'H',10,fontweight='bold')
    arr((5,82),(9.5,82));arr((32.5,82),(40.5,82));arr((79.5,82),(89,82));arr((93,82),(96,82))
    ax.plot([3,3,91,91],[84,92,92,84.3],color=GRAY,lw=.85);arr((91,85),(91,84))
    txt(3,58,'H',10,fontweight='bold')
    box(10,51,22,14,'Norm 2\nSC variance + FSM rsqrt\nOwn affine parameters','fsm',8)
    box(39,51,13,14,'SC linear\nW1','sc')
    box(58,51,13,14,'FSM\nGELU','fsm')
    box(77,51,13,14,'SC linear\nW2','sc')
    plus(95,58);txt(99,50,'Y',10,fontweight='bold',color=INK)
    arr((5,58),(9.5,58));arr((32.5,58),(38.5,58));arr((52.5,58),(57.5,58));arr((71.5,58),(76.5,58));arr((90.4,58),(93,58));arr((97,58),(99,52))
    ax.plot([3,3,95,95],[60,69,69,60.3],color=GRAY,lw=.85);arr((95,61),(95,60))
    txt(50,45,'Numerical helpers: centering, range scaling, mean, affine / bias / residual sums and stage conversions',7.8,color=GRAY)
    ax.plot([0,100],[41,41],color=GRID,lw=.8)
    txt(1,37,'b',12,fontweight='bold');ax.text(4,37,'Joint attention as an ancestral categorical draw',fontsize=9.5,fontweight='bold',va='center')
    box(.5,19,19.5,12,'Scores → sort + center\nBounded encoding\nInput-driven FSM bank','num',8)
    box(28,20,14,10,'CDF of\nG1(s1)','fsm',8.5);box(49,20,14,10,'CDF of\nG2(s2)','fsm',8.5);box(71,20,14,10,'CDF of\nGn(sn)','fsm',8.5)
    arr((20.5,25),(27.5,25));arr((42.5,25),(48.5,25));arr((63.5,25),(70.5,25));txt(45.5,28,'b1',7.9,color=GRAY);txt(67,28,'...',11,color=GRAY)
    for x,name in [(35,'U1'),(56,'U2'),(78,'Un')]:txt(x,14.6,f'Independent {name}',8,color=BLUE);arr((x,16.6),(x,19.5),BLUE)
    # FSM state inputs select the appropriate compiled CDF row at each stage.
    ax.plot([13,13,78],[31.4,33,33],lw=.8,color=BLUE)
    for x in [35,56,78]:arr((x,33),(x,30.5),BLUE)
    txt(94,25,'Sorted\ncategory c',8.5,color=TEAL);arr((85.5,25),(90,25))
    box(74,2,25,8,'Inverse permutation\nOne-hot count / L','sc',8.3)
    arr((94,21.5),(94,10.4));arr((73.5,6),(64,6));txt(52,6,'Attention probabilities',8.5,color=TEAL,fontweight='bold')
    txt(23,1.5,'Compiled offline: row probabilities → CDFs',7.9,color=GRAY)
    save(fig,'FROST_execution')

if __name__=='__main__':
    execution();dimensions();sampler();convergence();formation();bsds_distribution();reconstruction()
    (DATA/'figure_metadata.json').write_text(json.dumps(FIGMETA,indent=2)+'\n')
