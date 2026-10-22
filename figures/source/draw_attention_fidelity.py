from pathlib import Path
import csv,json,statistics,hashlib,re
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
import fitz
import numpy as np

ROOT=Path(__import__('os').environ['FROST_FIGURE_WORK'])
TMP=ROOT/'tmp/pdfs/frost_fig7_dumbbell'
OUT=ROOT/'outputs/frost_fig7_dumbbell'
RAW=ROOT/'outputs/frost_fig7_clean/Source/data/Fig_attention_fidelity_b.csv'
OLD=ROOT/'outputs/frost_fig7_clean/Images/Fig_attention_fidelity.pdf'
for name,filename in [('Arial','Arial.TTF'),('Arial-Bold','Arialbd.TTF')]:
    pdfmetrics.registerFont(TTFont(name,str(ROOT/'tmp/pdfs/frost_redesign/arial_fonts'/filename)))
W,H=248.26575,218.267715
INK='#26343D';GREY='#6B7280';CORAL='#C56B52';TEAL='#007F86'
X0,X1=80.,201.
XMAX=.35
def X(x):return X0+x/XMAX*(X1-X0)
text_boxes=[]
def text(c,x,y,s,size=7.3,bold=False,color=INK,align='left',tag=''):
    font='Arial-Bold' if bold else 'Arial'
    c.setFillColor(HexColor(color));c.setFont(font,size)
    width=pdfmetrics.stringWidth(s,font,size)
    if align=='right':
        c.drawRightString(x,H-y,s);left=x-width
    elif align=='center':
        c.drawCentredString(x,H-y,s);left=x-width/2
    else:
        c.drawString(x,H-y,s);left=x
    text_boxes.append({'tag':tag,'text':s,'rect':[left,y-size,left+width,y+size*.22]})
def line(c,x1,y1,x2,y2,color=INK,width=.6):
    c.setStrokeColor(HexColor(color));c.setLineWidth(width)
    c.line(x1,H-y1,x2,H-y2)
def mark(c,x,y,shape,color,size,filled=True):
    c.setStrokeColor(HexColor(color));c.setLineWidth(.65)
    c.setFillColor(HexColor(color if filled else '#FFFFFF'))
    if shape=='o':c.circle(x,H-y,size/2,stroke=1,fill=1)
    else:c.rect(x-size/2,H-y-size/2,size,size,stroke=1,fill=1)
def err(c,mean,sd,y,color):
    line(c,X(mean-sd),y,X(mean+sd),y,color,width=.8)
    for v in [mean-sd,mean+sd]:line(c,X(v),y-2.7,X(v),y+2.7,color,width=.7)

rows=list(csv.DictReader(RAW.open()))
assert len(rows)==40
stages=[('qk_only','QK only'),('hi_only','HI-SMURF only'),('av_only','AV only'),('full','Full core')]
methods=[('bipolar_iid','Bipolar IID',CORAL,'o'),('improved','Improved',TEAL,'s')]
c=canvas.Canvas(str(TMP/'panel_b.pdf'),pagesize=(W,H))
c.setTitle('Replacing individual stages: paired RMSE comparison')
text(c,24.17,17.6753,'b',size=9.5,bold=True,tag='panel')
text(c,55.6,17.6753,'Replacing individual stages',size=8.4,bold=True,tag='title')
for x,(method,label,color,shape) in zip([83.5,159.0],methods):
    mark(c,x,32.0,shape,color,3.7)
    text(c,x+6.5,34.3,label,size=7.1,tag='legend')
text(c,226,48.5,'RMSE',size=6.7,bold=True,color=GREY,align='center',tag='columnhead')
text(c,226,56.7,'reduction',size=6.7,bold=True,color=GREY,align='center',tag='columnhead')
summaries=[];gains=[]
for i,(stage,label) in enumerate(stages):
    y=65.+i*26.
    vals={}
    for method,_,color,shape in methods:
        group=sorted((r for r in rows if r['stage']==stage and r['method']==method),key=lambda r:int(r['seed']))
        values=[float(r['rmse']) for r in group]
        assert len(values)==5
        mean=statistics.mean(values);sd=statistics.stdev(values)
        vals[method]=(mean,sd)
        summaries.append({'stage':stage,'method':method,'n':5,'mean':mean,'sample_sd':sd,
            'values':values,'seeds':[int(r['seed']) for r in group],
            'display_x_pt':X(mean),'display_y_pt':y})
    base=vals['bipolar_iid'][0];imp=vals['improved'][0]
    gain=100*(1-imp/base)
    gains.append({'stage':stage,'relative_reduction_percent':gain,'computed_from_unrounded_means':True})
    text(c,75,y+2.4,label,size=7.1,align='right',tag='stage')
    if base!=imp:
        line(c,X(imp),y,X(base),y,'#A7B3BA',width=1.1)
        for method,_,color,shape in methods:
            mean,sd=vals[method]
            err(c,mean,sd,y,color)
            mark(c,X(mean),y,shape,color,4.0 if shape=='s' else 4.5)
            text(c,X(mean),y+11,f'{mean:.4f}',size=6.8,color=color,align='center',tag='value')
    else:
        # Both implementations have exactly the same five raw observations.
        # A ring and a square share the true x position; no fabricated separation.
        err(c,base,vals['bipolar_iid'][1],y,CORAL)
        mark(c,X(base),y,'o',CORAL,6.3,False)
        mark(c,X(imp),y,'s',TEAL,3.0)
        text(c,X(base),y+11,f'{base:.4f} (both)',size=6.7,color=GREY,align='center',tag='value')
    text(c,226,y+2.4,'0%' if gain==0 else f'{gain:.1f}%',size=7.8,
        bold=gain>0,color=TEAL if gain>0 else GREY,align='center',tag='gain')

line(c,X0,171,X1,171,width=.8)
for v in [0,.1,.2,.3]:
    x=X(v);line(c,x,171,x,174,width=.75)
    text(c,x,183,f'{v:.1f}',size=7.2,align='center',tag='tick')
text(c,(X0+X1)/2,195,'Attention output RMSE',size=8.0,align='center',tag='xlabel')
text(c,55.6,208,'Mean ± s.d.; 5 stream seeds',size=6.7,color=GREY,tag='footer')
c.showPage();c.save()

# Text blocks occupy disjoint regions. All x positions encode original RMSE.
for i,a in enumerate(text_boxes):
    ra=fitz.Rect(a['rect'])
    assert ra.x0>=0 and ra.x1<W and ra.y0>=0 and ra.y1<H,a
    for b in text_boxes[i+1:]:
        rb=fitz.Rect(b['rect'])
        assert not ra.intersects(rb),(a,b)
for s in summaries:
    assert abs((s['display_x_pt']-X0)/(X1-X0)*XMAX-s['mean'])<1e-15

# Preserve a exactly and replace b, retaining the original page geometry.
old=fitz.open(OLD);new=fitz.open();p=new.new_page(width=old[0].rect.width,height=old[0].rect.height)
cut=261.970455
left=fitz.Rect(0,0,cut,old[0].rect.height)
right=fitz.Rect(cut,0,old[0].rect.width,old[0].rect.height)
p.show_pdf_page(left,old,0,clip=left)
panel=fitz.open(TMP/'panel_b.pdf');p.show_pdf_page(right,panel,0)
pdf=OUT/'Fig_attention_fidelity.pdf'
new.save(pdf,garbage=4,deflate=True)
d=fitz.open(pdf)
assert len(d[0].get_images())==0
aa=np.frombuffer(old[0].get_pixmap(matrix=fitz.Matrix(2,2),clip=left).samples,dtype=np.uint8)
bb=np.frombuffer(d[0].get_pixmap(matrix=fitz.Matrix(2,2),clip=left).samples,dtype=np.uint8)
assert np.array_equal(aa,bb)
d[0].get_pixmap(matrix=fitz.Matrix(3,3)).save(TMP/'preview.png')
d[0].get_pixmap(matrix=fitz.Matrix(4,4),clip=right).save(TMP/'panel_b.png')

report={'source_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),'source_observations':40,
    'display':'horizontal paired means with sample SD','summary_ddof':1,
    'summaries':summaries,'reductions':gains,'panel_a_pixel_identical':True,
    'raster_images':0,'page_size_pt':list(d[0].rect),'text_boxes_disjoint':True,
    'x_limits':[0,.35],'new_experiments':False,'raw_values_unchanged':True}
(TMP/'validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='summaries'},indent=2))
