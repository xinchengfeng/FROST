"""Compact publication layout from verified Stage4/5 report JSON."""
from pathlib import Path
import json, hashlib
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
import fitz
import os
from figure_support import ROOT, TMP, OUTDIR, SOURCE, DATA, plot_data, register_fonts
d=plot_data()
register_fonts()
OUT=OUTDIR/'FROST_hardware_tradeoffs.pdf'
W,H=510.2362,170.
COLORS=['#00868C','#94B5CC','#557C9F','#9C8DB5','#D38B76']
INK='#24333C';MUTED='#52616B';GRID='#E8ECEF'
labels=[];shapes=[]
c=canvas.Canvas(str(OUT),pagesize=(W,H),initialFontName='Arial')
c.setTitle('Hardware energy and latency trade-offs')
c.setSubject('Compact layout of supplied results and uncertainty; no new experiments.')

def line(x0,y0,x1,y1,color=INK,width=1):
    c.setLineWidth(width);c.setStrokeColor(HexColor(color));c.line(x0,H-y0,x1,H-y1)

def text(x,y,s,size=7,bold=False,color=INK,align='left',tag=''):
    font='Arial-Bold' if bold else 'Arial'
    c.setFont(font,size);c.setFillColor(HexColor(color))
    width=pdfmetrics.stringWidth(s,font,size)
    if align=='center':left=x-width/2;c.drawCentredString(x,H-y,s)
    elif align=='right':left=x-width;c.drawRightString(x,H-y,s)
    else:left=x;c.drawString(x,H-y,s)
    labels.append({'text':s,'tag':tag,'bbox':[left,y-size,left+width,y+.22*size]})

def dot(x,y,color,frost=False):
    c.setFillColor(HexColor(color));c.setStrokeColor(HexColor('#FFFFFF'));c.setLineWidth(1)
    if frost:
        r=3.4;p=c.beginPath();p.moveTo(x,H-y+r);p.lineTo(x+r,H-y)
        p.lineTo(x,H-y-r);p.lineTo(x-r,H-y);p.close();c.drawPath(p,fill=1,stroke=1)
    else:c.circle(x,H-y,2.7,fill=1,stroke=1)

title_y=(1-.91386)*H
text(70,title_y,'a',size=8,bold=True,tag='panel')
text(84,title_y,'Energy per 16-query tile',size=7,bold=True,tag='title')
text(348,title_y,'b',size=8,bold=True,tag='panel')
text(362,title_y,'Area and latency',size=7,bold=True,tag='title')
text(281,24,'FROST energy',size=6.3,color=MUTED,align='center',tag='reduction header')
text(281,32.5,'reduction',size=6.3,color=MUTED,align='center',tag='reduction header')

ex0,ex1=70.,248.
def EX(v):return ex0+v/3.35*(ex1-ex0)
energy_records=[]
for i,name in enumerate(d['names']):
    y=36.+18*i
    color=COLORS[i];v=d['energy'][i];lo,hi=d['energy_error_bounds'][i]
    c.setFillColor(HexColor(color));c.rect(ex0,H-y-3.3,EX(v)-ex0,6.6,stroke=0,fill=1)
    line(EX(lo),y,EX(hi),y,INK)
    line(EX(lo),y-2,EX(lo),y+2,INK);line(EX(hi),y-2,EX(hi),y+2,INK)
    text(62,y+2.3,name,size=7,bold=i==0,align='right',tag='method')
    text(EX(v)+4,y+2.3,d['energy_labels'][i],size=7,bold=i==0,tag='energy label')
    if d['reduction_labels'][i]:text(281,y+2.3,d['reduction_labels'][i],size=7,bold=True,align='center',tag='reduction')
    energy_records.append({'name':name,'energy_uJ':v,'uncertainty_bounds_uJ':[lo,hi],
                           'y_pt':y,'bar_end_x_pt':EX(v),'error_x_pt':[EX(lo),EX(hi)]})
line(ex0,125,ex1,125)
for v in [0,1,2,3]:
    line(EX(v),125,EX(v),128)
    text(EX(v),138,str(v),size=7,align='center',tag='energy tick')
text((ex0+ex1)/2,151,'Estimated energy per tile (µJ)',size=7,align='center',tag='energy xlabel')

sx0,sx1=348.,498.
sy0,sy1=30.,125.
def SX(x):return sx0+x/110*(sx1-sx0)
def SY(y):return sy1-y/1150*(sy1-sy0)
line(sx0,sy0,sx0,sy1);line(sx0,sy1,sx1,sy1)
for v in [0,50,100]:
    line(SX(v),sy1,SX(v),sy1+3)
    text(SX(v),138,str(v),align='center',tag='scatter xtick')
for v in [0,500,1000]:
    line(sx0-3,SY(v),sx0,SY(v))
    text(sx0-6,SY(v)+2.3,str(v),align='right',tag='scatter ytick')
text((sx0+sx1)/2,151,'Cell area (10³ µm²)',align='center',tag='scatter xlabel')
c.saveState();c.translate(318.7,H-(sy0+sy1)/2);c.rotate(90)
c.setFillColor(HexColor(INK));c.setFont('Arial',7);c.drawCentredString(0,0,'Tile latency (µs)');c.restoreState()
positions=[(-7,11,'right'),(-8,-8,'right'),(-8,-8,'right'),(3,13,'center'),(0,-8,'center')]
scatter_records=[]
for i,(x,y) in enumerate(d['scatter']):
    xx,yy=SX(x),SY(y);dot(xx,yy,COLORS[i],i==0)
    dx,dy,align=positions[i]
    text(xx+dx,yy+dy,d['names'][i],size=7,bold=i==0,align=align,tag='scatter label')
    scatter_records.append({'name':d['names'][i],'area_1e3_um2':x,'tile_latency_us':y,'page_xy_pt':[xx,yy]})

text(70,164,'Nangate45  |  1.10 V, 25 °C, 8 MHz  |  Pre-layout estimates',size=6.4,color=MUTED,tag='footer')
c.showPage();c.save()

for label in labels:
    r=fitz.Rect(label['bbox'])
    assert r.x0>=0 and r.x1<=W and r.y0>=0 and r.y1<=H,label
overlaps=[]
for i,a in enumerate(labels):
    for b in labels[i+1:]:
        if fitz.Rect(a['bbox']).intersects(fitz.Rect(b['bbox'])):overlaps.append([a,b])
assert not overlaps,overlaps
for r in energy_records:
    assert abs((r['bar_end_x_pt']-ex0)/(ex1-ex0)*3.35-r['energy_uJ'])<1e-14
for r in scatter_records:
    assert abs((r['page_xy_pt'][0]-sx0)/(sx1-sx0)*110-r['area_1e3_um2'])<1e-12
    assert abs((sy1-r['page_xy_pt'][1])/(sy1-sy0)*1150-r['tile_latency_us'])<1e-12
doc=fitz.open(OUT);p=doc[0]
assert not p.get_images()
assert all(doc.extract_font(v[0])[3] for v in p.get_fonts())
p.get_pixmap(matrix=fitz.Matrix(2.6,2.6)).save(TMP/'tradeoffs_new.png')
report={'output':str(OUT),'dimensions_pt':[W,H], 'original_dimensions_pt':[W,236],
    'height_reduction_percent':100*(1-H/236), 'data_origin':str(DATA),
    'data_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),
    'means_and_bounds_preserved':True,'no_new_experiments':True,
    'energy_records':energy_records,'scatter_records':scatter_records,
    'font_family':os.environ.get('FROST_FIGURE_FONT_DIR', 'DejaVu Sans fallback'),'font_sizes_pt':[6.3,6.4,7,8],
    'raster_images':0,'all_fonts_embedded':True,'text_overlap_count':len(overlaps),
    'native_panel_baseline_fraction':.91386,'panel_x_fraction':[70/W,348/W]}
(TMP/'tradeoffs_validation.json').write_text(json.dumps(report,indent=2))
print(OUT)
