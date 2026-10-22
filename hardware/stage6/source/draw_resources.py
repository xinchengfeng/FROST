"""Compact publication layout from verified Stage4/5 report JSON."""
from pathlib import Path
import json, hashlib
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
import fitz
from figure_support import ROOT, TMP, OUTDIR, SOURCE, DATA, plot_data, register_fonts
d=plot_data()
register_fonts()
OUT=OUTDIR/'FROST_hardware_stage3.pdf'
W,H=510.2362,170.
INK='#24333C'; MUTED='#52616B'; GRID='#E8ECEF'
COLORS=['#00868C','#94B5CC','#557C9F','#9C8DB5','#D38B76']
box=[]
c=canvas.Canvas(str(OUT),pagesize=(W,H),initialFontName='Arial',initialFontSize=7)
c.setTitle('Hardware resource comparison')
c.setSubject('Compact layout of existing results, preserving every plotted resource value')

def text(x,y,s,size=7,bold=False,color=INK,align='left',tag=''):
    font='Arial-Bold' if bold else 'Arial'
    c.setFont(font,size);c.setFillColor(HexColor(color))
    width=pdfmetrics.stringWidth(s,font,size)
    left=x
    if align=='right':c.drawRightString(x,H-y,s);left=x-width
    elif align=='center':c.drawCentredString(x,H-y,s);left=x-width/2
    else:c.drawString(x,H-y,s)
    box.append({'s':s,'rect':[left,y-size,left+width,y+size*.2],'tag':tag})

def line(x0,y0,x1,y1,color=INK,width=1):
    c.setStrokeColor(HexColor(color));c.setLineWidth(width)
    c.line(x0,H-y0,x1,H-y1)

def marker(x,y,color,diamond=False):
    c.setStrokeColor(HexColor(color));c.setFillColor(HexColor(color));c.setLineWidth(1)
    if diamond:
        p=c.beginPath();p.moveTo(x,H-y+2.4);p.lineTo(x+2.4,H-y);p.lineTo(x,H-y-2.4);p.lineTo(x-2.4,H-y);p.close()
        c.drawPath(p,stroke=1,fill=1)
    else:c.circle(x,H-y,1.8,stroke=1,fill=1)

def panel(letter,title,subtitle,x0,x1,maxv,ticks,values,labels,xlabel):
    text(x0-12,14,letter,8,True,tag='panel')
    text(x0,14,title,7,True,tag='header')
    text(x0,25,subtitle,6.4,color=MUTED,tag='subtitle')
    X=lambda v:x0+v/maxv*(x1-x0)
    for i,(value,label) in enumerate(zip(values,labels)):
        y=42+i*18
        color=COLORS[i]
        if value:
            line(x0,y,X(value),y,color,1.15)
        marker(X(value),y,color,i==0)
        text(X(value)+5.3,y+2.15,label,7,i==0,color=INK,tag='value')
    line(x0,129,x1,129,width=1)
    for v,label in ticks:
        x=X(v);line(x,129,x,132,width=1)
        text(x,141,label,7,align='center',tag='tick')
    text((x0+x1)/2,153,xlabel,7,align='center',tag='axislabel')

for i,name in enumerate(d['names']):
    text(74,44.3+i*18,name,7,i==0,color=INK,align='right',tag='method')

panel('a','Standard-cell area','Nangate45',82,227,112,
      [(0,'0'),(25,'25'),(50,'50'),(75,'75'),(100,'100')],
      d['area'],d['area_labels'],'Cell area (10³ µm²)')
panel('b','FPGA COMB','Post-route ECP5',273,402,35000,
      [(0,'0'),(10000,'10,000'),(20000,'20,000'),(30000,'30,000')],
      d['resources'],[f'{v:,}' for v in d['resources']],'Combinational resources')
panel('c','FPGA DSP','Post-route ECP5',443,498,11,
      [(0,'0'),(4,'4'),(8,'8')],d['dsp'],[str(v) for v in d['dsp']],'DSP blocks')
text(82,166,'L = 64  |  Full score-to-output kernel  |  SExp labels: burn-in updates',6.4,color=MUTED,tag='footer')

for i,a in enumerate(box):
    ra=fitz.Rect(a['rect'])
    assert ra.x0>=0 and ra.x1<=W and ra.y0>=0 and ra.y1<=H,a
    for b in box[i+1:]:
        assert not ra.intersects(fitz.Rect(b['rect'])),(a,b)
c.showPage();c.save()
pdf=fitz.open(OUT);p=pdf[0]
assert not p.get_images()
used_fonts={s['font'] for b in p.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']}
assert all(pdf.extract_font(f[0])[3] for f in p.get_fonts()),p.get_fonts()
p.get_pixmap(matrix=fitz.Matrix(3,3)).save(TMP/'stage3_new.png')
validation={'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'names':d['names'],'area':d['area'],'resources':d['resources'],'dsp':d['dsp'],
    'labels':d['area_labels'],'new_experiments':False,'values_from_verified_reports':True,
    'height_pt':H,'original_height_pt':228,'height_reduction_percent':100*(1-H/228),
    'width_mm':W*25.4/72,'raster_images':0,'fonts_embedded':True,
    'text_boxes_disjoint':True,'background_grid':False,'all_text_neutral':True,
    'unused_helvetica_resource':False,'render':str(TMP/'stage3_new.png')}
(TMP/'stage3_validation.json').write_text(json.dumps(validation,indent=2))
print(json.dumps(validation,indent=2))
