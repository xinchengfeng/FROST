"""Restyle Fig.8 without altering its measured main-plot geometry.

Original curves, model points and sample-standard-deviation error bars
are reused as PDF vectors. The accuracy inset is an affine projection of
the original mean curve and error-bar vectors, with circular mean markers
kept circular. No experimental data or statistical summaries are inferred.
"""
from pathlib import Path
import re, math, json, csv, statistics, shutil, hashlib
import fitz

ROOT=Path(__import__('os').environ['FROST_FIGURE_WORK'])
SRC=ROOT/'tmp/pdfs/frost_layout_only/build/Images'
OUT=ROOT/'outputs/frost_plot_refresh/Images'
TMP=ROOT/'tmp/pdfs/frost_plot_refresh/fig8'
OUT.mkdir(parents=True,exist_ok=True);TMP.mkdir(parents=True,exist_ok=True)
FONT=ROOT/'tmp/pdfs/frost_redesign/arial_fonts'
RAW=Path(__file__).resolve().parent/'fig8_source_data'
RAW.mkdir(parents=True,exist_ok=True)
ORIGINAL_DATA=ROOT/'tmp/pdfs/frost_plot_refresh/fig8/raw/section3_3_transformer/source_data'
for n in ['Fig_transformer_performance.csv','Fig_transformer_ablation_a.csv',
          'Fig_transformer_ablation_b.csv','per_model_metrics.csv']:
 if not (RAW/n).exists():shutil.copy2(ORIGINAL_DATA/n,RAW/n)
FILES={'Arial':'Arial.TTF','ArialBold':'Arialbd.TTF','ArialItalic':'Ariali.TTF','ArialBoldItalic':'Arialbi.TTF'}
TEAL='#007F86';CORAL='#C56B52';BLUE='#426A90';GREY='#6B7280';INK='#26343D';LIGHT='#BBC5CE'

def rgb(h):return tuple(int(h[i:i+2],16)/255 for i in (1,3,5))
def close(a,b):return a is not None and max(abs(x-y) for x,y in zip(a,b))<2e-5
OLD_BLUE=(0,114/255,178/255)
OLD_GREY=(126/255,135/255,144/255)
OLD_ORANGE=(213/255,94/255,0)
OLD_GREEN=(0,158/255,115/255)
COLOR_RE=re.compile(rb'(?<![\w.])([+-]?(?:\d*\.\d+|\d+))\s+([+-]?(?:\d*\.\d+|\d+))\s+([+-]?(?:\d*\.\d+|\d+))\s+(rg|RG)\b')

def mapping(kind):
 return [(OLD_BLUE,rgb(TEAL)),(OLD_GREY,rgb(CORAL)),
         (OLD_ORANGE,rgb(GREY if kind=='performance' else BLUE)),
         (OLD_GREEN,rgb(BLUE)),((0,0,0),rgb(INK)),
         ((34/255,39/255,44/255),rgb(INK)),
         ((182/255,189/255,194/255),rgb(LIGHT)),
         ((220/255,224/255,227/255),rgb(LIGHT))]

def recolor(doc,kind):
 def replace(m):
  old=tuple(float(m[i]) for i in (1,2,3))
  for a,b in mapping(kind):
   if close(old,a):return (' '.join(f'{v:.7f}' for v in b)+' '+m[4].decode()).encode()
  return m[0]
 for x in range(1,doc.xref_length()):
  if doc.xref_is_stream(x):
   s=doc.xref_stream(x);n=COLOR_RE.sub(replace,s)
   if n!=s:doc.update_stream(x,n)

def txt(page,x,y,s,size=7.3,color=INK,bold=False,rot=0,align=None):
 f='ArialBold' if bold else 'Arial'
 if align=='right':x-=fitz.Font(fontfile=str(FONT/FILES[f])).text_length(s,fontsize=size)
 if align=='center':x-=fitz.Font(fontfile=str(FONT/FILES[f])).text_length(s,fontsize=size)/2
 page.insert_text((x,y),s,fontname=f,fontsize=size,color=rgb(color),rotate=rot,overlay=True)

def retype(doc,kind):
 p=doc[0];spans=[]
 for b in p.get_text('dict')['blocks']:
  for l in b.get('lines',[]):
   rot=int(round(math.degrees(math.atan2(-l['dir'][1],l['dir'][0]))))%360
   for s in l['spans']:spans.append((s,rot))
 # Text only; retain every graphics operator and marker.
 for x in set(p.get_contents())|set(x[0] for x in p.get_xobjects()):
  s=doc.xref_stream(x)
  doc.update_stream(x,re.sub(rb'\bBT\b.*?\bET\b',b'',s,flags=re.S))
  if x not in p.get_contents():doc.xref_set_key(x,'Resources/Font','null')
 for alias,file in FILES.items():p.insert_font(fontname=alias,fontfile=str(FONT/file))
 oldlegend={'Floating-point reference','Chance','Direct transfer','Calibrated'}
 for s,rot in spans:
  string=s['text'];x,y=s['origin'];oldfont=s['font']
  if kind=='performance' and x<260 and string in oldlegend:continue
  if kind=='performance' and string=='L' and y>190:continue
  if kind=='performance' and string=='Bitstream length, ':
   normal='Bitstream length, ';size=8.3
   nf=fitz.Font(fontfile=str(FONT/FILES['Arial']));it=fitz.Font(fontfile=str(FONT/FILES['ArialItalic']))
   nw=nf.text_length(normal,fontsize=size);iw=it.text_length('L',fontsize=size)
   centre=(53.57480239868164+238.06103515625)/2+(261.97045516967773 if x>260 else 0)
   left=centre-(nw+iw)/2
   p.insert_text((left,y),normal,fontname='Arial',fontsize=size,color=rgb(INK),overlay=True)
   p.insert_text((left+nw,y),'L',fontname='ArialItalic',fontsize=size,color=rgb(INK),overlay=True)
   continue
  bold='Bold' in oldfont
  italic='Oblique' in oldfont or 'Italic' in oldfont
  f='ArialBoldItalic' if bold and italic else 'ArialBold' if bold else 'ArialItalic' if italic else 'Arial'
  size=s['size']
  if string in ['a','b'] and y<25:
   string=string if kind=='performance' else {'a':'c','b':'d'}[string]
   size=9.5;f='ArialBold'
  elif y<26 and x>50:size=8.4;f='ArialBold'
  elif abs(size-7.1)<.1:size=7.3
  elif abs(size-8)<.1:size=8.3
  elif abs(size-6.8)<.1:size=7
  color=rgb(INK)
  p.insert_text((x,y),string,fontname=f,fontsize=size,color=color,rotate=rot,overlay=True)
 return len(spans)

def accuracy_zoom(page,drawings):
 # Clear the former in-panel legend only. This rectangle contains no data.
 page.draw_rect(fitz.Rect(139,91,238,132),color=None,fill=(1,1,1),overlay=True)
 # The inset uses the source's exact affine y mapping (0 and100% ticks).
 source100=39.27348327636719
 step=(171.3118133544922-source100)/100
 source_top=source100-.5*step;source_bottom=source100+3*step
 x0,x1=98.,223.;y0,y1=101.,126.
 sx0,sx1=69.3094482421875,220.9550018310547
 # Leave a little horizontal margin around both endpoint markers.
 def X(x):return x0+5+(x-sx0)/(sx1-sx0)*(x1-x0-10)
 def Y(y):return y0+(y-source_top)/(source_bottom-source_top)*(y1-y0)
 page.draw_rect(fitz.Rect(79,86,236,137),color=None,fill=(1,1,1),overlay=True)
 txt(page,92,95,'Zoom: 97–100.5%',size=6.8,bold=True)
 col=rgb(GREY)
 page.draw_line((x0,y0),(x0,y1),color=col,width=.45)
 page.draw_line((x0,y1),(x1,y1),color=col,width=.45)
 for value in [97,100]:
  yy=Y(source100+(100-value)*step)
  page.draw_line((x0-2,yy),(x0,yy),color=col,width=.45)
  txt(page,x0-4,yy+2,str(value),size=6.5,align='right')
 for x,label in [(sx0,'64'),((sx0+sx1)/2,'256'),(sx1,'1,024')]:
  xx=X(x);page.draw_line((xx,y1),(xx,y1+2),color=col,width=.45)
  txt(page,xx,135,label,size=6.5,align='center')
 # The unchanged100% reference is a dashed horizontal line.
 page.draw_line((x0,Y(source100)),(x1,Y(source100)),color=col,width=.7,dashes='[3 2]')
 # Raw, original model-level CSVs were subsequently recovered. Compute the
 # inset directly from those exact values, rather than from PDF digitization.
 rows=list(csv.DictReader(open(RAW/'Fig_transformer_performance.csv')))
 summary=[];previous=None
 for L in [64,256,1024]:
  vals=[100*float(r['accuracy']) for r in rows if r['config']=='calibrated_full' and int(r['L'])==L]
  mean=statistics.mean(vals);sd=statistics.stdev(vals)
  xx=x0+5+math.log2(L/64)/4*(x1-x0-10)
  yv=lambda value:y0+(100.5-value)/3.5*(y1-y0)
  yy=yv(mean)
  page.draw_line((xx,yv(mean-sd)),(xx,yv(mean+sd)),color=rgb(TEAL),width=.65)
  for cap in [mean-sd,mean+sd]:page.draw_line((xx-1.8,yv(cap)),(xx+1.8,yv(cap)),color=rgb(TEAL),width=.65)
  if previous:page.draw_line(previous,(xx,yy),color=rgb(TEAL),width=.8)
  page.draw_circle((xx,yy),1.4,color=rgb(TEAL),fill=rgb(TEAL),width=.65)
  previous=(xx,yy);summary.append({'L':L,'mean_accuracy_percent':mean,'sample_sd_percent':sd})
 # Move the complete panel-a legend to the row's previously unused bottom.
 for x,y,label,color,dash,mark in [
  (53,208,'Float reference',GREY,'[3 2]',False),
  (158,208,'Chance',LIGHT,'[1 2]',False),
  (53,220,'Direct transfer',CORAL,None,True),
  (158,220,'Calibrated',TEAL,None,True)]:
  page.draw_line((x,y-2.2),(x+12,y-2.2),color=rgb(color),width=.9,dashes=dash)
  if mark:
   if color==CORAL:page.draw_rect(fitz.Rect(x+4.5,y-3.7,x+7.5,y-.7),color=rgb(color),fill=rgb(color),width=.6)
   else:page.draw_circle((x+6,y-2.2),1.5,color=rgb(color),fill=rgb(color),width=.6)
  txt(page,x+16,y,label,size=7)
 return {'data_source':'Original Fig_transformer_performance.csv; three model means at each L',
         'summary':summary,
         'y_range_percent':[97,100.5],
         'original_models_preserved_on_main_plot':True,
         'calibration_source100_y_pt':source100,
         'source_percent_to_y_pt':step}

def validate_mean_curves(original):
 rows=list(csv.DictReader(open(RAW/'Fig_transformer_performance.csv')))
 paths=original[0].get_drawings();checks=[]
 for config,idxa,idxb in [('generic_full',30,99),('calibrated_full',34,103)]:
  for metric,index in [('accuracy',idxa),('centered_logit_relative_error',idxb)]:
   items=paths[index]['items'];pts=[items[0][1]]+[x[2] for x in items]
   for L,point in zip([64,256,1024],pts):
    values=[float(r[metric]) for r in rows if r['config']==config and int(r['L'])==L]
    mean=statistics.mean(values)
    if metric=='accuracy':expected=171.3118133544922-100*mean*1.32038330078125
    else:expected=51.178741455078125-math.log10(mean)*65.86573028564453
    error=abs(point.y-expected)
    assert error<.0002,(config,metric,L,error)
    checks.append({'config':config,'metric':metric,'L':L,'mean':mean,'PDF_y_error_pt':error})
 return checks

def build(kind):
 name='Fig_transformer_'+kind+'.pdf';original=fitz.open(SRC/name)
 drawings=original[0].get_drawings();orig_count=len(drawings)
 work=fitz.open();p=work.new_page(width=original[0].rect.width,height=original[0].rect.height)
 p.show_pdf_page(p.rect,original,0)
 count=retype(work,kind);recolor(work,kind)
 inset=accuracy_zoom(work[0],drawings) if kind=='performance' else None
 checks=validate_mean_curves(original) if kind=='performance' else []
 out=OUT/name;work.save(out,garbage=4,deflate=True)
 check=fitz.open(out);check[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(TMP/(name[:-4]+'.png'))
 return {'file':str(out),'kind':kind,'source_drawings':orig_count,'output_drawings':len(check[0].get_drawings()),
         'source_text_runs':count,'size_pt':[check[0].rect.width,check[0].rect.height],
         'images':len(check[0].get_images()),'inset':inset,
         'raw_CSV_validation':checks,
         'data_policy':'Main measured vectors unchanged; inset from original model-level CSV means and sample SD.'}

if __name__=='__main__':
 stats=[build('performance'),build('ablation')]
 (ROOT/'outputs/frost_plot_refresh/Source/fig8_refresh_validation.json').write_text(json.dumps(stats,indent=2))
 print(json.dumps(stats,indent=2))
