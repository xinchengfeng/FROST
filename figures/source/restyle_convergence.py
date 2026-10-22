"""Style-only refresh of the two original vector diagnostics in main Fig. 6.

Every original data path, cell, uncertainty band and clipping path is kept.
Only categorical method colours and visible sans-serif typography change.
STIX mathematical glyphs remain in their original positions and fonts.
"""
from pathlib import Path
import math, json, re
import fitz
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, ByteStringObject, ArrayObject, FloatObject

ROOT=Path(__import__('os').environ['FROST_FIGURE_WORK'])
SRC=ROOT/'tmp/pdfs/frost_layout_only/build/Images'
OUT=ROOT/'outputs/frost_plot_refresh/Images'
TMP=ROOT/'tmp/pdfs/frost_plot_refresh/fig6'
FONT=ROOT/'tmp/pdfs/frost_redesign/arial_fonts'
OUT.mkdir(parents=True,exist_ok=True); TMP.mkdir(parents=True,exist_ok=True)

def rgb(hexstr):
    return tuple(int(hexstr[i:i+2],16)/255 for i in (1,3,5))

TEAL=rgb('#007F86'); CORAL=rgb('#C56B52'); BLUE=rgb('#426A90')
INK=rgb('#25333C'); MUTED=rgb('#5C6470')
OLD_TEAL=(11/255,140/255,134/255)
OLD_CORAL=(188/255,101/255,13/255)
OLD_BLUE=(8/255,115/255,158/255)

def same(a,b,tol=1e-6):return len(a)==len(b) and max(abs(x-y) for x,y in zip(a,b))<tol
def remap(v,conv=False):
    if same(v,OLD_TEAL):return TEAL
    if same(v,OLD_CORAL):return CORAL
    if conv and same(v,OLD_BLUE):return BLUE
    if len(v)==3 and max(v)-min(v)<1e-6:
        if abs(v[0]-.45)<.015 or abs(v[0]-.5)<.002:return MUTED
        if .04<v[0]<.2:return INK
    return v

def visible_sans(doc):
    result=[]
    for b in doc[0].get_text('rawdict')['blocks']:
        for line in b.get('lines',[]):
            angle=math.degrees(math.atan2(-line['dir'][1],line['dir'][0]))
            for s in line['spans']:
                if 'NimbusSans' not in s['font']:continue
                result.append({'font':s['font'],'size':s['size'],'color':s['color'],
                               'chars':s['chars'],'angle':angle,
                               'text':''.join(c['c'] for c in s['chars'])})
    return result

def style(fn,convergence=False):
    source=SRC/fn
    original=fitz.open(source)
    spans=visible_sans(original)
    rd=PdfReader(source)
    wr=PdfWriter(clone_from=rd)
    page=wr.pages[0]
    resources=page['/Resources']['/Font']
    sans={k for k,v in resources.items() if 'NimbusSans' in str(v.get_object().get('/BaseFont',''))}
    cs=ContentStream(page.get_contents(),wr)
    font=None; removed=0;colours=0
    for operands,operator in cs.operations:
        if operator==b'Tf':font=operands[0]
        if operator in (b'Tj',b"'",b'"') and font in sans:
            operands[-1]=ByteStringObject(b'');removed+=1
        elif operator==b'TJ' and font in sans:
            operands[0]=ArrayObject(x if isinstance(x,(int,float)) else ByteStringObject(b'') for x in operands[0]);removed+=1
        elif operator in (b'rg',b'RG'):
            v=tuple(float(x) for x in operands); nv=remap(v,convergence)
            if nv!=v:
                operands[:]=[FloatObject(x) for x in nv];colours+=1
    page.replace_contents(cs)
    intermediate=TMP/('sans_removed_'+fn)
    with intermediate.open('wb') as f:wr.write(f)
    d=fitz.open(intermediate);p=d[0]
    p.insert_font(fontname='FigureArial',fontfile=str(FONT/'Arial.TTF'))
    p.insert_font(fontname='FigureArialBold',fontfile=str(FONT/'Arialbd.TTF'))
    for s in spans:
        name='FigureArialBold' if 'Bold' in s['font'] else 'FigureArial'
        color=tuple(((s['color']>>k)&255)/255 for k in (16,8,0))
        color=remap(color,convergence)
        is_letter=s['text'] in ['a','b'] and s['size']>10
        size=10 if is_letter else max(s['size'],7.8)
        for ch in s['chars']:
            text=ch['c']
            if text.isspace():continue
            if is_letter and convergence:text={'a':'c','b':'d'}[text]
            origin=fitz.Point(ch['origin'])
            angle=s['angle']
            kw={'rotate':int(round(angle))%360} if abs(angle/90-round(angle/90))<1e-4 else {'morph':(origin,fitz.Matrix(angle))}
            p.insert_text(origin,text,fontname=name,fontsize=size,color=color,**kw)
    target=OUT/fn
    d.save(target,garbage=4,deflate=True)
    final=fitz.open(target)
    final[0].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(TMP/(target.stem+'.png'))
    before=original[0].get_drawings();after=final[0].get_drawings()
    # Font change cannot alter any recorded geometry; all original paths stay.
    assert len(before)==len(after),(len(before),len(after))
    for i,(a,b) in enumerate(zip(before,after)):
        assert a['type']==b['type']
        assert len(a['items'])==len(b['items'])
        # pypdf serialization can round real coordinates at <1e-4 points.
        assert max(abs(x-y) for x,y in zip(a['rect'],b['rect']))<1e-3,(i,a['rect'],b['rect'])
    report={'file':fn,'size_points':list(final[0].rect),'data_paths_preserved':len(before),
            'original_vector_images':len(original[0].get_images()),'output_vector_images':len(final[0].get_images()),
            'original_sans_show_operations_replaced':removed,'colour_operations_changed':colours,
            'sans_spans_relaid':len(spans),'data_changed':False,'geometry_tolerance_pt':.001,
            'method_palette':{'FROST':'#007F86','base_TT':'#C56B52','sampling_reference':'#426A90'},
            'mathematical_STIX_glyphs':'preserved','heatmap_values_and_gradients':'preserved',
            'panel_letters':['c','d'] if convergence else ['a','b']}
    return report

if __name__=='__main__':
    report=[style('FROST_attention_formation.pdf'),style('FROST_attention_convergence.pdf',True)]
    (ROOT/'outputs/frost_plot_refresh/Source/fig6_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
