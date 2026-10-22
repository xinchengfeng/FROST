"""Fix archived Matplotlib CFF payloads mislabeled as TrueType in PDFs.

No data, text positioning, figure elements or glyph IDs are changed. Cubic
outlines are converted to quadratic TrueType outlines with <=0.5 font-unit
error; source font containers are retained in the author-only working archive.
"""
from pathlib import Path
from copy import deepcopy
from io import BytesIO
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject, NumberObject
from fontTools.ttLib import TTFont
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.cu2quPen import Cu2QuPen
import json, shutil

ROOT=Path(__file__).resolve().parents[1]

def ttf_bytes(data):
    original=TTFont(BytesIO(data))
    if original.sfntVersion!='OTTO':
        flags=original['OS/2'].fsSelection if 'OS/2' in original else 0
        style=(original['head'].macStyle&~3)|(1 if flags&32 else 0)|(2 if flags&1 else 0)
        if style==original['head'].macStyle:return None
        original['head'].macStyle=style
        out=BytesIO();original.save(out);return out.getvalue()
    glyphset=original.getGlyphSet();order=original.getGlyphOrder();glyphs={}
    for name in order:
        target=TTGlyphPen(glyphset)
        pen=Cu2QuPen(target,max_err=.5,reverse_direction=True)
        glyphset[name].draw(pen);glyphs[name]=target.glyph()
    fb=FontBuilder(original['head'].unitsPerEm,isTTF=True)
    fb.setupGlyphOrder(order);fb.setupCharacterMap(original.getBestCmap() or {})
    fb.setupGlyf(glyphs);fb.setupHorizontalMetrics(original['hmtx'].metrics)
    hh=original['hhea'];fb.setupHorizontalHeader(ascent=hh.ascent,descent=hh.descent,lineGap=hh.lineGap)
    fb.font['name']=deepcopy(original['name'])
    if 'OS/2' in original:fb.font['OS/2']=deepcopy(original['OS/2'])
    else:fb.setupOS2()
    fb.setupPost(keepGlyphNames=True)
    for tag in ['GPOS','GSUB','kern']:
        if tag in original:fb.font[tag]=deepcopy(original[tag])
    flags=fb.font['OS/2'].fsSelection
    fb.font['head'].macStyle=(1 if flags&32 else 0)|(2 if flags&1 else 0)
    out=BytesIO();fb.save(out);return out.getvalue()

def repair(path):
    reader=PdfReader(path);writer=PdfWriter(clone_from=reader)
    visited=set();changed=[]
    def walk(resources):
        resources=resources.get_object()
        fonts=resources.get('/Font',{})
        for key,ref in fonts.items():
            font=ref.get_object();descendants=font.get('/DescendantFonts',[font])
            for ref2 in descendants:
                descfont=ref2.get_object();descriptor=descfont.get('/FontDescriptor')
                if not descriptor:continue
                descriptor=descriptor.get_object();stream=descriptor.get('/FontFile2')
                if stream is None:continue
                ident=(stream.idnum,stream.generation) if hasattr(stream,'idnum') else id(stream)
                if ident in visited:continue
                visited.add(ident)
                data=stream.get_object().get_data();new=ttf_bytes(data)
                if new is None:continue
                replacement=DecodedStreamObject();replacement.set_data(new)
                replacement[NameObject('/Length1')]=NumberObject(len(new))
                descriptor[NameObject('/FontFile2')]=writer._add_object(replacement)
                changed.append(str(font.get('/BaseFont',key)))
        for ref in resources.get('/XObject',{}).values():
            item=ref.get_object()
            if item.get('/Subtype')=='/Form' and '/Resources' in item:walk(item['/Resources'])
    for page in writer.pages:walk(page['/Resources'])
    if changed:
        backup=ROOT/'author_planning/font_source';backup.mkdir(parents=True,exist_ok=True)
        if not (backup/path.name).exists():shutil.copy2(path,backup/path.name)
        temp=path.with_suffix('.fontfixed.pdf');writer.write(temp);temp.replace(path)
    return {'file':path.name,'fonts_corrected':changed}

if __name__=='__main__':
    records=[repair(p) for p in (ROOT/'Images').glob('*.pdf')]
    p=ROOT/'source_data/pdf_font_repairs.json'
    history=json.loads(p.read_text()) if p.exists() else []
    p.write_text(json.dumps(history+records,indent=2))
    for r in records:
        if r['fonts_corrected']:print(r['file'],len(r['fonts_corrected']),flush=True)
