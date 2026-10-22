"""Convert the bundled Nimbus Sans CFF outlines to real TrueType outlines.

This avoids a PDF font-subtype mismatch in Matplotlib's Type-42 embedding.
Family, names, metrics, character maps, and glyph appearance are preserved.
The original URW copyright notice accompanies these fonts.
"""
from pathlib import Path
from copy import deepcopy
from fontTools.ttLib import TTFont
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.cu2quPen import Cu2QuPen

ROOT=Path(__file__).resolve().parent.parent
for path in (ROOT/'fonts').glob('*.ttf'):
    original=TTFont(path)
    if original.sfntVersion!='OTTO':continue
    glyphset=original.getGlyphSet();order=original.getGlyphOrder();glyphs={}
    for name in order:
        target=TTGlyphPen(glyphset)
        pen=Cu2QuPen(target,max_err=.5,reverse_direction=True)
        glyphset[name].draw(pen);glyphs[name]=target.glyph()
    fb=FontBuilder(original['head'].unitsPerEm,isTTF=True)
    fb.setupGlyphOrder(order);fb.setupCharacterMap(original.getBestCmap())
    fb.setupGlyf(glyphs);fb.setupHorizontalMetrics(original['hmtx'].metrics)
    hh=original['hhea']
    fb.setupHorizontalHeader(ascent=hh.ascent,descent=hh.descent,lineGap=hh.lineGap)
    fb.font['name']=deepcopy(original['name']);fb.font['OS/2']=deepcopy(original['OS/2'])
    fb.setupPost(keepGlyphNames=True)
    for tag in ['GPOS','GSUB','kern']:
        if tag in original:fb.font[tag]=deepcopy(original[tag])
    fb.font['head'].created=original['head'].created
    fb.font['head'].modified=original['head'].modified
    fb.save(path)
    print('TrueType outlines prepared:',path.name)
