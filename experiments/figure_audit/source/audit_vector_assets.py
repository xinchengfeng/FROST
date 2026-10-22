"""Check final PDF/SVG structure, font embedding and physical figure sizes."""
from pathlib import Path
import hashlib, json, subprocess
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
report=[]
for pdf in sorted((ROOT/'Images').glob('FROST_*.pdf')):
    reader=PdfReader(pdf);assert len(reader.pages)==1
    page=reader.pages[0]
    font_report=subprocess.run(['pdffonts',str(pdf)],check=True,text=True,capture_output=True).stdout
    lines=font_report.splitlines()[2:]
    assert lines and all('Type 3' not in l for l in lines)
    assert all(' yes ' in l for l in lines)
    # Count PDF image objects without decoding their pixel content. In
    # particular, a valid single-colour Indexed palette need not contain
    # an unused second colour (some image decoders incorrectly require it).
    images=[ref for ref in page['/Resources'].get('/XObject',{}).values()
      if ref.get_object().get('/Subtype')=='/Image']
    photographic=(pdf.stem=='FROST_reconstruction')
    if not photographic:assert len(images)==0,pdf.name
    svg=pdf.with_suffix('.svg');text=svg.read_text()
    if not photographic:assert '<image' not in text
    assert '<text' in text
    width=float(page.mediabox.width)*25.4/72;height=float(page.mediabox.height)*25.4/72
    assert abs(width-180)<1e-5
    report.append(dict(file=pdf.name,width_mm=width,height_mm=height,
      embedded_font_report=font_report,embedded_pixel_images=len(images),
      scientific_charts_and_annotations_vector=True,
      pure_vector=(len(images)==0),svg_text_editable=True,
      raster_note='Original 64x64 reference/reconstruction crops, 16x16 ROIs and pixel error maps; text, boxes and colour bar remain vector.' if photographic else '',
      sha256=hashlib.sha256(pdf.read_bytes()).hexdigest()))
assert len(report)==7
(ROOT/'source_data/vector_asset_audit.json').write_text(json.dumps(dict(all_passed=True,figures=report),indent=2)+'\n')
print('Verified 7 PDFs: embedded non-Type-3 fonts; 6 pure-vector charts; photographic gallery has vector annotations.')
