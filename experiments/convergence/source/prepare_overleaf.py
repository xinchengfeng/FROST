"""Recover included figure assets from the USER-SUPPLIED manuscript PDF.

PDF figures are recovered as their original Form XObjects, retaining all
vector graphics and fonts. Embedded PNGs are losslessly decoded. The
uploaded manuscript text is then compiled against these exact assets.
"""
from pathlib import Path
import argparse,json,re,shutil
import fitz
from pypdf import PdfReader,PdfWriter
from pypdf.generic import NameObject,DictionaryObject,DecodedStreamObject

ROOT=Path(__file__).resolve().parent.parent

def main(source_pdf,class_dir):
    project=ROOT/'overleaf';images=project/'Images';images.mkdir(parents=True,exist_ok=True)
    reader=PdfReader(source_pdf);doc=fitz.open(source_pdf);done={}
    for pi,page in enumerate(reader.pages):
        resources=page.get('/Resources',{}).get('/XObject',{})
        for name,ref in resources.items():
            form=ref.get_object()
            original_name=form.get('/PTEX.FileName')
            if form.get('/Subtype')=='/Form' and original_name and '/Images/' in original_name:
                file_name=Path(str(original_name)).name
                if file_name in done:continue
                bbox=[float(v) for v in form['/BBox']]
                w=PdfWriter();out=w.add_blank_page(width=bbox[2]-bbox[0],height=bbox[3]-bbox[1])
                copied=ref.clone(w)
                out[NameObject('/Resources')]=DictionaryObject({NameObject('/XObject'):
                    DictionaryObject({NameObject('/OriginalFigure'):copied})})
                content=DecodedStreamObject()
                content.set_data(f'q 1 0 0 1 {-bbox[0]} {-bbox[1]} cm /OriginalFigure Do Q\n'.encode())
                out[NameObject('/Contents')]=w._add_object(content)
                with (images/file_name).open('wb') as f:w.write(f)
                done[file_name]=dict(source_page=pi+1,type='original vector Form XObject',bbox=bbox)
    # The manuscript has two PNG softmax panels; the include order identifies
    # them unambiguously within the page containing that figure caption.
    for pi,page in enumerate(reader.pages):
        if 'Softmax approximation and finite-stream accuracy' not in page.extract_text():continue
        items=[(name,ref) for name,ref in page.get('/Resources',{}).get('/XObject',{}).items()
            if ref.get_object().get('/Subtype')=='/Image']
        assert len(items)==2,items
        for (name,ref),file_name in zip(items,['Softmax_approximation.png','Softmax_bitstream.png']):
            info=doc.extract_image(ref.idnum);assert info['ext']=='png',info['ext']
            (images/file_name).write_bytes(info['image'])
            done[file_name]=dict(source_page=pi+1,type='losslessly decoded embedded PNG',width=info['width'],height=info['height'])
    for name in ['FROST_attention_formation.pdf', 'FROST_attention_convergence.pdf']:
        shutil.copy2(ROOT/name, images/name)
    shutil.copy2(ROOT/'FROST_with_convergence.tex',project/'FROST_with_convergence.tex')
    shutil.copy2(ROOT/'FROST_references.bib',project/'FROST_references.bib')
    class_dir=Path(class_dir)
    for name in ['sn-jnl.cls','sn-mathphys-num.bst']:shutil.copy2(class_dir/name,project/name)
    for name in ['sn-jnl.cls','sn-mathphys-num.bst']:shutil.copy2(class_dir/name,ROOT/name)
    text=(ROOT/'FROST_with_convergence.tex').read_text()
    required=set(re.findall(r'Images/[^\s{}]+\.(?:pdf|png)',text))
    missing=[s for s in required if not (project/s).exists()];assert not missing,missing
    (ROOT/'source_data/figure_asset_provenance.json').write_text(json.dumps(done,indent=2)+'\n')
    print('Recovered',len(done),'original figure assets; all',len(required),'required files present')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('original_pdf');p.add_argument('class_directory')
    a=p.parse_args();main(a.original_pdf,a.class_directory)
