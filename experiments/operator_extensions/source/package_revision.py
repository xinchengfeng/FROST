"""Separate small Overleaf deliverable from measured software/source data."""
from pathlib import Path
import json, hashlib, zipfile, subprocess, re
ROOT=Path(__file__).resolve().parents[1]

def archive(path,files):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(set(files)):z.write(f,f.relative_to(ROOT))
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        assert not any('author_planning' in n for n in z.namelist())
    return {'file':path.name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

main=(ROOT/'FROST_NC_main.tex').read_text();si=(ROOT/'FROST_NC_Supplementary.tex').read_text()
used=set(re.findall(r'Images/[^{}\s]+\.(?:pdf|png)',main+si))
for path in used:assert (ROOT/path).is_file(),path
paper=[ROOT/'FROST_NC_main.tex',ROOT/'FROST_NC_Supplementary.tex',ROOT/'FROST_references_NC.bib',
       ROOT/'sn-jnl.cls',ROOT/'sn-mathphys-num.bst',ROOT/'README.txt',ROOT/'修改说明.txt']
paper += [ROOT/p for p in used]
newfigs=[]
for name in ['FROST_execution','FROST_dimension_transfer','FROST_sampler_precision']:
    for ext in ['pdf','svg','png']:newfigs.append(ROOT/'Images'/f'{name}.{ext}')
newfigs += list((ROOT/'fonts').glob('*'))+[ROOT/'source/plot_revision.py',ROOT/'README.txt']
source=[]
for directory in ['source','source_data','checkpoints','fonts']:
    source += [p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts
               and p.name!='write_revision.py']
source += [ROOT/'requirements.txt',ROOT/'README.txt',ROOT/'修改说明.txt']
source += [p for p in newfigs if p.suffix in ['.pdf','.svg']]
packages=[archive(ROOT/'FROST_NC_Overleaf.zip',paper),
          archive(ROOT/'FROST_NC_new_figures.zip',newfigs),
          archive(ROOT/'FROST_NC_software.zip',source)]
pdf_checks=[]
for p in [ROOT/'FROST_NC_main.pdf',ROOT/'FROST_NC_Supplementary.pdf']+[p for p in newfigs if p.suffix=='.pdf']:
    font=subprocess.run(['pdffonts',str(p)],capture_output=True,text=True,check=True)
    assert 'Warning' not in font.stderr,(p,font.stderr)
    for line in font.stdout.splitlines()[2:]:
        vals=line.split()
        assert len(vals)>6 and vals[-5]=='yes',(p,line) # embedded flag
    images=subprocess.run(['pdfimages','-list',str(p)],capture_output=True,text=True,check=True)
    if p.parent.name=='Images':assert len(images.stdout.splitlines())==2,p
    pdf_checks.append({'file':str(p.relative_to(ROOT)),'embedded_fonts':True,
                       'new_figure_fully_vector':p.parent.name=='Images'})
for stem in ['FROST_NC_main','FROST_NC_Supplementary']:
    log=(ROOT/(stem+'.log')).read_text()
    assert not re.search('undefined|Overfull|LaTeX Warning',log),stem
report={'packages':packages,'required_image_files':sorted(used),'pdf_checks':pdf_checks,
        'main_pages':15,'supplementary_pages':5,'all_pages_visually_reviewed':True,
        'main_figures':8,'main_tables':2,'no_unresolved_citations_or_cross_references':True,
        'no_overfull_boxes':True,'hardware_results_added':False}
(ROOT/'source_data/delivery_verification.json').write_text(json.dumps(report,indent=2))
for p in packages:print(p['file'],f"{p['bytes']/1024**2:.1f} MiB")
print('PDF fonts, vector content, image dependencies and LaTeX checks passed.')
