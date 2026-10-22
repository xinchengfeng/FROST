#!/usr/bin/env python3
"""Reproduce publication vector figures from measured CSVs and author templates.

This command redraws figures; it does not rerun model training or hardware
synthesis. See docs/FIGURES.md for the distinction and the experiment commands.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parent


def selection(value):
    result=set()
    for part in value.split(','):
        lo,sep,hi=part.partition('-')
        result.update(range(int(lo),int(hi)+1) if sep else [int(lo)])
    if not result <= set(range(1,11)):
        raise argparse.ArgumentTypeError('Figure numbers must be between 1 and 10.')
    return result


def setup_fonts(work, font_dir, diagram_font_dir):
    from matplotlib import font_manager as fm
    styles=[('Arial.TTF','normal','normal'),('Arialbd.TTF','normal','bold'),
            ('Ariali.TTF','italic','normal'),('Arialbi.TTF','italic','bold')]
    target=work/'tmp/pdfs/frost_redesign/arial_fonts';target.mkdir(parents=True)
    exact=font_dir is not None
    for name,style,weight in styles:
        if exact:
            src=font_dir/name
            if not src.is_file():raise FileNotFoundError(f'Required external font is absent: {src}')
        else:
            src=Path(fm.findfont(fm.FontProperties(family='DejaVu Sans',style=style,weight=weight),fallback_to_default=False))
        shutil.copy2(src,target/name)
    diagrams=work/'tmp/pdfs/frost_redesign/source_sans_fonts';diagrams.mkdir()
    if diagram_font_dir:
        fonts=list(diagram_font_dir.glob('*.ttf'))
        if not fonts:raise FileNotFoundError(f'No .ttf files in {diagram_font_dir}')
        for src in fonts:shutil.copy2(src,diagrams/src.name)
        diagram_family='Source Sans 3'
    else:
        for name,_,_ in styles:shutil.copy2(target/name,diagrams/name)
        # Even if Arial is available, the schematic fallback is the portable
        # open-source font, with different font metrics explicitly recorded.
        for style,weight in [('normal','normal'),('normal','bold'),('italic','normal'),('italic','bold')]:
            src=Path(fm.findfont(fm.FontProperties(family='DejaVu Sans',style=style,weight=weight),fallback_to_default=False))
            shutil.copy2(src,diagrams/src.name)
        diagram_family='DejaVu Sans'
    return {'plot_font':'Arial' if exact else 'DejaVu Sans',
            'diagram_font':diagram_family,'exact_publication_fonts':bool(exact and diagram_font_dir),
            'font_notice':'Font substitution can change text metrics; retained template panels keep their embedded author fonts.'}


def stage(work):
    for folder in ['outputs/frost_plot_refresh/Source','outputs/frost_plot_refresh/Images',
        'tmp/pdfs/frost_plot_refresh','tmp/pdfs/frost_fig7_dumbbell','tmp/pdfs/frost_fig7_clean',
        'outputs/frost_fig7_dumbbell','outputs/frost_fig7_clean/Source',
        'outputs/frost_fig7_clean/Images','outputs/frost_fig7_overlap/Source/data',
        'tmp/pdfs/frost_layout_only/build/Images']:
        (work/folder).mkdir(parents=True,exist_ok=True)
    for name,dest in [('fig5','outputs/frost_plot_refresh/Source/fig5_data'),('fig8','outputs/frost_plot_refresh/Source/fig8_source_data')]:
        shutil.copytree(HERE/'data'/name,work/dest)
    for p in (HERE/'data/fig7').iterdir():
        for dest in ['outputs/frost_fig7_clean/Source/data','outputs/frost_fig7_overlap/Source/data']:
            (work/dest).mkdir(exist_ok=True,parents=True);shutil.copy2(p,work/dest/p.name)
    for p in (HERE/'templates').glob('*.pdf'):
        shutil.copy2(p,work/'tmp/pdfs/frost_layout_only/build/Images'/p.name)
    shutil.copy2(HERE/'templates/Fig_attention_fidelity.pdf',work/'outputs/frost_fig7_clean/Images/Fig_attention_fidelity.pdf')


def run_script(work, source, env, *args):
    # Scripts with adjacent data require this staging location. The source code
    # in figures/source remains readable and can also be run with the same env.
    dest=work/'outputs/frost_plot_refresh/Source'/source
    shutil.copy2(HERE/'source'/source,dest)
    result=subprocess.run([sys.executable,str(dest),*args],env=env,cwd=REPO,
                          text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    log=work/(source+'.log');log.write_text(result.stdout)
    if result.returncode:
        sys.stderr.write(result.stdout)
        raise RuntimeError(f'{source} failed ({result.returncode}); see {log}')
    print(f'  completed {source}',flush=True)


def compose_efficiency(work):
    import fitz
    import numpy as np
    source=fitz.open(HERE/'templates/Fig_attention_efficiency.pdf')
    panel=fitz.open(work/'tmp/pdfs/frost_fig7_clean/panel_c.pdf')
    old=source[0];cut=261.970455
    result=fitz.open();page=result.new_page(width=old.rect.width,height=old.rect.height)
    left=fitz.Rect(0,0,cut,old.rect.height);right=fitz.Rect(cut,0,old.rect.width,old.rect.height)
    page.show_pdf_page(left,panel,0)
    page.show_pdf_page(right,source,0,clip=right)
    dest=work/'outputs/frost_fig7_dumbbell/Fig_attention_efficiency.pdf'
    result.save(dest,garbage=4,deflate=True)
    check=fitz.open(dest)
    a=old.get_pixmap(matrix=fitz.Matrix(2,2),clip=right).samples
    b=check[0].get_pixmap(matrix=fitz.Matrix(2,2),clip=right).samples
    assert a==b,'Retained Fig7d template was changed.'
    assert not check[0].get_images(),'Unexpected rasterization in Fig7.'
    return {'file':dest.name,'panel_d_pixel_identical':True,'raster_images':0}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--figures',type=selection,default=selection('1-8'),help='Comma-separated numbers/ranges; default 1-8. Fig9/10 delegate to experiment/hardware packages.')
    p.add_argument('--output',type=Path,default=REPO/'build/figures')
    p.add_argument('--font-dir',type=Path,default=os.environ.get('FROST_FIGURE_FONT_DIR'),help='Optional external Arial TTF directory (four expected filenames).')
    p.add_argument('--diagram-font-dir',type=Path,default=os.environ.get('FROST_DIAGRAM_FONT_DIR'),help='Optional external Source Sans 3 TTF directory.')
    p.add_argument('--keep-work',action='store_true',help='Keep intermediate files at OUTPUT/work.')
    args=p.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    args.font_dir=Path(args.font_dir).resolve() if args.font_dir else None
    args.diagram_font_dir=Path(args.diagram_font_dir).resolve() if args.diagram_font_dir else None
    temporary=None
    if args.keep_work:
        work=output/'work'
        if work.exists():shutil.rmtree(work)
        work.mkdir()
    else:
        temporary=tempfile.TemporaryDirectory(prefix='frost_figures_');work=Path(temporary.name)
    stage(work)
    fonts=setup_fonts(work,args.font_dir,args.diagram_font_dir)
    env=dict(os.environ,FROST_FIGURE_WORK=str(work),FROST_DIAGRAM_FONT_FAMILY=fonts['diagram_font'],FROST_PLOT_FONT_FAMILY=fonts['plot_font'],MPLBACKEND='Agg')
    files=[];audits=[]
    diagrams={1:'overview',2:'representation',3:'operator',4:'block'}
    requested=[diagrams[n] for n in sorted(args.figures&diagrams.keys())]
    if requested:
        run_script(work,'draw_schematics.py',env,*requested)
        files.extend(work/'outputs/frost_figures_compact'/f'Fig_FROST_{n}.pdf' for n in requested)
    if 5 in args.figures:
        run_script(work,'draw_softmax.py',env)
        files.extend(work/'outputs/frost_plot_refresh/Images'/f'{n}.pdf' for n in ['Softmax_approximation','Softmax_bitstream'])
    if 6 in args.figures:
        run_script(work,'restyle_convergence.py',env)
        files.extend(work/'outputs/frost_plot_refresh/Images'/f'{n}.pdf' for n in ['FROST_attention_formation','FROST_attention_convergence'])
    if 7 in args.figures:
        run_script(work,'draw_attention_fidelity.py',env)
        run_script(work,'draw_attention_efficiency.py',env)
        audits.append(compose_efficiency(work))
        files.extend(work/'outputs/frost_fig7_dumbbell'/f'{n}.pdf' for n in ['Fig_attention_fidelity','Fig_attention_efficiency'])
    if 8 in args.figures:
        run_script(work,'restyle_retrieval.py',env)
        files.extend(work/'outputs/frost_plot_refresh/Images'/f'{n}.pdf' for n in ['Fig_transformer_performance','Fig_transformer_ablation'])
    if 9 in args.figures:
        folder=REPO/'experiments/reconstruction_scalar'
        script=folder/'source/make_figure.py'
        if not script.is_file():raise FileNotFoundError(f'Reconstruction plotting script is missing: {script}')
        # The reconstruction package validates the archived score JSON against
        # its restored target/reconstruction arrays before drawing.
        subprocess.run([sys.executable,str(script)],cwd=folder,env=dict(env,FROST_FIGURE_FONT_DIR=str(work/'tmp/pdfs/frost_redesign/arial_fonts')),check=True)
        files.append(folder/'figure/FROST_reconstruction_SC.pdf')
    if 10 in args.figures:
        script=REPO/'hardware/stage6/reproduce_figures.py'
        if not script.is_file():raise FileNotFoundError(f'Hardware wrapper is missing: {script}')
        hardware_output=work/'hardware_figures'
        subprocess.run([sys.executable,str(script),'--output-dir',str(hardware_output),'--qa-dir',str(work/'hardware_qa')],cwd=REPO,env=dict(env,FROST_FIGURE_FONT_DIR=str(work/'tmp/pdfs/frost_redesign/arial_fonts')),check=True)
        files.extend(hardware_output/n for n in ['FROST_hardware_stage3.pdf','FROST_hardware_tradeoffs.pdf'])
    import fitz
    checks=[]
    for src in files:
        dest=output/src.name;shutil.copy2(src,dest)
        with fitz.open(dest) as doc:
            assert len(doc)==1
            images=sum(len(page.get_images()) for page in doc)
            if dest.name!='FROST_reconstruction_SC.pdf':
                assert images==0,(dest,images)
            checks.append({'file':dest.name,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'page_size_pt':list(doc[0].rect),'raster_images':images,'text_characters':len(doc[0].get_text())})
    logdir=output/'source_audit';logdir.mkdir(exist_ok=True)
    for src in work.rglob('*validation.json'):
        # Preserve each source's built-in numerical/geometry checks.
        shutil.copy2(src,logdir/(src.parent.name+'_'+src.name))
    for src in work.glob('*.log'):shutil.copy2(src,logdir/src.name)
    report={'requested_figures':sorted(args.figures),'fonts':fonts,'vector_checks':checks,'composition_checks':audits,'reran_experiments':False}
    (output/'figure_build_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    if temporary:temporary.cleanup()
    print(f'Wrote {len(checks)} vector PDFs to {output}')
    print(json.dumps(fonts))

if __name__=='__main__':main()
