"""Only measured/estimated report JSON is used as the numeric source."""
from pathlib import Path
import json, os
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = Path(os.environ.get('FROST_FIGURE_OUTPUT', ROOT/'Images')).resolve()
TMP = Path(os.environ.get('FROST_FIGURE_QA', ROOT/'qa')).resolve()
OUTDIR.mkdir(parents=True,exist_ok=True)
TMP.mkdir(parents=True,exist_ok=True)
SOURCE = ROOT/'data/stage5_summary.json'
DATA = SOURCE

def plot_data():
    s=json.loads(SOURCE.read_text())
    r=json.loads((ROOT/'data/stage4_summary.json').read_text())
    p={row['design']:row for row in s['standard_cell_power']}
    f={row['design']:row['fpga'] for row in r['SC_results']}
    f['hot_reuse']=r['FROST_reference']['fpga']
    order=['hot_reuse','sexp16','sexp256','bern8','horner5']
    area=[p[n]['mapped_cell_area_um2']/1000 for n in order]
    energy=[p[n]['energy_per_tile_uJ']['mean'] for n in order]
    return {
      'source':'Verified Stage4/5 report JSON; no recovery from PDF geometry',
      'names':['FROST','SExp (16)','SExp (256)','Bernstein-8','Horner-5'],
      'area':area,'area_labels':[f'{x:.2f}' for x in area],
      'resources':[f[n]['TRELLIS_COMB'] for n in order],
      'dsp':[f[n]['MULT18X18D'] for n in order],
      'energy':energy,'energy_labels':[f'{x:.3f}' for x in energy],
      'energy_error_bounds':[[p[n]['energy_per_tile_uJ']['mean']-p[n]['energy_per_tile_uJ']['sample_sd'],p[n]['energy_per_tile_uJ']['mean']+p[n]['energy_per_tile_uJ']['sample_sd']] for n in order],
      'scatter':[[p[n]['mapped_cell_area_um2']/1000,p[n]['tile_latency_us']] for n in order],
      'reduction_labels':['']+[f"{s['optimized_FROST_vs_SC'][n]['energy_reduction_percent']:.1f}%" for n in order[1:]],
    }

def register_fonts():
    supplied=os.environ.get('FROST_FIGURE_FONT_DIR')
    if supplied:
        regular=Path(supplied)/'Arial.TTF'
        bold=Path(supplied)/'Arialbd.TTF'
        if not regular.is_file() or not bold.is_file():
            raise FileNotFoundError('FROST_FIGURE_FONT_DIR must contain Arial.TTF and Arialbd.TTF')
    else:
        from matplotlib import font_manager
        regular=font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans'))
        bold=font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans',weight='bold'))
    # Stable internal aliases allow layout code to use either font family.
    pdfmetrics.registerFont(TTFont('Arial',str(regular)))
    pdfmetrics.registerFont(TTFont('Arial-Bold',str(bold)))
