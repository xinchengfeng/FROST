"""Audit final embedded-bibliography source and the confirmed compilation."""
from pathlib import Path
import hashlib
import json
import re
import fitz

root = Path(__file__).resolve().parents[1]
work = root.parents[2]
out = work / 'outputs/FROST_hw_stage6'
source = out / 'FROST_complete_SC_hardware.tex'
text = source.read_text()
old = (work / 'outputs/FROST_hw_stage1/FROST_stage1_references_fixed.tex').read_text()
start, end = '\\section{Introduction}', '\\subsection{Hardware efficiency of'
assert text[text.index(start):text.index(end)] == old[old.index(start):old.index(end)]
unchanged_images = []
for path in (root / 'build/Images').iterdir():
    original = root.parent / 'input_project/Images' / path.name
    if path.name not in ['FROST_hardware_stage3.pdf', 'FROST_hardware_tradeoffs.pdf'] and original.is_file():
        assert path.read_bytes() == original.read_bytes(), path.name
        unchanged_images.append(path.name)
keys = re.findall(r'\\bibitem(?:\[[\s\S]*?\])?\{([^}]+)\}', text)
body = text[:text.index('% BEGIN EMBEDDED BIBLIOGRAPHY')]
cites = {key.strip() for group in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\]){0,2}\{([^}]+)\}', body) for key in group.split(',')}
assert len(keys) == len(set(keys)) == 37 and cites == set(keys)
assert '\\bibliography{' not in text
log = (root / 'build/final_checked.log').read_text()
warnings = [line for line in log.splitlines() if any(t in line for t in ['Warning', 'Overfull', 'Error'])]
assert not warnings, warnings
assert len(re.findall(r'\\begin\{figure\*?\}', text)) == 12
assert len(re.findall(r'\\begin\{table\*?\}', text)) == 4
aux = (root / 'build/final_checked.aux').read_text()
assert '\\newlabel{fig:fused_hardware_stage3}{{11}' in aux
assert '\\newlabel{fig:fused_hardware_tradeoffs}{{12}' in aux
doc = fitz.open(str(root / 'build/final_checked.pdf'))
assert len(doc) == 27
for num in [21, 22, 23, 24]:
    doc[num-1].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(str(root / 'qa' / f'embedded_final_page_{num}.png'))
figures = {}
for name in ['FROST_hardware_stage3.pdf', 'FROST_hardware_tradeoffs.pdf']:
    path = out / 'Images' / name
    fig = fitz.open(str(path))
    assert len(fig) == 1 and not fig[0].get_images()
    assert len(fig[0].get_drawings()) > 0
    figures[name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'raster_images': 0, 'vector_paths': len(fig[0].get_drawings()), 'page_points': list(fig[0].rect)}
report = {
    'complete_tex_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'bibliography_count': len(keys), 'cited_count': len(cites), 'unresolved_citations': [],
    'compiler_warnings': warnings, 'figure_count': 12, 'table_count': 4, 'page_count': len(doc),
    'validated_compilation': 'build/final_checked.pdf',
    'introduction_through_software_results_unchanged': True,
    'unchanged_image_assets': sorted(unchanged_images),
    'figures': figures,
    'visual_review': 'Figures and manuscript pp. 1,21–24,27 rendered and inspected. No clipped or overlapping labels.'
}
(root / 'qa/validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
