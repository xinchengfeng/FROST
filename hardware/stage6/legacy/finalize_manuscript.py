"""Embed the regenerated 37-entry bibliography in the complete deliverable."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
work = root.parents[2]
main = root / 'build/sn-article.tex'
text = main.read_text()
bbl = (root / 'build/sn-article.bbl').read_text()
assert len(re.findall(r'\\bibitem(?:\[[\s\S]*?\])?\{([^}]+)\}', bbl)) == 37
marker = '\\bibliography{FROST_reviewed_references}'
assert text.count(marker) == 1
text = text.replace(marker, '% BEGIN EMBEDDED BIBLIOGRAPHY\n' + bbl + '\n% END EMBEDDED BIBLIOGRAPHY')
assert '\\bibliography{' not in text
target = work / 'outputs/FROST_hw_stage6/FROST_complete_SC_hardware.tex'
target.write_text(text)
main.write_text(text)
print(target)
