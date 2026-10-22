"""Replace the two hardware figures and integrate the verified SC comparison."""
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parents[2]
SOURCE=WORK/'outputs/FROST_hw_stage1/FROST_stage1_references_fixed.tex'
text=SOURCE.read_text()
start=text.index('\\subsection{Hardware efficiency of 64-sample attention}')
end=text.index('\\section{Discussion}',start)
text=text[:start]+(ROOT/'source/hardware_section.tex').read_text()+'\n'+text[end:]
old='A separate 64-sample kernel uses 73.3\\% fewer combinational resources than a four-lane floating-point control on the same programmable device and 36.0\\% less estimated tile energy in a pre-layout standard-cell flow, without matched approximation error.'
new='A separate 64-sample kernel uses direct categorical value selection to reduce execution cycles by 33.1--67.5\\% and estimated tile energy by 45.8--83.7\\% relative to four implemented scalar stochastic controls in a common pre-layout standard-cell flow; approximation error is not matched.'
assert old in text;text=text.replace(old,new)
old='In the separate 64-sample hardware kernel, the optimized implementation uses 73.3\\% fewer combinational resources than four-lane FP32 on the evaluated FPGA and 36.0\\% less estimated tile energy in a common pre-layout standard-cell flow. The kernel comparison includes generators, parameter storage, buffering and conversions, making these savings attributable to the implemented computation rather than to arithmetic primitives considered in isolation.'
new='In the separate 64-sample hardware kernel, consuming categorical events directly reduces the mapped standard-cell area by 7.8--42.8\\% and estimated tile energy by 45.8--83.7\\% relative to the four implemented scalar stochastic controls. These are kernel-level operating points with different approximation errors. The comparison includes generators, parameter storage, buffering and conversions, linking the reported cost reductions to a complete score-to-output computation. On the evaluated FPGA, FROST eliminates the DSP blocks used by these controls, with a general-logic trade-off against SExp.'
assert old in text;text=text.replace(old,new)
old='These records are distinguished from the original 200-image evaluation\nand the hardware reports.'
new='These records are distinguished from the original 200-image evaluation\nand the hardware reports. The source data for\nFigs.~\\ref{fig:fused_hardware_stage3} and~\\ref{fig:fused_hardware_tradeoffs}\ninclude the complete scalar-SC comparisons, three placement-seed reports,\nthree stream-seed power records per design, exact quantized inputs and\nthe output-fidelity diagnostics. Gate waveforms, cell-library hashes and\ntool commands are retained in the associated reproducibility packages.'
assert old in text;text=text.replace(old,new)
# Update obsolete integration-stage comments, not author declarations.
text=text.replace('% No hardware experiment was rerun in this manuscript-integration stage.',
                  '% Hardware figures and claims use the verified stage-4/5 SC comparison.')
text=text.replace('% Stage 1: bibliography repair only; hardware results and figures are unchanged.',
                  '% Stage 6: updated hardware section, abstract, discussion and source-data description.')
text=text.replace('% New Fig. 10 compares five SC controls and the experimental FROST-LT variant.',
                  '% Fig. 10 is unchanged; Figs. 11 and 12 now show implemented SC hardware controls.')
text=text.replace('% All original numerical experiments and four tables are retained.',
                  '% Software experiments and figures are unchanged; the two hardware tables are updated.')
# Rebuild numeric bibliography in current citation order, then embed it.
a=text.index('% BEGIN EMBEDDED BIBLIOGRAPHY')
b=text.index('% END EMBEDDED BIBLIOGRAPHY',a)+len('% END EMBEDDED BIBLIOGRAPHY')
temp=text[:a]+'\\bibliography{FROST_reviewed_references}\n'+text[b:]
(ROOT/'build/sn-article.tex').write_text(temp)
(ROOT/'manuscript_with_old_bbl.tex').write_text(text)
shutil.copy2(ROOT.parent/'reference_build/FROST_reviewed_references.bib',ROOT/'build/FROST_reviewed_references.bib')
for name in ['FROST_hardware_stage3.pdf','FROST_hardware_tradeoffs.pdf']:
    shutil.copy2(WORK/'outputs/FROST_hw_stage6/Images'/name,ROOT/'build/Images'/name)
print('Integrated hardware results, synchronized claims, and prepared bibliography regeneration.')
