FROST resource-focused manuscript revision

This revision uses unchanged stage-3 hardware data. No new synthesis, routing or power runs were performed.
Upload the two Images/*.pdf files into the Overleaf Images/ directory. Replace the hardware section with FROST_hardware_stage3_section.tex. Retain the four entries in FROST_hardware_tools.bib.

Main claim: categorical value selection eliminates DSP and BRAM instances in this mapped kernel, with lower COMB than both tested FP32 controls. FF is higher and FX16 has lower COMB. All resource classes remain visible.
Same-flow result: at L64 and 15 MHz in Nangate45, mapped area, estimated active power and cycle-derived latency are all below the four-lane FP32 implementation. This does not establish matched-error or whole-network superiority.

To regenerate figures: python source/make_figures.py (numpy and matplotlib required). Raw underlying aggregate data and computed reductions are included. The prior complete hardware reproducibility package remains the source for RTL and implementation reports.

The PDFs are true vector figures; SVG copies are editable.
