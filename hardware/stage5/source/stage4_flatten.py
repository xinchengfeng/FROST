"""Keep every mapped cell; only rename nets for waveform annotation."""
import subprocess
from stage2_flow import ROOT, ENV

(ROOT/'reports/gate').mkdir(parents=True,exist_ok=True)
# Also regenerate the unchanged FROST netlist when rebuilding from source.
for name in ['hot_reuse','sexp16','sexp256','bern8','horner5']:
    script = ('read_liberty -lib third_party/nangate45/NangateOpenCellLibrary_typical.lib; '
              f'read_verilog reports/asic/{name}.v; hierarchy -top dut; '
              'select -module dut; rename -hide; '
              f'write_verilog -noattr -noexpr reports/asic/flat_{name}.v')
    with (ROOT/f'reports/gate/flatten_{name}.log').open('w') as f:
        subprocess.run(['yosys','-Q','-p',script], cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT, check=True)
