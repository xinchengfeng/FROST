"""Keep every mapped cell; only rename nets for waveform annotation."""
import subprocess
from stage2_flow import ROOT, ENV

for name in ['l64c','hot_reuse','fx16','fp32_1','fp32_4']:
    script = ('read_liberty -lib third_party/nangate45/NangateOpenCellLibrary_typical.lib; '
              f'read_verilog reports/asic/{name}.v; hierarchy -top dut; '
              'select -module dut; rename -hide; '
              f'write_verilog -noattr -noexpr reports/asic/flat_{name}.v')
    with (ROOT/f'reports/gate/flatten_{name}.log').open('w') as f:
        subprocess.run(['yosys','-Q','-p',script], cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT, check=True)
