"""Rename internal names only to make Verilator tracing compact; ports unchanged."""
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
for name in ['frost','fx16','fp32_1','fp32_4']:
 script=f'read_liberty -lib third_party/nangate45/NangateOpenCellLibrary_typical.lib; read_verilog reports/asic/{name}.v; hierarchy -top dut; select -module dut; rename -hide; write_verilog -noattr -noexpr reports/asic/flat_{name}.v'
 subprocess.run(['yosys','-Q','-p',script],cwd=ROOT,check=True)
