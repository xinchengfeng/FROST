from pathlib import Path
import subprocess
R=Path(__file__).resolve().parents[1]
for n in ["retime","l64","l64c"]:
 s=f"read_liberty -lib third_party/nangate45/NangateOpenCellLibrary_typical.lib; read_verilog reports/asic/{n}.v; hierarchy -top dut; select -module dut; rename -hide; write_verilog -noattr -noexpr reports/asic/flat_{n}.v"
 subprocess.run(["yosys","-Q","-p",s],cwd=R,check=True)
