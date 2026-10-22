"""Save reproducible stage-3 evidence and exact raw activity, excluding build clutter."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent
WORK=BASE.parents[1]
OUT=WORK/'outputs/FROST_hw_stage3'
OUT.mkdir(parents=True,exist_ok=True)
files={}

def add(p):
    if p.is_file() and '__pycache__' not in p.parts:
        files[str(p.relative_to(BASE))]=p

for d in ['rtl','mem','inputs','third_party','results']:
    for p in (ROOT/d).rglob('*'):
        add(p)
for name in ['stage2_flow.py','stage2_verify.py','stage3_aggregate.py','stage3_exec.py',
             'stage3_flatten.py','stage3_power.py','stage3_route.py','restore_stage3_tools.py',
             'build_gate_simulators.py','sim_main.cpp','fused_sampler.cpp','golden_fp32.cpp',
             'package_stage3.py']:
    add(ROOT/'source'/name)
for name in ['l64c','perm','fx16','fp32_1','fp32_4']:
    for suffix in ['.json','.ys','.log','_stat.json','_manifest.json']:
        add(ROOT/f'reports/ecp5/{name}{suffix}')
for name in ['l64c','hot_reuse','fx16','fp32_1','fp32_4']:
    for suffix in ['.v','.ys','.log','_stat.json','_manifest.json','_gate_build.log']:
        add(ROOT/f'reports/asic/{name}{suffix}')
    for f in [f'flat_{name}.v',f'flat_{name}.ys',f'flat_{name}.log',f'gate_build_{name}_manifest.json']:
        add(ROOT/'reports/asic'/f)
for d in ['route','power','build']:
    for p in (ROOT/'reports'/d).rglob('*'):
        add(p)
for p in (ROOT/'baseline').glob('frost_L64_seed*.csv'):
    add(p)
for p in (ROOT/'previous_stage_evidence').rglob('*'):
    add(p)
add(ROOT/'README_STAGE3.txt')
for name in ['native_packages.json','tcl_package.json','stage3_installed_packages.json',
             'stage3_tool_provenance.json','opensta_configure_command.json',
             'opensta_configure.log','opensta_build.log','OpenSTA_source.tar.gz','cudd_source.tar.gz']:
    add(BASE/'toolchain'/name)
for p in (BASE/'toolchain').glob('*.deb'):
    add(p)
add(BASE/'toolchain/opensta_build/sta')
digest={name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for name,p in sorted(files.items())}
(ROOT/'PACKAGE_CONTENTS_SHA256.json').write_text(json.dumps(digest,indent=2)+'\n')
add(ROOT/'PACKAGE_CONTENTS_SHA256.json')
archive=OUT/'FROST_hardware_stage3_verified_checkpoint.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name,p in sorted(files.items()):
        # Preserve original gzip bytes and their recorded SHA256 values.
        z.write(p,'FROST_hw_stage3/'+name,
                compress_type=zipfile.ZIP_STORED if p.suffix in ['.gz','.deb','.npz'] else zipfile.ZIP_DEFLATED)
result={'path':str(archive),'bytes':archive.stat().st_size,
        'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(files)}
(BASE/'stage3_checkpoint.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
