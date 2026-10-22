"""Package implemented SC controls, raw verification outputs and hardware reports."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent
WORK=BASE.parents[1]
OUT=WORK/'outputs/FROST_hw_stage4'
OUT.mkdir(parents=True,exist_ok=True)
summary=json.loads((ROOT/'results/stage4_summary.json').read_text())
assert len(summary['SC_results'])==4 and summary['checked_words']==626688
files={}
def add(p):
    if p.is_file() and '__pycache__' not in p.parts:
        files[str(p.relative_to(BASE))]=p
for d in ['rtl','source','mem','inputs','third_party','results','reports','literature','prior_frost']:
    for p in (ROOT/d).rglob('*'):
        add(p)
add(ROOT/'README_STAGE4.txt')
for name in ['native_packages.json','tcl_package.json','stage3_installed_packages.json',
             'stage3_tool_provenance.json','opensta_configure_command.json',
             'OpenSTA_source.tar.gz','cudd_source.tar.gz']:
    add(BASE/'toolchain'/name)
add(BASE/'toolchain/opensta_build/sta')
manifest={name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
          for name,p in sorted(files.items())}
(ROOT/'PACKAGE_CONTENTS_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
add(ROOT/'PACKAGE_CONTENTS_SHA256.json')
archive=OUT/'FROST_hardware_stage4_SC_verified_checkpoint.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name,p in sorted(files.items()):
        z.write(p,'FROST_hw_stage4/'+name,compress_type=zipfile.ZIP_STORED if p.suffix in ['.gz','.npz'] else zipfile.ZIP_DEFLATED)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
rec={'path':str(archive),'bytes':archive.stat().st_size,
     'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(files)}
(BASE/'stage4_checkpoint.json').write_text(json.dumps(rec,indent=2)+'\n')
print(json.dumps(rec,indent=2))
