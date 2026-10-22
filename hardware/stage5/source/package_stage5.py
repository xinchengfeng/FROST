"""Archive completed evidence, including exact activity, without build objects."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
OUT = BASE.parents[1]/'outputs/FROST_hw_stage5'
OUT.mkdir(parents=True, exist_ok=True)

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()

def main():
    summary = json.loads((ROOT/'results/stage5_summary.json').read_text())
    assert summary['total_new_gate_words_checked'] == 3840
    assert len(summary['standard_cell_power']) == 5
    assert all(r['clock_MHz']==8 for r in summary['standard_cell_power'])
    files = {}
    def add(p):
        if p.is_file() and '__pycache__' not in p.parts:
            files[str(p.relative_to(BASE))] = p
    for d in ['rtl','source','mem','inputs','third_party','results','reports','literature','prior_frost']:
        for p in (ROOT/d).rglob('*'):
            add(p)
    for n in ['README_STAGE4.txt','README_STAGE5.txt']:
        add(ROOT/n)
    for name in ['native_packages.json','tcl_package.json','stage3_installed_packages.json',
                 'stage3_tool_provenance.json','opensta_configure_command.json',
                 'OpenSTA_source.tar.gz','cudd_source.tar.gz']:
        add(BASE/'toolchain'/name)
    add(BASE/'toolchain/opensta_build/sta')
    manifest = {name:dict(bytes=p.stat().st_size, sha256=digest(p)) for name,p in sorted(files.items())}
    (ROOT/'PACKAGE_CONTENTS_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    add(ROOT/'PACKAGE_CONTENTS_SHA256.json')
    archive = OUT/'FROST_hardware_stage5_SC_power_verified_checkpoint.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,p in sorted(files.items()):
            z.write(p,'FROST_hw_stage5/'+name,
                    compress_type=zipfile.ZIP_STORED if p.suffix in ['.gz','.npz'] else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    result = dict(path=str(archive),bytes=archive.stat().st_size,sha256=digest(archive),files=len(files))
    (BASE/'stage5_checkpoint.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
