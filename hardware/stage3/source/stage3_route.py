"""Route every declared design/seed under the same ECP5 constraint."""
from pathlib import Path
import concurrent.futures
import json
import hashlib
import subprocess
import time
from stage2_flow import ROOT, ENV

OUT = ROOT/'reports/route'
OUT.mkdir(parents=True,exist_ok=True)

def run(job):
    name, seed = job
    stem = f'{name}_s{seed}'
    src = ROOT/f'reports/ecp5/{name}.json'
    cmd = ['nextpnr-ecp5', '--85k', '--package', 'CABGA381', '--speed', '6',
           '--json', str(src), '--freq', '8', '--seed', str(seed), '--lpf-allow-unconstrained',
           '--report', str(OUT/f'{stem}.json'), '--textcfg', str(OUT/f'{stem}.config')]
    t = time.monotonic()
    with (OUT/f'{stem}.log').open('w') as f:
        p = subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT)
    rec = dict(design=name, seed=seed, command=cmd, exit_code=p.returncode,
               seconds=time.monotonic()-t, netlist_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
               version=subprocess.check_output(['nextpnr-ecp5', '--version'], env=ENV, stderr=subprocess.STDOUT, text=True).strip())
    (OUT/f'{stem}_manifest.json').write_text(json.dumps(rec, indent=2)+'\n')
    print(json.dumps(rec), flush=True)
    if p.returncode:
        raise RuntimeError(stem)

if __name__ == '__main__':
    jobs = [(n,s) for n in ['l64c','perm','fx16','fp32_1','fp32_4'] for s in [1,2,3]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(run, jobs))
