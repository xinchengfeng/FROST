"""Reproducible stage-2 RTL experiment. No original reports are overwritten.

Uses the same Ubuntu Yosys 0.33 / ABC packages as the archived experiment.
Local tool paths are an installation detail; all synthesis commands and source
hashes are recorded. Stage 2 reports synthesis only, not routed area or power.
"""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
# GitHub portability adaptation: use installed tools, or an explicitly
# restored recorded toolchain; do not override system Verilator with a missing tree.
import shutil
_TOOLCHAIN = Path(os.environ.get('FROST_TOOLCHAIN_ROOT', ROOT.parent / 'toolchain'))
NATIVE = _TOOLCHAIN / 'native/usr'
ENV = dict(os.environ)
if (NATIVE / 'bin').is_dir():
    ENV['PATH'] = str(NATIVE / 'bin') + ':' + ENV['PATH']
    ENV['LD_LIBRARY_PATH'] = str(NATIVE / 'lib/x86_64-linux-gnu') + ':' + ENV.get('LD_LIBRARY_PATH', '')
    if (NATIVE / 'share/verilator').is_dir():
        ENV['VERILATOR_ROOT'] = str(NATIVE / 'share/verilator')
ABC = os.environ.get('FROST_ABC') or shutil.which('yosys-abc', path=ENV['PATH']) or shutil.which('berkeley-abc', path=ENV['PATH']) or 'yosys-abc'
for _d in ['build','results','reports/build','reports/asic','reports/ecp5','reports/route','reports/gate','reports/power']:
    (ROOT / _d).mkdir(parents=True, exist_ok=True)
LIB = 'third_party/nangate45/NangateOpenCellLibrary_typical.lib'


def sources(name):
    if name in ['sexp16','sexp256','bern8','horner5']:
        return [f'rtl/top_{name}.v', 'rtl/sc_count_attention.v'], ['rtl']
    if name == 'hot_reuse':
        return ['rtl/top_hot_reuse.v', 'rtl/frost_frontend_reuse.v', 'rtl/frost_hot_reuse.v'], ['rtl']
    raise ValueError(name)


def run(name, mode):
    src, inc = sources(name)
    report_dir = ROOT / 'reports' / mode
    report_dir.mkdir(parents=True,exist_ok=True)
    record = {'design': name, 'mode': mode,
              'sources': {s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in src}}
    # Include all constant tables and the pipeline include in provenance.
    record['constants'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted((ROOT/'mem').glob('tt*.hex'))}
    record['coefficient_sha256'] = hashlib.sha256((ROOT/'rtl/bernstein_coeff_case.vh').read_bytes()).hexdigest()
    if mode == 'build':
        (ROOT/'build').mkdir(parents=True,exist_ok=True)
        cmd = ['verilator', '--cc', '--exe', '--build', '-j', '2', '--trace',
               '-Wno-fatal', '--prefix', 'Vdut', '--top-module', 'dut',
               '--Mdir', str(ROOT/'build'/name), '-CFLAGS', '-O2']
        cmd += ['-I'+str(ROOT/i) for i in inc] + [str(ROOT/s) for s in src]
        cmd += [str(ROOT/'source/sim_main.cpp')]
    else:
        script = 'read_verilog ' + ' '.join('-I'+i for i in inc) + ' ' + ' '.join(src) + '; '
        if mode == 'ecp5':
            script += f'synth_ecp5 -top dut -json reports/ecp5/{name}.json; '
            script += f'tee -o reports/ecp5/{name}_stat.json stat -json'
        elif mode == 'asic':
            script += 'hierarchy -check -top dut; synth -top dut -flatten -noabc; '
            script += f'dfflibmap -liberty {LIB}; abc -exe {ABC} -liberty {LIB}; clean; '
            script += f'write_verilog -noattr -noexpr reports/asic/{name}.v; '
            script += f'tee -o reports/asic/{name}_stat.json stat -liberty {LIB} -json'
            record['library_sha256'] = hashlib.sha256((ROOT/LIB).read_bytes()).hexdigest()
        else:
            raise ValueError(mode)
        script_path = report_dir / f'{name}.ys'
        script_path.write_text(script+'\n')
        cmd = ['yosys', '-Q', '-s', str(script_path)]
    record['command'] = cmd
    record['tool_versions'] = {}
    for tool, args in [('yosys', ['-V']), ('verilator', ['--version'])]:
        record['tool_versions'][tool] = subprocess.check_output([tool]+args, env=ENV, text=True).strip()
    t = time.monotonic()
    with (report_dir/f'{name}.log').open('w') as log:
        result = subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    record.update(exit_code=result.returncode, elapsed_seconds=time.monotonic()-t)
    (report_dir/f'{name}_manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k: record[k] for k in ['design', 'mode', 'exit_code', 'elapsed_seconds']}), flush=True)
    if result.returncode:
        raise RuntimeError(str(report_dir/f'{name}.log'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['build', 'ecp5', 'asic'])
    p.add_argument('designs', nargs='+')
    p.add_argument('--jobs', type=int, default=2)
    args = p.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(lambda n: run(n, args.mode), args.designs))
