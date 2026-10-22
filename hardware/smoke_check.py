"""Compile independent integer models and check archived 16-query fixtures.

No EDA tools or large external record archives are required. This validates
functional model replay against archived outputs; it is not a new RTL/gate run.
"""
from pathlib import Path
import argparse, csv, json, shutil, subprocess, tempfile
import numpy as np

ROOT=Path(__file__).resolve().parent

def words(path):
    a=np.genfromtxt(path,delimiter=',',names=True)
    return a,np.stack([a[f'word{i}'] for i in range(16)],axis=-1).astype(np.uint32)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--json',type=Path,help='Optional output report; no report is written by default')
    args=p.parse_args()
    if not shutil.which('g++'):raise SystemExit('A C++17 compiler (g++) is required.')
    checks=[]
    with tempfile.TemporaryDirectory(prefix='frost-hardware-smoke-') as td:
        work=Path(td);sc=work/'golden_sc';frost=work/'fused_sampler'
        subprocess.run(['g++','-std=c++17','-O2',str(ROOT/'stage5/source/golden_sc.cpp'),'-o',str(sc)],check=True)
        subprocess.run(['g++','-std=c++17','-O2',str(ROOT/'stage3/source/fused_sampler.cpp'),'-o',str(frost)],check=True)
        config={'sexp16':(0,16),'sexp256':(0,256),'bern8':(1,0),'horner5':(2,0)}
        for design,(mode,burn) in config.items():
            for seed in [4004,5005,6006]:
                output=work/f'{design}_{seed}.csv'
                subprocess.run([str(sc),str(mode),str(burn),str(ROOT/'stage5/mem/scores.hex'),str(ROOT/'stage5/mem/values.hex'),str(ROOT/f'stage5/mem/seed{seed}.hex'),'16',str(output)],check=True)
                a,w=words(output);b,v=words(ROOT/f'smoke/expected_{design}_{seed}.csv')
                assert np.array_equal(w,v),f'{design}/{seed}: output mismatch'
                assert np.array_equal(a['cycles'],b['cycles']),f'{design}/{seed}: cycle mismatch'
                checks.append({'design':design,'stream_seed':seed,'rows':16,'words':int(w.size),'output_mismatches':0,'cycle_mismatches':0})
        for seed in [4004,5005,6006]:
            counts=work/f'frost_counts_{seed}.bin';sums=work/f'frost_sums_{seed}.bin';rng=work/f'frost_rng_{seed}.bin'
            subprocess.run([str(frost),str(ROOT/'stage3/inputs/scores.bin'),str(ROOT/'stage3/inputs/values.bin'),str(ROOT/'stage3/inputs/cdf_scaled.bin'),str(ROOT/f'stage3/inputs/seed{seed}.bin'),str(counts),str(sums),'16','2','64','16',str(rng)],check=True,stdout=subprocess.DEVNULL)
            decoded=(np.fromfile(sums,np.int32).reshape(-1,16).astype(np.int64)+32)//64
            _,expected=words(ROOT/f'smoke/expected_hot_reuse_{seed}.csv')
            assert np.array_equal(decoded.astype(np.int32),expected.view(np.int32)),f'FROST/{seed}: output mismatch'
            checks.append({'design':'hot_reuse','stream_seed':seed,'rows':16,'words':int(decoded.size),'output_mismatches':0,'cycle_mismatches':None,'note':'Independent functional model has no RTL timing scheduler; cycles not newly checked'})
    result={'status':'pass','scope':'C++ functional replay against archived fixtures, not a fresh RTL/gate simulation','checks':checks,'checked_output_words':sum(c['words'] for c in checks)}
    if args.json:args.json.parent.mkdir(parents=True,exist_ok=True);args.json.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
