"""Check public RTL outputs/cycles against an independent SC functional model."""
from pathlib import Path
import argparse
import json
import subprocess
import numpy as np
from stage2_flow import ROOT, ENV

CONFIG={'sexp16':(0,16),'sexp256':(0,256),'bern8':(1,0),'horner5':(2,0)}

def read(path):
    a=np.genfromtxt(path,delimiter=',',names=True)
    w=np.stack([a[f'word{i}'] for i in range(16)],-1).astype(np.uint32).view(np.int32)
    return a,w

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('designs',nargs='+',choices=list(CONFIG))
    parser.add_argument('--rows',type=int,default=3200)
    args=parser.parse_args()
    golden=ROOT/'build/golden_sc'
    subprocess.run(['g++','-std=c++17','-O2',str(ROOT/'source/golden_sc.cpp'),'-o',str(golden)],check=True)
    results=[]
    corner=ROOT/'corner_work'
    (corner/'mem').mkdir(parents=True,exist_ok=True)
    z=np.load(ROOT/'inputs/stage2_corner_vectors.npz')
    # Extend original corners to equal rows, exact extrema, ties and clipped gaps.
    scores=z['scores'].copy();scores[32:48]=np.tile([-32768,32767],8)
    for name,a in [('scores',scores),('values',z['values'])]:
        (corner/f'mem/{name}.hex').write_text(''.join(f'{int(x)&65535:04x}\n' for x in a.flat))
    for name in args.designs:
        mode,burn=CONFIG[name]
        for seed in [4004,5005,6006]:
            for label,work,rows in [('evaluation',ROOT,args.rows),('corners',corner,64)]:
                stem=f'{name}_{label}_seed{seed}'
                ref=ROOT/f'results/{stem}_golden.csv'
                out=ROOT/f'results/{stem}_rtl.csv'
                seedfile=ROOT/f'mem/seed{seed}.hex'
                subprocess.run([str(golden),str(mode),str(burn),str(work/'mem/scores.hex'),
                                str(work/'mem/values.hex'),str(seedfile),str(rows),str(ref)],check=True)
                with (ROOT/f'reports/build/{stem}_sim.log').open('w') as f:
                    subprocess.run([str(ROOT/'build'/name/'Vdut'),str(work/'mem/scores.hex'),str(seedfile),
                                    str(out),'64',str(rows)],cwd=work,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
                a,w=read(out);b,v=read(ref)
                record=dict(design=name,corpus=label,seed=seed,rows=rows,checked_words=w.size,
                            output_mismatches=int(np.count_nonzero(w!=v)),
                            cycle_mismatches=int(np.count_nonzero(a['cycles']!=b['cycles'])),
                            tile_cycles=int(a['cycles'][:16].sum()),
                            denominator_min=int(b['denominator'].min()),denominator_max=int(b['denominator'].max()))
                assert record['output_mismatches']==record['cycle_mismatches']==0,record
                if label=='corners':
                    assert (w[:16]==-32768).all() and (w[16:32]==32767).all()
                results.append(record)
                (ROOT/'results/rtl_verification.json').write_text(json.dumps(results,indent=2)+'\n')
                print(json.dumps(record),flush=True)
    print('PASS: exact output and cycle checks',flush=True)

if __name__=='__main__':main()
