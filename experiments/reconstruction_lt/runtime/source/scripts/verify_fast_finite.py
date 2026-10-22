"""Compare accelerated integer kernels against the original bit simulator."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from restoration_model import load_teacher, SCWindowRestorer
from run_quality30 import coords
import stochastic_model as sm
import packed_kernels as pk
from attention_core import FrozenHI
import fast_finite as fast


def main():
    root=Path(sys.argv[1]).resolve()
    rng=np.random.default_rng(997711)
    records=[]
    def record(name, a, b, seconds=None):
        difference=float(np.max(np.abs(a-b)))
        assert np.array_equal(a,b),(name,difference)
        records.append(dict(check=name,max_absolute_difference=difference,seconds=seconds))
        print(records[-1],flush=True)
    for L in [64,256]:
        x=rng.normal(size=(2,5,7));w=rng.normal(size=(9,7));bias=rng.normal(size=9)
        record(f'linear_L{L}',pk.linear(x,w,bias,L,121),fast.linear(x,w,bias,L,121))
        q=rng.normal(size=(2,4,7));k=rng.normal(size=(2,4,7))
        record(f'qk_L{L}',pk.stochastic_qk(q,k,L,122),fast.qk(q,k,L,122))
        a=rng.dirichlet(np.ones(4),size=(2,4));v=rng.normal(size=(2,4,7))
        record(f'av_L{L}',pk.stochastic_av(a,v,L,123,center=True),fast.av(a,v,L,123,center=True))
        for shape in [(2,5,1),(2,5,7)]:
            p=rng.uniform(size=shape);table=np.array(json.loads((root/'source/original/checkpoints/scalar_tables.json').read_text())['cdf']['weights'])
            record(f'FSM_{shape}_L{L}',sm.gated(x,p,table,L,124),fast.gated(x,p,table,L,124))
        hi=FrozenHI(root/'checkpoints/operator_render_2018.pt');z=rng.normal(size=(2,4,16))
        record(f'TT_L{L}',hi.stream(z,L,125),fast.tt_stream(hi,z,L,125))
    fast.install()
    for image_id in [2018,43033,235098]:
        model,_=load_teacher(root/'checkpoints'/f'render_{image_id}.pt')
        sc=SCWindowRestorer(model,root/'checkpoints'/f'operator_render_{image_id}.pt')
        old=np.load(root/'runs'/f'reference_previous_{image_id}.npz')
        stream_seed=101+image_id*9001+11*17011
        start=time.perf_counter()
        pred=np.clip(sc.run(coords(),L=4096,seed=stream_seed,mode='full')[0],0,1).astype(np.float32)
        record(f'full_FROST_image{image_id}_L4096_seed101',old['frost'],pred,time.perf_counter()-start)
        base=SCWindowRestorer(model)
        start=time.perf_counter()
        pred=np.clip(base.run(coords(),L=4096,seed=stream_seed,mode='full')[0],0,1).astype(np.float32)
        record(f'full_SC_image{image_id}_L4096_seed101',old['SC_base'],pred,time.perf_counter()-start)
        record(f'W8A8_image{image_id}',old['w8a8'],np.clip(sc.run(coords(),mode='w8a8')[0],0,1).astype(np.float32))
    (root/'source_data/exact_kernel_verification.json').write_text(json.dumps(records,indent=2))
    print('VERIFIED exact finite-stream counts and full original reconstructions',flush=True)


if __name__=='__main__':main()
