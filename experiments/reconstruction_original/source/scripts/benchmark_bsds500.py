"""Whole-partition image-specific FROST reconstruction, with resumable jobs.

The 200 official test-partition images are an evaluation image pool, not
held-out model inputs: every target participates in fitting its own model.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import argparse
import contextlib
import concurrent.futures
import json
import time
from pathlib import Path
import torch
from run_quality30 import train_render, calibrate


def fit_one(job):
    root_s, image_id = job
    root = Path(root_s)
    start = time.perf_counter()
    with (root/'logs'/f'fit_{image_id}.log').open('a', buffering=1) as log:
        with contextlib.redirect_stdout(log):
            if not (root/'checkpoints'/f'render_{image_id}.pt').exists():
                train_render(root/'data', root, image_id, 1500)
            ck = torch.load(root/'checkpoints'/f'render_{image_id}.pt', weights_only=True)
            assert ck['training_steps'] == 1500, ck
            assert ck['seed'] == 11 + image_id
            if not (root/'checkpoints'/f'operator_render_{image_id}.pt').exists():
                calibrate(root/'data', root, 'render', image_id)
    return dict(image_id=image_id, seconds=time.perf_counter()-start,
                training_steps=ck['training_steps'], best_step=ck['best_step'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=7)
    p.add_argument('--stage', choices=['fit'], default='fit')
    args = p.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root/'source_data/protocol.json').read_text())
    ids = [row['image_id'] for row in protocol['images']]
    assert len(ids) == len(set(ids)) == 200
    completed = []
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fit_one, (str(root), image_id)) for image_id in ids]
        for f in concurrent.futures.as_completed(futures):
            row = f.result()
            completed.append(row)
            print(json.dumps(dict(stage='fit', completed=len(completed), total=200,
                                  elapsed_seconds=round(time.perf_counter()-start, 1), **row)), flush=True)
            (root/'source_data/fit_progress.json').write_text(json.dumps(completed, indent=2))
    print('COMPLETE 200 fitted and calibrated image-specific models', flush=True)


if __name__ == '__main__':
    main()
