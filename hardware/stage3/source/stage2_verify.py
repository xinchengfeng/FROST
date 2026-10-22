"""Compare public RTL outputs and measured cycles with a fresh integer model.

Also checks archived outputs, three independent stream initializations, and
full-range signed-16 corner vectors. All data and conclusions remain kernel-
level; this script does not measure classification accuracy or board power.
"""
from pathlib import Path
import argparse
import json
import subprocess
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def csv_words(path):
    z = np.genfromtxt(path, delimiter=',', names=True)
    w = np.stack([z[f'word{i}'] for i in range(16)], -1).astype(np.uint32).view(np.int32)
    return z, w


def main():
    p = argparse.ArgumentParser()
    p.add_argument('designs', nargs='+')
    args = p.parse_args()
    golden = ROOT / 'build/fused_sampler'
    subprocess.run(['g++', '-std=c++17', '-O2', str(ROOT/'source/fused_sampler.cpp'), '-o', str(golden)], check=True)
    result_file = ROOT/'results/verification.json'
    records = json.loads(result_file.read_text()) if result_file.exists() else []
    records = [r for r in records if r['design'] not in args.designs]
    for seed in [4004, 5005, 6006]:
        stem = ROOT / f'results/fresh_golden_{seed}'
        subprocess.run([str(golden), 'inputs/scores.bin', 'inputs/values.bin', 'inputs/cdf_scaled.bin',
                        f'inputs/seed{seed}.bin', str(stem)+'_counts.bin', str(stem)+'_sums.bin',
                        '3200', '2', '64', '16', str(stem)+'_seedout.bin'], cwd=ROOT, check=True)
        ref = (np.fromfile(str(stem)+'_sums.bin', np.int32).reshape(-1, 16).astype(np.int64)+32)//64
        archived, archived_words = csv_words(ROOT/f'baseline/frost_L64_seed{seed}.csv')
        for design in args.designs:
            path = ROOT/f'results/{design}_seed{seed}.csv'
            subprocess.run([str(ROOT/'build'/design/'Vdut'), 'mem/scores.hex', f'mem/seed{seed}.hex',
                            str(path), '64', '3200'], cwd=ROOT, check=True)
            z, words = csv_words(path)
            record = dict(design=design, seed=seed, queries=len(words), checked_words=words.size,
                          golden_mismatches=int(np.count_nonzero(words != ref)),
                          archived_mismatches=int(np.count_nonzero(words != archived_words)),
                          cycle_mismatches=int(np.count_nonzero(z['cycles'] != archived['cycles'])),
                          tile_cycles=int(z['cycles'][:16].sum()))
            assert not any(record[k] for k in ['golden_mismatches', 'archived_mismatches', 'cycle_mismatches']), record
            records.append(record)
            (ROOT/'results/verification.json').write_text(json.dumps(records, indent=2)+'\n')
            print(json.dumps(record), flush=True)

    corner = ROOT/'corner_work'
    shutil.copytree(ROOT/'mem', corner/'mem', dirs_exist_ok=True)
    rng = np.random.default_rng(73021)
    scores = rng.integers(-32768, 32768, (64, 16), dtype=np.int32)
    scores[:16] = 0
    scores[16:32] = np.arange(16)*1024-8192
    values = rng.integers(-32768, 32768, (4, 16, 16), dtype=np.int32)
    values[0] = -32768
    values[1] = 32767
    values[2] = np.where(np.indices((16, 16)).sum(0)%2 == 0, -32768, 32767)
    np.savez_compressed(ROOT/'inputs/stage2_corner_vectors.npz', scores=scores, values=values)
    for name, a in [('scores', scores), ('values', values)]:
        (corner/f'mem/{name}.hex').write_text(''.join(f'{int(v)&65535:04x}\n' for v in a.flat))
    corner_file = ROOT/'results/corner_verification.json'
    corners = json.loads(corner_file.read_text()) if corner_file.exists() else []
    corners = [r for r in corners if r['design'] not in ['l64c']+args.designs]
    for seed in [4004, 5005, 6006]:
        reference = None
        # Always use the unmodified l64c RTL as the corner reference.
        for design in ['l64c'] + [n for n in args.designs if n != 'l64c']:
            path = ROOT/f'results/corner_{design}_{seed}.csv'
            subprocess.run([str(ROOT/'build'/design/'Vdut'), 'mem/scores.hex', f'mem/seed{seed}.hex',
                            str(path), '64', '64'], cwd=corner, check=True)
            z, w = csv_words(path)
            if reference is None:
                reference = z, w
            assert np.array_equal(w, reference[1]) and np.array_equal(z['cycles'], reference[0]['cycles'])
            assert (w[:16] == -32768).all() and (w[16:32] == 32767).all()
            corners.append(dict(design=design, seed=seed, queries=64, checked_words=w.size,
                                output_mismatches=0, cycle_mismatches=0))
    (ROOT/'results/corner_verification.json').write_text(json.dumps(corners, indent=2)+'\n')
    print('PASS: all normal and corner comparisons', flush=True)


if __name__ == '__main__':
    main()
