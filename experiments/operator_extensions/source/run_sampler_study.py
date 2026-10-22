"""CDF precision and latent-stage RNG independence on frozen CIFAR rows.

Core contraction is a diagnostic, not a hardware implementation. Independent
draws reproduce its conditional categorical law. A deliberately shared uniform
is a negative control. No image labels, weights or operator parameters are fit.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import csv, json, time
import numpy as np
from attention_core import FrozenHI

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = [64, 256, 1024, 4096]
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808]


def contract(cores, states):
    h = np.ones((len(states), 1))
    for j, c in enumerate(cores):
        h = np.einsum('bi,bij->bj', h, c[states[:, j]], optimize=True)
    return h


def stationary(cores, x, order):
    s = np.arange(16)
    pi = x[..., None]**s*(1-x[..., None])**(15-s)
    pi /= pi.sum(-1, keepdims=True)
    h = np.ones((len(x), 1))
    for j, c in enumerate(cores):
        h = np.einsum('bi,bs,sik->bk', h, pi[:, j], c, optimize=True)
    out = np.empty_like(h)
    np.put_along_axis(out, order, h, -1)
    return out


def shared_exact(cdfs, state):
    # A common U creates a piecewise-constant deterministic path. Partition at
    # every potentially selected threshold, integrate interval widths exactly.
    cuts = [np.array([0., 1.])]
    for j, c in enumerate(cdfs):
        cuts.append(c[state[j]].ravel())
    edges = np.unique(np.clip(np.concatenate(cuts), 0, 1))
    u = (edges[1:]+edges[:-1])/2
    width = np.diff(edges)
    bond = np.zeros(len(u), dtype=np.int64)
    for j, c in enumerate(cdfs):
        bond = (c[state[j], bond] <= u[:, None]).sum(-1)
    return np.bincount(bond, weights=width, minlength=cdfs[-1].shape[-1])


def metric(p, q):
    return {'mean_row_tv': float(.5*np.abs(p-q).sum(-1).mean()),
            'probability_rmse': float(np.sqrt(np.mean((p-q)**2)))}


def main():
    start = time.time()
    hi = FrozenHI(ROOT/'checkpoints/calibrated_cifar10_11.pt')
    archive = np.load(ROOT/'source_data/prior_inputs.npz')
    print('Available fixed inputs:', archive.files, flush=True)
    z = archive['scores'] if 'scores' in archive.files else archive['selected_scores']
    z = z.reshape(-1, 16)
    if len(z) != 128:
        raise ValueError(f'Expected the archived 128 rows, got {z.shape}')
    x, order = hi.encode(z)
    cores = [c/c.sum(-1, keepdims=True) for c in hi.cores]
    cdfs = [np.cumsum(c, -1) for c in cores]
    for c in cdfs:
        c[..., -1] = 1
    target = np.exp(z-z.max(-1, keepdims=True)); target /= target.sum(-1, keepdims=True)
    floating = stationary(cores, x, order)
    states = np.random.default_rng(620999).integers(0, 16, size=(512, 16))
    conditional = contract(cores, states)
    shared = np.array([shared_exact(cdfs, s) for s in states])
    methods = {'float_independent': {'cdfs': cdfs, 'cores': cores, 'bits': None},
               'float_shared': {'cdfs': cdfs, 'cores': cores, 'bits': None}}
    precision = []
    conditional_outputs = {'states': states, 'floating': conditional, 'shared': shared}
    for b in [8, 12, 16]:
        M = 2**(b-1)
        qcdf = [np.rint(c*M).astype(np.int64) for c in cdfs]
        for c in qcdf:
            c[..., -1] = M
            if np.any(np.diff(c, axis=-1) < 0):
                raise AssertionError('Nonmonotone CDF')
        qcores = [np.diff(np.concatenate([np.zeros((*c.shape[:-1], 1), dtype=np.int64), c], -1), axis=-1)/M for c in qcdf]
        qcond = contract(qcores, states)
        qanalytic = stationary(qcores, x, order)
        delta = [float(.5*np.abs(c-q).sum(-1).max()) for c, q in zip(cores, qcores)]
        tv = .5*np.abs(qcond-conditional).sum(-1)
        precision.append({'word_bits': b, 'fraction_bits': b-1,
            'conditional_tv_mean': float(tv.mean()), 'conditional_tv_max': float(tv.max()),
            'sum_core_tv_bound': float(sum(delta)), 'core_maximum_tv': delta,
            'stationary_tv_to_float': metric(qanalytic, floating)['mean_row_tv'],
            'stationary_tv_to_softmax': metric(qanalytic, target)['mean_row_tv'],
            'parameters': sum(c.size for c in cores),
            'packed_threshold_payload_kib': sum(c.size for c in cores)*b/(8*1024)})
        methods[f'cdf{b}_independent'] = {'cdfs': qcdf, 'cores': qcores, 'bits': b}
        conditional_outputs[f'cdf{b}'] = qcond
    conditional_outputs['shared_row_tv'] = .5*np.abs(shared-conditional).sum(-1)
    np.savez_compressed(ROOT/'source_data/sampler_conditional_outputs.npz', **conditional_outputs)
    rows = []
    for seed in SEEDS:
        drive = np.random.default_rng(seed)
        state = np.full(x.shape, 7, dtype=np.int16)
        for t in range(256):
            state = np.clip(state+2*(drive.random(x.shape)<x)-1, 0, 15)
        counts = {m: np.zeros_like(x, dtype=np.int64) for m in methods}
        traces = []
        events = {m: [] for m in methods}
        emission = np.random.default_rng(seed+900019)
        for a in range(0, 4096, 16):
            trace = []
            for t in range(16):
                state = np.clip(state+2*(drive.random(x.shape)<x)-1, 0, 15)
                trace.append(state.copy())
            trace = np.array(trace)
            if seed == 101:
                traces.append(trace.astype(np.uint8))
            ss = trace.reshape(-1, 16)
            # The float and quantized controls share underlying independent U's;
            # only CDF rounding and threshold resolution differ.
            uniforms = emission.random((16, len(ss)))
            for method, setting in methods.items():
                bond = np.zeros(len(ss), dtype=np.int64)
                for j, c in enumerate(setting['cdfs']):
                    u = uniforms[0] if method == 'float_shared' else uniforms[j]
                    if setting['bits'] is not None:
                        u = np.floor(u*2**(setting['bits']-1)).astype(np.int64)
                    bond = (c[ss[:, j], bond] <= u[:, None]).sum(-1)
                ev = bond.reshape(16, len(x))
                events[method].append(ev.astype(np.uint8))
                for t in range(16):
                    counts[method][np.arange(len(x)), ev[t]] += 1
                L = a+16
                if L in LENGTHS:
                    probability = np.empty_like(x)
                    np.put_along_axis(probability, order, counts[method]/L, -1)
                    own = stationary(setting['cores'], x, order)
                    rows.append({'seed': seed, 'L': L, 'method': method,
                        **metric(probability, target),
                        'tv_to_float_stationary': metric(probability, floating)['mean_row_tv'],
                        'tv_to_own_stationary': metric(probability, own)['mean_row_tv'],
                        'maximum_mass_error': float(np.abs(probability.sum(-1)-1).max())})
        payload = {m: np.concatenate(ev) for m, ev in events.items()}
        if seed == 101:
            payload['fsm_states_sorted'] = np.concatenate(traces)
        np.savez_compressed(ROOT/'source_data'/f'sampler_events_seed{seed}.npz', **payload)
        print(f'Sampler stream seed {seed} finished.', flush=True)
    with (ROOT/'source_data/sampler_stream_metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    groups = []
    for method in methods:
        for L in LENGTHS:
            r = [v for v in rows if v['method']==method and v['L']==L]
            group = {'method': method, 'L': L}
            for k in ['mean_row_tv', 'probability_rmse', 'tv_to_float_stationary', 'tv_to_own_stationary']:
                group[k+'_mean'] = float(np.mean([v[k] for v in r]))
                group[k+'_stream_sd'] = float(np.std([v[k] for v in r], ddof=1))
            groups.append(group)
    shared_tv = conditional_outputs['shared_row_tv']
    summary = {'precision': precision, 'groups': groups,
        'floating_stationary': metric(floating, target),
        'shared_conditional_tv_mean': float(shared_tv.mean()),
        'shared_conditional_tv_median': float(np.median(shared_tv)),
        'shared_conditional_tv_max': float(shared_tv.max()),
        'maximum_stream_mass_error': max(v['maximum_mass_error'] for v in rows),
        'compiled_normalization_difference': metric(floating, hi.analytic(z)),
        'seconds': time.time()-start,
        'warning': 'Shared-stage control violates conditional independence; its comparison with the independent stationary law is a bias diagnostic, not its own stationary error.'}
    (ROOT/'source_data/sampler_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ['groups', 'precision']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
