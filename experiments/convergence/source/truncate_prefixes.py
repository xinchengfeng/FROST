"""Retain measured nested prefixes up to 4096 and recompute all metrics.

No trajectory is resampled or interpolated. The source arrays contain actual
categorical events from the original frozen-model experiment.
"""
from pathlib import Path
import hashlib, json
import numpy as np
from run_convergence import LENGTHS, SEEDS, summarize

ROOT = Path(__file__).resolve().parent.parent
data = ROOT / 'source_data'
protocol = json.loads((data / 'protocol.json').read_text())
old_lengths = protocol['lengths']
indices = [old_lengths.index(int(L)) for L in LENGTHS]
reference = np.load(data / 'inputs.npz')['reference']
rows = []
hashes = {}
for method in ['SC_base', 'FROST']:
    for seed in SEEDS:
        path = data / f'trace_{method}_seed{seed}.npz'
        with np.load(path) as saved:
            a = {key: saved[key].copy() for key in saved.files}
        for key in ['probabilities', 'row_tv', 'sampling_row_tv']:
            a[key] = a[key][indices]
        a['events_sorted'] = a['events_sorted'][:int(LENGTHS[-1])]
        assert a['events_sorted'].shape == (4096, 128)
        for li, length in enumerate(LENGTHS):
            p = a['probabilities'][li]
            analytic = a['analytic']
            tv = .5 * np.abs(p - reference).sum(-1)
            sampling = .5 * np.abs(p - analytic).sum(-1)
            assert np.array_equal(tv, a['row_tv'][li])
            assert np.array_equal(sampling, a['sampling_row_tv'][li])
            rows.append(dict(method=method, stream_seed=seed, L=int(length),
                mean_row_tv=float(tv.mean()),
                probability_rmse=float(np.sqrt(np.mean((p-reference)**2))),
                sampling_mean_row_tv=float(sampling.mean()),
                sampling_rmse=float(np.sqrt(np.mean((p-analytic)**2))),
                analytic_mean_row_tv=float((.5*np.abs(analytic-reference).sum(-1)).mean()),
                analytic_rmse=float(np.sqrt(np.mean((analytic-reference)**2))),
                row_sum_error=float(np.max(np.abs(p.sum(-1)-1))),
                min_probability=float(p.min())))
        hashes[path.name] = hashlib.sha256(a['events_sorted'].tobytes()).hexdigest()
        np.savez_compressed(path, **a)
protocol['lengths'] = LENGTHS.tolist()
protocol['prefixes'] = ('nested prefixes of one 4096-cycle trajectory for every '
    'method/seed; 16-cycle emission chunks')
protocol['figure_revision'] = 'two separate two-panel vector figures; maximum sampling length 4096'
(data / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
summarize(rows)
(data / 'prefix_revision.json').write_text(json.dumps(dict(
    maximum_sampling_length=4096, unchanged_recorded_event_prefixes=True,
    per_method_seed_event_sha256=hashes, metric_sets_recomputed=len(rows)), indent=2) + '\n')
print('Retained 16 measured 4096-cycle traces; recomputed', len(rows), 'metric sets.')
