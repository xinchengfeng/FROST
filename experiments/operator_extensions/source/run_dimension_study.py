"""Fixed-budget, held-out operator adaptation at n=16,32,64 (CPU).

This is a software study, not a device synthesis. All dimensions start from
the same archived n=16 operator, resized by core-position and output-bin
interpolation. Three separately optimized fits use different perturbations,
training data and minibatch schedules. Test rows are shared across fits.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from pathlib import Path
import copy, csv, json, math, time, argparse, concurrent.futures
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class Operator(nn.Module):
    def __init__(self, cores, bound):
        super().__init__()
        self.bound = float(bound)
        self.cs = nn.ParameterList([nn.Parameter(c.clone()) for c in cores])

    def forward(self, z):
        z, order = z.sort(-1, descending=True)
        z = z - z.mean(-1, keepdim=True)
        x = ((z + self.bound) / (2 * self.bound)).clamp(0, 1)
        s = torch.arange(16, dtype=x.dtype)
        pi = x[..., None] ** s * (1 - x[..., None]) ** (15 - s)
        pi = pi / pi.sum(-1, keepdim=True)
        h = torch.ones(len(z), 1, dtype=z.dtype)
        for j, c in enumerate(self.cs):
            v = torch.einsum('bs,asr->bar', pi[:, j], c.softmax(-1))
            h = torch.bmm(h[:, None], v).squeeze(1)
        return torch.zeros_like(h).scatter(1, order, h)


def resize_initialization(base, n, seed):
    # Rank positions in the original chain are mapped by normalized position.
    cs = [base['state_dict']['cs.0'].clone()]
    for j in range(1, n - 1):
        k = int(round(1 + 13 * (j - 1) / max(n - 3, 1)))
        cs.append(base['state_dict'][f'cs.{k}'].clone())
    p = base['state_dict']['cs.15'].softmax(-1).numpy()
    if n != 16:
        cdf = np.concatenate([np.zeros((*p.shape[:-1], 1)), p.cumsum(-1)], -1)
        cdf[..., -1] = 1
        old = np.linspace(0, 1, 17)
        new = np.linspace(0, 1, n + 1)
        q = np.array([np.interp(new, old, r) for r in cdf.reshape(-1, 17)])
        p = np.diff(q.reshape(*p.shape[:-1], n + 1), axis=-1)
    cs.append(torch.tensor(np.log(np.maximum(p, 1e-9)), dtype=torch.float32))
    generator = torch.Generator().manual_seed(610000 + n * 100 + seed)
    return [c + 0.05 * torch.randn(c.shape, generator=generator) for c in cs]


def scores(n, count, family, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(count, n))
    if family == 'diffuse':
        z *= 0.5
    elif family == 'sharp':
        z *= 2
    elif family == 'correlated':
        for j in range(1, n):
            z[:, j] = .7 * z[:, j-1] + math.sqrt(1-.7**2) * z[:, j]
    elif family != 'gaussian':
        raise ValueError(family)
    return torch.tensor(z, dtype=torch.float32)


def predict(model, rows):
    with torch.no_grad():
        return torch.cat([model(z) for z in rows.split(1024)])


def fit(model, n, seed, family, steps, lr, suffix):
    offset = 100000 if family == 'sharp' else 0
    train = scores(n, 16384, family, 630000 + offset + 100*n + seed)
    val = scores(n, 2048, family, 640000 + offset + 100*n + seed)
    target, vtarget = train.softmax(-1), val.softmax(-1)
    generator = torch.Generator().manual_seed(650000 + offset + 100*n + seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    initial = (predict(model, val)-vtarget).square().mean().sqrt().item()
    best, beststep = initial, 0
    beststate = copy.deepcopy(model.state_dict())
    hist = [{'step': 0, 'validation_rmse': initial}]
    for step in range(steps):
        idx = torch.randint(len(train), (256,), generator=generator)
        loss = n*n*(model(train[idx])-target[idx]).square().mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()
        opt.param_groups[0]['lr'] = lr*(.1+.9*.5*(1+math.cos(math.pi*(step+1)/steps)))
        if (step+1) % 100 == 0:
            error = (predict(model, val)-vtarget).square().mean().sqrt().item()
            hist.append({'step': step+1, 'validation_rmse': error})
            if error < best:
                best, beststep = error, step+1
                beststate = copy.deepcopy(model.state_dict())
    model.load_state_dict(beststate)
    ck = {'config': {'n': n, 'states': 16, 'rank': 8, 'bound': model.bound,
          'seed': seed, 'steps': steps, 'selected_step': beststep,
          'encoding': 'sorted_center', 'training_family': family,
          'initialization': 'dimension-resized archived N16/r8 operator'},
          'state_dict': beststate}
    torch.save(ck, ROOT/'checkpoints'/f'dimension_n{n}_seed{seed}_{suffix}.pt')
    return {'selected_step': beststep, 'validation_rmse': best, 'history': hist}


def job(args):
    n, seed = args
    begin = time.time()
    base = torch.load(ROOT/'checkpoints/softmax_N16_r8.pt', weights_only=True)
    model = Operator(resize_initialization(base, n, seed), 4)
    bfit = fit(model, n, seed, 'gaussian', 700, .03, 'base')
    results = []
    families = ['gaussian', 'diffuse', 'sharp', 'correlated']
    test = {f: scores(n, 4096, f, 620000 + n*10 + k) for k, f in enumerate(families)}

    def evaluate(stage):
        for family, z in test.items():
            y, p = predict(model, z), z.softmax(-1)
            row_tv = .5*(y-p).abs().sum(-1)
            centered = z-z.mean(-1, keepdim=True)
            record = {'n': n, 'seed': seed, 'stage': stage, 'family': family,
                'mean_row_tv': row_tv.mean().item(),
                'probability_rmse': (y-p).square().mean().sqrt().item(),
                'maximum_mass_error': (y.sum(-1)-1).abs().max().item(),
                'clipped_score_fraction': (centered.abs()>model.bound).float().mean().item(),
                'parameters': sum(c.numel() for c in model.cs), 'bound': model.bound}
            results.append(record)
            np.savez_compressed(ROOT/'source_data'/f'dimension_n{n}_seed{seed}_{stage}_{family}.npz',
                scores=z.numpy(), target=p.numpy(), prediction=y.numpy(), row_tv=row_tv.numpy())

    evaluate('base')
    model.bound = 8
    cfit = fit(model, n, seed, 'sharp', 400, .01, 'calibrated')
    evaluate('calibrated')
    record = {'n': n, 'seed': seed, 'base_fit': bfit, 'calibration_fit': cfit,
              'seconds': time.time()-begin, 'test_results': results}
    path = ROOT/'source_data'/f'dimension_n{n}_seed{seed}_fit.json'
    path.write_text(json.dumps(record, indent=2))
    print(f'n={n} seed={seed}: base val={bfit["validation_rmse"]:.5f}, '
          f'calibrated val={cfit["validation_rmse"]:.5f}, {record["seconds"]:.1f}s', flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()
    jobs = [(n, s) for n in [16, 32, 64] for s in [11, 22, 33]]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        allfits = list(pool.map(job, jobs))
    rows = [r for fitrow in allfits for r in fitrow['test_results']]
    with (ROOT/'source_data/dimension_metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    groups = []
    for n in [16, 32, 64]:
        for stage in ['base', 'calibrated']:
            for family in ['gaussian', 'diffuse', 'sharp', 'correlated']:
                group = [r for r in rows if (r['n'], r['stage'], r['family']) == (n, stage, family)]
                summary = {'n': n, 'stage': stage, 'family': family,
                           'parameters': group[0]['parameters']}
                for metric in ['mean_row_tv', 'probability_rmse', 'clipped_score_fraction']:
                    vals = [r[metric] for r in group]
                    summary[metric+'_mean'] = float(np.mean(vals))
                    summary[metric+'_fit_sd'] = float(np.std(vals, ddof=1))
                groups.append(summary)
    summary = {'groups': groups, 'fits': allfits,
        'torch_version': str(torch.__version__), 'numpy_version': str(np.__version__),
        'statistical_unit': 'three separately optimized operator fits, shared resized warm start; common held-out test rows'}
    (ROOT/'source_data/dimension_summary.json').write_text(json.dumps(summary, indent=2))
    print('Dimension study completed.', flush=True)


if __name__ == '__main__':
    main()
