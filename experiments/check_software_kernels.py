"""Small dependency-light checks for the released software/native interfaces.

Uses synthetic values and a temporary C++ shared library. It does not train,
download a benchmark, modify checkpoints, or claim reproduction of paper scores.
"""
from pathlib import Path
import argparse
import ctypes
import importlib.util
import json
import shutil
import subprocess
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parent


def check_native():
    compiler = shutil.which('c++') or shutil.which('g++')
    if compiler is None:
        return {'status': 'skipped', 'reason': 'C++17 compiler not found'}
    source = ROOT / 'convergence/source/trace_kernels.cpp'
    with tempfile.TemporaryDirectory(prefix='frost-kernel-check-') as temp:
        library = Path(temp) / 'trace_kernels.so'
        subprocess.run([compiler, '-std=c++17', '-O2', '-shared', '-fPIC', str(source), '-o', str(library)], check=True)
        lib = ctypes.CDLL(str(library))
        pointer = ctypes.c_void_p
        lib.fsm_trace.argtypes = [pointer, pointer, pointer, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32]
        lib.categorical_trace.argtypes = [pointer, pointer, pointer, pointer, pointer, pointer, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        rng = np.random.default_rng(20261022)
        width, rows, n, states, pad = 7, 3, 16, 4, 16
        bits = rng.integers(0, 2, size=(width, rows, n), dtype=np.uint8)
        bits[:, 0, 0] = 0
        bits[:, 0, 1] = 1
        initial = np.full((rows, n), 1, dtype=np.int16)
        actual_state = initial.copy()
        actual_trace = np.empty((width, rows, n), dtype=np.int16)
        lib.fsm_trace(actual_state.ctypes.data, bits.ctypes.data, actual_trace.ctypes.data, rows*n, width, states)
        expected = initial.copy()
        expected_trace = []
        for b in bits:
            expected = np.clip(expected + 2*b.astype(np.int16) - 1, 0, states-1)
            expected_trace.append(expected.copy())
        assert np.array_equal(actual_state, expected)
        assert np.array_equal(actual_trace, np.stack(expected_trace))
        right_size = np.array([2]*(n-1) + [n], dtype=np.int32)
        cdf = np.zeros((n, states, pad, pad), dtype=np.float64)
        for j, right in enumerate(right_size):
            probabilities = rng.random((states, pad, int(right)))
            probabilities /= probabilities.sum(-1, keepdims=True)
            cdf[j, :, :, :right] = probabilities.cumsum(-1)
            cdf[j, :, :, right-1] = 1
        uniforms = rng.random((n, width*rows))
        actual_event = np.empty(width*rows, dtype=np.uint8)
        actual_counts = np.zeros((rows, n), dtype=np.int32)
        lib.categorical_trace(actual_trace.ctypes.data, uniforms.ctypes.data, cdf.ctypes.data,
                              right_size.ctypes.data, actual_event.ctypes.data, actual_counts.ctypes.data,
                              rows, width, states, n, pad)
        flat_states = actual_trace.reshape(-1, n)
        expected_event = np.empty(width*rows, dtype=np.uint8)
        expected_counts = np.zeros_like(actual_counts)
        for event in range(width*rows):
            bond = 0
            for j in range(n):
                p = cdf[j, flat_states[event, j], bond, :right_size[j]]
                bond = min(int(np.searchsorted(p, uniforms[j, event], side='left')), int(right_size[j])-1)
            expected_event[event] = bond
            expected_counts[event % rows, bond] += 1
        assert np.array_equal(actual_event, expected_event)
        assert np.array_equal(actual_counts, expected_counts)
        assert np.all(actual_counts.sum(-1) == width)
        return {'status': 'passed', 'fsm_values_checked': int(actual_trace.size),
                'categorical_events_checked': int(actual_event.size), 'every_prefix_emits_one_event': True}


def check_data_helpers():
    path = ROOT / 'classification/data.py'
    spec = importlib.util.spec_from_file_location('frost_classification_data_check', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    labels = np.repeat(np.arange(10), 100)
    a = module.stratified_indices(labels, 100, 42)
    b = module.stratified_indices(labels, 100, 42)
    assert np.array_equal(a, b) and len(np.unique(a)) == 100
    assert np.array_equal(np.bincount(labels[a]), np.full(10, 10))
    pixels = np.arange(2*3*32*32, dtype=np.uint8).reshape(2, 3, 32, 32)
    patches = module.patchify(pixels, np.zeros(3), np.ones(3))
    rebuilt = patches.reshape(2, 4, 4, 3, 8, 8).transpose(0, 3, 1, 4, 2, 5).reshape(2, 3, 32, 32)
    assert np.allclose(rebuilt, pixels.astype(np.float32)/255)
    return {'status': 'passed', 'partition_unique_indices': 100, 'tokens': list(patches.shape)}


def check_torch():
    if importlib.util.find_spec('torch') is None:
        return {'status': 'skipped', 'reason': 'PyTorch is not installed in this environment'}
    commands = [
        ('frost_direct_hi_smurf', "import torch; from categorical_study import CategoricalTT; torch.manual_seed(42); m=CategoricalTT(states=16,rank=8); p=m(torch.randn(4,16)); assert p.shape==(4,16); assert torch.all(p>=0); assert torch.allclose(p.sum(-1),torch.ones(4),atol=2e-6); assert sum(v.numel() for v in m.parameters())==16512"),
        ('classification', "import torch; from vision import VisionTransformer; m=VisionTransformer(192); y=m(torch.zeros(2,16,192)); assert y.shape==(2,10); assert sum(p.numel() for p in m.parameters())==41258"),
        ('retrieval', "import torch; from teacher import Transformer,inputs; x,y,*_=inputs(2,42); logits=Transformer()(torch.tensor(x)); assert logits.shape==(2,4)"),
    ]
    import sys
    for directory, command in commands:
        subprocess.run([sys.executable, '-c', command], cwd=ROOT/directory, check=True)
    return {'status': 'passed', 'checks': ['categorical simplex/parameter count', 'classification graph/parameter count', 'retrieval graph shape']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-torch', action='store_true')
    args = parser.parse_args()
    report = {'scope': 'synthetic software-interface smoke checks; not full experimental reproduction',
              'data_helpers': check_data_helpers(), 'native_kernels': check_native(),
              'torch_models': {'status': 'skipped', 'reason': '--skip-torch'} if args.skip_torch else check_torch()}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
