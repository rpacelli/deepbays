"""Optional disk cache of CNN prior patch blocks, shared across widths and Q.

The key contains the prepared data, resolved architecture, precisions, output
scale, backend and kernel source fingerprint. Runtime batching is not scientific
input. Files are NumPy arrays with no pickle and can be opened read-only.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np


class CNNKernelCache:
    def __init__(self, directory, *, namespace=''):
        self.directory = Path(directory)
        self.namespace = str(namespace)
        root = Path(__file__).resolve().parents[1]
        source = hashlib.sha256()
        for name in ('conv_geometry.py', 'kernels/cnn_cache.py', 'kernels/conv_kernels.py',
                     'kernels/conv_diagonal.py', 'kernels/kernels.py'):
            source.update((root / name).read_bytes())
        self.source_digest = source.hexdigest()
        self.hits = self.misses = 0

    @staticmethod
    def _data_digest(prepared):
        digest = hashlib.sha256()
        for value in (prepared[0], *prepared[1]):
            array = np.ascontiguousarray(value)
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(memoryview(array).cast('B'))
        return digest.hexdigest()

    def blocks(self, builder, left, right=None, *, diagonal=False):
        if diagonal and right is not None:
            raise ValueError('self blocks take one input batch')
        left_key = self._data_digest(left)
        right_key = left_key if right is None else self._data_digest(right)
        settings = dict(schema=1, namespace=self.namespace, source=self.source_digest,
                        backend=type(builder).__name__, layers=[asdict(x) for x in builder.layers],
                        channels=builder.input_channels, priors=builder.priors, act=builder.act,
                        output_scale=builder.output_scale, left=left_key, right=right_key,
                        diagonal=diagonal)
        key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        path = self.directory / (key + '.npy')
        shape = ((len(left[0]), builder.d, builder.d) if diagonal else
                 (len(left[0]), len(left[0]) if right is None else len(right[0]), builder.d, builder.d))
        if path.exists():
            value = np.load(path, mmap_mode='r', allow_pickle=False)
            if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
                raise ValueError(f'invalid cached kernel: {path}')
            self.hits += 1
            return value
        value = (builder.self_blocks(left) if diagonal else
                 builder.cross(left, None if left_key == right_key else right))
        self.directory.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.' + key, dir=self.directory)
        try:
            with os.fdopen(fd, 'wb') as stream:
                np.save(stream, value, allow_pickle=False)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.misses += 1
        return value
