"""CPU arrays or disk-backed .npy arrays, with explicit valid draw counts.

The manifest is written only after array data is flushed. Incomplete runs remain
readable; unfilled array tails are never returned by get_samples() or chain().
This stores draws, not the internal adaptive state required to resume a chain.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import tempfile
import numpy as np


def _write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


@dataclass
class SamplingResult:
    """Samples are CPU NumPy arrays, possibly read-only memory maps on reload.

    Prefer get_samples(name) for complete chains or chain(name, i) to inspect a
    partial chain. ``arrays`` includes preallocated tails and is a low-level view.
    Chain and draw axes are always retained; warmup is never stored.
    """
    arrays: dict
    metadata: dict
    directory: Path | None = None

    @property
    def chain_info(self):
        return self.metadata['chains']

    @property
    def completed_draws(self):
        return tuple(self.metadata['completed_draws'])

    def chain(self, name, index):
        return self.arrays[name][index, :self.completed_draws[index]]

    def get_samples(self, name='theta'):
        """Uniform array (completed chains, draws, ...), without copying a prefix."""
        count = 0
        for row in self.chain_info:
            if row.get('status') != 'complete':
                break
            count += 1
        if count == 0:
            raise ValueError('no complete chains; use chain(name, index) for partial draws')
        return self.arrays[name][:count]

    def save(self, directory):
        """Export a fully completed result once, in the existing loadable format.

        Useful with sample_posterior(output_dir=None): no files are written while
        chains run. Incomplete results are rejected. A new destination is required;
        arrays and manifest are written in a temporary sibling, then published.
        This saves draws, not resumable HMC state. Returns the destination Path.
        """
        metadata = self.metadata
        chains, draws = metadata['config']['chains'], metadata['config']['draws']
        if (metadata.get('status') != 'complete' or len(self.chain_info) != chains
                or any(row.get('status') != 'complete' for row in self.chain_info)
                or self.completed_draws != (draws,) * chains):
            raise ValueError('only fully completed results can be exported')
        if not self.arrays or set(self.arrays) != set(metadata['arrays']):
            raise ValueError('missing or inconsistent sample arrays')
        for name, value in self.arrays.items():
            spec = metadata['arrays'][name]
            if (not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name)
                    or spec['file'] != name + '.npy'
                    or list(value.shape) != spec['shape'] or value.dtype.str != spec['dtype']
                    or value.shape[:2] != (chains, draws)):
                raise ValueError(f'invalid sample specification: {name}')
        path = Path(directory)
        if path.exists():
            raise FileExistsError(f'export destination already exists: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix='.' + path.name + '-', dir=path.parent))
        try:
            for name, value in self.arrays.items():
                np.save(temporary / (name + '.npy'), value, allow_pickle=False)
            _write_json(temporary / 'manifest.json', metadata)
            if path.exists():
                raise FileExistsError(path)
            temporary.rename(path)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return path

    @classmethod
    def load(cls, directory):
        """Open a completed or interrupted run without loading its arrays into RAM."""
        directory = Path(directory)
        metadata = json.loads((directory/'manifest.json').read_text())
        if metadata.get('schema_version') != 1:
            raise ValueError('unsupported sampling manifest version')
        arrays = {}
        for name, spec in metadata['arrays'].items():
            filename = spec['file']
            if Path(filename).name != filename:
                raise ValueError('invalid sample filename in manifest')
            value = np.load(directory/filename, mmap_mode='r', allow_pickle=False)
            if list(value.shape) != spec['shape'] or value.dtype.str != spec['dtype']:
                raise ValueError(f'sample array does not match manifest: {name}')
            arrays[name] = value
        expected = (metadata['config']['chains'], metadata['config']['draws'])
        counts = metadata['completed_draws']
        if len(counts) != expected[0] or any(not 0 <= n <= expected[1] for n in counts):
            raise ValueError('invalid completed draw counts')
        if any(a.shape[:2] != expected for a in arrays.values()):
            raise ValueError('inconsistent chain/draw dimensions')
        return cls(arrays, metadata, directory)


class SampleStore:
    """Internal writer; one bounded device buffer is flushed at a time."""

    def __init__(self, metadata, output_dir):
        self.result = SamplingResult({}, metadata)
        if output_dir is not None:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            if any(path.iterdir()):
                raise FileExistsError(f'sampling output directory must be empty: {path}')
            self.result.directory = path
        self.checkpoint()

    def append(self, chain, values):
        metadata, arrays = self.result.metadata, self.result.arrays
        count = len(next(iter(values.values())))
        start = metadata['completed_draws'][chain]
        stop = start+count
        if stop > metadata['config']['draws']:
            raise ValueError('too many draws for the allocated store')
        if arrays and set(values) != set(arrays):
            raise ValueError('observable names changed during sampling')
        # Validate the entire chunk before advancing any valid-count marker.
        for name, value in values.items():
            if len(value) != count or not np.isfinite(value).all():
                raise ValueError(f'non-finite or inconsistent observable: {name}')
            if name in arrays and (value.shape[1:] != arrays[name].shape[2:] or value.dtype != arrays[name].dtype):
                raise ValueError(f'observable shape/dtype changed: {name}')
        for name, value in values.items():
            if name not in arrays:
                shape = (metadata['config']['chains'], metadata['config']['draws'], *value.shape[1:])
                if self.result.directory is None:
                    arrays[name] = np.empty(shape, dtype=value.dtype)
                else:
                    arrays[name] = np.lib.format.open_memmap(self.result.directory/f'{name}.npy',
                                                           mode='w+', dtype=value.dtype, shape=shape)
                metadata['arrays'][name] = dict(file=f'{name}.npy', shape=list(shape), dtype=value.dtype.str)
            arrays[name][chain, start:stop] = value
        metadata['completed_draws'][chain] = stop
        self.checkpoint()
        return start

    def checkpoint(self):
        if self.result.directory is not None:
            for value in self.result.arrays.values():
                value.flush()
            _write_json(self.result.directory/'manifest.json', self.result.metadata)
