"""Quick look at an unknown radar dataset: file types, array shapes/dtypes, value ranges, filename tokens.

  python scripts/inspect_dataset.py <folder> [--n 5]

Reads .npy/.npz/.mat (scipy) and .h5/.hdf5 (h5py if installed). Use it on a new dataset (e.g. the KIT
FC-FMCW gesture set) to find the raw-data layout before writing a loader.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np


def arrays_in(path: Path):
    s = path.suffix.lower()
    if s == ".npy":
        yield "", np.load(path, allow_pickle=False)
    elif s == ".npz":
        z = np.load(path, allow_pickle=False)
        for k in z.files:
            yield k, z[k]
    elif s == ".mat":
        from scipy.io import loadmat
        try:
            for k, v in loadmat(path).items():
                if not k.startswith("__"):
                    yield k, np.asarray(v)
        except NotImplementedError:      # MATLAB v7.3 = HDF5
            yield from _h5(path)
    elif s in (".h5", ".hdf5"):
        yield from _h5(path)


def _h5(path):
    import h5py
    with h5py.File(path, "r") as f:
        def walk(g, pre=""):
            for k, v in g.items():
                if isinstance(v, h5py.Dataset):
                    yield pre + k, v[()]
                else:
                    yield from walk(v, pre + k + "/")
        yield from walk(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    files = [p for p in Path(a.folder).rglob("*") if p.is_file()]
    print(f"{len(files)} files, {sum(p.stat().st_size for p in files) / 1e9:.2f} GB")
    print("extensions:", collections.Counter(p.suffix.lower() for p in files).most_common())
    print("top folders:", collections.Counter(p.relative_to(a.folder).parts[0] for p in files).most_common(10))
    data = [p for p in files if p.suffix.lower() in (".npy", ".npz", ".mat", ".h5", ".hdf5")]
    toks = [p.stem.split("_") for p in data]
    if toks:
        L = collections.Counter(len(t) for t in toks).most_common(1)[0][0]
        print(f"filename tokens (most common length {L}):")
        for i in range(L):
            vals = collections.Counter(t[i] for t in toks if len(t) == L)
            print(f"  token {i}: {len(vals)} distinct, e.g. {list(vals.most_common(8))}")
    for p in data[: a.n]:
        print(f"\n== {p.relative_to(a.folder)}  ({p.stat().st_size / 1e6:.1f} MB)")
        for k, v in arrays_in(p):
            if v.dtype.kind in "fiucb" and v.size > 1:
                print(f"  {k:30s} {v.shape} {v.dtype}  min={v.min():.4g} max={v.max():.4g} "
                      f"mean={v.mean():.4g}")
            else:
                print(f"  {k:30s} {v.dtype} {v.shape} :: {str(v)[:200]}")
    others = [p for p in files if p.suffix.lower() in (".txt", ".md", ".json", ".csv", ".pdf", ".xlsx")]
    if others:
        print("\ndocs/legend files:", [str(p.relative_to(a.folder)) for p in others[:20]])


if __name__ == "__main__":
    main()
