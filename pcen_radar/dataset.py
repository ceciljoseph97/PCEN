"""Turn labelled recordings into micro-Doppler windows for classification (numpy only).

Folder convention (either works):
  data/<anything>/<session>/<class>/*.npz            (class/session from folders)
  data/<session>/*.npz with labels {"class":..., "session":...} inside the file (record.py)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import load_recording
from .processing import micro_doppler


def find_recordings(root):
    items = []
    for p in sorted(Path(root).rglob("*.npz")):
        d = load_recording(p)
        lab = d["labels"]
        cls = lab.get("class") or p.parent.name
        ses = lab.get("session") or p.parent.parent.name
        items.append((p, cls, ses))
    return items


def windows_from_recording(path, win: int = 64, hop: int = 32, clutter: str = "mean",
                           skip_near_bins: int = 3):
    """-> array (n_windows, doppler_bins, win) of micro-Doppler power."""
    d = load_recording(path)
    fr = d["frames"]
    R = d["config"].num_samples // 2
    md = micro_doppler(fr, range_gate=(skip_near_bins, R), clutter=clutter)   # (D, T)
    T = md.shape[1]
    if T < win:
        return np.empty((0, md.shape[0], win), np.float32)
    starts = range(0, T - win + 1, hop)
    return np.stack([md[:, s:s + win] for s in starts]).astype(np.float32)


def build_dataset(root, win=64, hop=32, clutter="mean", cache=None):
    if cache and Path(cache).exists():
        z = np.load(cache, allow_pickle=False)
        return z["X"], z["y"], z["sessions"], list(z["classes"])
    items = find_recordings(root)
    classes = sorted({c for _, c, _ in items})
    X, y, S = [], [], []
    for p, c, s in items:
        w = windows_from_recording(p, win, hop, clutter)
        X.append(w); y += [classes.index(c)] * len(w); S += [s] * len(w)
    X = np.concatenate(X)
    y, S = np.array(y), np.array(S)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, y=y, sessions=S, classes=np.array(classes))
    return X, y, S, classes
