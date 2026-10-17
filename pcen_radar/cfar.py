"""2-D CFAR detectors on power range-Doppler maps.

Every detector here returns a *test statistic* map (CUT power / background estimate).
Detection = statistic > threshold factor. Returning the statistic instead of a binary map
lets the benchmark sweep the threshold and draw ROC curves for every method on equal terms;
the analytic factors (`ca_cfar_alpha`, ...) give the nominal operating point.

Axes: (range, doppler). Doppler is treated as circular (wrap), range is not.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.optimize import brentq


def _ring(guard: tuple[int, int], train: tuple[int, int]) -> np.ndarray:
    gr, gd = guard
    tr, td = train
    H, W = 2 * (gr + tr) + 1, 2 * (gd + td) + 1
    k = np.ones((H, W), dtype=bool)
    k[tr:tr + 2 * gr + 1, td:td + 2 * gd + 1] = False
    return k


def _conv(P, k):
    # range: reflect, doppler: wrap  -> do it by padding manually
    pr, pd = k.shape[0] // 2, k.shape[1] // 2
    Pp = np.pad(P, ((0, 0), (pd, pd)), mode="wrap")
    Pp = np.pad(Pp, ((pr, pr), (0, 0)), mode="reflect")
    out = ndimage.correlate(Pp, k.astype(np.float64), mode="constant")
    return out[pr:-pr or None, pd:-pd or None]


def ca_cfar_stat(P: np.ndarray, guard=(2, 2), train=(8, 4)) -> np.ndarray:
    k = _ring(guard, train)
    noise = _conv(P.astype(np.float64), k) / k.sum()
    return (P / (noise + 1e-30)).astype(np.float32)


def os_cfar_stat(P: np.ndarray, guard=(2, 2), train=(8, 4), k_frac: float = 0.75) -> np.ndarray:
    fp = _ring(guard, train)
    N = int(fp.sum())
    rank = int(np.clip(round(k_frac * N), 1, N)) - 1
    pr, pd = fp.shape[0] // 2, fp.shape[1] // 2
    Pp = np.pad(P, ((0, 0), (pd, pd)), mode="wrap")
    Pp = np.pad(Pp, ((pr, pr), (0, 0)), mode="reflect")
    noise = ndimage.rank_filter(Pp, rank=rank, footprint=fp, mode="constant")
    noise = noise[pr:-pr or None, pd:-pd or None]
    return (P / (noise + 1e-30)).astype(np.float32)


def clutter_map_stat(P_seq: np.ndarray, w: float = 0.05, init_frames: int = 10) -> np.ndarray:
    """Clutter-map CFAR: per-cell exponential average over frames, CUT compared to the map
    *before* it is updated with the current frame. P_seq: (T, R, D)."""
    P_seq = P_seq.astype(np.float64)
    Q = P_seq[:init_frames].mean(axis=0)
    out = np.empty_like(P_seq, dtype=np.float32)
    for t in range(len(P_seq)):
        out[t] = P_seq[t] / (Q + 1e-30)
        Q = (1 - w) * Q + w * P_seq[t]
    return out


# ---------- analytic thresholds (exponential / Swerling-0-in-noise assumptions) ----------

def ca_cfar_alpha(pfa: float, n_train: int) -> float:
    return n_train * (pfa ** (-1.0 / n_train) - 1.0)


def os_cfar_alpha(pfa: float, n_train: int, k: int) -> float:
    def f(T):
        p = 1.0
        for i in range(k):
            p *= (n_train - i) / (n_train - i + T)
        return p - pfa
    return brentq(f, 1e-6, 1e6)


def clutter_map_alpha(pfa: float, w: float, n_terms: int = 2000) -> float:
    def f(T):
        k = np.arange(n_terms)
        return np.prod(1.0 / (1.0 + T * w * (1 - w) ** k)) - pfa
    return brentq(f, 1e-6, 1e6)


def n_train_cells(guard=(2, 2), train=(8, 4)) -> int:
    return int(_ring(guard, train).sum())
