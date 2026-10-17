"""Standard FMCW processing: range FFT, Doppler FFT, static clutter removal, micro-Doppler.

Raw frame layout (as delivered by the Infineon Radar SDK): (num_rx, num_chirps, num_samples).
A recording is (num_frames, num_rx, num_chirps, num_samples).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import windows


def range_fft(frame: np.ndarray, n_fft: int | None = None) -> np.ndarray:
    """(..., chirps, samples) real IF -> (..., chirps, range_bins) complex. Keeps positive half."""
    ns = frame.shape[-1]
    n_fft = n_fft or ns
    x = frame - frame.mean(axis=-1, keepdims=True)          # remove DC per chirp
    win = windows.blackmanharris(ns).astype(np.float32)
    X = np.fft.fft(x * win, n=n_fft, axis=-1)
    return X[..., : n_fft // 2]


def remove_static(rf: np.ndarray, mode: str = "mean", axis: int = -2) -> np.ndarray:
    """Static clutter removal along slow time within a frame.

    mode='mean'  : subtract the mean over chirps (zero-Doppler suppression)
    mode='mti2'  : 2-pulse canceller (first difference over chirps)
    mode='none'  : passthrough
    """
    if mode == "none":
        return rf
    if mode == "mean":
        return rf - rf.mean(axis=axis, keepdims=True)
    if mode == "mti2":
        return np.diff(rf, axis=axis)
    raise ValueError(mode)


def range_doppler(frame: np.ndarray, clutter: str = "none", n_doppler: int | None = None,
                  combine_rx: str = "noncoherent") -> np.ndarray:
    """Raw frame (rx, chirps, samples) -> power RD map (range_bins, doppler_bins), Doppler fftshifted."""
    rf = range_fft(frame)                                   # (rx, chirps, rbins)
    rf = remove_static(rf, clutter, axis=-2)
    nc = rf.shape[-2]
    n_doppler = n_doppler or nc
    win = windows.chebwin(nc, at=60).astype(np.float32)[:, None]
    D = np.fft.fftshift(np.fft.fft(rf * win, n=n_doppler, axis=-2), axes=-2)  # (rx, dop, rbins)
    P = np.abs(D) ** 2
    if P.ndim == 3:
        P = P.mean(axis=0) if combine_rx == "noncoherent" else P[0]
    return P.T.astype(np.float32)                            # (rbins, dop)


def rd_sequence(frames: np.ndarray, clutter: str = "none", **kw) -> np.ndarray:
    """(T, rx, chirps, samples) -> (T, range_bins, doppler_bins) power."""
    return np.stack([range_doppler(f, clutter=clutter, **kw) for f in frames])


def frame_static_reference(frames: np.ndarray) -> np.ndarray:
    """Long-term static background in range-FFT domain (complex), averaged over all frames/chirps.
    Subtracting it is the 'background subtraction' baseline used in many BGT60 pipelines."""
    rf = range_fft(frames)                                  # (T, rx, chirps, rbins)
    return rf.mean(axis=(0, 2), keepdims=False)             # (rx, rbins)


def micro_doppler(frames: np.ndarray, range_gate: tuple[int, int] | None = None,
                  clutter: str = "mean", n_doppler: int | None = None) -> np.ndarray:
    """Frame-based micro-Doppler spectrogram: (doppler_bins, T) power.

    Each frame's chirps give one Doppler spectrum; the spectra of the range bins inside
    `range_gate` (or all bins) are summed. Time resolution = frame repetition time.
    """
    cols = []
    for f in frames:
        P = range_doppler(f, clutter=clutter, n_doppler=n_doppler)   # (rbins, dop)
        if range_gate is not None:
            P = P[range_gate[0]:range_gate[1]]
        cols.append(P.sum(axis=0))
    return np.stack(cols, axis=1).astype(np.float32)


def to_db(P: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    return 10 * np.log10(np.maximum(P, floor))
