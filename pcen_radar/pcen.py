"""Per-Channel Energy Normalization for radar maps.

PCEN(t, c) = (E / (eps + M)^alpha + delta)^r - delta^r
M(t, c)    = (1 - s) M(t-1, c) + s E(t, c)          (first-order IIR over *frames*)

Here a "channel" is any non-time index: a range-Doppler cell (for RD-map sequences) or a
Doppler bin (for micro-Doppler spectrograms). Time is always axis 0 for `pcen`.

Notes for radar:
- Radar power has an arbitrary scale. `pcen` divides by a global reference (median of the
  first frames) so that eps/delta keep the same meaning across recordings and gain settings.
- s relates to a time constant tau via s = 1 - exp(-T_frame / tau).
- Anything that stays in a cell longer than ~tau gets absorbed into M (like MTI). This is
  the stopping-target failure mode; measure it.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


def s_from_tau(tau_s: float, frame_period_s: float) -> float:
    return float(1.0 - np.exp(-frame_period_s / tau_s))


def pcen(E: np.ndarray, s: float = 0.05, alpha: float = 0.98, delta: float = 2.0,
         r: float = 0.5, eps: float = 1e-6, ref: float | str = "auto",
         init_frames: int = 10, return_M: bool = False):
    """Apply PCEN along axis 0 (time / frame index) of a non-negative power array.

    E   : (T, ...) power (linear, not dB)
    ref : global scale. 'auto' = median of the first `init_frames` frames.
    The IIR state is initialised with the mean of the first `init_frames` frames to avoid
    a start-up transient that would otherwise look like a burst of detections.
    """
    E = np.asarray(E, dtype=np.float64)
    if ref == "auto":
        ref = float(np.median(E[:init_frames])) + 1e-30
    E = E / ref
    m0 = E[:init_frames].mean(axis=0)
    # lfilter with initial condition: y[t] = (1-s) y[t-1] + s x[t], y[-1] = m0
    zi = ((1 - s) * m0)[None, ...]
    M, _ = lfilter([s], [1, -(1 - s)], E, axis=0, zi=zi)
    out = (E / (eps + M) ** alpha + delta) ** r - delta ** r
    out = out.astype(np.float32)
    return (out, M.astype(np.float32)) if return_M else out


class ClutterMapState:
    """Streaming PCEN (for live use): call .step(E_t) once per frame."""

    def __init__(self, s=0.05, alpha=0.98, delta=2.0, r=0.5, eps=1e-6, ref=None):
        self.s, self.alpha, self.delta, self.r, self.eps = s, alpha, delta, r, eps
        self.ref = ref
        self.M = None

    def step(self, E: np.ndarray) -> np.ndarray:
        E = np.asarray(E, dtype=np.float64)
        if self.ref is None:
            self.ref = float(np.median(E)) + 1e-30
        E = E / self.ref
        self.M = E.copy() if self.M is None else (1 - self.s) * self.M + self.s * E
        return ((E / (self.eps + self.M) ** self.alpha + self.delta) ** self.r
                - self.delta ** self.r).astype(np.float32)


def pcen_gated(E: np.ndarray, s: float = 0.05, alpha: float = 0.98, delta: float = 2.0, r: float = 0.5,
               eps: float = 1e-6, ref: float | str = "auto", init_frames: int = 10, kappa_db: float = 6.0,
               s_fast: float = 0.1, hold_frames: int | None = None, M0: np.ndarray | None = None,
               return_M: bool = False):
    """Censored ("gated") PCEN: the background estimate learns only from background frames.

    Per cell, a fast energy average A (s_fast, time constant ~0.5 s) decides whether the current frame
    belongs to a target:
        A(t) <  kappa * M(t-1):  M(t) = (1 - s) M(t-1) + s E(t)     (background frame: plain PCEN update)
        otherwise:               M(t) = M(t-1)                      (target frame: censored, M held)
    like censoring in clutter-map CFAR. A cell held for more than hold_frames consecutive frames is
    released (M = a parallel, always-updating PCEN smoother), so a new permanent object eventually
    becomes background. A stopped target or a seated, breathing person is therefore not absorbed,
    while steady clutter is normalized as in PCEN. Fixed values: kappa = 6 dB, fast average 0.5 s,
    hold 600 s. Limitation: clutter that switches on after M was learned is held as a target too.
    History: a first version that compared two smoothers never triggered (the background followed
    the target gradually) - see results/adaptive/NOTES.md.
    """
    E = np.asarray(E, dtype=np.float64)
    if ref == "auto":
        ref = float(np.median(E[:init_frames])) + 1e-30
    En = E / ref
    M = (En[:init_frames].mean(axis=0) if M0 is None else np.asarray(M0, np.float64) / ref).copy()
    Mf = M.copy(); A = M.copy()
    cnt = np.zeros(M.shape, np.int64)
    hold = int(1e12) if hold_frames is None else int(hold_frames)
    kappa = 10 ** (kappa_db / 10)
    out = np.empty_like(En)
    Ms = np.empty_like(En) if return_M else None
    for t in range(len(En)):
        e = En[t]
        A = (1 - s_fast) * A + s_fast * e
        Mf = (1 - s) * Mf + s * e
        tgt = A >= kappa * M
        cnt = np.where(tgt, cnt + 1, 0)
        M = np.where(tgt, M, (1 - s) * M + s * e)
        rel = cnt > hold
        M = np.where(rel, Mf, M); cnt = np.where(rel, 0, cnt)
        out[t] = (e / (eps + M) ** alpha + delta) ** r - delta ** r
        if return_M:
            Ms[t] = M
    out = out.astype(np.float32)
    return (out, Ms.astype(np.float32)) if return_M else out
