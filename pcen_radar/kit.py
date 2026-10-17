"""Loader + feature extraction for the KIT FC-FMCW gesture data set (BGT60TR13C, raw IF).

Antes, Bekker, Bhutani, Zwick: "A Flexible Data Set for Radar-based Gesture Recognition with an
FC-FMCW Radar", KIT 2024, doi:10.35097/hCNNxXlJBdFTSYse (CC BY 4.0).

File: GM_<gesture>_<setup>_<age>_<disability>_<hand>_<jewelry>_<gender>_<id>.mat
  rx0, rx1, rx2 : (frames, 600 chirps, 128 samples) float32 raw IF (0..1 ADC scale)
  57.5-64.5 GHz (7 GHz), fs 1 MHz, chirp repetition 299.8 us, frame repetition 185.9 ms
  -> 600 chirps take 179.9 ms of each 185.9 ms frame: an almost continuous chirp stream.

Feature extraction treats the recording as one chirp stream, cuts it into sliding windows
(`win` chirps, `hop` chirps) and builds per window a range-Doppler map. From the RD sequence two
time-frequency maps are kept (both power, linear):
  md : (T, n_doppler)  micro-Doppler, summed over the hand range gate
  rt : (T, n_range)    range-time, summed over Doppler
each with and without static removal (mean over the chirps of the window subtracted).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import windows

C0 = 299_792_458.0
KNOWN_DUPLICATES = {"955", "942", "1429"}   # legend: replications of the previous recording
SETUP_INFO = {   # legend Sec. 4: hand distance [m], angle [deg], nearest background [m], type
    "1": (0.3, 0, 1.3, "wooden wall"), "3v1": (0.4, 0, 1.7, "wooden wall"),
    "3v2": (0.4, 40, 1.6, "wooden wall"), "4v1": (0.3, 0, 1.0, "window"),
    "4v2": (0.5, 0, 1.0, "window"), "4v3": (0.3, 0, 1.3, "window"), "4v4": (0.5, 0, 1.3, "window"),
    "5": (0.3, 0, 2.2, "shelves"), "6": (0.3, 0, 1.7, "shelves"),
    "7v1": (0.3, 0, None, "free space"), "7v2": (0.5, 0, None, "free space"),
}


def parse_name(name: str) -> dict:
    t = Path(name).stem.split("_")
    return {"gesture": t[1], "setup": t[2], "age": t[3], "disability": t[4], "hand": t[5],
            "jewelry": t[6], "gender": t[7], "id": t[8], "person_attr": "_".join(t[3:8])}


def load_mat(path):
    from scipy.io import loadmat
    m = loadmat(path)
    x = np.stack([m["rx0"], m["rx1"], m["rx2"]], 0).astype(np.float32)   # (3, F, 600, 128)
    meta = {k: float(np.asarray(m[k]).ravel()[0]) for k in
            ["sample_rate_Hz", "lower_frequency_Hz", "upper_frequency_Hz", "chirp_repetition_time_s",
             "frame_repetition_time_s", "if_gain_dB"]}
    return x, meta


@dataclass
class FeatCfg:
    win: int = 128          # chirps per Doppler window (38 ms)
    hop: int = 64           # chirps between windows (19 ms -> ~52 time steps / s)
    gate_m: tuple = (0.15, 0.75)   # hand range gate for micro-Doppler
    n_range: int = 64


def extract(x: np.ndarray, meta: dict, cfg: FeatCfg = FeatCfg()):
    """x: (3, F, 600, 128) -> dict of maps (float32) and axes."""
    nrx, F, nc, ns = x.shape
    s = x.reshape(nrx, F * nc, ns)
    s = s - s.mean(-1, keepdims=True)
    rf = np.fft.rfft(s * windows.hann(ns, sym=False).astype(np.float32), axis=-1)[..., : cfg.n_range]
    rf = rf.astype(np.complex64)                                       # (3, N, 64)
    B = meta["upper_frequency_Hz"] - meta["lower_frequency_Hz"]
    dr = C0 / (2 * B)
    g0, g1 = int(round(cfg.gate_m[0] / dr)), int(round(cfg.gate_m[1] / dr))
    N = rf.shape[1]
    starts = np.arange(0, N - cfg.win + 1, cfg.hop)
    dw = windows.hann(cfg.win, sym=False).astype(np.float32)[None, :, None]
    out = {k: [] for k in ["md_raw", "md_sr", "rt_raw", "rt_sr"]}
    for st in starts:
        blk = rf[:, st:st + cfg.win]                                   # (3, win, R)
        for tag, b in (("raw", blk), ("sr", blk - blk.mean(1, keepdims=True))):
            D = np.fft.fftshift(np.fft.fft(b * dw, axis=1), axes=1)    # (3, dop, R)
            P = (np.abs(D) ** 2).mean(0)                               # (dop, R) Rx-averaged power
            out["md_" + tag].append(P[:, g0:g1].sum(1))
            out["rt_" + tag].append(P.sum(0))
    res = {k: np.asarray(v, np.float32) for k, v in out.items()}
    lam = C0 / (0.5 * (meta["upper_frequency_Hz"] + meta["lower_frequency_Hz"]))
    tc = meta["chirp_repetition_time_s"]
    res["dt_s"] = np.float32(cfg.hop * tc * (meta["frame_repetition_time_s"] / (nc * tc)))
    res["dr_m"] = np.float32(dr)
    res["v_axis"] = ((np.arange(cfg.win) - cfg.win // 2) * lam / (2 * cfg.win * tc)).astype(np.float32)
    res["gate_bins"] = np.array([g0, g1])
    return res
