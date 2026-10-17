"""Recording format + label handling.

A recording is one .npz file:
    frames   : (T, rx, chirps, samples) float32 raw IF (as from the Infineon SDK, 0..1 scaled)
    config   : JSON string of RadarConfig
    t        : (T,) timestamps in seconds
    labels   : JSON string (optional), see below
    meta     : JSON string (optional, free text: room, person, notes ...)

Labels (all optional, per recording):
    {
      "class": "walking",                 # for classification
      "session": "room_A_day1",           # for cross-session splits
      "target_range_m": [[f0, f1, r_min, r_max], ...],   # frame spans + range gate of target
      "moving": [[f0, f1], ...],          # frame spans in which the target moves
    }
Frame indices f1 = -1 means "until the end".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import RadarConfig


def save_recording(path, frames, cfg: RadarConfig, t=None, labels=None, meta=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames, dtype=np.float32)
    if t is None:
        t = np.arange(len(frames)) * cfg.frame_repetition_time_s
    np.savez_compressed(path, frames=frames, t=np.asarray(t, np.float64),
                        config=cfg.to_json(), labels=json.dumps(labels or {}),
                        meta=json.dumps(meta or {}))


def load_recording(path):
    path = Path(path)
    if path.suffix == ".npy":
        return load_raw_npy(path)
    z = np.load(path, allow_pickle=False)
    cfg = RadarConfig.from_dict(json.loads(str(z["config"])))
    labels = json.loads(str(z["labels"])) if "labels" in z else {}
    meta = json.loads(str(z["meta"])) if "meta" in z else {}
    if "frames_i16" in z:                       # compact format of record_fan_session.py (signed, no clipping)
        frames = z["frames_i16"].astype(np.float32) / float(z["adc_scale"])
    elif "frames_u16" in z:                     # first version: values below 0 were clipped to 0 when saving
        frames = z["frames_u16"].astype(np.float32) / float(z["adc_scale"])
    else:
        frames = z["frames"]
    return {"frames": frames, "t": z["t"], "config": cfg, "labels": labels, "meta": meta}


def load_raw_npy(path, config_json=None):
    """Raw .npy from another tool (e.g. Infineon Radar Fusion GUI export).
    Expected shape (T, rx, chirps, samples). A sibling config.json (our RadarConfig keys)
    is used if present, otherwise defaults -> check the axes!"""
    path = Path(path)
    frames = np.load(path).astype(np.float32)
    if frames.ndim == 3:                       # (T, chirps, samples) single Rx
        frames = frames[:, None]
    cj = Path(config_json) if config_json else path.with_name("config.json")
    cfg = RadarConfig.from_dict(json.loads(cj.read_text())) if cj.exists() else RadarConfig(
        num_chirps=frames.shape[2], num_samples=frames.shape[3])
    lj = path.with_name("labels.json")
    labels = json.loads(lj.read_text()) if lj.exists() else {}
    return {"frames": frames, "t": np.arange(len(frames)) * cfg.frame_repetition_time_s,
            "config": cfg, "labels": labels, "meta": {}}


def _span(f0, f1, T):
    f1 = T if f1 is None or f1 < 0 else min(f1, T)
    return max(0, f0), f1


def make_masks(labels: dict, cfg: RadarConfig, shape, tol_m: float = 0.15,
               guard_m: float = 0.30, exclude_zero_doppler: int = 0):
    """Build (target, guard, present, moving) from labels for a (T, R, D) map sequence.

    target  : range gate [r_min - tol, r_max + tol] over all Doppler bins (optionally
              excluding +-exclude_zero_doppler bins around 0 m/s)
    guard   : extra guard_m around the gate (excluded from Pfa)
    """
    T, R, D = shape
    dr = cfg.range_resolution_m
    target = np.zeros(shape, bool)
    guard = np.zeros(shape, bool)
    present = np.zeros(T, bool)
    moving = np.zeros(T, bool)
    for f0, f1, rmin, rmax in labels.get("target_range_m", []):
        a, b = _span(int(f0), int(f1), T)
        r0 = max(0, int(np.floor((rmin - tol_m) / dr)))
        r1 = min(R, int(np.ceil((rmax + tol_m) / dr)) + 1)
        g0 = max(0, int(np.floor((rmin - tol_m - guard_m) / dr)))
        g1 = min(R, int(np.ceil((rmax + tol_m + guard_m) / dr)) + 1)
        target[a:b, r0:r1, :] = True
        guard[a:b, g0:g1, :] = True
        present[a:b] = True
    if exclude_zero_doppler:
        c = D // 2
        zd = slice(c - exclude_zero_doppler, c + exclude_zero_doppler + 1)
        guard[:, :, zd] |= target[:, :, zd]
        target[:, :, zd] = False
    guard &= ~target
    for f0, f1 in labels.get("moving", []):
        a, b = _span(int(f0), int(f1), T)
        moving[a:b] = True
    if "moving" not in labels:
        moving = present.copy()
    return target, guard, present, moving


def per_frame_range_masks(range_m: np.ndarray, cfg: RadarConfig, shape, tol_m=0.15, guard_m=0.30,
                          exclude_zero_doppler: int = 0):
    """Masks from a per-frame target range track (NaN = absent), e.g. from simulation or a
    reference sensor."""
    T, R, D = shape
    dr = cfg.range_resolution_m
    target = np.zeros(shape, bool)
    guard = np.zeros(shape, bool)
    for t in range(T):
        r = range_m[t]
        if not np.isfinite(r):
            continue
        r0, r1 = max(0, int((r - tol_m) / dr)), min(R, int(np.ceil((r + tol_m) / dr)) + 1)
        g0, g1 = max(0, int((r - tol_m - guard_m) / dr)), min(R, int(np.ceil((r + tol_m + guard_m) / dr)) + 1)
        target[t, r0:r1] = True
        guard[t, g0:g1] = True
    if exclude_zero_doppler:
        c = D // 2
        zd = slice(c - exclude_zero_doppler, c + exclude_zero_doppler + 1)
        guard[:, :, zd] |= target[:, :, zd]
        target[:, :, zd] = False
    guard &= ~target
    return target, guard, np.isfinite(range_m)


def load_parts(labels_json):
    """Join the 60-s parts of one breathing block (record_breathing_session.py) -> dict like load_recording."""
    labels_json = Path(labels_json)
    lab = json.loads(labels_json.read_text(encoding="utf-8"))
    recs = [load_recording(labels_json.parent / p) for p in lab["parts"]]
    return {"frames": np.concatenate([r["frames"] for r in recs]), "t": np.concatenate([r["t"] for r in recs]),
            "config": recs[0]["config"], "labels": lab, "meta": lab.get("meta", {})}
