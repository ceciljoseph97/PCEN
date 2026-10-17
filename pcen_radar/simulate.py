"""Raw-IF simulator matching the BGT60TR13C frame layout, for controlled experiments.

Scene ingredients
- static scatterers (walls, furniture) with slight amplitude fluctuation
- a 'fan' region: moving clutter with broad Doppler spread at a fixed range, switchable on/off
- one target: walk-in, walk, stop (dwell), walk-out; optional limb micro-Doppler
- white receiver noise + mild near-range noise (leakage)

All units SI. Output (T, rx, chirps, samples) float32, plus the true per-frame target range.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import RadarConfig, C


@dataclass
class Scene:
    n_frames: int = 400
    static: list = field(default_factory=lambda: [(0.6, 30.0), (2.2, 12.0), (3.4, 20.0), (4.3, 8.0)])
    static_fluct: float = 0.02         # relative amplitude jitter per frame
    fan_range_m: float = 1.6
    fan_amp: float = 1.5
    fan_frames: tuple = (150, 400)     # moving clutter switches on mid-recording
    target_amp: float = 0.6
    target_path: str = "walk_stop"     # walk_stop | approach | none
    limbs: bool = True
    noise_std: float = 0.02
    seed: int = 0


def _beat(cfg: RadarConfig, R, amp, phase0, t_chirp):
    """IF signal of one point scatterer. R: (chirps,) range per chirp. -> (chirps, samples)."""
    Tc = cfg.num_samples / cfg.sample_rate_Hz
    slope = cfg.bandwidth_Hz / Tc
    n = np.arange(cfg.num_samples) / cfg.sample_rate_Hz
    fb = 2 * R[:, None] * slope / C
    ph = 4 * np.pi * R[:, None] / cfg.wavelength_m
    return amp * np.cos(2 * np.pi * fb * n[None] + ph + phase0)


def target_track(sc: Scene, cfg: RadarConfig):
    T = sc.n_frames
    t = np.arange(T) * cfg.frame_repetition_time_s
    r = np.full(T, np.nan)
    v = np.zeros(T)
    if sc.target_path == "walk_stop":
        # absent 0-40, walk 3.8 -> 2.6 (40-140), stop (140-240), walk 2.6 -> 1.0 (240-340), absent after
        segs = [(40, 140, 3.8, 2.6), (140, 240, 2.6, 2.6), (240, 340, 2.6, 1.0)]
    elif sc.target_path == "approach":
        segs = [(20, T - 20, 4.0, 0.8)]
    else:
        segs = []
    for a, b, r0, r1 in segs:
        a, b = min(a, T), min(b, T)
        if b <= a:
            continue
        r[a:b] = np.linspace(r0, r1, b - a)
        v[a:b] = (r1 - r0) / ((b - a) * cfg.frame_repetition_time_s)
    return t, r, v


def simulate(cfg: RadarConfig | None = None, sc: Scene | None = None, extra_fn=None):
    """extra_fn(frame_idx, t_frame, t_chirp) -> (chirps, samples) extra IF signal (optional)."""
    cfg = cfg or RadarConfig()
    sc = sc or Scene()
    rng = np.random.default_rng(sc.seed)
    T, Nrx, Nc, Ns = sc.n_frames, cfg.num_rx, cfg.num_chirps, cfg.num_samples
    tc = np.arange(Nc) * cfg.chirp_repetition_time_s
    out = np.zeros((T, Nrx, Nc, Ns), np.float32)
    _, r_tgt, v_tgt = target_track(sc, cfg)

    fan_blades = [(rng.uniform(0.05, 0.15), rng.uniform(4, 9), rng.uniform(0, 2 * np.pi),
                   rng.uniform(0.3, 1.0)) for _ in range(8)]
    leak = 0.3 * np.cos(2 * np.pi * 0.01 * np.arange(Ns))[None]

    for f in range(T):
        tf = f * cfg.frame_repetition_time_s
        sig = np.zeros((Nc, Ns))
        for R0, A in sc.static:
            a = A * (1 + sc.static_fluct * rng.standard_normal())
            sig += _beat(cfg, np.full(Nc, R0), a, 0.3 * R0, tc)
        if sc.fan_frames[0] <= f < sc.fan_frames[1]:
            for rad, rps, ph, a in fan_blades:   # rotating scatterers: sinusoidal range
                R = sc.fan_range_m + rad * np.sin(2 * np.pi * rps * (tf + tc) + ph)
                sig += _beat(cfg, R, sc.fan_amp * a, ph, tc)
        if np.isfinite(r_tgt[f]):
            moving = abs(v_tgt[f]) > 1e-3
            R = r_tgt[f] + v_tgt[f] * tc
            if not moving:                         # standing: breathing ~0.25 Hz, 5 mm
                R = R + 0.005 * np.sin(2 * np.pi * 0.25 * (tf + tc))
            sig += _beat(cfg, R, sc.target_amp, 0.0, tc)
            if sc.limbs and moving:                # arms/legs swinging at ~1 Hz gait
                for k, sw in enumerate([0.35, -0.35, 0.6]):
                    vl = sw * np.sin(2 * np.pi * 1.0 * (tf + tc) + k)
                    Rl = R + np.cumsum(vl) * cfg.chirp_repetition_time_s
                    sig += _beat(cfg, Rl, (0.25, 0.25, 0.15)[k] * sc.target_amp / 0.6, k, tc)
        if extra_fn is not None:
            sig += extra_fn(f, tf, tc)
        for rx in range(Nrx):
            noise = sc.noise_std * rng.standard_normal((Nc, Ns))
            out[f, rx] = (sig + noise + leak * (1 + 0.01 * rng.standard_normal((Nc, 1)))).astype(np.float32)
    # fixed scaling into the SDK's 0..1 ADC range (fixed, so amplitudes stay comparable)
    out = 0.5 + out / 250.0
    return out.astype(np.float32), r_tgt, v_tgt


def simulate_activity(label: str, cfg: RadarConfig, n_frames: int = 64, seed: int = 0,
                      session_gain: float = 1.0, session_clutter: float = 1.0):
    """Short synthetic clips for the classifier demo. Classes differ in micro-Doppler:
    'walk' (translation + limb swing), 'wave' (in place, fast hand), 'still' (breathing only),
    'empty' (clutter only). session_* vary the background to emulate different rooms."""
    rng = np.random.default_rng(seed)
    r0 = rng.uniform(1.2, 3.5)
    amp = 0.6 * session_gain * rng.uniform(0.6, 1.4)
    v = rng.uniform(0.4, 1.2) * rng.choice([-1, 1]) if label == "walk" else 0.0
    sc = Scene(n_frames=n_frames, seed=seed,
               fan_frames=(0, n_frames) if rng.random() < 0.5 else (0, 0),
               fan_range_m=rng.uniform(0.8, 4.0), fan_amp=1.5 * session_clutter,
               static=[(rng.uniform(0.4, 4.5), rng.uniform(5, 30) * session_clutter) for _ in range(4)],
               target_path="none", limbs=False)

    def extra(f, tf, tc):
        if label == "empty":
            return 0.0
        R = np.clip(r0 + v * (tf + tc), 0.3, 4.6)
        sig = _beat(cfg, R, amp, 0, tc)
        if label == "walk":
            for k, sw in enumerate([0.4, -0.4, 0.7]):
                vl = sw * np.sin(2 * np.pi * 0.9 * (tf + tc) + k)
                sig += _beat(cfg, R + np.cumsum(vl) * cfg.chirp_repetition_time_s, 0.3 * amp, k, tc)
        elif label == "wave":
            vl = 1.2 * np.sin(2 * np.pi * 2.0 * (tf + tc))
            sig += _beat(cfg, R - 0.2 + np.cumsum(vl) * cfg.chirp_repetition_time_s, 0.5 * amp, 1, tc)
        elif label == "still":
            sig = _beat(cfg, R + 0.006 * np.sin(2 * np.pi * 0.3 * (tf + tc)), amp, 2, tc)
        return sig

    frames, _, _ = simulate(cfg, sc, extra_fn=extra)
    return frames
