"""Radar point clouds from BGT60TR13C frames, with exchangeable detection front ends.

Re-implementation of the processing chain of rsazid99/inf_radarpoint (radar_data/utils.py, Apache-2.0,
RadarPoint.get_pointcloud) for offline use, parameterized for our radar configuration:
  per Rx: DC removal over fast time, Hamming window, range FFT (2x zero padding, positive half);
  static removal = mean over chirps; Hamming window, Doppler FFT, fftshift;
  detection: per-Rx CA-CFAR (2 guard, 4 training cells per side, square ring), AND over the 3 Rx;
  angles from Rx phase differences (L-shaped array: Rx1/Rx3 azimuth, Rx3/Rx2 elevation, d = lambda/2),
  elevation correction pi/8 for the FOV gate, FOV +-45 deg, minimum range 0.2 m, SNR gate (amplitude of
  Rx3 / (median |Rx3| / 0.6745) > 8); point = (r cos(el) sin(az), -r sin(el), r cos(el) cos(az)).

With use_azimuth=False (our fan session: Rx1 delivered only zeros) the azimuth is set to 0, detection uses
Rx2 and Rx3 only, and the FOV gate acts on elevation only; points then lie in the range-elevation plane.

Two deviations of the original are kept only in the "as published" mode and fixed otherwise:
  (1) its CFAR compares the complex Doppler spectrum with a threshold, which numpy does lexicographically
      (real part first), so it is not a power test; here the CFAR acts on power |X|^2;
  (2) its velocity axis uses the sampling time of one chirp instead of the chirp repetition time.
In "as published" mode the original CFAR (pfa 0.05, +0.10 offset, complex comparison) and SNR gate are used.

For the front-end comparison every method produces a statistic per Rx and cell; a cell is a point when the
statistic exceeds the threshold in all three Rx (min over Rx > thr) and the cell passes the geometric gates
(range, FOV). The SNR gate is a global amplitude threshold and is therefore dropped in comparison mode;
each method's threshold is set to the same false-point rate in an empty room (fan off).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve

C0 = 299_792_458.0


class PointChain:
    def __init__(self, cfg, az_sign=1.0, el_sign=1.0, nfft_range=None, nfft_doppler=None, use_azimuth=True):
        self.cfg = cfg
        self.use_azimuth = use_azimuth     # False when Rx1 delivers no data: elevation (Rx3/Rx2) only
        self.ns, self.nc = cfg.num_samples, cfg.num_chirps
        self.nfr = nfft_range or 2 * self.ns
        self.nfd = nfft_doppler or self.nc
        B = cfg.end_frequency_Hz - cfg.start_frequency_Hz
        Tc = self.ns / cfg.sample_rate_Hz                      # ramp ~ sampling time (as in the original)
        S = B / Tc
        f = np.fft.fftfreq(self.nfr, d=1 / cfg.sample_rate_Hz)[: self.nfr // 2]
        self.ranges = C0 * f / (2 * S)
        self.lam = C0 / (0.5 * (cfg.start_frequency_Hz + cfg.end_frequency_Hz))
        fd = np.fft.fftshift(np.fft.fftfreq(self.nfd, d=cfg.chirp_repetition_time_s))
        self.velocities = fd * self.lam / 2
        fd_pub = np.fft.fftshift(np.fft.fftfreq(self.nfd, d=Tc))
        self.velocities_published = fd_pub * self.lam / 2
        self.az_sign, self.el_sign = az_sign, el_sign
        # CFAR kernel of the original: square ring, 4 training + 2 guard cells per side
        nt, ng = 4, 2
        k = 2 * (nt + ng) + 1
        K = np.ones((k, k), np.float32); c = nt + ng
        K[c - ng: c + ng + 1, c - ng: c + ng + 1] = 0
        self.T = float(K.sum()); self.K = K / self.T

    # ------------------------------------------------------------------ spectra
    def rd(self, frame, static_removal=True):
        """frame (rx, chirps, samples) -> complex RD (rx, doppler, range)."""
        x = frame - frame.mean(axis=2, keepdims=True)
        x = x * np.hamming(self.ns)[None, None, :]
        R = np.fft.fft(x, n=self.nfr, axis=2)[:, :, : self.nfr // 2]
        if static_removal:
            R = R - R.mean(axis=1, keepdims=True)             # static removal (mean over chirps)
        R = R * np.hamming(self.nc)[None, :, None]
        return np.fft.fftshift(np.fft.fft(R, n=self.nfd, axis=1), axes=1)

    def rd_sequence(self, frames, static_removal=True):
        return np.stack([self.rd(f, static_removal) for f in frames]).astype(np.complex64)   # (T, rx, dop, rng)

    # ------------------------------------------------------------------ geometry
    def angles(self, X):
        """X (rx, dop, rng) complex -> azimuth, elevation (rad) per cell."""
        def est(a, b):
            s = np.angle(a * np.conj(b)) * self.lam / (2 * np.pi * (self.lam / 2))
            return np.arcsin(np.clip(s, -1, 1))
        az = self.az_sign * est(X[0], X[2]) if self.use_azimuth else np.zeros(X.shape[1:])
        el = self.el_sign * est(X[2], X[1])
        return az, el

    def gates(self, X):
        az, el = self.angles(X)
        rng = self.ranges[None, :] * np.ones((X.shape[1], 1))
        fov = (np.abs(az) <= np.pi / 4) & (np.abs(el + np.pi / 8) <= np.pi / 4)
        return (rng > 0.2) & fov, az, el

    def to_xyz(self, r, az, el):
        return np.stack([r * np.cos(el) * np.sin(az), -r * np.sin(el), r * np.cos(el) * np.cos(az)], axis=-1)

    # ------------------------------------------------------------------ original detector
    def cfar_published(self, s):
        noise = convolve(s, self.K, mode="nearest")           # complex convolution, as in the original
        alpha = self.T * (0.05 ** (-1 / self.T) - 1.0)
        return s > alpha * noise + 0.10                       # numpy: lexicographic complex comparison

    def points_published(self, X):
        det = self.cfar_published(X[0]) & self.cfar_published(X[1]) & self.cfar_published(X[2])
        g, az, el = self.gates(X)
        noise = np.median(np.abs(X[2])) / 0.6745
        snr = np.abs(X[2]) / noise
        m = det & g & (snr > 8)
        return m, az, el

    # ------------------------------------------------------------------ power CA-CFAR statistic
    def ca_stat(self, P):
        """P (dop, rng) power -> P / ring mean (same ring as the original, edges 'nearest')."""
        return P / (convolve(P, self.K, mode="nearest") + 1e-30)
