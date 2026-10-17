"""Radar configuration and derived quantities for the BGT60TR13C (58-63.5 GHz, 1 Tx / 3 Rx)."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field

C = 299_792_458.0


@dataclass
class RadarConfig:
    # Chirp
    start_frequency_Hz: float = 59.5e9
    end_frequency_Hz: float = 61.5e9
    num_samples: int = 128            # fast-time samples per chirp
    sample_rate_Hz: float = 1.0e6
    # Frame
    num_chirps: int = 64              # slow-time chirps per frame (-> Doppler bins)
    chirp_repetition_time_s: float = 391.55e-6
    frame_repetition_time_s: float = 0.05   # 20 Hz frame rate
    # Front end
    rx_mask: int = 0b111              # all 3 Rx
    tx_mask: int = 0b1
    tx_power_level: int = 31
    if_gain_dB: int = 33
    lp_cutoff_Hz: int = 500_000
    hp_cutoff_Hz: int = 80_000
    extra: dict = field(default_factory=dict)

    # ---- derived ----
    @property
    def bandwidth_Hz(self) -> float:
        return abs(self.end_frequency_Hz - self.start_frequency_Hz)

    @property
    def center_frequency_Hz(self) -> float:
        return 0.5 * (self.start_frequency_Hz + self.end_frequency_Hz)

    @property
    def wavelength_m(self) -> float:
        return C / self.center_frequency_Hz

    @property
    def num_rx(self) -> int:
        return bin(self.rx_mask).count("1")

    @property
    def range_resolution_m(self) -> float:
        return C / (2 * self.bandwidth_Hz)

    @property
    def max_range_m(self) -> float:
        # real-valued IF sampling -> num_samples/2 usable range bins
        return self.range_resolution_m * self.num_samples / 2

    @property
    def velocity_resolution_m_s(self) -> float:
        return self.wavelength_m / (2 * self.num_chirps * self.chirp_repetition_time_s)

    @property
    def max_velocity_m_s(self) -> float:
        return self.wavelength_m / (4 * self.chirp_repetition_time_s)

    @property
    def frame_rate_Hz(self) -> float:
        return 1.0 / self.frame_repetition_time_s

    def range_axis(self, n_bins: int | None = None):
        import numpy as np
        n = n_bins or self.num_samples // 2
        return np.arange(n) * self.range_resolution_m

    def velocity_axis(self):
        import numpy as np
        n = self.num_chirps
        return (np.arange(n) - n // 2) * self.velocity_resolution_m_s

    def range_bin(self, r_m: float) -> int:
        return int(round(r_m / self.range_resolution_m))

    def summary(self) -> str:
        return (f"BW={self.bandwidth_Hz/1e9:.2f} GHz  dR={self.range_resolution_m*100:.1f} cm  "
                f"Rmax={self.max_range_m:.2f} m  dv={self.velocity_resolution_m_s:.3f} m/s  "
                f"vmax=±{self.max_velocity_m_s:.2f} m/s  frame rate={self.frame_rate_Hz:.1f} Hz  "
                f"Rx={self.num_rx}")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "RadarConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)
