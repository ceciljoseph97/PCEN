"""Live ADC sanity check (10 s): per Rx min / max / mean / share of exact zeros of the float frames the
Infineon SDK returns, before any storage. Use it to find out why the fan session stored Rx1 as all zeros and
Rx2/Rx3 with ~70 % zeros.

  python scripts/check_adc.py            # 200 frames with the repo default RadarConfig

Reading the output:
  - min < 0 on any Rx     -> the SDK delivers signed values; the old uint16 storage (clip at 0) cut them.
                             Fixed in record_fan_session.py (signed int16, no clipping).
  - min = 0 and many zeros -> the ADC input itself rails low (gain / offset problem), not the storage.
  - an Rx with std = 0    -> that receiver delivers no data (rx_mask, hardware).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcen_radar import RadarConfig
from pcen_radar.acquire import open_device, next_frame, close_device


def main(n=200):
    cfg = RadarConfig()
    dev = open_device(cfg)
    try:
        F = np.stack([next_frame(dev) for _ in range(n)])     # (n, rx, chirps, samples) float32
    finally:
        close_device(dev)
    print(f"{n} frames, shape {F.shape}, dtype {F.dtype}")
    for i in range(F.shape[1]):
        x = F[:, i].ravel()
        print(f"Rx{i + 1}: min {x.min():+.5f}  max {x.max():+.5f}  mean {x.mean():+.5f}  std {x.std():.5f}  "
              f"exact zeros {(x == 0).mean() * 100:5.1f} %  negative {(x < 0).mean() * 100:5.1f} %")
    if (F < 0).any():
        print("-> signed values: the old uint16 storage clipped them. Re-record with the fixed recorder.")
    elif (F == 0).mean() > 0.05:
        print("-> values clip at 0 inside the ADC: lower if_gain_dB or check the board.")
    else:
        print("-> ADC values look healthy.")


if __name__ == "__main__":
    main()
