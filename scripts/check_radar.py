"""Acquisition-only check of the BGT60TR13C (no plotting): does the link keep up with the config?

  python scripts/check_radar.py              # 400 frames (20 s) with the repo default config
  python scripts/check_radar.py --frames 1200

Prints the frame period, the number of dropped frames, and a coarse range profile so you can see
that the radar sees something (strongest bins).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcen_radar import RadarConfig
from pcen_radar.acquire import open_device, next_frame, close_device
from pcen_radar.processing import range_fft


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=400)
    a = ap.parse_args()
    cfg = RadarConfig()
    print(cfg.summary())
    print(f"frame = {cfg.num_rx} Rx x {cfg.num_chirps} chirps x {cfg.num_samples} samples, "
          f"chirp block {cfg.num_chirps * cfg.chirp_repetition_time_s * 1e3:.1f} ms of "
          f"{cfg.frame_repetition_time_s * 1e3:.0f} ms frame period")
    dev = open_device(cfg)
    stats = {"dropped": 0}
    ts, prof = [], 0.0
    try:
        t0 = time.perf_counter()
        for i in range(a.frames):
            f = next_frame(dev, stats)
            ts.append(time.perf_counter() - t0)
            if i % 20 == 0:
                prof = prof + np.abs(range_fft(f[0])).mean(axis=0)
                print(f"\r  frame {i:5d}  dropped {stats['dropped']}", end="", flush=True)
    finally:
        close_device(dev)
    ts = np.array(ts)
    print(f"\n\nframes {a.frames}   dropped {stats['dropped']}   "
          f"mean period {np.diff(ts).mean() * 1e3:.1f} ms (target {cfg.frame_repetition_time_s * 1e3:.0f} ms)   "
          f"max gap {np.diff(ts).max() * 1e3:.0f} ms")
    rr = cfg.range_axis(len(prof))
    top = np.argsort(prof[2:])[::-1][:5] + 2
    print("strongest range bins:", ", ".join(f"{rr[k]:.2f} m" for k in sorted(top)))
    print("OK" if stats["dropped"] == 0 else
          "Frames were dropped: use another USB port (USB 3, no hub), close the Radar Fusion GUI, retry.")


if __name__ == "__main__":
    main()
