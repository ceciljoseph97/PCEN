"""How strongly does the radar see the fan? Fan-ON minus fan-OFF moving power per range bin.

  python scripts/fan_check.py --fan-range 0.8                 # live: 10 s fan off, then 10 s fan on
  python scripts/fan_check.py --data C:/radar_data/fan/P01    # from the recorded empty blocks

Rule of thumb for the experiment: the fan should add >= 10 dB of moving (non-zero Doppler) power in
its range gate. At 60 GHz plastic blades reflect weakly; see MEASUREMENT_PROTOCOL.md for fixes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcen_radar import RadarConfig
from pcen_radar.processing import rd_sequence
from pcen_radar.fan_scene import gates


def moving_profile(frames):
    P = rd_sequence(frames, clutter="mean")                 # (T, R, D)
    D = P.shape[2]
    nz = np.r_[0:D // 2 - 1, D // 2 + 2:D]
    return P[:, :, nz].mean(axis=(0, 2))                    # (R,)


def live(cfg, seconds):
    from pcen_radar.acquire import open_device, next_frame, close_device
    n = int(seconds / cfg.frame_repetition_time_s)
    out = {}
    for state in ["off", "on"]:
        input(f"\nSwitch the fan {state.upper()} (nobody in front of the radar), wait 5 s, then press Enter ...")
        dev = open_device(cfg); st = {"dropped": 0}
        try:
            fr = np.stack([next_frame(dev, st) for _ in range(n)])
        finally:
            close_device(dev)
        print(f"  recorded {n} frames, dropped {st['dropped']}")
        out[state] = fr
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="participant folder with recorded blocks (uses the empty blocks)")
    ap.add_argument("--fan-range", dest="fan_range", type=float, default=0.8)
    ap.add_argument("--seconds", type=float, default=10)
    a = ap.parse_args()
    cfg = RadarConfig()
    if a.data:
        from pcen_radar.io import load_recording
        prof = {}
        for state in ["off", "on"]:
            fs = sorted(Path(a.data).glob(f"*_{state}_empty.npz"))
            if not fs:
                sys.exit(f"no *_{state}_empty.npz in {a.data}")
            ps = []
            for f in fs:
                d = load_recording(f); cfg = d["config"]
                ps.append(moving_profile(d["frames"][int(2 / cfg.frame_repetition_time_s):]))
                print(f"  {f.name}")
            prof[state] = np.mean(ps, axis=0)
    else:
        fr = live(cfg, a.seconds)
        prof = {k: moving_profile(v) for k, v in fr.items()}

    rr = cfg.range_axis(len(prof["on"]))
    diff = 10 * np.log10((prof["on"] + 1e-30) / (prof["off"] + 1e-30))
    g = gates(a.fan_range)["fan"]
    print(f"\nfan ON - fan OFF, moving power per range bin (fan gate {g[0]:.2f}-{g[1]:.2f} m marked *)")
    for k in range(len(rr)):
        if rr[k] > 2.5:
            break
        bar = "#" * int(max(0, min(diff[k], 40)))
        mark = "*" if g[0] <= rr[k] <= g[1] else " "
        print(f" {mark}{rr[k]:5.2f} m  {diff[k]:+6.1f} dB  {bar}")
    sel = (rr >= g[0]) & (rr <= g[1])
    c = 10 * np.log10(prof["on"][sel].sum() / prof["off"][sel].sum())
    peak = rr[np.argmax(diff[:int(2.5 / cfg.range_resolution_m)])]
    print(f"\nfan contrast in its gate: {c:+.1f} dB   (strongest increase at {peak:.2f} m)")
    if c < 10:
        print("WEAK: below 10 dB the fan hardly disturbs any method -> boost the fan echo "
              "(see MEASUREMENT_PROTOCOL.md, 'If the fan is weak').")
    else:
        print("OK: the fan is clearly visible.")
    if abs(peak - a.fan_range) > 0.2:
        print(f"NOTE: the strongest increase is at {peak:.2f} m, not at --fan-range {a.fan_range} m. "
              "Re-measure the distance radar -> fan hub.")


if __name__ == "__main__":
    main()
