"""Record raw frames from the BGT60TR13C with labels for benchmarking.

Examples
  # empty room, 30 s at 20 Hz -> false-alarm statistics only
  python scripts/record.py --session roomA --name empty --frames 600

  # person walking back and forth between 1.5 m and 3.0 m for the whole recording
  python scripts/record.py --session roomA --name walk --frames 600 --target-range 1.5 3.0 --class walk

  # dwell test: walk in (frames 0-200), stand still at 2 m (200-400), walk out (400-600)
  python scripts/record.py --session roomA --name dwell --frames 600 \
      --target-range 1.0 3.5 --moving 0 200 --moving 400 -1

  # pendulum / corner reflector swinging at 2.0-2.3 m (repeatable detection target)
  python scripts/record.py --session roomA --name pendulum --frames 600 --target-range 2.0 2.3

Files go to data/<session>/<name>_<YYYYmmdd_HHMMSS>.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcen_radar import RadarConfig
from pcen_radar.acquire import record
from pcen_radar.io import save_recording


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True, help="room/day identifier, used for cross-session splits")
    ap.add_argument("--name", required=True)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--config", help="JSON file with RadarConfig fields to override")
    ap.add_argument("--class", dest="cls", help="activity label for classification")
    ap.add_argument("--target-range", nargs=2, type=float, metavar=("RMIN", "RMAX"),
                    help="range gate the target stays in [m] (for the whole recording)")
    ap.add_argument("--target-span", nargs=2, type=int, default=[0, -1], metavar=("F0", "F1"),
                    help="frames during which the target is present (default: all)")
    ap.add_argument("--moving", nargs=2, type=int, action="append", metavar=("F0", "F1"),
                    help="frame span where the target moves (repeatable). Default: whole target span")
    ap.add_argument("--notes", default="")
    ap.add_argument("--countdown", type=int, default=3)
    ap.add_argument("--root", default="data")
    a = ap.parse_args()

    cfg = RadarConfig()
    if a.config:
        cfg = RadarConfig.from_dict(cfg.to_dict() | json.loads(Path(a.config).read_text()))
    print(cfg.summary())
    if a.target_range and a.target_range[1] > cfg.max_range_m:
        print(f"WARNING: target gate beyond max range {cfg.max_range_m:.2f} m")

    labels = {"session": a.session}
    if a.cls:
        labels["class"] = a.cls
    if a.target_range:
        labels["target_range_m"] = [[a.target_span[0], a.target_span[1], *a.target_range]]
        if a.moving:
            labels["moving"] = [list(m) for m in a.moving]

    for i in range(a.countdown, 0, -1):
        print(f"recording in {i} ...", flush=True)
        time.sleep(1)
    print(f"RECORDING {a.frames} frames ({a.frames * cfg.frame_repetition_time_s:.1f} s)")

    def tick(i, _):
        if i % int(cfg.frame_rate_Hz) == 0:
            print(f"\r  t = {i * cfg.frame_repetition_time_s:5.1f} s", end="", flush=True)

    frames, ts = record(cfg, a.frames, on_frame=tick)
    print()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(a.root) / a.session / f"{a.name}_{stamp}.npz"
    save_recording(out, frames, cfg, ts, labels,
                   meta={"notes": a.notes, "sensor": "BGT60TR13C", "recorded": stamp})
    dt = ts[-1] / max(1, len(ts) - 1)
    print(f"saved {out}  shape={frames.shape}  measured frame period={dt * 1e3:.1f} ms")


if __name__ == "__main__":
    main()
