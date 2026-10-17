"""Synthetic activity dataset with 3 'rooms' (sessions) that differ in gain and clutter level,
so the cross-session classifier pipeline can be tested before real recordings exist.

  python scripts/make_synthetic_dataset.py --out data/synth --clips 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcen_radar import RadarConfig
from pcen_radar.io import save_recording
from pcen_radar.simulate import simulate_activity

SESSIONS = {"roomA": (1.0, 1.0), "roomB": (0.5, 2.5), "roomC": (2.0, 0.4)}   # (gain, clutter)
CLASSES = ["empty", "still", "walk", "wave"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--clips", type=int, default=30, help="clips per class per session")
    ap.add_argument("--frames", type=int, default=96)
    a = ap.parse_args()
    cfg = RadarConfig()
    seed = 0
    for ses, (g, c) in SESSIONS.items():
        for cls in CLASSES:
            for k in range(a.clips):
                fr = simulate_activity(cls, cfg, a.frames, seed=seed, session_gain=g, session_clutter=c)
                save_recording(Path(a.out) / ses / cls / f"{cls}_{k:03d}.npz", fr, cfg,
                               labels={"class": cls, "session": ses})
                seed += 1
        print("done", ses)


if __name__ == "__main__":
    main()
