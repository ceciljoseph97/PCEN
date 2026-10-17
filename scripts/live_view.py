"""Live side-by-side view: static removal | CA-CFAR statistic | PCEN, straight from the BGT60TR13C.
Use it to set up scenes (where is the clutter? is the pendulum visible?) before recording.

  python scripts/live_view.py              # default config
  python scripts/live_view.py --tau 2.0    # slower PCEN adaptation
  python scripts/live_view.py --file data/roomA/walk_*.npz   # replay a recording instead
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcen_radar import RadarConfig, range_doppler, ca_cfar_stat
from pcen_radar.pcen import ClutterMapState, s_from_tau
from pcen_radar.processing import to_db


def frame_source(a, cfg):
    if a.file:
        from pcen_radar.io import load_recording
        d = load_recording(a.file)
        for f in d["frames"]:
            yield f
        return
    import queue
    import time
    from pcen_radar.acquire import FrameReader
    rd = FrameReader(cfg).start()                  # radar read in a background thread
    t_last, t_print, shown = time.perf_counter(), time.perf_counter(), 0
    try:
        while True:
            try:
                f = rd.latest(timeout=0.1)
            except queue.Empty:
                plt.pause(0.05)                    # keep the window responsive while waiting
                if time.perf_counter() - t_last > 3:
                    print(f"\n  no frame from the radar for {time.perf_counter() - t_last:.0f} s "
                          f"(read {rd.stats['read']}, dropped {rd.stats['dropped']}, "
                          f"reader alive {rd._t.is_alive()})", flush=True)
                    t_last = time.perf_counter()
                if not plt.get_fignums() and shown:
                    return
                continue
            t_last = time.perf_counter(); shown += 1
            if t_last - t_print > 2:
                print(f"\r  radar frames read {rd.stats['read']:6d}  dropped {rd.stats['dropped']:3d}  "
                      f"displayed {shown:6d}", end="", flush=True)
                t_print = t_last
            yield f
    finally:
        rd.stop()
        print(f"\nframes read {rd.stats['read']}, dropped {rd.stats['dropped']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.98)
    a = ap.parse_args()
    cfg = RadarConfig()
    if a.file:
        from pcen_radar.io import load_recording
        cfg = load_recording(a.file)["config"]
    print(cfg.summary())
    st = ClutterMapState(s=s_from_tau(a.tau, cfg.frame_repetition_time_s), alpha=a.alpha)

    plt.ion()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    ext = [cfg.velocity_axis()[0], cfg.velocity_axis()[-1], 0, cfg.max_range_m]
    ims = None
    for i, f in enumerate(frame_source(a, cfg)):
        P_raw = range_doppler(f, clutter="none")
        P_mr = range_doppler(f, clutter="mean")
        imgs = [to_db(P_mr), 10 * np.log10(np.maximum(ca_cfar_stat(P_mr), 1e-3)), st.step(P_raw)]
        if ims is None:
            ims = [ax.imshow(x, aspect="auto", origin="lower", extent=ext) for ax, x in zip(axs, imgs)]
            for ax, t in zip(axs, ["static removal [dB]", "CA-CFAR stat [dB]", f"PCEN (tau={a.tau}s)"]):
                ax.set_title(t); ax.set_xlabel("v [m/s]"); ax.set_ylabel("R [m]")
        for im, x in zip(ims, imgs):
            im.set_data(x)
            im.set_clim(np.percentile(x, 5), np.percentile(x, 99.9))
        fig.suptitle(f"frame {i}")
        plt.pause(0.001)
        if not plt.fignum_exists(fig.number):
            break


if __name__ == "__main__":
    main()
