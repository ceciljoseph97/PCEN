"""3-D point clouds and gesture trails on the public KIT data (BGT60TR13C, all 3 Rx). POST HOC.

  python scripts/kit_pointcloud.py --kit <folder with GM_*.mat> --out results/pointcloud_kit

Same chain and front ends as scripts/pointcloud_study.py (pcen_radar/pointcloud.py, re-implementation of
rsazid99/inf_radarpoint), here with azimuth (Rx1/Rx3) and elevation (Rx3/Rx2) and AND over all 3 Rx.
KIT records an almost continuous chirp stream (600 chirps per 186-ms frame, 0.3-ms chirps); it is cut into
point-cloud frames of 64 chirps (hop 64, 19 ms, ~52 frames/s). 57.5-64.5 GHz, so range bins are 1.07 cm
(2x zero padding) and the unambiguous range is 1.37 m.
Calibration: one threshold per front end so that the no-gesture recordings of setup 1 (training setup,
wooden wall) give TARGET points per frame. Test: setups 4v1 (window behind the hand at 1.0 m) and 7v1
(free space), gestures 0-4 and no gesture. Hand at 0.3 m.
Metrics per test setup: hand points / frame (0.15-0.60 m), frames with a hand point, background points / frame
(range >= 0.75 m), points / frame in the no-gesture recordings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pcen_radar.kit import load_mat, parse_name
from pcen_radar.pointcloud import PointChain
import pointcloud_study as PS

TARGET = 0.1
HAND, BG = (0.15, 0.60), 0.75
NC = 64


def frames_from_stream(x):
    """(3, F, 600, 128) -> (T, 3, 64, 128) frames of 64 chirps from the concatenated chirp stream."""
    s = x.reshape(3, -1, x.shape[-1])
    T = s.shape[1] // NC
    return s[:, : T * NC].reshape(3, T, NC, -1).transpose(1, 0, 2, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "pointcloud_kit"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.kit).glob("GM_*.mat"))
    x0, meta = load_mat(files[0])
    cfg = SimpleNamespace(start_frequency_Hz=meta["lower_frequency_Hz"], end_frequency_Hz=meta["upper_frequency_Hz"],
                          num_samples=x0.shape[-1], sample_rate_Hz=meta["sample_rate_Hz"], num_chirps=NC,
                          chirp_repetition_time_s=meta["chirp_repetition_time_s"])
    fpm = NC * meta["chirp_repetition_time_s"] * meta["frame_repetition_time_s"] / (600 * meta["chirp_repetition_time_s"])
    PS.RX = [0, 1, 2]
    ch = PointChain(cfg, use_azimuth=True)
    rr = ch.ranges
    w = 10                                       # warm-up frames (PCEN initialisation)
    info = {f.stem: parse_name(f.name) for f in files}
    calib = [f for f in files if info[f.stem]["setup"] == "1" and info[f.stem]["gesture"] == "n"]
    test = [f for f in files if info[f.stem]["setup"] in ("4v1", "7v1")]

    def prep(f):
        x, _ = load_mat(f)
        return PS.prepare({"frames": frames_from_stream(x)}, ch)

    print(f"{len(calib)} calibration, {len(test)} test recordings, frame {fpm * 1e3:.1f} ms", flush=True)
    cal = {f.stem: prep(f) for f in calib}
    ref = float(np.median(np.concatenate([c[0][w:].ravel()[::50] for c in cal.values()])))
    thr = {}
    for m in PS.METHODS:
        vals, nfr = [], 0
        for c in cal.values():
            Z = PS.stats(c[0], m, ch, fpm, ref)
            vals.append(Z[w:][c[1][w:]]); nfr += len(Z) - w
        v = np.concatenate(vals); k = max(1, int(round(TARGET * nfr)))
        thr[m] = float(np.sort(v)[-k])
        print(f"thr {m:12s} {thr[m]:.4g}", flush=True)
    del cal
    allm = ["published"] + PS.METHODS
    pts, Tn = {}, {}
    for f in test:
        P, G, AZ, EL, pub = prep(f); Tn[f.stem] = len(P)
        for m in allm:
            M = pub if m == "published" else (PS.stats(P, m, ch, fpm, ref) > thr[m]) & G
            t, di, ri = np.nonzero(M)
            pts[(f.stem, m)] = np.stack([t, rr[ri], AZ[t, di, ri], EL[t, di, ri], ch.velocities[di]], 1).astype(np.float32)
        print("  ", f.stem, flush=True)
    met = {}
    for m in allm:
        row = {}
        for setup in ("4v1", "7v1"):
            g = {"hand_pts": 0, "hand_frames": 0, "bg_pts": 0, "frames": 0, "n_pts": 0, "n_frames": 0,
                 "x_std": [], "y_std": []}
            for f in test:
                if info[f.stem]["setup"] != setup:
                    continue
                p = pts[(f.stem, m)]; p = p[p[:, 0] >= w]; T = Tn[f.stem] - w
                if info[f.stem]["gesture"] == "n":
                    g["n_pts"] += len(p); g["n_frames"] += T
                    continue
                hand = p[(p[:, 1] >= HAND[0]) & (p[:, 1] <= HAND[1])]
                g["hand_pts"] += len(hand); g["frames"] += T
                g["hand_frames"] += len(np.unique(hand[:, 0])); g["bg_pts"] += int((p[:, 1] >= BG).sum())
                if len(hand) > 5:
                    xyz = ch.to_xyz(hand[:, 1], hand[:, 2], hand[:, 3])
                    g["x_std"].append(float(xyz[:, 0].std())); g["y_std"].append(float(xyz[:, 1].std()))
            row[setup] = {"hand_points_per_frame": g["hand_pts"] / g["frames"],
                          "frames_with_hand_point": g["hand_frames"] / g["frames"],
                          "background_points_per_frame": g["bg_pts"] / g["frames"],
                          "no_gesture_points_per_frame": g["n_pts"] / max(g["n_frames"], 1),
                          "trail_lateral_spread_m": float(np.median(g["x_std"])) if g["x_std"] else None,
                          "trail_vertical_spread_m": float(np.median(g["y_std"])) if g["y_std"] else None}
        met[m] = row
        print(m, json.dumps({s: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for s, r in row.items()}), flush=True)
    res = {"_params": {"target_points_per_frame_calibration": TARGET, "frame_period_s": fpm, "hand_gate_m": HAND,
                       "background_from_m": BG, "thresholds": thr, "ref_power": ref,
                       "calibration": [f.stem for f in calib], "test": [f.stem for f in test]},
           "metrics": met}
    (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez_compressed(out / "points.npz", **{f"{n}__{m}": v for (n, m), v in pts.items()})
    print("done", flush=True)


if __name__ == "__main__":
    main()
