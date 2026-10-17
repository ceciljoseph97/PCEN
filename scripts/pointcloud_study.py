"""Point clouds and dynamic trails with different detection front ends (fan session). POST HOC.

  python scripts/pointcloud_study.py --fan <fan/P01> --out results/pointcloud

Chain: pcen_radar/pointcloud.py (re-implementation of rsazid99/inf_radarpoint for our configuration).
Front ends (detection stage only; spectra, angles and gates identical):
  published      the original detector: complex CA-CFAR per Rx (pfa 0.05, +0.10), AND, SNR > 8 (not calibrated)
  ca_cfar        SR + CA-CFAR on power (same ring as the original), AND over Rx
  global         SR + global threshold on power
  clutter_map    SR + clutter-map CFAR (tau 1 s)
  pcen           SR + PCEN (tau 1 s, alpha .98, delta 2, r .5)
  cpcen          SR + C-PCEN (hold 600 s, kappa 6 dB, fast average 0.5 s)
  pcen_ca        SR + PCEN + CA-CFAR
Calibration: one threshold per front end such that the empty room with the fan OFF (front and side
recordings pooled, after 2 s warm-up) gives TARGET false points per frame. Same false-point budget for all.
Metrics (fan ON unless stated, after warm-up):
  fan points / frame      empty room, fan on, points in the fan gate (all are false)
  Pd / FA window          gesture windows (0.2-1.7 s after the cue) / rest windows with >= 1 point in the
                          hand gate; trail points = mean points per gesture window in the hand gate
  Rx1 delivered only zeros in this session, so azimuth is unavailable: detection uses Rx2+Rx3 (AND), points
  carry range, elevation and radial velocity; the 'published' detector (AND over all 3 Rx) is reported as is.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pcen_radar.io import load_recording
from pcen_radar.pcen import pcen, pcen_gated, s_from_tau
from pcen_radar.cfar import clutter_map_stat
from pcen_radar.pointcloud import PointChain
from analyze_fan import windows, WARM_S

TARGET = 0.1          # false points per frame, empty room, fan off
RX = [1, 2]           # Rx used for detection (Rx1 = index 0 is all zeros in the fan session)
METHODS = ["ca_cfar", "global", "clutter_map", "pcen", "cpcen", "pcen_ca"]


def stats(P, method, ch, fpm, ref):
    """P (T, rx, dop, rng) power -> statistic, min over Rx: (T, dop, rng)."""
    s1 = s_from_tau(1.0, fpm)
    if method == "ca_cfar":
        S = np.stack([[ch.ca_stat(P[t, i]) for i in range(P.shape[1])] for t in range(len(P))])
    elif method == "global":
        S = P / ref
    elif method == "clutter_map":
        S = clutter_map_stat(P, w=s1)
    elif method == "pcen":
        S = pcen(P, s=s1, ref=ref)
    elif method == "cpcen":
        S = pcen_gated(P, s=s1, ref=ref, hold_frames=int(600 / fpm), kappa_db=6.0, s_fast=s_from_tau(0.5, fpm))
    elif method == "pcen_ca":
        Q = pcen(P, s=s1, ref=ref).astype(np.float64); Q = Q - Q.min() + 1e-6
        S = np.stack([[ch.ca_stat(Q[t, i]) for i in range(Q.shape[1])] for t in range(len(Q))])
    else:
        raise ValueError(method)
    return S[:, RX].min(axis=1).astype(np.float32)


def prepare(d, ch, static_removal=True):
    X = ch.rd_sequence(d["frames"], static_removal)                         # (T, rx, dop, rng) complex
    P = (np.abs(X) ** 2).astype(np.float32)
    G = np.empty(P.shape[:1] + P.shape[2:], bool); AZ = np.empty(G.shape, np.float32); EL = np.empty(G.shape, np.float32)
    for t in range(len(X)):
        G[t], AZ[t], EL[t] = ch.gates(X[t])
    pub = np.stack([ch.points_published(X[t])[0] for t in range(len(X))])
    return P, G, AZ, EL, pub


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fan", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "pointcloud"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.fan).glob("*.npz"))
    recs = {f.stem: load_recording(f) for f in files}
    cfg = next(iter(recs.values()))["config"]
    fp = cfg.frame_repetition_time_s
    fpm = float(np.median([d["meta"].get("measured_frame_period_s", fp) for d in recs.values()]))
    ch = PointChain(cfg, use_azimuth=False)
    w = int(WARM_S / fp)
    L = {n: d["labels"] for n, d in recs.items()}
    empties_off = [n for n in L if L[n]["gesture"] == "empty" and L[n]["fan"] == "off"]

    # pass 1: reference power and thresholds from the empty room with the fan off
    print("pass 1: thresholds", flush=True)
    prep_off = {n: prepare(recs[n], ch) for n in empties_off}
    ref = float(np.median(np.concatenate([prep_off[n][0][w:].ravel()[::50] for n in empties_off])))
    thr = {}
    for m in METHODS:
        Z = {n: stats(prep_off[n][0], m, ch, fpm, ref) for n in empties_off}
        vals = np.concatenate([Z[n][w:][prep_off[n][1][w:]] for n in empties_off])
        nfr = sum(len(Z[n]) - w for n in empties_off)
        k = max(1, int(round(TARGET * nfr)))
        thr[m] = float(np.sort(vals)[-k])
        print(f"thr {m:12s} {thr[m]:.4g}", flush=True)
    del prep_off

    rr = ch.ranges
    res = {"_params": {"target_false_points_per_frame": TARGET, "frame_period_s": fpm, "warmup_s": WARM_S,
                       "range_bin_m": float(rr[1] - rr[0]), "thresholds": thr, "ref_power": ref,
                       "chain": "re-implementation of rsazid99/inf_radarpoint, see pcen_radar/pointcloud.py"}}
    pts, Tn = {}, {}
    allm = ["published"] + METHODS
    print("pass 2: points", flush=True)
    for n in recs:
        P, G, AZ, EL, pub = prepare(recs[n], ch); Tn[n] = len(P)
        for m in allm:
            M = pub if m == "published" else (stats(P, m, ch, fpm, ref) > thr[m]) & G
            t, di, ri = np.nonzero(M)
            pts[(n, m)] = np.stack([t, rr[ri], AZ[t, di, ri], EL[t, di, ri], ch.velocities[di]], 1).astype(np.float32)
        del P, G, AZ, EL, pub
        print("  ", n, flush=True)

    def gate_mask(p, g):
        return (p[:, 1] >= g[0]) & (p[:, 1] <= g[1])

    # hand side of the fan (sign of azimuth) from fan-off side gestures, any method (use global)
    side_off = [n for n in L if L[n]["position"] == "side" and L[n]["fan"] == "off" and L[n]["gesture"] != "empty"]
    az_hand = []
    for n in side_off:
        p = pts[(n, "global")]; lab = L[n]
        for (w0, w1, pos, kind) in windows(lab, Tn[n], fp):
            if pos:
                q = p[(p[:, 0] >= w0) & (p[:, 0] < w1) & gate_mask(p, lab["hand_gate_m"])]
                az_hand += q[:, 3].tolist()
    fan_on_empty = {pos: next(n for n in L if L[n]["gesture"] == "empty" and L[n]["fan"] == "on" and L[n]["position"] == pos)
                    for pos in ["front", "side"]}
    az_fan = pts[(fan_on_empty["side"], "global")]
    az_fan = az_fan[gate_mask(az_fan, L[fan_on_empty["side"]]["fan_gate_m"]), 3]
    res["elevation_deg"] = {"hand_side_fan_off_median": float(np.degrees(np.median(az_hand))) if az_hand else None,
                          "fan_empty_median": float(np.degrees(np.median(az_fan))) if len(az_fan) else None}
    split = 0.5 * (np.median(az_hand) + np.median(az_fan)) if az_hand and len(az_fan) else 0.0
    hand_sign = np.sign(np.median(az_hand) - split) if az_hand else 1.0
    res["elevation_deg"]["split"] = float(np.degrees(split))

    table = {}
    for m in allm:
        row = {}
        for n in recs:
            lab = L[n]; T = Tn[n]; p = pts[(n, m)]; p = p[p[:, 0] >= w]
            key = f"{lab['position']}_{lab['fan']}"
            if lab["gesture"] == "empty":
                row[f"{key}_empty_points_per_frame"] = float(len(p) / (T - w))
                row[f"{key}_empty_fan_gate_points_per_frame"] = float(gate_mask(p, lab["fan_gate_m"]).sum() / (T - w))
                continue
            g = row.setdefault(f"{key}_gest", {"gw": 0, "gw_det": 0, "rw": 0, "rw_det": 0, "trail_pts": 0,
                                                "hand_side": 0, "in_gate": 0})
            for (w0, w1, pos, kind) in windows(lab, T, fp):
                q = p[(p[:, 0] >= w0) & (p[:, 0] < w1) & gate_mask(p, lab["hand_gate_m"])]
                if pos:
                    g["gw"] += 1; g["gw_det"] += int(len(q) > 0); g["trail_pts"] += len(q); g["in_gate"] += len(q)
                    g["hand_side"] += int((np.sign(q[:, 3] - split) == hand_sign).sum())
                else:
                    g["rw"] += 1; g["rw_det"] += int(len(q) > 0)
        for k, g in list(row.items()):
            if isinstance(g, dict):
                row[k] = {"Pd_window": g["gw_det"] / g["gw"], "FA_window": g["rw_det"] / max(g["rw"], 1),
                          "trail_points_per_gesture": g["trail_pts"] / g["gw"],
                          "hand_side_share": g["hand_side"] / g["in_gate"] if g["in_gate"] else None,
                          "n_gesture_windows": g["gw"], "n_rest_windows": g["rw"]}
        table[m] = row
        print(m, json.dumps({k: (round(v, 3) if isinstance(v, float) else {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}) for k, v in row.items()}), flush=True)
    res["metrics"] = table
    (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez_compressed(out / "points.npz", **{f"{n}__{m}": v for (n, m), v in pts.items()})
    print("done", flush=True)


if __name__ == "__main__":
    main()
