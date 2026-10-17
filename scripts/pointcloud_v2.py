"""Point clouds / dynamic trails on fan session 2 (all 3 Rx: azimuth + elevation). POST HOC.

  python scripts/pointcloud_v2.py --fan <fan/P02> --out results/pointcloud_v2

Chain: pcen_radar/pointcloud.py (re-implementation of rsazid99/inf_radarpoint). Detection: statistic > threshold
in all 3 Rx, threshold per front end at TARGET false points per frame in the empty room with the fan off
(front and near pooled). Front ends with static removal (moving gestures) and without (hold: a still hand is
erased by static removal): raw_global, raw_pcen, raw_cpcen.
Metrics: fan points/frame in the fan gate (empty room, fan on); gesture / rest windows with >= 1 point in the
hand gate (Pd / FA window); points per gesture window; hold: still-phase windows (1.5-3.5 s) vs hand lowered.
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
from pcen_radar.pointcloud import PointChain
import pointcloud_study as PS
from analyze_fan_v2 import windows

TARGET = 0.1
SR_M = ["ca_cfar", "global", "clutter_map", "pcen", "cpcen", "pcen_ca"]
RAW_M = ["raw_global", "raw_pcen", "raw_cpcen"]


def stat(P, m, ch, fpm, ref):
    if m in SR_M:
        return PS.stats(P, m, ch, fpm, ref)
    s1 = s_from_tau(1.0, fpm)
    if m == "raw_global":
        S = P / ref
    elif m == "raw_pcen":
        S = pcen(P, s=s1, ref=ref)
    else:
        S = pcen_gated(P, s=s1, ref=ref, hold_frames=int(600 / fpm), kappa_db=6.0, s_fast=s_from_tau(0.5, fpm))
    return S.min(axis=1).astype(np.float32)


class LazyRecs(dict):
    """Recording loader that keeps only labels/meta in memory and loads frames on access."""
    def __init__(self, files):
        super().__init__()
        import json as _j
        for f in files:
            z = np.load(f, allow_pickle=False)
            self[Path(f).stem] = {"_path": f, "labels": _j.loads(str(z["labels"])), "meta": _j.loads(str(z["meta"]))}
        self._cfg = load_recording(files[0])["config"]

    def full(self, n):
        return load_recording(self[n]["_path"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fan", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "pointcloud_v2"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.fan).glob("*.npz"))
    recs = LazyRecs(files)
    L = {n: d["labels"] for n, d in recs.items()}
    cfg = recs._cfg
    fpm = float(np.median([d["meta"]["measured_frame_period_s"] for d in recs.values()]))
    PS.RX = [0, 1, 2]
    ch = PointChain(cfg, use_azimuth=True)
    rr = ch.ranges; w = int(round(2.0 / fpm))
    off = [n for n in L if L[n]["gesture"] == "empty" and L[n]["fan"] == "off"]
    thr, ref = {}, {}
    for sr, ms in ((True, SR_M), (False, RAW_M)):
        prep = {n: PS.prepare(recs.full(n), ch, static_removal=sr) for n in off}
        ref[sr] = float(np.median(np.concatenate([prep[n][0][w:].ravel()[::50] for n in off])))
        for m in ms:
            Z = {n: stat(prep[n][0], m, ch, fpm, ref[sr]) for n in off}
            v = np.concatenate([Z[n][w:][prep[n][1][w:]] for n in off]); nfr = sum(len(Z[n]) - w for n in off)
            thr[m] = float(np.sort(v)[-max(1, int(round(TARGET * nfr)))])
            print(f"thr {m:12s} {thr[m]:.4g}", flush=True)
        del prep
    pts, Tn = {}, {}
    for n in recs:
        for sr, ms in ((True, ["published"] + SR_M), (False, RAW_M)):
            if not sr and L[n]["gesture"] not in ("hold", "empty"):
                continue
            P, G, AZ, EL, pub = PS.prepare(recs.full(n), ch, static_removal=sr); Tn[n] = len(P)
            for m in ms:
                M = pub if m == "published" else (stat(P, m, ch, fpm, ref[sr]) > thr[m]) & G
                t, di, ri = np.nonzero(M)
                pts[(n, m)] = np.stack([t, rr[ri], AZ[t, di, ri], EL[t, di, ri], ch.velocities[di]], 1).astype(np.float32)
            del P, G, AZ, EL, pub
        print("  ", n, flush=True)
    met = {}
    for m in ["published"] + SR_M + RAW_M:
        row = {}
        for n in recs:
            if (n, m) not in pts:
                continue
            lab = L[n]; p = pts[(n, m)]; p = p[p[:, 0] >= w]; T = Tn[n]
            key = f"{lab['position']}_{lab['fan']}_{lab['gesture']}"
            if lab["gesture"] == "empty":
                fg = (p[:, 1] >= lab["fan_gate_m"][0]) & (p[:, 1] <= lab["fan_gate_m"][1])
                row[key] = {"points_per_frame": len(p) / (T - w), "fan_gate_points_per_frame": float(fg.sum() / (T - w))}
                continue
            hg = lab["hand_gate_m"]; gd = rd = gw = rw = tp = allp = 0; lat = []
            for (s0, s1, pos, kind) in windows(lab, T, fpm):
                q = p[(p[:, 0] >= s0) & (p[:, 0] < s1)]
                h = q[(q[:, 1] >= hg[0]) & (q[:, 1] <= hg[1])]
                if pos:
                    gw += 1; gd += int(len(h) > 0); tp += len(h); allp += len(q)
                    if len(h) > 2:
                        lat.append(float(np.std(h[:, 1] * np.cos(h[:, 3]) * np.sin(h[:, 2]))))
                else:
                    rw += 1; rd += int(len(h) > 0)
            row[key] = {"Pd_window": gd / gw, "FA_window": rd / max(rw, 1), "hand_points_per_window": tp / gw,
                        "hand_share_of_points": tp / allp if allp else None,
                        "lateral_spread_m": float(np.median(lat)) if lat else None}
        met[m] = row
        print(m, json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in row.items()}), flush=True)
    res = {"_params": {"target_false_points_per_frame": TARGET, "frame_period_s": fpm, "thresholds": thr,
                       "ref": {str(k): v for k, v in ref.items()}}, "metrics": met}
    (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez_compressed(out / "points.npz", **{f"{n}__{m}": v for (n, m), v in pts.items()})
    print("done", flush=True)


if __name__ == "__main__":
    main()
