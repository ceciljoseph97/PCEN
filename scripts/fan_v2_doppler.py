"""Range-time maps and micro-Doppler signatures, fan session 2 (clean data). POST HOC (descriptive).

  python scripts/fan_v2_doppler.py --data <fan/P02> --out results/fan_v2_doppler

For push-pull and circle at the near position (hand 0.8 m, fan at 1.0 m) with the fan on and off, and the
SR front ends global / CA-CFAR / clutter map / PCEN / C-PCEN (statistics as in analyze_fan_v2.py):
  range-time  : per frame the maximum statistic over the non-zero Doppler bins, per range bin; cue-locked mean
                (-0.5 .. 2.5 s) of the 20 repetitions, log10
  micro-Doppler: per frame the mean statistic over the hand bins (0.60-0.90 m), per Doppler bin; cue-locked mean, log10
Fidelity: Pearson correlation between the mean fan-on and the mean fan-off signature of the same gesture after
removing each signature's pre-cue mean per Doppler bin (0-2.5 s after the cue); also per repetition.
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
from pcen_radar.processing import rd_sequence
import analyze_fan_v2 as V

METHODS = ["sr_global", "sr_cacfar", "sr_clutmap", "sr_pcen1", "sr_cpcen"]
PRE, POST = 0.5, 2.5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "fan_v2_doppler"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.data).glob("*.npz"))
    lab = {f.stem: json.loads(str(np.load(f)["labels"])) for f in files}
    fpm = float(np.median([json.loads(str(np.load(f)["meta"]))["measured_frame_period_s"] for f in files]))
    w0 = int(2 / fpm)
    offs = [f for f in files if lab[f.stem]["gesture"] == "empty" and lab[f.stem]["fan"] == "off"]
    ref = float(np.median(np.concatenate([rd_sequence(load_recording(f)["frames"], clutter="mean")[w0:].ravel() for f in offs])))
    blocks = [f for f in files if lab[f.stem]["position"] == "near" and lab[f.stem]["gesture"] in ("pushpull", "circle")]
    pre, post = int(round(PRE / fpm)), int(round(POST / fpm))
    maps, reps = {}, {}
    for f in blocks:
        d = load_recording(f); L = d["labels"]; cfg = d["config"]
        P = rd_sequence(d["frames"], clutter="mean"); R, D = P.shape[1:]
        rr = cfg.range_axis(R); nz = np.r_[0:D // 2 - 1, D // 2 + 2:D]
        hb = (rr >= L["hand_gate_m"][0]) & (rr <= L["hand_gate_m"][1])
        S = V.stats(P, "sr", fpm, ref)
        for m in METHODS:
            X = S[m]
            rt = X[:, :, nz].max(axis=2)                     # (T, R)
            md = X[:, hb].mean(axis=1)                       # (T, D)
            sr, sm = [], []
            for c in L["cue_frames"]:
                if c - pre >= 0 and c + post <= len(X):
                    sr.append(rt[c - pre:c + post]); sm.append(md[c - pre:c + post])
            key = f"{L['gesture']}|{L['fan']}|{m}"
            maps[key + "|rt"] = np.log10(np.maximum(np.mean(sr, 0), 1e-12)).astype(np.float32)
            maps[key + "|md"] = np.log10(np.maximum(np.mean(sm, 0), 1e-12)).astype(np.float32)
            reps[key] = np.log10(np.maximum(np.array(sm), 1e-12)).astype(np.float32)
        print("  ", f.stem, flush=True)
        axes = {"range_m": rr.tolist(), "v_mps": cfg.velocity_axis().tolist() if hasattr(cfg, "velocity_axis") else None}

    def bc(x):                                               # remove pre-cue mean per Doppler bin, keep post-cue part
        return (x - x[:pre].mean(axis=0, keepdims=True))[pre:]
    res = {"_params": {"frame_period_s": fpm, "pre_s": PRE, "post_s": POST, "axes": axes}, "corr_avg": {}, "corr_per_rep": {}}
    for m in METHODS:
        res["corr_avg"][m] = {}; res["corr_per_rep"][m] = {}
        for g in ("pushpull", "circle"):
            on, off = maps[f"{g}|on|{m}|md"], maps[f"{g}|off|{m}|md"]
            res["corr_avg"][m][g] = float(np.corrcoef(bc(on).ravel(), bc(off).ravel())[0, 1])
            t = bc(off).ravel()
            res["corr_per_rep"][m][g] = float(np.mean([np.corrcoef(bc(r).ravel(), t)[0, 1] for r in reps[f"{g}|on|{m}"]]))
        res["corr_avg"][m]["mean"] = float(np.mean([res["corr_avg"][m][g] for g in ("pushpull", "circle")]))
        res["corr_per_rep"][m]["mean"] = float(np.mean([res["corr_per_rep"][m][g] for g in ("pushpull", "circle")]))
        print(m, {k: round(v, 3) for k, v in res["corr_avg"][m].items()}, {k: round(v, 3) for k, v in res["corr_per_rep"][m].items()})
    (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez_compressed(out / "maps.npz", **{k.replace("|", "__"): v for k, v in maps.items()})


if __name__ == "__main__":
    main()
