"""Breathing session (plan v3, normal breathing): seated person at 1.0 m, fan behind at ~1.3 m (on / off).

  python scripts/analyze_breath.py --data <breath/P03> --repmon <folder containing RepMon> --out results/breath

Signal path (as in the respiration case study, scripts/repmon_case.py): per frame the complex range profile
(range FFT, mean over the 64 chirps, 3 Rx as channels), resampled to 10 Hz on the recorded timestamps;
residual power E(t, r) after causal slow-time static removal (20-s complex running mean, initialised in the
empty room). Front ends on E: SR (E / ref), CA-CFAR along range (guard 2, train 4 bins per side),
PCEN tau = 10 s / 100 s / inf (inf = division by the empty-room background), C-PCEN tau = 10 s (hold 600 s,
kappa 6 dB, fast average 0.5 s); background M0 = mean of E over the empty phase (5-28 s).
Fixed before looking at the data (2026-10-17):
 1 contrast of the person bins (rest vs empty >= 6 dB with SR, fan-off block) vs the median of the other bins,
   20-s average, at 10 / 60 / 120 / 200 s after sitting down; expected: PCEN (10 s) falls to ~0 dB, C-PCEN stays.
 2 localisation in 30-s windows (hop 5 s) of the free-breathing phase: maximum of the 20-s profile (>= 0.3 m)
   within the person bins; share of windows locked on the fan (fan range +-0.15 m).
 3 breathing rate (RepMon DSP, linear demodulation, 6-45 bpm) at the detected bin vs the rate at the person
   bins (oracle range): agreement within 10 %. There is no external reference (normal breathing).
Fan off and on are one block each (no repetition): fan state is confounded with recording order.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pcen_radar.io import load_parts
from pcen_radar.processing import range_fft
from pcen_radar.pcen import pcen_gated
import repmon_case as RC

FS = RC.FS
METHODS = [("SR", "SR", None), ("CA-CFAR", "CA", None), ("PCEN tau=10s", "PCEN", 10.0),
           ("PCEN tau=100s", "PCEN", 100.0), ("PCEN tau=inf", "PCEN", np.inf), ("C-PCEN tau=10s", "CPCEN", 10.0)]


def slowtime(d):
    st = np.stack([range_fft(f).mean(axis=1) for f in d["frames"]])        # (T, rx, R) complex
    t = d["t"] - d["t"][0]
    tu = np.arange(0, t[-1], 1 / FS)
    C, R = st.shape[1], st.shape[2]
    out = np.empty((len(tu), C, R), np.complex128)
    for c in range(C):
        for r in range(R):
            out[:, c, r] = np.interp(tu, t, st[:, c, r].real) + 1j * np.interp(tu, t, st[:, c, r].imag)
    return out, tu                                                             # (Tu, rx, R)


def ca_range(E, guard=2, train=4):
    R = E.shape[1]; out = np.empty_like(E)
    for r in range(R):
        idx = [i for i in range(r - guard - train, r + guard + train + 1) if 0 <= i < R and abs(i - r) > guard]
        out[:, r] = E[:, r] / (E[:, idx].mean(axis=1) + 1e-30)
    return out


def front_end(E, kind, tau, M0, ref):
    if kind == "SR":
        return E / ref
    if kind == "CA":
        return ca_range(E)
    if kind == "PCEN":
        return RC.normalize(E, "PCEN", tau, M0, ref)
    s = 1 - np.exp(-1 / (FS * tau))
    return pcen_gated(E, s=s, ref=ref, M0=M0, hold_frames=int(600 * FS), kappa_db=6.0,
                      s_fast=1 - np.exp(-1 / (FS * 0.5))).astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True); ap.add_argument("--repmon", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "breath"))
    a = ap.parse_args()
    sys.path.insert(0, a.repmon)
    from RepMon.dsp_user import features
    from RepMon.localize import person_cells
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfgd = {"demod": "linear", "band_lo_bpm": 6.0, "band_hi_bpm": 45.0}
    blocks = sorted(Path(a.data).glob("*_labels.json"))
    data = {}
    for lj in blocks:
        d = load_parts(lj); lab = d["labels"]
        st, tu = slowtime(d)
        rm = d["config"].range_axis(st.shape[2])
        ph = {p["name"]: p for p in lab["phases"]}
        sit = ph["settle"]["start_s"]
        e0, e1 = 5.0, ph["empty"]["start_s"] + ph["empty"]["dur_s"] - 2.0
        E, _ = RC.residual_power(st, init=st[int(e0 * FS): int(e0 * FS) + 20].mean(0))
        M0 = E[int(e0 * FS): int(e1 * FS)].mean(0); ref = float(np.median(M0))
        data[lab["block"]] = dict(st=st, E=E, M0=M0, ref=ref, rm=rm, sit=sit, fan=lab["fan"], lab=lab,
                                  free=(ph["free"]["start_s"], ph["free"]["start_s"] + ph["free"]["dur_s"]))
        print(lab["block"], st.shape, f"sit at {sit:.0f} s", flush=True)
    # person bins from the fan-off block with SR (rest vs empty >= 6 dB), else from the first block
    key = next((k for k in data if data[k]["fan"] == "off"), next(iter(data)))
    D0 = data[key]; W = int(20 * FS)
    emp = D0["E"][int(5 * FS): int((D0["sit"] - 2) * FS)].mean(0)
    rst = D0["E"][int((D0["sit"] + 20) * FS): int(D0["free"][1] * FS)].mean(0)
    rm = D0["rm"]
    person = (10 * np.log10(rst / emp) >= 6.0) & (rm >= 0.3)
    pc = int(np.round(np.average(np.where(person)[0], weights=(rst / emp)[person])))
    fan_r = D0["lab"]["fan_range_m"]
    fanbins = np.abs(rm - fan_r) <= 0.15
    half = int(round(0.15 / float(np.median(np.diff(rm)))))
    res = {"_params": {"person_bins_m": rm[person].round(3).tolist(), "person_center_m": float(rm[pc]),
                       "fan_range_m": fan_r, "fs_hz": FS, "methods": [m[0] for m in METHODS]}, "blocks": {}}
    for b, D in data.items():
        E, st = D["E"], D["st"]; sit = D["sit"]; f0, f1 = D["free"]
        r = {"fan": D["fan"], "contrast_db": {}, "loc": {}, "rr": {}, "curves": {}}
        # oracle breathing rate per window (person centre)
        wins = np.arange(f0 + 30, f1 + 1e-9, 5.0)
        orr = []
        for s_end in wins:
            e = int(s_end * FS)
            orr.append(features(person_cells(st[e - int(30 * FS): e], pc, half), cfgd)["rr_dsp"])
        r["rr_oracle_bpm"] = {"median": float(np.median(orr)), "iqr": [float(np.percentile(orr, 25)), float(np.percentile(orr, 75))]}
        for name, kind, tau in METHODS:
            Y = front_end(E, kind, tau, D["M0"], D["ref"])
            secs = np.arange(int(sit - 25), int(f1)); secs = secs[secs * FS >= W]
            P = np.array([Y[int(s * FS) - W: int(s * FS)].mean(0) for s in secs])
            P = P - min(0.0, P.min()) + 1e-12
            c = 10 * np.log10(P[:, person].mean(1) / (np.median(P[:, ~person & (rm >= 0.3)], axis=1)) + 1e-12)
            r["curves"][name] = {"t_after_sit_s": (secs - sit).tolist(), "contrast_db": c.round(2).tolist()}
            r["contrast_db"][name] = {f"{k}s": float(c[np.argmin(np.abs(secs - (sit + k)))]) for k in (10, 60, 120, 200)}
            hit = fanlock = agree = 0
            for k, s_end in enumerate(wins):
                e = int(s_end * FS)
                prof = Y[e - W: e].mean(0).copy(); prof[rm < 0.3] = -np.inf
                rb = int(np.argmax(prof))
                hit += int(person[rb]); fanlock += int(fanbins[rb])
                rr = features(person_cells(st[e - int(30 * FS): e], rb, half), cfgd)["rr_dsp"]
                agree += int(person[rb] and abs(rr - orr[k]) <= 0.1 * orr[k])
            n = len(wins)
            r["loc"][name] = {"person": hit / n, "fan_lock": fanlock / n, "n_windows": n}
            r["rr"][name] = {"localized_and_within10pct_of_oracle": agree / n}
            print(f"  {b} {name:15s} contrast {r['contrast_db'][name]}  loc {hit / n:.2f} fan {fanlock / n:.2f}  rr {agree / n:.2f}", flush=True)
        res["blocks"][b] = r
    (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["_params"]), flush=True)


if __name__ == "__main__":
    main()
