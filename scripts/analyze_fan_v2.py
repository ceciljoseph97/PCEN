"""Session 2 (plan v2) of the fan experiment: push-pull, circle, hold at "front" (0.4 m) and "near"
(0.8 m, directly in front of the fan at 1.0 m), fan off / on, ABBA. All 3 Rx valid (signed int16 storage).

  python scripts/analyze_fan_v2.py --data <fan/P02> --out results/fan_v2

Fixed before looking at the results (2026-10-17, after recording):
 T1 moving gestures (push-pull + circle pooled), near, fan on: SR + PCEN (tau 1 s) has a higher window AUC
    than SR + CA-CFAR and than SR + global threshold (2 one-sided paired bootstrap tests, Bonferroni 0.025).
 T2 hold, near, fan on, front ends WITHOUT static removal (static removal erases a still hand):
    C-PCEN > PCEN and C-PCEN > global threshold in AUC of still-phase windows (1.5-3.5 s after GO) vs
    hand-lowered windows (4.0-4.9 s) (2 tests, Bonferroni 0.025).
 Reported without tests: the same at "front" and with the fan off, Pd at 5 % window false alarms, fan
 false-alarm increase in the empty room (threshold for Pfa 1e-3 with the fan off), all front ends.
Window score = maximum statistic in the hand gate (front 0.20-0.65 m, near 0.60-0.90 m); SR front ends use
the non-zero Doppler bins (as in session 1), raw front ends all Doppler bins.
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
from pcen_radar.pcen import pcen, pcen_gated, s_from_tau
from pcen_radar.cfar import ca_cfar_stat, clutter_map_stat
from analyze_fan import auc

WARM_S = 2.0
SR_METHODS = ["sr_global", "sr_cacfar", "sr_clutmap", "sr_pcen1", "sr_pcen013", "sr_pcen1_ca", "sr_cpcen"]
RAW_METHODS = ["raw_global", "raw_clutmap", "raw_pcen1", "raw_cpcen"]
NAMES = {"sr_global": "SR + global", "sr_cacfar": "SR + CA-CFAR", "sr_clutmap": "SR + clutter map",
         "sr_pcen1": "SR + PCEN (1 s)", "sr_pcen013": "SR + PCEN (0.13 s)", "sr_pcen1_ca": "SR + PCEN + CA-CFAR",
         "sr_cpcen": "SR + C-PCEN", "raw_global": "global (no SR)", "raw_clutmap": "clutter map (no SR)",
         "raw_pcen1": "PCEN (no SR)", "raw_cpcen": "C-PCEN (no SR)"}


def stats(P, kind, fp, ref):
    s1, s013 = s_from_tau(1.0, fp), s_from_tau(0.13, fp)
    sf = s_from_tau(0.5, fp); hold = int(600 / fp)
    out = {}
    pre = "sr_" if kind == "sr" else "raw_"
    out[pre + "global"] = (P / ref).astype(np.float32)
    out[pre + "clutmap"] = clutter_map_stat(P, w=s1)
    out[pre + "pcen1"] = pcen(P, s=s1, ref=ref)
    out[pre + "cpcen"] = pcen_gated(P, s=s1, ref=ref, hold_frames=hold, kappa_db=6.0, s_fast=sf)
    if kind == "sr":
        out["sr_cacfar"] = np.stack([ca_cfar_stat(x) for x in P])
        out["sr_pcen013"] = pcen(P, s=s013, ref=ref)
        out["sr_pcen1_ca"] = np.stack([ca_cfar_stat(x - x.min() + 1e-6) for x in out["sr_pcen1"]])
    return out


def windows(lab, n, fp):
    """(start, stop, positive, kind); hold: positive = still phase."""
    w = []
    if lab["gesture"] == "empty":
        L = int(round(1.5 / fp)); s = int(round(WARM_S / fp))
        while s + L <= n:
            w.append((s, s + L, 0, "empty")); s += L
        return w
    gw = lab["still_window_s"] if lab["gesture"] == "hold" else lab["gesture_window_s"]
    rw = lab["rest_window_s"]
    for c in lab["cue_frames"]:
        a, b = c + int(round(gw[0] / fp)), c + int(round(gw[1] / fp))
        if b <= n:
            w.append((a, b, 1, "gesture"))
        a, b = c + int(round(rw[0] / fp)), c + int(round(rw[1] / fp))
        if b <= n:
            w.append((a, b, 0, "rest"))
    return w


def gate(rr, g):
    return (rr >= g[0]) & (rr <= g[1])


def boot_diff(P, N, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        ip = rng.integers(0, len(P[a]), len(P[a])); i_n = rng.integers(0, len(N[a]), len(N[a]))
        d.append(auc(P[a][ip], N[a][i_n]) - auc(P[b][ip], N[b][i_n]))
    d = np.array(d)
    return {"diff": auc(P[a], N[a]) - auc(P[b], N[b]), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_one_sided": float((np.sum(d <= 0) + 1) / (len(d) + 1))}


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
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "fan_v2"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    recs = LazyRecs(sorted(Path(a.data).glob("*.npz")))
    L = {n: d["labels"] for n, d in recs.items()}
    cfg = recs._cfg
    fpm = float(np.median([d["meta"]["measured_frame_period_s"] for d in recs.values()]))
    w0 = int(round(WARM_S / fpm))
    print(f"{len(recs)} recordings, measured frame period {fpm * 1e3:.1f} ms", flush=True)
    off_empty = [n for n in L if L[n]["gesture"] == "empty" and L[n]["fan"] == "off"]
    ref = {}
    for k, cl in (("sr", "mean"), ("raw", "none")):
        ref[k] = float(np.median(np.concatenate([rd_sequence(recs.full(n)["frames"], clutter=cl)[w0:].ravel()
                                                 for n in off_empty])))
    allm = SR_METHODS + RAW_METHODS
    rows, S_empty, hold_curve, T_n = [], {}, {}, {}
    for n in list(recs):
        fr = recs.full(n)["frames"]
        Psr = rd_sequence(fr, clutter="mean"); Praw = rd_sequence(fr, clutter="none"); del fr
        R, D = Psr.shape[1:]
        rr = cfg.range_axis(R); nz = np.r_[0:D // 2 - 1, D // 2 + 2:D]
        Sn = {**stats(Psr, "sr", fpm, ref["sr"]), **stats(Praw, "raw", fpm, ref["raw"])}
        lab = L[n]; hg = gate(rr, lab["hand_gate_m"]); T_n[n] = len(Psr)
        for (s0, s1, pos, kind) in windows(lab, len(Psr), fpm):
            r = {"block": n, "position": lab["position"], "fan": lab["fan"], "gesture": lab["gesture"], "pos": pos, "kind": kind}
            for m in allm:
                X = Sn[m][s0:s1][:, hg]
                r[m] = float((X[:, :, nz] if m.startswith("sr_") else X).max())
            rows.append(r)
        if lab["gesture"] == "empty":
            S_empty[n] = Sn
        if lab["gesture"] == "hold":
            cyc = int(round(lab["cycle_s"] / fpm))
            segs = {m: [] for m in RAW_METHODS + ["sr_pcen1", "sr_cpcen"]}
            for c in lab["cue_frames"]:
                if c + cyc <= len(Psr):
                    for m in segs:
                        X = Sn[m][c:c + cyc][:, hg]
                        segs[m].append((X[:, :, nz] if m.startswith("sr_") else X).max(axis=(1, 2)))
            hold_curve[n] = {m: np.mean(v, axis=0).tolist() for m, v in segs.items()}
        del Sn, Psr, Praw
        print("  ", n, flush=True)
    res = {"_params": {"frame_period_s": fpm, "warmup_s": WARM_S, "hand_gates": {k: L[k]["hand_gate_m"] for k in L},
                       "fan_gate_m": L[n]["fan_gate_m"], "ref": ref, "range_bin_m": float(rr[1] - rr[0])}, "auc": {}, "pd_at_far5": {}, "n": {}}

    def sel(group, pos_, fan):
        gs = ["pushpull", "circle"] if group == "moving" else ["hold"]
        return [r for r in rows if r["position"] == pos_ and r["fan"] == fan and r["gesture"] in gs]
    for group in ["moving", "hold"]:
        for pos_ in ["front", "near"]:
            for fan in ["off", "on"]:
                rs = sel(group, pos_, fan); key = f"{group}_{pos_}_{fan}"
                Pw = {m: np.array([r[m] for r in rs if r["pos"]]) for m in allm}
                Nw = {m: np.array([r[m] for r in rs if not r["pos"]]) for m in allm}
                res["n"][key] = [len(Pw[allm[0]]), len(Nw[allm[0]])]
                res["auc"][key] = {m: auc(Pw[m], Nw[m]) for m in allm}
                res["pd_at_far5"][key] = {m: float((Pw[m] > np.quantile(Nw[m], 0.95)).mean()) for m in allm}
                if key == "moving_near_on":
                    res["T1"] = {f"sr_pcen1_vs_{b}": boot_diff(Pw, Nw, "sr_pcen1", b, seed=1) for b in ["sr_cacfar", "sr_global"]}
                    res["T1_extra"] = {f"sr_cpcen_vs_sr_pcen1": boot_diff(Pw, Nw, "sr_cpcen", "sr_pcen1", seed=2)}
                if key == "hold_near_on":
                    res["T2"] = {f"raw_cpcen_vs_{b}": boot_diff(Pw, Nw, "raw_cpcen", b, seed=3) for b in ["raw_pcen1", "raw_global"]}
                if key == "hold_front_on":
                    res["T2_front"] = {f"raw_cpcen_vs_{b}": boot_diff(Pw, Nw, "raw_cpcen", b, seed=4) for b in ["raw_pcen1", "raw_global"]}
    for t in ("T1", "T2"):
        for k in res[t]:
            res[t][k]["significant_bonferroni"] = res[t][k]["p_one_sided"] < 0.025
    # fan false alarms, empty room
    fg = gate(rr, res["_params"]["fan_gate_m"])
    res["fan_fa_increase_db"] = {}
    for pos_ in ["front", "near"]:
        off = next(n for n in L if L[n]["gesture"] == "empty" and L[n]["fan"] == "off" and L[n]["position"] == pos_)
        on = next(n for n in L if L[n]["gesture"] == "empty" and L[n]["fan"] == "on" and L[n]["position"] == pos_)
        res["fan_fa_increase_db"][pos_] = {}
        for m in allm:
            dsel = nz if m.startswith("sr_") else slice(None)
            So = S_empty[off][m][w0:][:, :, dsel]; Sn_ = S_empty[on][m][w0:][:, fg][:, :, dsel]
            thr = np.quantile(So.ravel(), 1 - 1e-3)
            res["fan_fa_increase_db"][pos_][m] = float(10 * np.log10(max((Sn_ > thr).mean(), 1e-6) / 1e-3))
    res["hold_curve"] = hold_curve
    (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    # summary
    Ls = [f"# Fan session 2 ({Path(a.data).name})", "", f"frame period {fpm * 1e3:.1f} ms", ""]
    for key in res["auc"]:
        Ls.append(f"## {key}  (pos/neg windows {res['n'][key]})")
        Ls += [f"- {NAMES[m]:22s} AUC {res['auc'][key][m]:.3f}  Pd@5% {res['pd_at_far5'][key][m]:.2f}" for m in allm]
        Ls.append("")
    Ls.append("## Tests"); Ls.append("")
    for t in ("T1", "T1_extra", "T2", "T2_front"):
        for k, v in res[t].items():
            Ls.append(f"- {t} {k}: dAUC {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] p={v['p_one_sided']:.4f}"
                      + (f"  significant={v['significant_bonferroni']}" if 'significant_bonferroni' in v else ""))
    Ls += ["", "## Fan false-alarm increase [dB], empty room"]
    for pos_, d in res["fan_fa_increase_db"].items():
        Ls.append(f"- {pos_}: " + ", ".join(f"{NAMES[m]} {v:+.0f}" for m, v in d.items()))
    (out / "results.md").write_text("\n".join(Ls), encoding="utf-8")
    print("\n".join(Ls))


if __name__ == "__main__":
    main()
