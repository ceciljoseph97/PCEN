"""Gated (adaptive) PCEN: evaluation on all data sets + end-to-end respiration check.

  python scripts/adaptive_eval.py --fan <fan/P01> --calib <RepMon session.npz> --walabot <walabot_slowtime>
                                  --repmon <4_code> --out results/adaptive

G-PCEN (pcen_radar.pcen.pcen_gated): PCEN whose background stops adapting in cells where a target is
present (0.5-s energy average >= 6 dB above the background: censored update), for at most 600 s. Parameters fixed
before this evaluation (2026-10-16): kappa = 6 dB, hold = 600 s, everything else as plain PCEN.
Expectations stated before running:
  A1 fan session: G-PCEN keeps the fan suppression of PCEN (fan false-alarm increase ~0 dB) and detects
     gestures beside the fan at least as well as PCEN (AUC, Pd at 5 % window false alarms).
  A2 simulator (walk - stop - walk): G-PCEN detects the stopped target, PCEN does not.
  A3 RepMon, own BGT60 calibration session: G-PCEN keeps the seated person's contrast (like SR),
     PCEN (tau 10 s) loses it.
  A4 Walabot localization: G-PCEN ~ SR, PCEN (tau <= 100 s) fails.
  A5 Respiration end to end: the person is localized by each front end, the breathing rate is then
     estimated with RepMon's DSP (dsp_user.features, linear demodulation) in the cells around the
     detected range. Metric: share of person-windows with a detection AND a rate within 10 % of the
     reference (Walabot: chest band; BGT60: the paced rate 10/15/20/25 bpm), MAE on detected windows.
     Expected: PCEN (finite tau) fails because the person is not localized; G-PCEN ~ SR.
All of this is post hoc relative to the pre-registered studies and is reported as such.
"""

from __future__ import annotations

import argparse
import csv
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
from analyze_fan import windows, auc, gate, WARM_S
import repmon_case as RC

KAPPA_DB, HOLD_S = 6.0, 600.0


# ------------------------------------------------------------------ A1 fan
def run_fan(folder):
    recs = [(f.stem, load_recording(f)) for f in sorted(Path(folder).glob("*.npz"))]
    cfg = recs[0][1]["config"]; fp = cfg.frame_repetition_time_s
    fpm = float(np.median([d["meta"].get("measured_frame_period_s", fp) for _, d in recs]))
    s1 = s_from_tau(1.0, fpm); hold = int(HOLD_S / fpm)
    P_all = {n: rd_sequence(d["frames"], clutter="mean") for n, d in recs}
    ref_name = next(n for n, d in recs if d["labels"]["gesture"] == "empty" and d["labels"]["fan"] == "off")
    ref = float(np.median(P_all[ref_name][int(WARM_S / fp):]))
    R, D = P_all[recs[0][0]].shape[1:]
    rr = cfg.range_axis(R); nz = np.r_[0:D // 2 - 1, D // 2 + 2:D]
    fns = {"SR + global": lambda P: P / ref,
           "SR + CA-CFAR": lambda P: np.stack([ca_cfar_stat(x) for x in P]),
           "SR + clutter-map": lambda P: clutter_map_stat(P, w=s1),
           "SR + PCEN": lambda P: pcen(P, s=s1, ref=ref),
           "SR + G-PCEN": lambda P: pcen_gated(P, s=s1, ref=ref, hold_frames=hold, kappa_db=KAPPA_DB, s_fast=s_from_tau(0.5, fpm))}
    rows, S_cache = [], {}
    for name, d in recs:
        lab = d["labels"]; P = P_all[name]; hand = gate(rr, lab["hand_gate_m"])
        S = {k: f(P) for k, f in fns.items()}
        S_cache[name] = S
        for (w0, w1, pos, kind) in windows(lab, len(P), fp):
            row = dict(position=lab["position"], fan=lab["fan"], gesture=lab["gesture"], pos=pos)
            for k in fns:
                row[k] = float(S[k][w0:w1][:, hand][:, :, nz].max())
            rows.append(row)
    res = {"auc": {}, "pd_at_far5": {}, "fan_fa_increase_db": {}}
    for p in ["front", "side"]:
        for f in ["off", "on"]:
            sel = [r for r in rows if r["position"] == p and r["fan"] == f]
            res["auc"][f"{p}_{f}"] = {}; res["pd_at_far5"][f"{p}_{f}"] = {}
            for k in fns:
                P_ = np.array([r[k] for r in sel if r["pos"]]); N_ = np.array([r[k] for r in sel if not r["pos"]])
                res["auc"][f"{p}_{f}"][k] = auc(P_, N_)
                res["pd_at_far5"][f"{p}_{f}"][k] = float((P_ > np.quantile(N_, 0.95)).mean())
        off = next(n for n, d in recs if d["labels"]["gesture"] == "empty" and d["labels"]["fan"] == "off" and d["labels"]["position"] == p)
        on = next(n for n, d in recs if d["labels"]["gesture"] == "empty" and d["labels"]["fan"] == "on" and d["labels"]["position"] == p)
        fg = gate(rr, recs[0][1]["labels"]["fan_gate_m"]); w = int(WARM_S / fp)
        res["fan_fa_increase_db"][p] = {}
        for k in fns:
            Soff = S_cache[off][k][w:][:, :, nz]; Son = S_cache[on][k][w:][:, fg][:, :, nz]
            thr = np.quantile(Soff.ravel(), 1 - 1e-3)
            res["fan_fa_increase_db"][p][k] = float(10 * np.log10(max((Son > thr).mean(), 1e-6) / 1e-3))
    return res, S_cache, recs, rr, nz, fpm


# ------------------------------------------------------------------ A2 simulator
def run_sim():
    from pcen_radar import RadarConfig
    from pcen_radar.simulate import simulate, Scene
    from pcen_radar.io import per_frame_range_masks
    from pcen_radar.metrics import thr_at_pfa, detection_over_time
    cfg = RadarConfig(); out = {}
    for seed in range(5):
        frames, r_true, v_true = simulate(cfg, Scene(seed=seed))
        P = rd_sequence(frames, clutter="mean"); fp = cfg.frame_repetition_time_s
        s1 = s_from_tau(1.0, fp)
        tm, gm, present = per_frame_range_masks(r_true, cfg, P.shape, 0.15, 0.30, 1)
        ta, _, _ = per_frame_range_masks(r_true, cfg, P.shape, 0.15, 0.30, 0)
        moving = present & (np.abs(v_true) > 1e-3); stopped = present & ~moving
        meth = {"SR + CA-CFAR": np.stack([ca_cfar_stat(x) for x in P]),
                "SR + clutter-map": clutter_map_stat(P, w=s1),
                "SR + PCEN": pcen(P, s=s1),
                "SR + G-PCEN": pcen_gated(P, s=s1, hold_frames=int(HOLD_S / fp), kappa_db=KAPPA_DB, s_fast=s_from_tau(0.5, fp))}
        w = 20
        for k, S in meth.items():
            S, tm_, gm_, ta_ = S[w:], tm[w:], gm[w:], ta[w:]
            thr = thr_at_pfa(S, tm_, gm_, 1e-3)
            det = detection_over_time(S, ta_, thr)
            o = out.setdefault(k, {"pd_moving": [], "pd_stopped": []})
            o["pd_moving"].append(float(det[moving[w:]].mean())); o["pd_stopped"].append(float(det[stopped[w:]].mean()))
    return {k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in out.items()}


# ------------------------------------------------------------------ RepMon helpers
def norm_ext(E, kind, tau, M0, ref):
    if kind == "GPCEN":
        s = 1 - np.exp(-1 / (RC.FS * tau))
        return pcen_gated(E, s=s, ref=ref, M0=M0, hold_frames=int(HOLD_S * RC.FS), kappa_db=KAPPA_DB, s_fast=1 - np.exp(-1 / (RC.FS * 0.5))).astype(np.float64)
    return RC.normalize(E, kind, tau, M0, ref)


REP_METHODS = [("SR", "SR", None), ("PCEN tau=10s", "PCEN", 10.0), ("PCEN tau=100s", "PCEN", 100.0),
               ("PCEN tau=inf", "PCEN", np.inf), ("G-PCEN tau=10s", "GPCEN", 10.0), ("G-PCEN tau=30s", "GPCEN", 30.0)]


def run_bgt60(path, features, person_cells):
    z = np.load(path); st = z["slowtime"].astype(np.complex128); rm = z["range_m"]
    ph = json.loads(str(z["phases"])); t0 = 0; spans = {}
    for p in ph:
        spans[p["name"]] = (t0, t0 + p["dur_s"], p.get("rate_bpm")); t0 += p["dur_s"]
    e0, e1, _ = spans["empty"]; sit = spans["rest"][0]
    E, _ = RC.residual_power(st, init=st[int(e0 * RC.FS): int(e0 * RC.FS) + 20].mean(0))
    M0 = E[int((e0 + 5) * RC.FS): int(e1 * RC.FS)].mean(0); ref = float(np.median(M0))
    W = int(20 * RC.FS)
    secs = np.arange(int(e0 + 25), int(t0))
    prof_sr = np.array([E[int(s * RC.FS) - W: int(s * RC.FS)].mean(0) for s in secs])
    emp = prof_sr[(secs >= e0 + 25) & (secs <= e1)].mean(0)
    rst = prof_sr[(secs >= sit + 20) & (secs <= spans["rest"][1])].mean(0)
    person = 10 * np.log10(rst / emp) >= 6.0
    half = int(round(0.15 / float(np.median(np.diff(rm)))))
    cfgd = {"demod": "linear", "band_lo_bpm": 7.0, "band_hi_bpm": 45.0}
    out = {"contrast": {}, "rr": {}, "curves": {}, "curve_secs": (secs - sit).tolist()}
    for name, kind, tau in REP_METHODS:
        Y = norm_ext(E, kind, tau, M0, ref)
        P = np.array([Y[int(s * RC.FS) - W: int(s * RC.FS)].mean(0) for s in secs])
        c = 10 * np.log10(P[:, person].mean(1) / (np.median(P[:, ~person], axis=1) + 1e-12) + 1e-12)
        def at(dt):
            return float(c[np.argmin(np.abs(secs - (sit + dt)))])
        out["contrast"][name] = {"10s": at(10), "60s": at(60), "300s": at(300)}
        out["curves"][name] = c.tolist()
        # respiration end to end: paced phases, 30-s windows, hop 5 s
        errs, ok, n = [], 0, 0
        for ph_name in ["paced_10", "paced_15", "paced_20", "paced_25"]:
            a_, b_, rate = spans[ph_name]
            for s_end in np.arange(a_ + 30, b_ + 1e-9, 5.0):
                e = int(s_end * RC.FS)
                prof = Y[e - W: e].mean(0)
                rb = int(np.argmax(prof))
                n += 1
                f = features(person_cells(st[e - int(30 * RC.FS): e], rb, half), cfgd)
                err = abs(f["rr_dsp"] - rate); errs.append(err)
                ok += int(person[rb] and err <= 0.1 * rate)
        out["rr"][name] = {"success_within10pct_and_localized": ok / n, "mae_bpm": float(np.mean(errs)), "n": n}
    # oracle: cells at the centre of the person bins
    rb0 = int(np.round(np.mean(np.where(person)[0])))
    errs = []
    for ph_name in ["paced_10", "paced_15", "paced_20", "paced_25"]:
        a_, b_, rate = spans[ph_name]
        for s_end in np.arange(a_ + 30, b_ + 1e-9, 5.0):
            e = int(s_end * RC.FS)
            f = features(person_cells(st[e - int(30 * RC.FS): e], rb0, half), cfgd)
            errs.append(abs(f["rr_dsp"] - rate))
    errs = np.array(errs)
    rates = np.repeat([10, 15, 20, 25], len(errs) // 4)
    out["rr"]["oracle (person bins)"] = {"success_within10pct_and_localized": float((errs <= 0.1 * rates).mean()),
                                         "mae_bpm": float(errs.mean()), "n": len(errs)}
    return out


def run_walabot(folder, features, person_cells):
    rows = list(csv.DictReader(open(Path(folder) / "index.csv", newline="")))
    bg = {}
    for ob, f in [("free", "m0"), ("obst1", "m0-1"), ("obst2", "m0-2")]:
        zz = np.load(Path(folder) / f"{f}.npz"); s0 = zz["slowtime"].astype(np.complex128)
        Eb, _ = RC.residual_power(s0, init=s0.mean(0)); bg[ob] = (Eb[20:].mean(0), s0.mean(0))
    scales = np.arange(0.0, 24.5, 0.5)
    cfgd = {"demod": "linear", "band_lo_bpm": 6.0, "band_hi_bpm": 45.0}
    cache = []   # per window: refs [(range, rr)], per method: per scale list of detected ranges
    for row in rows:
        z = np.load(Path(folder) / row["file"]); refs = json.loads(str(z["refs"]))
        if not refs:
            continue
        st = z["slowtime"].astype(np.complex128); rm = z["range_m"].astype(np.float64)
        ob = RC.obstacle_class(row["obstacle"]); M0, mean0 = bg[ob]
        E, _ = RC.residual_power(st, init=mean0); ref = float(np.median(M0))
        min_bin = int(np.searchsorted(rm, 0.25)); half = max(1, int(round(0.15 / float(np.median(np.diff(rm))))))
        Ys = {name: norm_ext(E, kind, tau, M0, ref) for name, kind, tau in REP_METHODS}
        for s_end in np.arange(30.0, st.shape[0] / RC.FS + 1e-9, 5.0):
            e = int(s_end * RC.FS)
            rf = []
            for k, r in refs.items():
                i = min(int(s_end), len(r["range_m"]) - 1)
                v, q = r["range_m"][i], r["rr"][i]
                if v is not None and np.isfinite(v):
                    rf.append((v, q))
            dets = {}
            for name in Ys:
                p = Ys[name][e - int(20 * RC.FS): e].mean(0); p = p - min(0.0, p.min()) + 1e-9
                dets[name] = {}
                for sc in scales:
                    idx = RC.merge_peaks(RC.cfar_peaks(p, scale_db=sc, min_bin=min_bin), p, rm, 0.3)
                    dets[name][float(sc)] = rm[sorted(idx, key=lambda i: -p[i])[:5]]
            cache.append({"refs": rf, "dets": dets, "win": st[e - int(30 * RC.FS): e], "rm": rm, "half": half,
                          "ob": ob})
        print(f"  {row['id']:6s} {ob}", flush=True)
    out = {}
    for name, _, _ in REP_METHODS:
        # operating point: largest detection rate with false detections per window <= 0.1
        best = None
        for sc in scales:
            nref = hit = fa = 0
            for c in cache:
                dr = c["dets"][name][float(sc)]
                nref += len(c["refs"]); hit += sum(1 for v, _ in c["refs"] if np.any(np.abs(dr - v) <= 0.3))
                fa += sum(1 for d in dr if all(abs(d - v) > 0.3 for v, _ in c["refs"]))
            fpw = fa / len(cache)
            if fpw <= 0.1 and (best is None or hit / nref > best[1]):
                best = (float(sc), hit / nref, fpw)
        sc = best[0]
        ok = n = 0; errs = []
        for c in cache:
            dr = c["dets"][name][sc]
            for v, q in c["refs"]:
                if q is None or not (5 <= q <= 45):
                    continue
                n += 1
                near = [d for d in dr if abs(d - v) <= 0.3]
                if not near:
                    continue
                d = min(near, key=lambda x: abs(x - v)); rb = int(np.argmin(np.abs(c["rm"] - d)))
                f = features(person_cells(c["win"], rb, c["half"]), cfgd)
                err = abs(f["rr_dsp"] - q); errs.append(err); ok += int(err <= 0.1 * q)
        out[name] = {"cfar_scale_db": sc, "det_rate": best[1], "fpw": best[2],
                     "rr_success_within10pct": ok / n, "rr_mae_detected_bpm": float(np.mean(errs)) if errs else None,
                     "n_person_windows": n}
        print(f"  {name:16s} det {best[1]:.2f} @fpw {best[2]:.2f}  RR ok {ok / n:.2f}  MAE {out[name]['rr_mae_detected_bpm']}", flush=True)
    # oracle: reference range
    ok = n = 0; errs = []
    for c in cache:
        for v, q in c["refs"]:
            if q is None or not (5 <= q <= 45):
                continue
            n += 1; rb = int(np.argmin(np.abs(c["rm"] - v)))
            f = features(person_cells(c["win"], rb, c["half"]), cfgd)
            err = abs(f["rr_dsp"] - q); errs.append(err); ok += int(err <= 0.1 * q)
    out["oracle (reference range)"] = {"rr_success_within10pct": ok / n, "rr_mae_detected_bpm": float(np.mean(errs)),
                                       "det_rate": 1.0, "n_person_windows": n}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fan", required=True); ap.add_argument("--calib", required=True)
    ap.add_argument("--walabot", required=True); ap.add_argument("--repmon", required=True,
                                                              help="folder that contains the RepMon package")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "adaptive"))
    a = ap.parse_args()
    sys.path.insert(0, a.repmon)
    from RepMon.dsp_user import features
    from RepMon.localize import person_cells
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {"_params": {"kappa_db": KAPPA_DB, "hold_s": HOLD_S, "note": "post hoc"}}
    print("A2 simulator", flush=True); res["A2_sim"] = run_sim(); print(res["A2_sim"], flush=True)
    print("A3/A5 BGT60 calibration", flush=True); res["A3_bgt60"] = run_bgt60(a.calib, features, person_cells)
    for k in res["A3_bgt60"]["rr"]:
        print(f"  {k:22s} contrast {res['A3_bgt60']['contrast'].get(k, {})}  RR {res['A3_bgt60']['rr'][k]}", flush=True)
    print("A4/A5 Walabot", flush=True); res["A4_walabot"] = run_walabot(a.walabot, features, person_cells)
    print("A1 fan", flush=True); fan, S_cache, recs, rr, nz, fpm = run_fan(a.fan); res["A1_fan"] = fan
    (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    # micro-Doppler signatures: hand bins 0.60-0.85 m, cue-locked mean per gesture, side position
    md = {}
    hb = (rr >= 0.60) & (rr <= 0.85)
    pre, post = int(0.5 / 0.05), int(2.5 / 0.05)
    fid = {}
    for k in ["SR + global", "SR + CA-CFAR", "SR + PCEN", "SR + G-PCEN"]:
        for g in ["swipe", "push", "circle"]:
            sig = {}
            for fan_state in ["off", "on"]:
                name = next(n for n, d in recs if d["labels"]["gesture"] == g and d["labels"]["fan"] == fan_state
                            and d["labels"]["position"] == "side")
                S = S_cache[name][k]; lab = dict(recs)[name]["labels"]
                Z = np.log10(np.maximum(S[:, hb].mean(1), 1e-12))        # (T, D) log statistic
                segs = [Z[c - pre: c + post] for c in lab["cue_frames"] if c - pre >= 0 and c + post <= len(Z)]
                reps = np.array(segs)
                sig[fan_state] = reps
                md[f"{k}|{g}|{fan_state}"] = reps.mean(0).astype(np.float32)
            tmpl = sig["off"].mean(0).ravel()
            cs = [np.corrcoef(rp.ravel(), tmpl)[0, 1] for rp in sig["on"]]
            fid.setdefault(k, {})[g] = {"mean_corr_on_vs_off_template": float(np.mean(cs)),
                                        "per_rep": [float(x) for x in cs]}
    res["A1_fan_microdoppler_fidelity"] = fid
    np.savez_compressed(out / "fan_microdoppler.npz", **{k.replace("|", "__").replace(" ", "_").replace("+", "p"): v
                                                         for k, v in md.items()})
    (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: {g: round(v["mean_corr_on_vs_off_template"], 3) for g, v in d.items()} for k, d in fid.items()}, indent=1))
    print(json.dumps(res["A1_fan"], indent=1))


if __name__ == "__main__":
    main()
