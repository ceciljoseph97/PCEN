"""Ablation of the censored PCEN (C-PCEN, code name G-PCEN) parameters. POST HOC.

  python scripts/cpcen_ablation.py --fan <fan/P01> --calib <RepMon session.npz> --walabot <walabot_slowtime>
                                   --repmon <folder containing RepMon> --out results/cpcen_ablation

Defaults (fixed before the first C-PCEN evaluation): hold 600 s, kappa 6 dB, fast average 0.5 s.
One factor at a time:
  hold     : 1, 2, 5, 15, 30, 60, 120, 300, 600 s, infinite   (how long M may stay frozen)
  kappa    : 3, 6, 10 dB                                  (censoring threshold)
  tau_fast : 0.25, 0.5, 1, 2 s                            (time constant of the fast average A)
Data:
  S  simulator (walk - stop 5 s - walk, 5 seeds): Pd of the stopped and of the moving target at Pfa 1e-3,
     scene 'switch_on' (fan switches on at 7.5 s, the default scene) and 'from_start' (fan on from the start)
  B  own BGT60 seated session: contrast 10/60/300 s after sitting down, breathing rate ok (tau = 10 s)
  W  Walabot, 59 recordings of ~174 s: localization rate at 0.1 false detections/window, breathing rate ok
  F  fan session: AUC and Pd (5 % window FAR) beside the running fan, fan false-alarm increase (empty room)
  L  fan switched on LATER (real recordings spliced: 45 s empty room fan off, then 45 s fan on, same
     position): false-alarm rate in the fan gate 5-40 s after switch-on and time until it returns to the
     fan-off level. Plain PCEN adapts within ~tau; C-PCEN holds the new clutter as a target until release.
     The splice is synthetic in time (two recordings back to back), not a recorded switch-on.
Expectation stated before running: hold only matters below the target duration (sim 5 s, seated 300 s,
Walabot ~174 s) and in L, where it sets how long the switched-on fan stays a false target.
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
import adaptive_eval as AE
import repmon_case as RC
from analyze_fan import windows, auc, gate, WARM_S

BASE = dict(hold=600.0, kappa=6.0, tf=0.5)
HOLDS = [1.0, 2.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, np.inf]
KAPPAS = [3.0, 6.0, 10.0]
TFS = [0.25, 0.5, 1.0, 2.0]


def configs():
    c = [("hold", h, dict(BASE, hold=h)) for h in HOLDS]
    c += [("kappa", k, dict(BASE, kappa=k)) for k in KAPPAS if k != BASE["kappa"]]
    c += [("tau_fast", t, dict(BASE, tf=t)) for t in TFS if t != BASE["tf"]]
    return c


def label(axis, v):
    return f"{axis}=" + ("inf" if not np.isfinite(v) else f"{v:g}")


def gp(E, s, fp, p, ref="auto", M0=None):
    hold = None if not np.isfinite(p["hold"]) else int(round(p["hold"] / fp))
    return pcen_gated(E, s=s, ref=ref, M0=M0, hold_frames=hold, kappa_db=p["kappa"], s_fast=s_from_tau(p["tf"], fp))


# ------------------------------------------------------------------ S simulator
def run_sim(cfgs):
    return {"switch_on": _run_sim(cfgs, None), "from_start": _run_sim(cfgs, (0, 400))}


def _run_sim(cfgs, fan_frames):
    from pcen_radar import RadarConfig
    from pcen_radar.simulate import simulate, Scene
    from pcen_radar.io import per_frame_range_masks
    from pcen_radar.metrics import thr_at_pfa, detection_over_time
    cfg = RadarConfig(); fp = cfg.frame_repetition_time_s; s1 = s_from_tau(1.0, fp); out = {}
    for seed in range(5):
        frames, r_true, v_true = simulate(cfg, Scene(seed=seed) if fan_frames is None else Scene(seed=seed, fan_frames=fan_frames))
        P = rd_sequence(frames, clutter="mean")
        tm, gm, present = per_frame_range_masks(r_true, cfg, P.shape, 0.15, 0.30, 1)
        ta, _, _ = per_frame_range_masks(r_true, cfg, P.shape, 0.15, 0.30, 0)
        moving = present & (np.abs(v_true) > 1e-3); stopped = present & ~moving
        meth = {"PCEN": pcen(P, s=s1)}
        meth.update({label(a, v): gp(P, s1, fp, p) for a, v, p in cfgs})
        w = 20
        for k, S in meth.items():
            thr = thr_at_pfa(S[w:], tm[w:], gm[w:], 1e-3)
            det = detection_over_time(S[w:], ta[w:], thr)
            o = out.setdefault(k, {"pd_moving": [], "pd_stopped": []})
            o["pd_moving"].append(float(det[moving[w:]].mean())); o["pd_stopped"].append(float(det[stopped[w:]].mean()))
    return {k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in out.items()}


# ------------------------------------------------------------------ B / W via adaptive_eval with patched methods
def patch_rep(cfgs, tau=10.0):
    AE.REP_METHODS = [("SR", "SR", None), ("PCEN tau=10s", "PCEN", 10.0)] + \
                     [(label(a, v), "GP", (tau, p)) for a, v, p in cfgs]
    orig = RC.normalize

    def norm_ext(E, kind, tau_, M0, ref):
        if kind == "GP":
            t, p = tau_
            return gp(E, 1 - np.exp(-1 / (RC.FS * t)), 1 / RC.FS, p, ref=ref, M0=M0).astype(np.float64)
        return orig(E, kind, tau_, M0, ref)
    AE.norm_ext = norm_ext


# ------------------------------------------------------------------ F / L fan
def run_fan(folder, cfgs):
    files = sorted(Path(folder).glob("*.npz"))
    lab0 = {f.stem: json.loads(str(np.load(f)["labels"])) for f in files}
    pos2 = "near" if any(l["position"] == "near" for l in lab0.values()) else "side"     # session 2 / session 1
    keep = [f for f in files if lab0[f.stem]["gesture"] == "empty" or
            (lab0[f.stem]["position"] == pos2 and lab0[f.stem]["fan"] == "on" and lab0[f.stem]["gesture"] != "hold")]
    recs = [(f.stem, load_recording(f)) for f in keep]
    cfg = recs[0][1]["config"]; fp = cfg.frame_repetition_time_s
    fpm = float(np.median([d["meta"].get("measured_frame_period_s", fp) for _, d in recs]))
    s1 = s_from_tau(1.0, fpm)
    P_all = {n: rd_sequence(d["frames"], clutter="mean") for n, d in recs}
    L = {n: d["labels"] for n, d in recs}
    ref_name = next(n for n in L if L[n]["gesture"] == "empty" and L[n]["fan"] == "off")
    ref = float(np.median(P_all[ref_name][int(WARM_S / fp):]))
    R, D = P_all[recs[0][0]].shape[1:]
    rr = cfg.range_axis(R); nz = np.r_[0:D // 2 - 1, D // 2 + 2:D]
    fg = gate(rr, recs[0][1]["labels"]["fan_gate_m"]); w = int(WARM_S / fp)
    fns = {"PCEN": lambda P: pcen(P, s=s1, ref=ref)}
    fns.update({label(a, v): (lambda p: (lambda P: gp(P, s1, fpm, p, ref=ref)))(p) for a, v, p in cfgs})
    res = {"auc_side_on": {}, "pd5_side_on": {}, "fan_fa_increase_db": {"front": {}, pos2: {}}, "later": {},
           "second_position": pos2}
    # F: side position, fan on, gestures; empty-room fan false alarms
    rows = []
    side_on = [n for n in L if L[n]["position"] == pos2 and L[n]["fan"] == "on" and L[n]["gesture"] != "empty"]
    S_cache = {}
    for n in side_on:
        hand = gate(rr, L[n]["hand_gate_m"]); S_cache[n] = {k: f(P_all[n]) for k, f in fns.items()}
        for (w0, w1, pos, kind) in windows(L[n], len(P_all[n]), fp):
            rows.append({"pos": pos, **{k: float(S_cache[n][k][w0:w1][:, hand][:, :, nz].max()) for k in fns}})
    for k in fns:
        Pp = np.array([r[k] for r in rows if r["pos"]]); Nn = np.array([r[k] for r in rows if not r["pos"]])
        res["auc_side_on"][k] = auc(Pp, Nn); res["pd5_side_on"][k] = float((Pp > np.quantile(Nn, 0.95)).mean())
    for pos in ["front", pos2]:
        off = next(n for n in L if L[n]["gesture"] == "empty" and L[n]["fan"] == "off" and L[n]["position"] == pos)
        on = next(n for n in L if L[n]["gesture"] == "empty" and L[n]["fan"] == "on" and L[n]["position"] == pos)
        for k, f in fns.items():
            Soff = f(P_all[off])[w:][:, :, nz]; Son = f(P_all[on])[w:][:, fg][:, :, nz]
            thr = np.quantile(Soff.ravel(), 1 - 1e-3)
            res["fan_fa_increase_db"][pos][k] = float(10 * np.log10(max((Son > thr).mean(), 1e-6) / 1e-3))
        # L: fan off (45 s) then fan on (45 s), spliced
        Poff, Pon = P_all[off], P_all[on]
        X = np.concatenate([Poff, Pon]); t_on = len(Poff)
        res["later"][pos] = {}
        for k, f in fns.items():
            S = f(X)[:, :, nz]
            thr = np.quantile(S[w:t_on].ravel(), 1 - 1e-3)        # fan-off part sets the threshold
            fa_t = (S[t_on:][:, fg] > thr).mean(axis=(1, 2))         # per frame, fan gate
            a, b = int(5 / fpm), int(40 / fpm)
            fa = float(fa_t[a:b].mean())
            # time until the 2-s moving average of the fan-gate FA rate falls below 10x the nominal 1e-3
            k2 = max(1, int(2 / fpm)); sm = np.convolve(fa_t, np.ones(k2) / k2, mode="valid")
            below = np.where(sm < 1e-2)[0]
            t_rel = float(below[0] * fpm) if len(below) else None
            res["later"][pos][k] = {"fa_increase_db_5_40s": float(10 * np.log10(max(fa, 1e-6) / 1e-3)),
                                    "t_back_s": t_rel, "len_after_on_s": float(len(Pon) * fpm)}
    if pos2 != "side":                                        # alias for the table script
        res["later"]["side"] = res["later"][pos2]; res["fan_fa_increase_db"]["side"] = res["fan_fa_increase_db"][pos2]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fan", required=True); ap.add_argument("--calib", required=True)
    ap.add_argument("--walabot", required=True); ap.add_argument("--repmon", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "cpcen_ablation"))
    ap.add_argument("--only", default="SBWFL")
    a = ap.parse_args()
    sys.path.insert(0, a.repmon)
    from RepMon.dsp_user import features
    from RepMon.localize import person_cells
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fp_ = out / "results.json"
    res = json.loads(fp_.read_text(encoding="utf-8")) if fp_.exists() else {}
    res["_params"] = {"base": BASE, "holds": [str(h) for h in HOLDS], "kappas": KAPPAS, "tau_fast": TFS,
                      "note": "post hoc ablation, one factor at a time"}
    cf = configs()
    if "S" in a.only:
        print("S simulator", flush=True); res["S_sim"] = run_sim(cf)
        fp_.write_text(json.dumps(res, indent=1), encoding="utf-8")
    if "F" in a.only or "L" in a.only:
        print("F/L fan", flush=True); res["F_fan"] = run_fan(a.fan, cf)
        fp_.write_text(json.dumps(res, indent=1), encoding="utf-8")
    patch_rep(cf)
    if "B" in a.only:
        print("B BGT60 seated", flush=True); b = AE.run_bgt60(a.calib, features, person_cells)
        b.pop("curves", None); b.pop("curve_secs", None); res["B_bgt60"] = b
        fp_.write_text(json.dumps(res, indent=1), encoding="utf-8")
    if "W" in a.only:
        print("W Walabot", flush=True); res["W_walabot"] = AE.run_walabot(a.walabot, features, person_cells)
        fp_.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()
