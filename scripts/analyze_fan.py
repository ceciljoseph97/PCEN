"""Analysis of the fan-background gesture session (record_fan_session.py). numpy/scipy/sklearn only.

  conda activate lecture_radar_systems
  python scripts/analyze_fan.py --data C:/radar_data/fan/P01 --out results/fan

PRE-REGISTERED (fixed 2026-10-15, before any data was recorded)
----------------------------------------------------------------
Maps: per frame, range FFT -> static removal (mean over the 64 chirps) -> Doppler FFT -> power,
averaged over the 3 Rx (non-coherent). RD sequence (T, 64 range, 64 Doppler) per block, 20 Hz.
Detection statistics (all on the static-removed maps; IIR methods restart in every block):
  sr_global     power / session reference (one global threshold)
  sr_cacfar     2-D CA-CFAR, guard 2x2, training 8x4 (as in the paper)
  sr_clutmap    clutter-map CFAR, X_t / Q_(t-1), w = 1 - exp(-T/1 s)
  sr_pcen1      PCEN tau = 1 s, alpha .98, delta 2, r .5   (paper default)
  sr_pcen013    PCEN tau = 0.13 s, otherwise identical      (value learned on KIT)
  sr_pcen1_ca   PCEN tau = 1 s followed by CA-CFAR
Windows (1.5 s): positive = [cue+0.2, cue+1.7] s; negative = rest [cue+2.3, cue+3.8] s, the lead-in
[2.5, 4.0] s of every gesture block, and consecutive windows of the empty block of the same
(position, fan) group after the first 2 s. Window score = max of the statistic over the window's
frames, the hand range gate (front 0.20-0.65 m, side 0.80-1.25 m) and all Doppler bins except
zero-Doppler +-1. Gates move with the actual fan distance (pcen_radar/fan_scene.py, --fan-range);
for the fan at 1.0 m they are exactly the values above.
PRIMARY (H1): with the fan ON, window-level detection AUC of sr_pcen1 is higher than that of
  sr_cacfar and of sr_global, at both positions -> 4 one-sided paired bootstrap tests (2000
  resamples of windows, stratified pos/neg), Bonferroni alpha = 0.05/4.
  Expectation stated in advance: at "front" the hand is range-separated from the fan, so all
  methods may saturate near AUC = 1 there; "side" (hand in the fan's range bins) is the
  informative condition. A dry run on simulated data (no real measurement) gave no PCEN advantage
  at "side", so the outcome is open.
SECONDARY (reported whatever the outcome, no correction):
  H2 fan-induced AUC loss (AUC_off - AUC_on) per method;
  H3 sr_pcen1 vs sr_clutmap (expected close: PCEN = clutter map + compression);
  H4 sr_pcen013 vs sr_pcen1 (confirmation of the KIT-learned time constant on new hardware);
  M1 fan-region false-alarm increase: threshold at Pfa = 1e-3 on the fan-OFF empty blocks,
     Pfa in the fan range gate (non-zero Doppler) of the fan-ON empty blocks, in dB over 1e-3;
  M2 background identifiability: extra-trees (300 trees) predicting fan on/off from features of
     the gesture windows (per Doppler bin: mean and max over the window of the statistic averaged
     over the hand gate; log10 for sr_global/sr_cacfar/sr_clutmap), 5-fold stratified CV,
     balanced accuracy, mean of 10 seeds; lower = background less visible;
  M3 cross-background gesture accuracy (3 classes, same features): train fan OFF -> test fan ON
     and reverse, mean of both directions, 10 seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcen_radar.io import load_recording
from pcen_radar.processing import rd_sequence
from pcen_radar.pcen import pcen, s_from_tau
from pcen_radar.cfar import ca_cfar_stat, clutter_map_stat
from pcen_radar.fan_scene import gates as scene_gates

METHODS = ["sr_global", "sr_cacfar", "sr_clutmap", "sr_pcen1", "sr_pcen013", "sr_pcen1_ca"]
LOGFEAT = {"sr_global", "sr_cacfar", "sr_clutmap"}
WARM_S = 2.0


def statistics(P, ref, fp):
    s1, s013 = s_from_tau(1.0, fp), s_from_tau(0.13, fp)
    out = {"sr_global": (P / ref).astype(np.float32),
           "sr_cacfar": np.stack([ca_cfar_stat(x) for x in P]),
           "sr_clutmap": clutter_map_stat(P, w=s1)}
    out["sr_pcen1"] = pcen(P, s=s1, alpha=0.98, delta=2.0, r=0.5, ref=ref)
    out["sr_pcen013"] = pcen(P, s=s013, alpha=0.98, delta=2.0, r=0.5, ref=ref)
    out["sr_pcen1_ca"] = np.stack([ca_cfar_stat(x - x.min() + 1e-6) for x in out["sr_pcen1"]])
    return out


def gate(rr, g):
    return (rr >= g[0]) & (rr <= g[1])


def windows(lab, n, fp, empty_len_s=1.5):
    """-> list of (start, stop, is_positive, kind)."""
    w = []
    L = int(round(1.5 / fp))
    if lab["gesture"] == "empty":
        s = int(round(WARM_S / fp))
        while s + L <= n:
            w.append((s, s + L, 0, "empty")); s += L
        return w
    a, b = int(round(2.5 / fp)), int(round(2.5 / fp)) + L
    w.append((a, b, 0, "leadin"))
    for c in lab["cue_frames"]:
        p0 = c + int(round(0.2 / fp)); n0 = c + int(round(2.3 / fp))
        if p0 + L <= n:
            w.append((p0, p0 + L, 1, "gesture"))
        if n0 + L <= n:
            w.append((n0, n0 + L, 0, "rest"))
    return w


def auc(pos, neg):
    from scipy.stats import rankdata
    x = np.concatenate([pos, neg]); r = rankdata(x)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pd_at_far(pos, neg, far=0.05):
    thr = np.quantile(neg, 1 - far)
    return float((pos > thr).mean())


def window_features(S, fe, hand, nz, w0, w1):
    x = S[w0:w1][:, hand].mean(axis=1)[:, nz]                 # (frames, doppler)
    if fe in LOGFEAT:
        x = np.log10(np.maximum(x, 1e-12))
    return np.concatenate([x.mean(0), x.max(0)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with the block .npz files of one participant")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "fan"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--gate-shift", type=float, default=0.0,
                    help="shift all range gates by this many metres (sensitivity analysis)")
    ap.add_argument("--fan-range", dest="fan_range", type=float, default=None,
                    help="actual radar->fan distance [m]; overrides the gates stored in the files "
                         "(use when the scene was set up differently from the recording default)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.data).glob("*.npz"))
    if not files:
        sys.exit(f"no .npz in {a.data}")
    recs = []
    for f in files:
        d = load_recording(f)
        recs.append((f.stem, d))
    cfg = recs[0][1]["config"]; fp = cfg.frame_repetition_time_s
    fp_meas = float(np.median([d["meta"].get("measured_frame_period_s", fp) for _, d in recs]))
    print(f"nominal frame period {fp * 1e3:.1f} ms, measured {fp_meas * 1e3:.1f} ms (used for the PCEN/clutter-map time constants)")
    print(f"{len(recs)} blocks, {cfg.summary()}")

    # session reference: median static-removed power of the first fan-OFF empty block
    t0 = time.time()
    P_all = {}
    for name, d in recs:
        P_all[name] = rd_sequence(d["frames"], clutter="mean")
        print(f"  RD {name:24s} {P_all[name].shape}  ({time.time() - t0:5.0f} s)", flush=True)
    ref_name = next((n for n, d in recs if d["labels"]["gesture"] == "empty" and d["labels"]["fan"] == "off"),
                    recs[0][0])
    ref = float(np.median(P_all[ref_name][int(WARM_S / fp):]))
    T0, R, D = P_all[recs[0][0]].shape
    rr = cfg.range_axis(R)
    nz = np.r_[0:D // 2 - 1, D // 2 + 2:D]

    rows = []            # one row per window
    feats = {fe: [] for fe in METHODS}; frows = []
    fan_cells = {}       # (fan, position) -> method -> values in fan gate (empty blocks)
    maps = {}
    gates_used = {}
    for name, d in recs:
        lab = d["labels"]; P = P_all[name]; n = len(P)
        S = statistics(P, ref, fp_meas)
        sh = a.gate_shift
        if a.fan_range is not None:                       # documented scene adjustment
            G = scene_gates(a.fan_range)
            lab["hand_gate_m"], lab["fan_gate_m"], lab["fan_range_m"] = G[lab["position"]], G["fan"], a.fan_range
        hand = gate(rr, [g + sh for g in lab["hand_gate_m"]]); fan = gate(rr, [g + sh for g in lab["fan_gate_m"]])
        gates_used[lab["position"]] = {"hand_gate_m": list(lab["hand_gate_m"]), "fan_gate_m": list(lab["fan_gate_m"]),
                                       "fan_range_m": lab.get("fan_range_m", 1.0)}
        wins = windows(lab, n, fp)
        for (w0, w1, pos, kind) in wins:
            row = dict(block=name, position=lab["position"], fan=lab["fan"], gesture=lab["gesture"],
                       kind=kind, pos=pos)
            for fe in METHODS:
                row[fe] = float(S[fe][w0:w1][:, hand][:, :, nz].max())
            rows.append(row)
            if pos:
                for fe in METHODS:
                    feats[fe].append(window_features(S[fe], fe, hand, nz, w0, w1))
                frows.append(row)
        if lab["gesture"] == "empty":
            w = int(WARM_S / fp)
            fan_cells[(lab["fan"], lab["position"])] = {
                fe: S[fe][w:][:, fan][:, :, nz].ravel() for fe in METHODS}
            fan_cells[(lab["fan"], lab["position"], "all")] = {fe: S[fe][w:][:, :, nz].ravel() for fe in METHODS}
        # compact maps for the paper figure: range-time, log10 of max over non-zero Doppler
        maps[name] = {fe: np.log10(np.maximum(S[fe][:, :, nz].max(axis=2), 1e-12)).astype(np.float16)
                      for fe in METHODS}                                  # log10 of the statistic
        maps[name]["_cues"] = np.array(lab["cue_frames"])
        print(f"  stats {name:24s} {len(wins)} windows  ({time.time() - t0:5.0f} s)", flush=True)

    res = {"_params": {"files": [n for n, _ in recs], "ref_block": ref_name, "ref": ref,
                       "frame_period_s": fp, "frame_period_measured_s": fp_meas,
                       "gate_shift_m": a.gate_shift, "boot": a.boot, "seeds": a.seeds}}
    res["_params"]["gates_used"] = gates_used
    res["_params"]["fan_range_override_m"] = a.fan_range
    rng = np.random.default_rng(0)
    groups = sorted({(r["position"], r["fan"]) for r in rows})
    # ---------- detection AUC per group
    det = {}
    for pos_, fan_ in groups:
        sel = [r for r in rows if r["position"] == pos_ and r["fan"] == fan_]
        P_ = {fe: np.array([r[fe] for r in sel if r["pos"]]) for fe in METHODS}
        N_ = {fe: np.array([r[fe] for r in sel if not r["pos"]]) for fe in METHODS}
        g = f"{pos_}_{fan_}"
        det[g] = {"n_pos": int(len(P_[METHODS[0]])), "n_neg": int(len(N_[METHODS[0]]))}
        if det[g]["n_pos"] == 0 or det[g]["n_neg"] == 0:
            continue
        ip = rng.integers(0, len(P_[METHODS[0]]), (a.boot, len(P_[METHODS[0]])))
        ineg = rng.integers(0, len(N_[METHODS[0]]), (a.boot, len(N_[METHODS[0]])))
        boots = {}
        for fe in METHODS:
            boots[fe] = np.array([auc(P_[fe][ip[b]], N_[fe][ineg[b]]) for b in range(a.boot)])
            det[g][fe] = {"auc": auc(P_[fe], N_[fe]),
                          "auc_ci95": [float(np.quantile(boots[fe], .025)), float(np.quantile(boots[fe], .975))],
                          "pd_at_far5": pd_at_far(P_[fe], N_[fe], 0.05)}
        det[g]["_boot"] = boots
    # ---------- tests
    tests = {}
    def one_sided(g, a_, b_):
        dlt = det[g]["_boot"][a_] - det[g]["_boot"][b_]
        return {"diff": det[g][a_]["auc"] - det[g][b_]["auc"],
                "ci95": [float(np.quantile(dlt, .025)), float(np.quantile(dlt, .975))],
                "p_one_sided": float((np.sum(dlt <= 0) + 1) / (len(dlt) + 1))}
    for pos_ in ["front", "side"]:
        g = f"{pos_}_on"
        if g in det and "_boot" in det[g]:
            for ref_m in ["sr_cacfar", "sr_global"]:
                t = one_sided(g, "sr_pcen1", ref_m)
                t["significant_bonferroni"] = t["p_one_sided"] < 0.05 / 4
                tests[f"H1_{g}_pcen1_vs_{ref_m}"] = t
            tests[f"H3_{g}_pcen1_vs_clutmap"] = one_sided(g, "sr_pcen1", "sr_clutmap")
            tests[f"H4_{g}_pcen013_vs_pcen1"] = one_sided(g, "sr_pcen013", "sr_pcen1")
        if f"{pos_}_off" in det and f"{pos_}_on" in det and METHODS[0] in det.get(f"{pos_}_on", {}):
            tests[f"H2_{pos_}_auc_loss"] = {fe: det[f"{pos_}_off"][fe]["auc"] - det[f"{pos_}_on"][fe]["auc"]
                                           for fe in METHODS}
    for g in det:
        det[g].pop("_boot", None)
    res["detection"] = det; res["tests"] = tests

    # ---------- M1 fan-region false-alarm increase
    m1 = {}
    for pos_ in ["front", "side"]:
        k_off, k_on = ("off", pos_, "all"), ("on", pos_)
        if k_off in fan_cells and k_on in fan_cells:
            m1[pos_] = {}
            for fe in METHODS:
                thr = np.quantile(fan_cells[k_off][fe], 1 - 1e-3)
                pfa = float((fan_cells[k_on][fe] > thr).mean())
                m1[pos_][fe] = {"pfa_fan_region": pfa, "excess_db": float(10 * np.log10(max(pfa, 1e-6) / 1e-3))}
    res["M1_fan_false_alarm"] = m1

    # ---------- M2 / M3 classifiers
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import balanced_accuracy_score
    m2, m3 = {}, {}
    fpos = np.array([r["position"] for r in frows]); ffan = np.array([r["fan"] for r in frows])
    fges = np.array([r["gesture"] for r in frows])
    for pos_ in ["front", "side"]:
        sel = (fpos == pos_) & np.isin(ffan, ["off", "on"])
        if sel.sum() < 20 or len(set(ffan[sel])) < 2:
            continue
        m2[pos_], m3[pos_] = {}, {}
        for fe in METHODS:
            X = np.array(feats[fe])[sel]; yb = ffan[sel]; yg = fges[sel]
            b2, b3 = [], []
            for sd in range(a.seeds):
                clf = ExtraTreesClassifier(300, random_state=sd, n_jobs=-1)
                pr = cross_val_predict(clf, X, yb, cv=StratifiedKFold(5, shuffle=True, random_state=sd))
                b2.append(balanced_accuracy_score(yb, pr))
                accs = []
                for tr_, te_ in [("off", "on"), ("on", "off")]:
                    clf = ExtraTreesClassifier(300, random_state=sd, n_jobs=-1)
                    clf.fit(X[yb == tr_], yg[yb == tr_])
                    accs.append(float((clf.predict(X[yb == te_]) == yg[yb == te_]).mean()))
                b3.append(np.mean(accs))
            m2[pos_][fe] = {"bal_acc": float(np.mean(b2)), "std": float(np.std(b2))}
            m3[pos_][fe] = {"acc": float(np.mean(b3)), "std": float(np.std(b3))}
    res["M2_background_id"] = m2; res["M3_cross_background_acc"] = m3

    (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez_compressed(out / "fan_maps.npz", **{f"{b}__{k}": v for b, mm in maps.items() for k, v in mm.items()})
    json.dump(rows, open(out / "windows.json", "w"))

    # ---------- markdown summary
    L = ["# Fan-background measurement: results", "",
         "Gates used: " + "; ".join(f"{p}: hand {g['hand_gate_m']} m, fan {g['fan_gate_m']} m (fan at {g['fan_range_m']} m)"
                                   for p, g in gates_used.items()), "",
         "## Detection AUC (window level, hand gate)  [95% bootstrap CI]", "",
         "| Method | " + " | ".join(det) + " |", "|---|" + "---|" * len(det)]
    for fe in METHODS:
        L.append(f"| {fe} | " + " | ".join(
            f"{det[g][fe]['auc']:.3f} [{det[g][fe]['auc_ci95'][0]:.2f}, {det[g][fe]['auc_ci95'][1]:.2f}]"
            if fe in det[g] else "-" for g in det) + " |")
    L += ["", "Windows per group: " + ", ".join(f"{g}: {det[g]['n_pos']}+/{det[g]['n_neg']}-" for g in det), "",
          "## Tests", "", "| Test | ΔAUC | 95% CI | p (one-sided) | note |", "|---|---|---|---|---|"]
    for k, t in tests.items():
        if k.startswith("H2"):
            continue
        note = ("PRIMARY, " + ("significant" if t["significant_bonferroni"] else "n.s.") + " (Bonferroni 0.0125)"
                if k.startswith("H1") else "secondary")
        L.append(f"| {k} | {t['diff']:+.3f} | [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] | {t['p_one_sided']:.4f} | {note} |")
    L += ["", "## Secondary metrics", "", "| Method | " + " | ".join(
        [f"AUC loss {p}" for p in ["front", "side"]] + [f"fan Pfa excess {p} [dB]" for p in m1] +
        [f"bg-ID {p} (↓)" for p in m2] + [f"cross-bg acc {p}" for p in m3]) + " |"]
    ncol = 2 + len(m1) + len(m2) + len(m3)
    L.append("|---|" + "---|" * ncol)
    for fe in METHODS:
        c = [f"{tests[f'H2_{p}_auc_loss'][fe]:+.3f}" if f"H2_{p}_auc_loss" in tests else "-" for p in ["front", "side"]]
        c += [f"{m1[p][fe]['excess_db']:+.1f}" for p in m1]
        c += [f"{m2[p][fe]['bal_acc']:.3f}" for p in m2]
        c += [f"{m3[p][fe]['acc']:.3f}" for p in m3]
        L.append(f"| {fe} | " + " | ".join(c) + " |")
    (out / "results.md").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))
    print("\nsaved", out.resolve())


if __name__ == "__main__":
    main()
