"""EXPLORATORY follow-up analyses of the fan session (post hoc, after the pre-registered analyze_fan.py).

  python scripts/fan_extra.py --data C:/radar_data/fan/P01 --out results/fan_extra [--fan-range 1.0]

1. CFAR fairness  : is CA-CFAR's failure beside the fan specific to our training ring? Variants:
                    CA-CFAR with other rings, GO-/SO-CFAR, OS-CFAR (robust to clutter in the ring).
2. PCEN alpha/tau : alpha interpolates between a global threshold (alpha=0) and a clutter map (alpha=1);
                    tau sweep for PCEN and clutter-map CFAR. (delta and r are monotone per cell and
                    cannot change a max-statistic detector's AUC - checked numerically.)
3. Operating pts  : Pd at window false-alarm rates of 1/5/10 %, ROC curves.
4. Per gesture    : AUC beside the fan for swipe / push / circle separately.
5. Cluster CI     : bootstrap over time-contiguous clusters of 5 repetitions instead of single windows.
6. Runtime        : streaming cost per 64x64 frame of each detector (this machine).
Window definitions, gates and the reference are identical to analyze_fan.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pcen_radar.io import load_recording
from pcen_radar.processing import rd_sequence
from pcen_radar.pcen import pcen, s_from_tau, ClutterMapState
from pcen_radar.cfar import ca_cfar_stat, os_cfar_stat, clutter_map_stat, _ring, _conv
from pcen_radar.fan_scene import gates as scene_gates
from analyze_fan import windows, auc, gate, WARM_S


# ------------------------------------------------------------------ extra CFAR variants
def gos_cfar_stat(P, guard=(2, 2), train=(8, 4), mode="go"):
    """Greatest-of / smallest-of CFAR: separate means of the leading and lagging halves in range."""
    gr, gd = guard; tr, td = train
    H, W = 2 * (gr + tr) + 1, 2 * (gd + td) + 1
    k = _ring(guard, train)
    lead = k.copy(); lead[H // 2:, :] = False
    lag = k.copy(); lag[:H // 2 + 1, :] = False
    a = _conv(P.astype(np.float64), lead) / lead.sum()
    b = _conv(P.astype(np.float64), lag) / lag.sum()
    noise = np.maximum(a, b) if mode == "go" else np.minimum(a, b)
    return (P / (noise + 1e-30)).astype(np.float32)


def methods_cfar():
    return {
        "CA 2x2/8x4 (paper)": lambda P: np.stack([ca_cfar_stat(x, (2, 2), (8, 4)) for x in P]),
        "CA 1x1/4x4": lambda P: np.stack([ca_cfar_stat(x, (1, 1), (4, 4)) for x in P]),
        "CA 2x2/4x8": lambda P: np.stack([ca_cfar_stat(x, (2, 2), (4, 8)) for x in P]),
        "CA 2x2/16x4": lambda P: np.stack([ca_cfar_stat(x, (2, 2), (16, 4)) for x in P]),
        "CA Doppler-only 0x2/0x8": lambda P: np.stack([ca_cfar_stat(x, (0, 2), (0, 8)) for x in P]),
        "GO-CFAR 2x2/8x4": lambda P: np.stack([gos_cfar_stat(x, mode="go") for x in P]),
        "SO-CFAR 2x2/8x4": lambda P: np.stack([gos_cfar_stat(x, mode="so") for x in P]),
        "OS-CFAR k=0.75": lambda P: np.stack([os_cfar_stat(x, (2, 2), (8, 4), 0.75) for x in P]),
        "OS-CFAR k=0.5": lambda P: np.stack([os_cfar_stat(x, (2, 2), (8, 4), 0.5) for x in P]),
    }


def score_rows(P_all, recs, rr, nz, fp, fn_by_name, gate_override=None):
    """-> list of window rows with one score per method."""
    rows = []
    for name, d in recs:
        lab = dict(d["labels"])
        if gate_override is not None:
            G = scene_gates(gate_override)
            lab["hand_gate_m"], lab["fan_gate_m"] = G[lab["position"]], G["fan"]
        P = P_all[name]
        hand = gate(rr, lab["hand_gate_m"])
        S = {k: f(P) for k, f in fn_by_name.items()}
        for (w0, w1, pos, kind) in windows(lab, len(P), fp):
            row = dict(block=name, position=lab["position"], fan=lab["fan"], gesture=lab["gesture"],
                       kind=kind, pos=pos, w0=w0)
            for k in S:
                row[k] = float(S[k][w0:w1][:, hand][:, :, nz].max())
            rows.append(row)
    return rows


def group_auc(rows, keys, pos_, fan_):
    sel = [r for r in rows if r["position"] == pos_ and r["fan"] == fan_]
    return {k: auc(np.array([r[k] for r in sel if r["pos"]]), np.array([r[k] for r in sel if not r["pos"]]))
            for k in keys}


def cluster_boot(rows, a_, b_, pos_, fan_, n=2000, reps_per_cluster=5, seed=0):
    """Paired bootstrap of AUC(a) - AUC(b) over time-contiguous clusters of windows."""
    sel = [r for r in rows if r["position"] == pos_ and r["fan"] == fan_]
    clus = {}
    for r in sel:                              # cluster id: block + window start // (5 cycles of 4 s)
        cid = (r["block"], r["w0"] // int(round(reps_per_cluster * 4.0 / 0.05)))
        clus.setdefault(cid, []).append(r)
    ids = list(clus)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        pick = [clus[ids[i]] for i in rng.integers(0, len(ids), len(ids))]
        rs = [r for c in pick for r in c]
        p = [r for r in rs if r["pos"]]; q = [r for r in rs if not r["pos"]]
        if not p or not q:
            continue
        d.append(auc(np.array([r[a_] for r in p]), np.array([r[a_] for r in q])) -
                 auc(np.array([r[b_] for r in p]), np.array([r[b_] for r in q])))
    d = np.array(d)
    return {"n_clusters": len(ids), "ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))],
            "p_one_sided": float((np.sum(d <= 0) + 1) / (len(d) + 1))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "fan_extra"))
    ap.add_argument("--fan-range", dest="fan_range", type=float, default=1.0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    recs = [(f.stem, load_recording(f)) for f in sorted(Path(a.data).glob("*.npz"))]
    cfg = recs[0][1]["config"]; fp = cfg.frame_repetition_time_s
    fpm = float(np.median([d["meta"].get("measured_frame_period_s", fp) for _, d in recs]))
    t0 = time.time()
    P_all = {n: rd_sequence(d["frames"], clutter="mean") for n, d in recs}
    print(f"RD maps ({time.time() - t0:.0f} s)", flush=True)
    ref_name = next(n for n, d in recs if d["labels"]["gesture"] == "empty" and d["labels"]["fan"] == "off")
    ref = float(np.median(P_all[ref_name][int(WARM_S / fp):]))
    R, D = P_all[recs[0][0]].shape[1:]
    rr = cfg.range_axis(R); nz = np.r_[0:D // 2 - 1, D // 2 + 2:D]
    s1 = s_from_tau(1.0, fpm)
    G = a.fan_range
    res = {"_params": {"fan_range_for_gates": G, "frame_period_measured_s": fpm,
                       "note": "EXPLORATORY, post hoc"}}
    groups = [("front", "off"), ("front", "on"), ("side", "off"), ("side", "on")]

    # ---------- 1. CFAR variants (+ references)
    fns = {"SR + global": lambda P: P / ref, "SR + PCEN tau=1s": lambda P: pcen(P, s=s1, ref=ref),
           "SR + clutter-map": lambda P: clutter_map_stat(P, w=s1)}
    fns.update(methods_cfar())
    rows = score_rows(P_all, recs, rr, nz, fp, fns, G)
    res["cfar_variants_auc"] = {f"{p}_{f}": group_auc(rows, list(fns), p, f) for p, f in groups}
    print(f"1. CFAR variants ({time.time() - t0:.0f} s)", flush=True)
    rows_tape = score_rows(P_all, recs, rr, nz, fp,
                           {k: fns[k] for k in ["SR + PCEN tau=1s", "CA 2x2/8x4 (paper)", "OS-CFAR k=0.75",
                                                "GO-CFAR 2x2/8x4", "SO-CFAR 2x2/8x4"]}, 0.8)
    res["cfar_variants_auc_tape_gates"] = {"side_on": group_auc(rows_tape, ["SR + PCEN tau=1s", "CA 2x2/8x4 (paper)",
                                                                           "OS-CFAR k=0.75", "GO-CFAR 2x2/8x4",
                                                                           "SO-CFAR 2x2/8x4"], "side", "on")}
    print(f"   tape gates ({time.time() - t0:.0f} s)", flush=True)

    # ---------- 2. alpha / tau sweeps
    alphas = [0.0, 0.25, 0.5, 0.75, 0.9, 0.98, 1.0]
    taus = [0.05, 0.1, 0.13, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    sw = {f"pcen_a{al}": (lambda al: (lambda P: pcen(P, s=s1, alpha=al, ref=ref)))(al) for al in alphas}
    sw.update({f"pcen_t{t}": (lambda t: (lambda P: pcen(P, s=s_from_tau(t, fpm), ref=ref)))(t) for t in taus})
    sw.update({f"clut_t{t}": (lambda t: (lambda P: clutter_map_stat(P, w=s_from_tau(t, fpm))))(t) for t in taus})
    sw.update({"pcen_d0.5_r0.25": lambda P: pcen(P, s=s1, delta=0.5, r=0.25, ref=ref),
               "pcen_d10_r1": lambda P: pcen(P, s=s1, delta=10.0, r=1.0, ref=ref)})
    rows_sw = score_rows(P_all, recs, rr, nz, fp, sw, G)
    res["sweep_auc"] = {f"{p}_{f}": group_auc(rows_sw, list(sw), p, f) for p, f in groups}
    # fan false-alarm increase for the sweeps (empty blocks, fan gate, threshold at 1e-3 fan off)
    fa = {}
    for pos_ in ["front", "side"]:
        off = next(n for n, d in recs if d["labels"]["gesture"] == "empty" and d["labels"]["fan"] == "off"
                   and d["labels"]["position"] == pos_)
        on = next(n for n, d in recs if d["labels"]["gesture"] == "empty" and d["labels"]["fan"] == "on"
                  and d["labels"]["position"] == pos_)
        fg = gate(rr, scene_gates(G)["fan"]); w = int(WARM_S / fp)
        fa[pos_] = {}
        for k, f in list(sw.items()) + list(fns.items()):
            Soff = f(P_all[off])[w:][:, :, nz]; Son = f(P_all[on])[w:][:, fg][:, :, nz]
            thr = np.quantile(Soff.ravel(), 1 - 1e-3)
            fa[pos_][k] = float(10 * np.log10(max((Son > thr).mean(), 1e-6) / 1e-3))
    res["fan_fa_increase_db"] = fa
    print(f"2. sweeps ({time.time() - t0:.0f} s)", flush=True)

    # ---------- 3. operating points + ROC (side, fan on)
    keys = ["SR + global", "CA 2x2/8x4 (paper)", "OS-CFAR k=0.75", "SR + clutter-map", "SR + PCEN tau=1s"]
    op, roc = {}, {}
    for p, f in groups:
        sel = [r for r in rows if r["position"] == p and r["fan"] == f]
        op[f"{p}_{f}"] = {}
        for k in keys:
            P_ = np.array([r[k] for r in sel if r["pos"]]); N_ = np.array([r[k] for r in sel if not r["pos"]])
            op[f"{p}_{f}"][k] = {f"pd_at_far{int(far * 100)}": float((P_ > np.quantile(N_, 1 - far)).mean())
                                 for far in [0.01, 0.05, 0.10]}
            if (p, f) == ("side", "on"):
                thr = np.unique(np.concatenate([P_, N_]))[::-1]
                roc[k] = {"far": [float((N_ > t).mean()) for t in thr] + [1.0],
                          "pd": [float((P_ > t).mean()) for t in thr] + [1.0]}
    res["operating_points"] = op
    json.dump(roc, open(out / "roc_side_on.json", "w"))

    # ---------- 4. per gesture (side & front, fan on/off)
    pg = {}
    for p, f in groups:
        sel = [r for r in rows if r["position"] == p and r["fan"] == f]
        N_ = {k: np.array([r[k] for r in sel if not r["pos"]]) for k in keys}
        pg[f"{p}_{f}"] = {g_: {k: auc(np.array([r[k] for r in sel if r["pos"] and r["gesture"] == g_]), N_[k])
                               for k in keys} for g_ in ["swipe", "push", "circle"]}
    res["per_gesture_auc"] = pg

    # ---------- 5. cluster bootstrap for the primary contrasts
    cb = {}
    for p in ["front", "side"]:
        for b in ["CA 2x2/8x4 (paper)", "SR + global", "OS-CFAR k=0.75"]:
            cb[f"{p}_on: PCEN vs {b}"] = cluster_boot(rows, "SR + PCEN tau=1s", b, p, "on")
    res["cluster_bootstrap"] = cb
    print(f"3-5 ({time.time() - t0:.0f} s)", flush=True)

    # ---------- 6. runtime per frame (streaming, 64x64)
    X = P_all[recs[1][0]][:400]
    rt = {}
    st = ClutterMapState(s=s1); t = time.perf_counter()
    for x in X: st.step(x)
    rt["PCEN (streaming)"] = (time.perf_counter() - t) / len(X) * 1e3
    Q = X[0].astype(np.float64); t = time.perf_counter()
    for x in X:
        _ = x / (Q + 1e-30); Q = (1 - s1) * Q + s1 * x
    rt["clutter-map (streaming)"] = (time.perf_counter() - t) / len(X) * 1e3
    for k in ["CA 2x2/8x4 (paper)", "GO-CFAR 2x2/8x4", "OS-CFAR k=0.75"]:
        f = methods_cfar()[k]; t = time.perf_counter(); f(X[:100])
        rt[k] = (time.perf_counter() - t) / 100 * 1e3
    res["runtime_ms_per_frame"] = rt
    (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    json.dump(rows, open(out / "windows_cfar.json", "w"))

    # ---------- summary
    L = ["# Fan session: exploratory follow-up analyses (post hoc)", "",
         f"Gates from fan distance {G} m on the radar axis (as in the primary analysis).", "",
         "## 1. CFAR fairness: detection AUC", "", "| Method | front off | front on | side off | side on |",
         "|---|---|---|---|---|"]
    for k in fns:
        L.append(f"| {k} | " + " | ".join(f"{res['cfar_variants_auc'][f'{p}_{f}'][k]:.3f}" for p, f in groups) + " |")
    L += ["", "Tape-distance gates (fan 0.8 m), side, fan on: " + ", ".join(
        f"{k} {v:.3f}" for k, v in res["cfar_variants_auc_tape_gates"]["side_on"].items()), "",
          "## 2. PCEN alpha and tau (AUC side on / fan FA increase side [dB])", "",
          "| Setting | AUC side on | AUC front on | FA increase front | FA increase side |", "|---|---|---|---|---|"]
    for k in sw:
        L.append(f"| {k} | {res['sweep_auc']['side_on'][k]:.3f} | {res['sweep_auc']['front_on'][k]:.3f} | "
                 f"{fa['front'][k]:+.1f} | {fa['side'][k]:+.1f} |")
    L += ["", "## 3. Pd at fixed window false-alarm rate (side, fan on)", "", "| Method | Pd@1% | Pd@5% | Pd@10% |",
          "|---|---|---|---|"]
    for k in keys:
        o = op["side_on"][k]
        L.append(f"| {k} | {o['pd_at_far1']:.2f} | {o['pd_at_far5']:.2f} | {o['pd_at_far10']:.2f} |")
    L += ["", "## 4. AUC per gesture (side, fan on)", "", "| Method | swipe | push | circle |", "|---|---|---|---|"]
    for k in keys:
        L.append(f"| {k} | " + " | ".join(f"{pg['side_on'][g_][k]:.3f}" for g_ in ["swipe", "push", "circle"]) + " |")
    L += ["", "## 5. Cluster bootstrap (clusters of 5 repetitions)", "", "| Contrast | clusters | 95% CI | p one-sided |",
          "|---|---|---|---|"]
    for k, v in cb.items():
        L.append(f"| {k} | {v['n_clusters']} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | {v['p_one_sided']:.4f} |")
    L += ["", "## 6. Runtime per 64x64 frame (numpy, this machine)", ""]
    L += [f"- {k}: {v:.3f} ms" for k, v in rt.items()]
    (out / "results.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
