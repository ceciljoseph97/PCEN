"""Ablation of the PCEN components on the KIT gesture data (hand-crafted features, numpy/sklearn only).

  python scripts/kit_ablation.py --maps results/kit/maps --out results/kit_ablation

Input: the packed maps of kit_train_cnn.py (per recording 256 time steps x 64 Doppler / 64 range bins,
static-removed (SR) power, dt = 0.0235 s). Every variant starts from SR maps; only the normalization
changes. Features, splits and classifiers as in kit_experiments.py (E2: train setup 1 -> test setups
4-7, ET 500 trees (5 seeds) and LR; E4: background-type prediction, LR, 5-fold, lower = better).

Variants (fixed before running, 2026-10-16):
  A0 SR + log                         no normalization (reference)
  A1 SR + PCEN full                   tau 1 s, alpha .98, delta 2, r .5   (paper)
  A2 - compression                    log(E / M^alpha)        (temporal normalization only)
  A3 - temporal normalization         alpha = 0 (compression only)
  A4 partial normalization            alpha = 0.5
  A5 linear output                    r = 1 (E / M^alpha, no root, no log)
  A6 stronger compression             r = 0.25
  A7 tau = 0.13 s   A8 tau = 4 s       time constant
  A9 clutter-map CFAR on SR           log(E_t / Q_(t-1))     (alpha = 1, causal, no compression)
Reported: E2 accuracy (ET mean of 5 seeds, LR), paired bootstrap CI of A_k - A1 (ET seed 0), E4.
Note: this uses the resampled maps, so absolute numbers differ slightly from Table I.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcen_radar.pcen import pcen, s_from_tau

DT = 0.0235
BG = {"4v1": "window", "4v2": "window", "4v3": "window", "4v4": "window", "5": "shelves",
      "6": "shelves", "7v1": "free space", "7v2": "free space"}


def clutmap(E, s):
    Q = E[:10].mean(0)
    out = np.empty_like(E)
    for t in range(len(E)):
        out[t] = E[t] / (Q + 1e-30)
        Q = (1 - s) * Q + s * E[t]
    return out


def variant(name, E):
    """E: (T, C) SR power of one map. Returns (T, C) front-end output."""
    ref = float(np.median(E[:10])) + 1e-30
    s1 = s_from_tau(1.0, DT)
    if name == "A0 SR + log":
        return np.log10(E / (np.median(E) + 1e-30) + 1e-6)
    if name == "A1 PCEN full":
        return pcen(E, s=s1, ref=ref)
    if name == "A2 - compression (log E/M^a)":
        _, M = pcen(E, s=s1, ref=ref, return_M=True)
        return np.log10((E / ref) / (1e-6 + M) ** 0.98 + 1e-6)
    if name == "A3 - temporal norm. (alpha=0)":
        return pcen(E, s=s1, alpha=0.0, ref=ref)
    if name == "A4 alpha=0.5":
        return pcen(E, s=s1, alpha=0.5, ref=ref)
    if name == "A5 linear output (r=1)":
        return pcen(E, s=s1, r=1.0, ref=ref)
    if name == "A6 r=0.25":
        return pcen(E, s=s1, r=0.25, ref=ref)
    if name == "A7 tau=0.13 s":
        return pcen(E, s=s_from_tau(0.13, DT), ref=ref)
    if name == "A8 tau=4 s":
        return pcen(E, s=s_from_tau(4.0, DT), ref=ref)
    if name == "A9 clutter-map (log)":
        return np.log10(clutmap(E, s1) + 1e-6)
    raise ValueError(name)


VARIANTS = ["A0 SR + log", "A1 PCEN full", "A2 - compression (log E/M^a)", "A3 - temporal norm. (alpha=0)",
            "A4 alpha=0.5", "A5 linear output (r=1)", "A6 r=0.25", "A7 tau=0.13 s", "A8 tau=4 s",
            "A9 clutter-map (log)"]


def feats(Y, n=32):
    T, C = Y.shape
    Z = Y.reshape(T, n, C // n).mean(2)
    return np.concatenate([Z.mean(0), Z.std(0), Z.max(0), np.percentile(Z, 90, axis=0)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", default=str(Path(__file__).resolve().parents[1] / "results" / "kit" / "maps"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "kit_ablation"))
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    m = Path(a.maps)
    meta = np.load(m / "meta.npz")
    md = np.power(10.0, np.load(m / "md_sr.npz")["log10_power"].astype(np.float64))
    rt = np.power(10.0, np.load(m / "rt_sr.npz")["log10_power"].astype(np.float64))
    g, su, part = meta["gesture"].astype(str), meta["setup"].astype(str), meta["part"].astype(str)
    tr = (part == "part2") & (su == "1"); te = (part == "part2") & (su != "1")
    bgl = np.array([BG[s] for s in su[te]])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res, correct = {}, {}
    for v in VARIANTS:
        X = np.nan_to_num(np.array([np.concatenate([feats(variant(v, md[i])), feats(variant(v, rt[i]))])
                                    for i in range(len(g))], np.float32))
        et = []
        for sd in range(a.seeds):
            clf = ExtraTreesClassifier(500, random_state=sd, n_jobs=-1).fit(X[tr], g[tr])
            p = clf.predict(X[te]); et.append(float((p == g[te]).mean()))
            if sd == 0:
                correct[v] = (p == g[te]).astype(float)
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X[tr], g[tr])
        pr = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
                               X[te], bgl, cv=StratifiedKFold(5, shuffle=True, random_state=0))
        res[v] = {"E2_ET_mean": float(np.mean(et)), "E2_ET_std": float(np.std(et)),
                  "E2_LR": float(lr.score(X[te], g[te])), "E4_bg_id": float((pr == bgl).mean())}
        print(f"{v:32s} E2 ET {np.mean(et):.3f}±{np.std(et):.3f}  LR {res[v]['E2_LR']:.3f}  E4 {res[v]['E4_bg_id']:.3f}",
              flush=True)
    rng = np.random.default_rng(0); n = int(te.sum()); idx = rng.integers(0, n, (2000, n))
    for v in VARIANTS:
        d = correct[v] - correct["A1 PCEN full"]; bs = d[idx].mean(1)
        res[v]["diff_vs_A1_ET_seed0"] = {"diff": float(d.mean()),
                                         "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    L = ["# KIT ablation of PCEN components (SR maps, hand-crafted features)", "",
         "| Variant | E2 ET (5 seeds) | E2 LR | E4 bg-ID (lower better) | ET seed0 − A1 [95% CI] |",
         "|---|---|---|---|---|"]
    for v in VARIANTS:
        r = res[v]; d = r["diff_vs_A1_ET_seed0"]
        L.append(f"| {v} | {r['E2_ET_mean']:.3f} ± {r['E2_ET_std']:.3f} | {r['E2_LR']:.3f} | {r['E4_bg_id']:.3f} | "
                 f"{d['diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] |")
    (out / "results.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
