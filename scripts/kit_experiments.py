"""PCEN vs. static removal / CFAR front ends on the KIT FC-FMCW gesture data set (BGT60TR13C, raw IF).

PRE-REGISTERED (fixed before any result was looked at)
------------------------------------------------------
Input maps per recording (pcen_radar/kit.py): micro-Doppler md (T x 128, hand gate 0.15-0.75 m) and
range-time rt (T x 64), each without ("raw") and with ("sr") static removal; ~52 time steps/s.

Front ends (applied identically to md and rt):
  log_raw         log10(X / median(X))
  log_sr          same on static-removed maps
  cacfar_sr       log10(CA-CFAR statistic) on sr maps; 2-D ring in (time, channel): guard (2,2), train (8,4)
  clutmap_raw     log10(X_t / Q_{t-1}), Q = exponential average over time (w = s below)
  pcen_raw        PCEN over time per channel: tau = 1.0 s, alpha 0.98, delta 2, r 0.5
  pcen_sr         same PCEN on sr maps
  pcen_sr_cacfar  CA-CFAR statistic (log) computed on the pcen_sr output

Features: channels average-pooled to 32 (md: 128->32, rt: 64->32); per channel mean, std, max, 90th pct
over time -> 2 x 32 x 4 = 256 features (time-shift invariant).
Classifiers: (a) standardised logistic regression, C=1; (b) ExtraTrees, 500 trees. Seeds fixed.

Experiments:
  E1 within-setup  : part 2, setup 1 (wooden wall), 240 rec., 6 classes, 5-fold stratified CV
  E2 cross-setup   : MAIN. train setup 1 (240) -> test setups 4/5/6/7 (231; window, shelves, free space)
  E3 cross-angle   : part 1, train 3v1 (0 deg) -> test 3v2 (40 deg) and reverse, 7 classes, 105 each
  E4 background identifiability: predict background type (window / shelves / free space) from the
                     features of the 231 setup-4..7 recordings, 5-fold CV. LOWER = background suppressed
  E5 gesture-vs-none detection without training: score = max over time of the hand-gate (md) output
                     minus its median; AUC gestures vs. 'n' on the 231 cross-setup recordings
Statistics: paired bootstrap (2000 resamples) of accuracy differences vs. log_raw and log_sr on E2.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcen_radar.pcen import pcen, s_from_tau

FEAT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/claude/kit/feat")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/claude/kit/results")
OUT.mkdir(parents=True, exist_ok=True)
TAU, ALPHA, DELTA, R = 1.0, 0.98, 2.0, 0.5
FRONTENDS = ["log_raw", "log_sr", "cacfar_sr", "clutmap_raw", "pcen_raw", "pcen_sr", "pcen_sr_cacfar"]
BG = {"4v1": "window", "4v2": "window", "4v3": "window", "4v4": "window", "5": "shelves",
      "6": "shelves", "7v1": "free space", "7v2": "free space"}


def ring_mean(X, guard=(2, 2), train=(8, 4), wrap_ch=False):
    gr, gd = guard; tr, td = train
    k = np.ones((2 * (gr + tr) + 1, 2 * (gd + td) + 1))
    k[tr:tr + 2 * gr + 1, td:td + 2 * gd + 1] = 0
    k /= k.sum()
    if wrap_ch:
        Xp = np.pad(X, ((0, 0), (gd + td,) * 2), mode="wrap")
        m = ndimage.correlate(Xp, k, mode="reflect")[:, gd + td:-(gd + td)]
    else:
        m = ndimage.correlate(X, k, mode="reflect")
    return m


def clutter_map(X, w):
    Q = X[:10].mean(0)
    out = np.empty_like(X)
    for t in range(len(X)):
        out[t] = X[t] / (Q + 1e-30)
        Q = (1 - w) * Q + w * X[t]
    return out


def frontend(name, raw, sr, dt, wrap_ch):
    s = s_from_tau(TAU, dt)
    eps = 1e-30
    if name == "log_raw":
        return np.log10(raw / (np.median(raw) + eps) + 1e-6)
    if name == "log_sr":
        return np.log10(sr / (np.median(sr) + eps) + 1e-6)
    if name == "cacfar_sr":
        return np.log10(sr / (ring_mean(sr, wrap_ch=wrap_ch) + eps) + 1e-6)
    if name == "clutmap_raw":
        return np.log10(clutter_map(raw.astype(np.float64), s) + 1e-6)
    if name == "pcen_raw":
        return pcen(raw, s=s, alpha=ALPHA, delta=DELTA, r=R)
    if name == "pcen_sr":
        return pcen(sr, s=s, alpha=ALPHA, delta=DELTA, r=R)
    if name == "pcen_sr_cacfar":
        P = pcen(sr, s=s, alpha=ALPHA, delta=DELTA, r=R).astype(np.float64)
        P = P - P.min() + 1e-6
        return np.log10(P / (ring_mean(P, wrap_ch=wrap_ch) + eps))
    raise ValueError(name)


def pool_ch(X, n=32):
    T, C = X.shape
    return X.reshape(T, n, C // n).mean(2)


def feats(Y):
    Z = pool_ch(Y)
    return np.concatenate([Z.mean(0), Z.std(0), Z.max(0), np.percentile(Z, 90, axis=0)])


def load_all():
    recs = []
    for p in sorted(glob.glob(str(FEAT_DIR / "*.npz"))):
        z = np.load(p)
        recs.append({k: z[k] for k in z.files} | {"name": Path(p).stem})
    return recs


def main():
    recs = load_all()
    print(f"{len(recs)} recordings")
    meta = [{"name": r["name"], "gesture": str(r["gesture"]), "setup": str(r["setup"]),
             "part": str(r["part"])} for r in recs]
    F, det = {}, {}
    for fe in FRONTENDS:
        rows, dscore = [], []
        for r in recs:
            dt = float(r["dt_s"])
            ymd = frontend(fe, r["md_raw"].astype(np.float64), r["md_sr"].astype(np.float64), dt, True)
            yrt = frontend(fe, r["rt_raw"].astype(np.float64), r["rt_sr"].astype(np.float64), dt, False)
            rows.append(np.concatenate([feats(ymd), feats(yrt)]))
            e = ymd.mean(1)                                  # hand-gate activity over time
            dscore.append(float(e.max() - np.median(e)))
        F[fe] = np.nan_to_num(np.asarray(rows, np.float32))
        det[fe] = np.asarray(dscore)
        print("  front end", fe, F[fe].shape)

    g = np.array([m["gesture"] for m in meta]); su = np.array([m["setup"] for m in meta])
    part = np.array([m["part"] for m in meta])
    p2s1 = (part == "part2") & (su == "1")
    p2x = (part == "part2") & (su != "1")
    v1 = (part == "part1") & (su == "3v1"); v2 = (part == "part1") & (su == "3v2")
    print("E1", p2s1.sum(), "E2 test", p2x.sum(), "E3", v1.sum(), v2.sum())

    def clfs(seed=0):
        return {"LR": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
                "ET": ExtraTreesClassifier(n_estimators=500, random_state=seed, n_jobs=-1)}

    res = {fe: {} for fe in FRONTENDS}
    preds_e2 = {}
    for fe in FRONTENDS:
        X = F[fe]
        for cn in ["LR", "ET"]:
            # E1
            cv = StratifiedKFold(5, shuffle=True, random_state=0)
            pr = cross_val_predict(clfs()[cn], X[p2s1], g[p2s1], cv=cv)
            res[fe][f"E1_{cn}"] = float((pr == g[p2s1]).mean())
            # E2
            m = clfs()[cn].fit(X[p2s1], g[p2s1]); pe = m.predict(X[p2x])
            preds_e2[(fe, cn)] = (pe == g[p2x])
            res[fe][f"E2_{cn}"] = float((pe == g[p2x]).mean())
            for bg in ["window", "shelves", "free space"]:
                sel = np.array([BG[s] == bg for s in su[p2x]])
                res[fe][f"E2_{cn}_{bg}"] = float((pe[sel] == g[p2x][sel]).mean())
            # E3
            a = clfs()[cn].fit(X[v1], g[v1]).score(X[v2], g[v2])
            b = clfs()[cn].fit(X[v2], g[v2]).score(X[v1], g[v1])
            res[fe][f"E3_{cn}"] = float((a + b) / 2)
        # E4 background identifiability (LR only, pre-registered)
        bgl = np.array([BG[s] for s in su[p2x]])
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        pr = cross_val_predict(clfs()["LR"], F[fe][p2x], bgl, cv=cv)
        res[fe]["E4_bg_id_acc"] = float((pr == bgl).mean())
        # E5 detection AUC
        y = (g[p2x] != "n").astype(int)
        res[fe]["E5_auc"] = float(roc_auc_score(y, det[fe][p2x]))
        print(f"{fe:15s} " + " ".join(f"{k}={v:.3f}" for k, v in res[fe].items() if "_" in k and k.count("_") <= 2))

    # paired bootstrap on E2
    rng = np.random.default_rng(0)
    n = p2x.sum(); B = 2000
    idx = rng.integers(0, n, size=(B, n))
    boot = {}
    for cn in ["LR", "ET"]:
        for ref in ["log_raw", "log_sr"]:
            for fe in FRONTENDS:
                if fe == ref:
                    continue
                d = preds_e2[(fe, cn)].astype(float) - preds_e2[(ref, cn)].astype(float)
                bs = d[idx].mean(1)
                boot[f"{cn}:{fe}-{ref}"] = {"diff": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)),
                                            float(np.percentile(bs, 97.5))]}
    chance = {"E2": float(max(np.mean(g[p2x] == c) for c in np.unique(g[p2x]))),
              "E4": float(max(np.mean(np.array([BG[s] for s in su[p2x]]) == c)
                              for c in ["window", "shelves", "free space"]))}
    json.dump({"results": res, "bootstrap_E2": boot, "majority_baseline": chance,
               "n": {"E1": int(p2s1.sum()), "E2_test": int(n), "E3": [int(v1.sum()), int(v2.sum())]},
               "params": {"tau": TAU, "alpha": ALPHA, "delta": DELTA, "r": R}},
              open(OUT / "kit_results.json", "w"), indent=1)
    np.savez_compressed(OUT / "kit_features.npz", **{fe: F[fe] for fe in FRONTENDS},
                        gesture=g, setup=su, part=part)
    print("bootstrap:")
    for k, v in boot.items():
        print(f"  {k:32s} {v['diff']:+.3f}  [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    print("majority baselines", chance)


if __name__ == "__main__":
    main()
