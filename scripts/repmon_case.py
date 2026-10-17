"""Case study: PCEN in radar vital-sign monitoring (RepMon) - persistent, nearly static targets.

  python scripts/repmon_case.py --calib <RepMon>/calibration/cecil/session.npz \
                                --walabot <5_data>/processed/walabot_slowtime --out results/repmon_case

Why: gestures are transient, so a temporal normalizer (PCEN, clutter map) helps. A seated, breathing
person is the opposite: present for minutes with mm-scale motion. Hypothesis (fixed 2026-10-16,
before running): PCEN "forgets" a still person after about tau, so a short tau (<= 10 s) hurts presence
detection / localization, and only tau much longer than the monitoring time (or no adaptation,
tau = inf: per-cell normalization by an empty-room background) can match static removal.

Common signal path (RepMon): slow-time complex range profiles (T, channels, range) at 10 Hz, static
clutter removal, residual power per frame E(t, r) summed over channels; profile = mean of E over the
last 20 s (RepMon's "dynamic profile"). Variants change only what is averaged:
  SR          E itself (RepMon baseline; causal running mean with 20 s time constant as clutter estimate)
  PCEN tau    PCEN(E) with alpha .98, delta 2, r .5, M initialized from the empty-room background
  CMAP tau    clutter map E / Q (same initialization), for reference
  tau in {1, 10, 30, 100, inf} s

R1 (own BGT60TR13C, RepMon calibration session "cecil"): 35 s empty room, then the participant sits at
   the monitoring position for 5 min (rest + paced breathing). Person bins = range bins where the SR
   profile during rest exceeds the empty-room profile by >= 6 dB (method-neutral definition).
   Metrics: presence AUC (1-s steps, all sitting phases vs the empty phase) of the person-bin contrast
   (mean profile in person bins / median of the other bins), and the contrast 10, 60 and 300 s after
   the participant sat down.
R2 (Walabot multi-person data set, Schroth et al., 59 recordings, free space / two obstacles, 1-5
   persons, reference ranges): RepMon's localization (OS-CFAR along range on the profile, local maxima,
   merged within 0.3 m; 30-s windows, hop 5 s, profile over the last 20 s; hit = detection within
   0.3 m of a reference range). Background for M: the empty recording of the same obstacle condition
   (m0 / m0-1 / m0-2). Metric: detection rate at matched false detections per window (0.1 and 0.5),
   obtained by sweeping the CFAR scale; plus the rate at RepMon's default 6 dB.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FS = 10.0
TAUS = [1.0, 10.0, 30.0, 100.0, np.inf]
ALPHA, DELTA, R_ = 0.98, 2.0, 0.5


# ---------------------------------------------------------------- common signal path
def residual_power(st, tau_sr=20.0, init=None):
    """Causal static removal: running complex mean (EMA, tau_sr); E(t, r) = sum_ch |x - mean_(t-1)|^2."""
    a = 1 - np.exp(-1 / (FS * tau_sr))
    m = st[: int(FS * 2)].mean(0) if init is None else init
    E = np.empty((st.shape[0], st.shape[2]))
    for t in range(st.shape[0]):
        d = st[t] - m
        E[t] = (np.abs(d) ** 2).sum(0)
        m = (1 - a) * m + a * st[t]
    return E, m


def normalize(E, kind, tau, M0, ref):
    """E: (T, R) residual power. kind: 'SR' | 'PCEN' | 'CMAP'. M0: (R,) background energy."""
    if kind == "SR":
        return E / ref
    s = 0.0 if not np.isfinite(tau) else 1 - np.exp(-1 / (FS * tau))
    M = M0.copy() / ref
    out = np.empty_like(E)
    for t in range(len(E)):
        e = E[t] / ref
        if kind == "PCEN":
            M = (1 - s) * M + s * e
            out[t] = (e / (1e-6 + M) ** ALPHA + DELTA) ** R_ - DELTA ** R_
        else:  # clutter map: compare with the map before the update
            out[t] = e / (M + 1e-12)
            M = (1 - s) * M + s * e
    return out


def methods():
    m = [("SR", "SR", None)]
    for t in TAUS:
        lab = "inf" if not np.isfinite(t) else f"{t:g}"
        m.append((f"PCEN tau={lab}s", "PCEN", t))
    for t in [10.0, np.inf]:
        lab = "inf" if not np.isfinite(t) else f"{t:g}"
        m.append((f"CMAP tau={lab}s", "CMAP", t))
    return m


# ---------------------------------------------------------------- RepMon OS-CFAR (copied from localize.py)
def cfar_peaks(p, guard=2, train=10, scale_db=6.0, min_bin=0, os_pct=50.0):
    R = len(p); out = []; k = 10 ** (scale_db / 10)
    for i in range(max(min_bin, 1), R - 1):
        if not (p[i] >= p[i - 1] and p[i] >= p[i + 1]):
            continue
        ref = np.concatenate([p[max(0, i - guard - train): max(0, i - guard)], p[i + guard + 1: i + guard + 1 + train]])
        if len(ref) < 3 or p[i] < k * np.percentile(ref, os_pct):
            continue
        out.append(i)
    return out


def merge_peaks(idx, p, rm, min_sep):
    keep = []
    for i in sorted(idx, key=lambda j: -p[j]):
        if all(abs(rm[i] - rm[j]) >= min_sep for j in keep):
            keep.append(i)
    return sorted(keep)


# ---------------------------------------------------------------- R1: own BGT60 calibration session
def run_r1(path):
    z = np.load(path)
    st = z["slowtime"].astype(np.complex128); rm = z["range_m"]
    ph = json.loads(str(z["phases"])); t0 = 0; spans = {}
    for p in ph:
        spans[p["name"]] = (t0, t0 + p["dur_s"]); t0 += p["dur_s"]
    e0, e1 = spans["empty"]; sit = spans["rest"][0]
    # background from the empty phase (causal SR initialized on it)
    E, _ = residual_power(st, init=st[int(e0 * FS): int(e0 * FS) + 20].mean(0))
    M0 = E[int((e0 + 5) * FS): int(e1 * FS)].mean(0)
    ref = float(np.median(M0))
    W = int(20 * FS)
    secs = np.arange(int(e0 + 25), int(t0))                   # profile needs 20 s of history
    prof_sr = np.array([E[int(s * FS) - W: int(s * FS)].mean(0) for s in secs])
    emp = prof_sr[(secs >= e0 + 25) & (secs <= e1)].mean(0)
    rst = prof_sr[(secs >= spans["rest"][0] + 20) & (secs <= spans["rest"][1])].mean(0)
    person = 10 * np.log10(rst / emp) >= 6.0
    if person.sum() == 0:
        person = 10 * np.log10(rst / emp) >= np.sort(10 * np.log10(rst / emp))[-3]
    out = {"person_bins_m": [float(x) for x in rm[person]], "methods": {}}
    sitting = secs >= sit
    is_empty = (secs >= e0 + 25) & (secs <= e1)
    curves = {}
    for name, kind, tau in methods():
        Y = normalize(E, kind, tau, M0, ref)
        P = np.array([Y[int(s * FS) - W: int(s * FS)].mean(0) for s in secs])
        c = 10 * np.log10(P[:, person].mean(1) / (np.median(P[:, ~person], axis=1) + 1e-12) + 1e-12)
        pos, neg = c[sitting], c[is_empty]
        auc = float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())
        def at(dt):
            i = np.argmin(np.abs(secs - (sit + dt))); return float(c[i])
        out["methods"][name] = {"presence_auc": auc, "contrast_db_10s": at(10), "contrast_db_60s": at(60),
                                "contrast_db_300s": at(min(300, t0 - sit - 1)),
                                "contrast_db_empty_median": float(np.median(neg))}
        curves[name] = c.tolist()
    out["curve_secs"] = secs.tolist(); out["curves"] = curves; out["sit_down_s"] = sit
    return out


# ---------------------------------------------------------------- R2: Walabot localization
def obstacle_class(s):
    s = (s or "").strip()
    return {"0": "free", "1": "obst1", "2": "obst2"}.get(s, "free")


def run_r2(folder):
    rows = list(csv.DictReader(open(Path(folder) / "index.csv", newline="")))
    bg = {}
    for ob, f in [("free", "m0"), ("obst1", "m0-1"), ("obst2", "m0-2")]:
        zz = np.load(Path(folder) / f"{f}.npz"); s0 = zz["slowtime"].astype(np.complex128)
        Eb, mb = residual_power(s0, init=s0.mean(0))
        bg[ob] = (Eb[20:].mean(0), s0.mean(0))
    scales = np.arange(0.0, 24.5, 0.5)
    tally = {}   # (method, obstacle, scale) -> [ref, hit, false, windows]
    for row in rows:
        z = np.load(Path(folder) / row["file"]); refs = json.loads(str(z["refs"]))
        if not refs:
            continue
        st = z["slowtime"].astype(np.complex128); rm = z["range_m"].astype(np.float64)
        ob = obstacle_class(row["obstacle"])
        M0, mean0 = bg[ob]
        E, _ = residual_power(st, init=mean0)
        ref = float(np.median(M0)); min_bin = int(np.searchsorted(rm, 0.25))
        Ys = {name: normalize(E, kind, tau, M0, ref) for name, kind, tau in methods()}
        for s_end in np.arange(30.0, st.shape[0] / FS + 1e-9, 5.0):
            e = int(s_end * FS)
            refr = []
            for k, r in refs.items():
                i = min(int(s_end), len(r["range_m"]) - 1); v = r["range_m"][i]
                if v is not None and np.isfinite(v):
                    refr.append(v)
            for name in Ys:
                p = Ys[name][e - int(20 * FS): e].mean(0)
                p = p - min(0.0, p.min()) + 1e-9
                for sc in scales:
                    idx = merge_peaks(cfar_peaks(p, scale_db=sc, min_bin=min_bin), p, rm, 0.3)
                    idx = sorted(idx, key=lambda i: -p[i])[:5]
                    dr = rm[idx]
                    hit = sum(1 for v in refr if np.any(np.abs(dr - v) <= 0.3))
                    matched = sum(1 for d in dr if any(abs(d - v) <= 0.3 for v in refr))
                    key = (name, ob, float(sc))
                    t = tally.setdefault(key, [0, 0, 0, 0])
                    t[0] += len(refr); t[1] += hit; t[2] += len(dr) - matched; t[3] += 1
        print(f"  {row['id']:6s} {ob:5s} persons {len(refs)}", flush=True)
    res = {}
    for name, _, _ in methods():
        res[name] = {}
        for ob in ["free", "obst1", "obst2", "all"]:
            obs = ["free", "obst1", "obst2"] if ob == "all" else [ob]
            pts = []
            for sc in scales:
                t = np.sum([tally[(name, o, float(sc))] for o in obs if (name, o, float(sc)) in tally], axis=0)
                pts.append((t[2] / t[3], t[1] / t[0], sc))
            pts = sorted(pts)
            fpw = np.array([q[0] for q in pts]); dr = np.array([q[1] for q in pts])
            def at(f):
                ok = fpw <= f
                return float(dr[ok].max()) if ok.any() else 0.0
            d6 = [q for q in pts if abs(q[2] - 6.0) < 1e-9][0]
            res[name][ob] = {"det_at_fpw0.1": at(0.1), "det_at_fpw0.5": at(0.5), "det_at_6dB": float(d6[1]),
                             "fpw_at_6dB": float(d6[0]), "curve": [[float(a), float(b)] for a, b, _ in pts]}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--walabot", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "repmon_case"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r1 = run_r1(a.calib)
    print("R1 person bins", r1["person_bins_m"])
    for k, v in r1["methods"].items():
        print(f"  {k:18s} AUC {v['presence_auc']:.3f}  contrast 10s {v['contrast_db_10s']:+5.1f}  60s {v['contrast_db_60s']:+5.1f}"
              f"  300s {v['contrast_db_300s']:+5.1f}  empty {v['contrast_db_empty_median']:+5.1f} dB", flush=True)
    r2 = run_r2(a.walabot)
    (out / "results.json").write_text(json.dumps({"R1_bgt60_calibration": r1, "R2_walabot": r2}, indent=1),
                                      encoding="utf-8")
    L = ["# RepMon case study: PCEN on persistent (breathing) targets", "",
         f"## R1 own BGT60TR13C calibration session (person bins {r1['person_bins_m'][0]:.2f}-{r1['person_bins_m'][-1]:.2f} m)", "",
         "| Front end | presence AUC | contrast 10 s | 60 s | 300 s after sitting | empty room |", "|---|---|---|---|---|---|"]
    for k, v in r1["methods"].items():
        L.append(f"| {k} | {v['presence_auc']:.3f} | {v['contrast_db_10s']:+.1f} dB | {v['contrast_db_60s']:+.1f} dB | "
                 f"{v['contrast_db_300s']:+.1f} dB | {v['contrast_db_empty_median']:+.1f} dB |")
    L += ["", "## R2 Walabot localization (detection rate of reference persons)", "",
          "| Front end | free @0.1 FPW | obst1 @0.1 | obst2 @0.1 | all @0.1 | all @0.5 | all @6 dB (FPW) |", "|---|---|---|---|---|---|---|"]
    for k, v in r2.items():
        L.append(f"| {k} | {v['free']['det_at_fpw0.1']:.2f} | {v['obst1']['det_at_fpw0.1']:.2f} | {v['obst2']['det_at_fpw0.1']:.2f} | "
                 f"{v['all']['det_at_fpw0.1']:.2f} | {v['all']['det_at_fpw0.5']:.2f} | {v['all']['det_at_6dB']:.2f} ({v['all']['fpw_at_6dB']:.2f}) |")
    (out / "results.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
