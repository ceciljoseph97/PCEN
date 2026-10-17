"""Detection metrics computed from test-statistic maps + target masks.

Conventions
-----------
stat  : (T, R, D) test statistic (bigger = more target-like)
target: (T, R, D) bool, cells where the target may legitimately be (e.g. its range gate)
guard : (T, R, D) bool, cells excluded from both Pd and Pfa (target spillover / sidelobes)
present: (T,) bool, frames in which the target is actually there

Pd(thr)  = fraction of target-present frames with max(stat in target region) > thr
Pfa(thr) = fraction of background cells (not target, not guard) with stat > thr
"""

from __future__ import annotations

import numpy as np


def roc(stat, target, guard, present, n_points: int = 400, pfa_min: float = 1e-6):
    stat = np.asarray(stat, dtype=np.float64)
    bg = ~(target | guard)
    bg_vals = stat[bg]
    T = stat.shape[0]
    tmax = np.full(T, -np.inf)
    for t in range(T):
        if present[t] and target[t].any():
            tmax[t] = stat[t][target[t]].max()
    tmax = tmax[present & target.reshape(T, -1).any(1)]

    # thresholds spanning background quantiles (log-spaced in Pfa)
    pfas = np.logspace(np.log10(max(pfa_min, 1.0 / bg_vals.size)), 0, n_points)
    thr = np.quantile(bg_vals, 1 - pfas)
    thr = np.unique(np.concatenate([thr, [bg_vals.max() + 1e-9]]))[::-1]
    srt = np.sort(bg_vals)
    pfa = 1 - np.searchsorted(srt, thr, side="right") / srt.size
    pd = np.array([(tmax > t).mean() if tmax.size else np.nan for t in thr])
    return {"thr": thr, "pfa": pfa, "pd": pd}


def pd_at_pfa(r, target_pfa: float) -> float:
    pfa, pd = r["pfa"], r["pd"]
    idx = np.where(pfa <= target_pfa)[0]
    return float(pd[idx].max()) if idx.size else float("nan")


def thr_at_pfa(stat, target, guard, target_pfa: float) -> float:
    bg_vals = np.asarray(stat)[~(target | guard)]
    return float(np.quantile(bg_vals, 1 - target_pfa))


def regional_pfa(stat, target, guard, thr, regions: dict[str, np.ndarray]):
    """Pfa inside named background regions at one global threshold.
    regions: name -> (R, D) bool mask. A CFAR-like method keeps these roughly equal."""
    stat = np.asarray(stat)
    bg = ~(target | guard)
    out = {}
    for name, m in regions.items():
        sel = bg & m[None]
        out[name] = float((stat[sel] > thr).mean()) if sel.any() else float("nan")
    return out


def pfa_excess_db(regional: dict, nominal: float) -> float:
    """Worst-region false-alarm excess: 10*log10(max regional Pfa / nominal Pfa).
    0 dB = no region exceeds the design Pfa; +10 dB = some region has 10x the false alarms."""
    v = np.array([x for x in regional.values() if np.isfinite(x)])
    return float(10 * np.log10(max(v.max(), 1e-9) / nominal)) if v.size else float("nan")


def detection_over_time(stat, target, thr):
    """(T,) bool: was the target detected in each frame (any cell in target region > thr)."""
    T = stat.shape[0]
    return np.array([(stat[t][target[t]] > thr).any() if target[t].any() else False
                     for t in range(T)])


def default_regions(R: int, D: int, zero_dop_halfwidth: int = 2, near_range_bins: int = 6):
    zd = np.zeros((R, D), bool)
    c = D // 2
    zd[:, c - zero_dop_halfwidth:c + zero_dop_halfwidth + 1] = True
    near = np.zeros((R, D), bool)
    near[:near_range_bins] = True
    far = np.zeros((R, D), bool)
    far[R // 2:] = True
    other = ~(zd | near | far)
    return {"zero-Doppler": zd & ~near, "near-range": near, "far-range": far & ~zd,
            "mid (moving)": other}
