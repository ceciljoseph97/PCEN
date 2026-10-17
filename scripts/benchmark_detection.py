"""Experiment A: PCEN vs. static clutter removal + CA/OS-CFAR vs. clutter-map CFAR (detection).

Usage
  python scripts/benchmark_detection.py --sim                       # synthetic scene, known truth
  python scripts/benchmark_detection.py data/det/*.npz              # your BGT60TR13C recordings
  python scripts/benchmark_detection.py data/det/*.npz --out results/room_A

Recordings need labels (see pcen_radar/io.py): a target range gate per frame span and,
ideally, which spans are 'moving'. Record them with scripts/record.py --target-range ...

Outputs (in --out): roc.png, regional_pfa.png, dwell.png, example_maps.png, results.json, results.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcen_radar import RadarConfig, rd_sequence, pcen, ca_cfar_stat, os_cfar_stat, clutter_map_stat
from pcen_radar.cfar import n_train_cells, ca_cfar_alpha
from pcen_radar.io import load_recording, make_masks, per_frame_range_masks
from pcen_radar.metrics import (roc, pd_at_pfa, thr_at_pfa, regional_pfa, pfa_excess_db,
                                detection_over_time, default_regions)
from pcen_radar.processing import to_db
from pcen_radar.pcen import s_from_tau


def build_methods(a):
    g, tr = (a.guard_r, a.guard_d), (a.train_r, a.train_d)
    s = s_from_tau(a.tau, a.frame_period)
    pc = dict(s=s, alpha=a.alpha, delta=a.delta, r=a.r)
    # each method: name -> (input key, function over the whole (T,R,D) sequence)
    return {
        "Raw (global thr)":         ("raw", lambda X: X),
        "Static removal (global)":  ("mr", lambda X: X),
        "Static removal + CA-CFAR": ("mr", lambda X: np.stack([ca_cfar_stat(x, g, tr) for x in X])),
        "Static removal + OS-CFAR": ("mr", lambda X: np.stack([os_cfar_stat(x, g, tr, a.os_k) for x in X])),
        "Clutter-map CFAR":         ("raw", lambda X: clutter_map_stat(X, w=s)),
        "PCEN (raw)":               ("raw", lambda X: pcen(X, **pc)),
        "Static removal + PCEN":    ("mr", lambda X: pcen(X, **pc)),
        "Static removal + PCEN + CA-CFAR": ("mr", lambda X: np.stack(
            [ca_cfar_stat(x - x.min() + 1e-6, g, tr) for x in pcen(X, **pc)])),
    }


def run(P_raw, P_mr, target_mv, guard_mv, target_all, guard_all, moving, present, cfg, a, out: Path,
        tag=""):
    out.mkdir(parents=True, exist_ok=True)
    methods = build_methods(a)
    T, R, D = P_raw.shape
    regions = default_regions(R, D)
    res, stats = {}, {}
    for name, (key, fn) in methods.items():
        X = P_raw if key == "raw" else P_mr
        t0 = time.perf_counter()
        S = fn(X)
        dt = (time.perf_counter() - t0) / T * 1e3
        # skip the first frames (IIR / clutter-map warm-up) for every method alike
        w = a.warmup
        S, tm, gm, mv = S[w:], target_mv[w:], guard_mv[w:], moving[w:]
        r = roc(S, tm, gm, mv)
        thr = thr_at_pfa(S, tm, gm, a.pfa)
        reg = regional_pfa(S, tm, gm, thr, regions)
        det = detection_over_time(S, target_all[w:], thr)
        res[name] = {
            "pd_at_1e-2": pd_at_pfa(r, 1e-2), "pd_at_1e-3": pd_at_pfa(r, 1e-3),
            "pd_at_1e-4": pd_at_pfa(r, 1e-4), "regional_pfa": reg,
            "pfa_excess_db": pfa_excess_db(reg, a.pfa), "ms_per_frame": dt,
            "pd_moving_frames": float(det[mv].mean()) if mv.any() else float("nan"),
            "pd_static_frames": float(det[present[w:] & ~mv].mean()) if (present[w:] & ~mv).any() else float("nan"),
            "_roc": r, "_det": det,
        }
        stats[name] = S
        print(f"  {name:28s} Pd@1e-3={res[name]['pd_at_1e-3']:.3f}  Pd@1e-4={res[name]['pd_at_1e-4']:.3f}"
              f"  worst-region Pfa excess={res[name]['pfa_excess_db']:+5.1f} dB  {dt:6.2f} ms/frame")

    _plots(res, stats, P_raw[a.warmup:], P_mr[a.warmup:], present[a.warmup:], moving[a.warmup:],
           target_all[a.warmup:], cfg, a, out)
    clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in res.items()}
    clean["_params"] = vars(a) | {"tag": tag, "config": cfg.to_dict()}
    (out / "results.json").write_text(json.dumps(clean, indent=2, default=str), encoding="utf-8")
    _table(clean, out / "results.md", a)
    return clean


def _plots(res, stats, P_raw, P_mr, present, moving, target_all, cfg, a, out):
    names = list(res)
    cols = plt.cm.tab10(np.arange(len(names)))
    # ROC
    fig, ax = plt.subplots(figsize=(7, 5))
    for c, n in zip(cols, names):
        r = res[n]["_roc"]
        ax.semilogx(np.maximum(r["pfa"], 1e-7), r["pd"], label=n, color=c,
                    lw=2.2 if "PCEN" in n else 1.4, ls="-" if "PCEN" in n else "--")
    ax.axvline(a.pfa, color="0.6", lw=0.8)
    ax.set(xlabel="Pfa (background cells)", ylabel="Pd (moving-target frames)",
           title="ROC", xlim=(1e-6, 1), ylim=(0, 1.02))
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "roc.png", dpi=130); plt.close(fig)

    # Regional Pfa (CFAR property)
    regs = list(res[names[0]]["regional_pfa"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    wdt = 0.8 / len(names)
    for i, (c, n) in enumerate(zip(cols, names)):
        v = [max(res[n]["regional_pfa"][g], 1e-7) for g in regs]
        ax.bar(np.arange(len(regs)) + i * wdt, v, wdt, color=c, label=n)
    ax.set_yscale("log"); ax.axhline(a.pfa, color="k", lw=1, ls=":")
    ax.set_xticks(np.arange(len(regs)) + 0.4 - wdt / 2); ax.set_xticklabels(regs)
    ax.set(ylabel=f"Pfa at global Pfa={a.pfa:g} threshold", title="False-alarm rate per region (flat = CFAR)")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(out / "regional_pfa.png", dpi=130); plt.close(fig)

    # Dwell / detection over time
    t = np.arange(len(present)) / cfg.frame_rate_Hz
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (c, n) in enumerate(zip(cols, names)):
        d = res[n]["_det"].astype(float)
        ax.plot(t, d * 0.8 + i, color=c, drawstyle="steps-mid")
    ax.fill_between(t, -0.5, len(names), where=present & ~moving, color="orange", alpha=0.15,
                    label="target present, not moving")
    ax.fill_between(t, -0.5, len(names), where=moving, color="green", alpha=0.08, label="target moving")
    ax.set_yticks(np.arange(len(names)) + 0.4); ax.set_yticklabels(names, fontsize=8)
    ax.set(xlabel="time [s]", title=f"Target detected per frame (thr at Pfa={a.pfa:g})")
    ax.legend(fontsize=8, loc="upper right"); fig.tight_layout(); fig.savefig(out / "dwell.png", dpi=130); plt.close(fig)

    # Example maps
    idx = np.where(moving)[0]
    f = int(idx[len(idx) // 2]) if idx.size else len(P_raw) // 2
    fig, axs = plt.subplots(2, 4, figsize=(16, 7))
    panels = [("Raw RD [dB]", to_db(P_raw[f])), ("Static removed [dB]", to_db(P_mr[f]))]
    panels += [(n, stats[n][f] if "PCEN" in n else 10 * np.log10(np.maximum(stats[n][f], 1e-6)))
               for n in names if n not in ("Raw (global thr)", "Static removal (global)")]
    vx = cfg.velocity_axis(); rx = cfg.range_axis(P_raw.shape[1])
    for axx, (ttl, X) in zip(axs.flat, panels):
        axx.imshow(X, aspect="auto", origin="lower", extent=[vx[0], vx[-1], rx[0], rx[-1]],
                   cmap="viridis")
        rows = np.where(target_all[f].any(1))[0]
        if rows.size:
            axx.axhspan(rx[rows[0]], rx[rows[-1]], color="r", alpha=0.12)
        axx.set_title(ttl, fontsize=9); axx.set_xlabel("v [m/s]"); axx.set_ylabel("R [m]")
    for axx in axs.flat[len(panels):]:
        axx.axis("off")
    fig.suptitle(f"Frame {f + a.warmup} (red band = labelled target gate)")
    fig.tight_layout(); fig.savefig(out / "example_maps.png", dpi=110); plt.close(fig)


def _table(clean, path, a):
    rows = ["| Method | Pd@1e-2 | Pd@1e-3 | Pd@1e-4 | Worst-region Pfa excess [dB] | Pd moving | Pd stopped | ms/frame |",
            "|---|---|---|---|---|---|---|---|"]
    for n, v in clean.items():
        if n.startswith("_"):
            continue
        rows.append(f"| {n} | {v['pd_at_1e-2']:.3f} | {v['pd_at_1e-3']:.3f} | {v['pd_at_1e-4']:.3f} | "
                    f"{v['pfa_excess_db']:+.1f} | {v['pd_moving_frames']:.3f} | {v['pd_static_frames']:.3f} | "
                    f"{v['ms_per_frame']:.2f} |")
    rows.append("")
    rows.append(f"Worst-region Pfa excess = 10·log10(max regional Pfa / {a.pfa:g}) at the threshold giving global Pfa={a.pfa:g}. "
                f"'Pd moving/stopped' use that same threshold.")
    path.write_text("\n".join(rows), encoding="utf-8")
    print("\n" + "\n".join(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="*")
    ap.add_argument("--sim", action="store_true", help="use a synthetic scene")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/detection")
    ap.add_argument("--pfa", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=20, help="frames ignored at the start")
    # CFAR
    ap.add_argument("--guard-r", type=int, default=2); ap.add_argument("--guard-d", type=int, default=2)
    ap.add_argument("--train-r", type=int, default=8); ap.add_argument("--train-d", type=int, default=4)
    ap.add_argument("--os-k", type=float, default=0.75)
    # PCEN
    ap.add_argument("--tau", type=float, default=1.0, help="PCEN / clutter-map time constant [s]")
    ap.add_argument("--alpha", type=float, default=0.98)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--r", type=float, default=0.5)
    ap.add_argument("--tol-m", type=float, default=0.15)
    ap.add_argument("--guard-m", type=float, default=0.30)
    a = ap.parse_args()

    if a.sim:
        from pcen_radar.simulate import simulate, Scene
        cfg = RadarConfig()
        print("Simulating:", cfg.summary())
        frames, r_true, v_true = simulate(cfg, Scene(seed=a.seed))
        recs = [("sim", frames, cfg, dict(r_true=r_true, v_true=v_true))]
    else:
        if not a.recordings:
            ap.error("give recordings or --sim")
        recs = []
        for p in a.recordings:
            d = load_recording(p)
            recs.append((Path(p).stem, d["frames"], d["config"], d["labels"]))

    a.frame_period = recs[0][2].frame_repetition_time_s
    all_res = {}
    for tag, frames, cfg, lab in recs:
        print(f"\n== {tag}: {frames.shape}  ({cfg.summary()})")
        P_raw = rd_sequence(frames, clutter="none")
        P_mr = rd_sequence(frames, clutter="mean")
        shape = P_raw.shape
        if "r_true" in lab:
            tm, gm, present = per_frame_range_masks(lab["r_true"], cfg, shape, a.tol_m, a.guard_m, 1)
            ta, ga, _ = per_frame_range_masks(lab["r_true"], cfg, shape, a.tol_m, a.guard_m, 0)
            moving = present & (np.abs(lab["v_true"]) > 1e-3)
        else:
            if not lab.get("target_range_m"):
                print("  no target labels -> only false-alarm statistics are meaningful")
            tm, gm, present, moving = make_masks(lab, cfg, shape, a.tol_m, a.guard_m, exclude_zero_doppler=1)
            ta, ga, _, _ = make_masks(lab, cfg, shape, a.tol_m, a.guard_m, exclude_zero_doppler=0)
        out = Path(a.out) / tag if len(recs) > 1 else Path(a.out)
        all_res[tag] = run(P_raw, P_mr, tm, gm, ta, ga, moving, present, cfg, a, out, tag)
    print(f"\nResults written to {Path(a.out).resolve()}")


if __name__ == "__main__":
    main()
