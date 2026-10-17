"""Sweep PCEN time constant tau and gain exponent alpha; plot Pd at fixed Pfa.

  python scripts/sweep_pcen.py --sim
  python scripts/sweep_pcen.py data/roomA/walk_*.npz --pfa 1e-3

Gives the sensitivity figure reviewers will ask for ("how did you pick the parameters?").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcen_radar import RadarConfig, rd_sequence, pcen
from pcen_radar.io import load_recording, make_masks, per_frame_range_masks
from pcen_radar.metrics import roc, pd_at_pfa
from pcen_radar.pcen import s_from_tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recordings", nargs="*")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--pfa", type=float, default=1e-3)
    ap.add_argument("--clutter", default="mean", choices=["none", "mean"])
    ap.add_argument("--taus", type=float, nargs="+", default=[0.25, 0.5, 1, 2, 4, 8])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.8, 0.9, 0.95, 0.98, 1.0])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default="results/sweep")
    a = ap.parse_args()

    data = []
    if a.sim:
        from pcen_radar.simulate import simulate, Scene
        cfg = RadarConfig()
        fr, r, v = simulate(cfg, Scene())
        P = rd_sequence(fr, clutter=a.clutter)
        tm, gm, present = per_frame_range_masks(r, cfg, P.shape, exclude_zero_doppler=1)
        data.append((P, tm, gm, present & (np.abs(v) > 1e-3), cfg))
    for p in a.recordings:
        d = load_recording(p)
        P = rd_sequence(d["frames"], clutter=a.clutter)
        tm, gm, present, moving = make_masks(d["labels"], d["config"], P.shape, exclude_zero_doppler=1)
        data.append((P, tm, gm, moving, d["config"]))
    if not data:
        ap.error("give recordings or --sim")

    w = a.warmup
    grid = np.zeros((len(a.alphas), len(a.taus)))
    for i, al in enumerate(a.alphas):
        for j, tau in enumerate(a.taus):
            pds = []
            for P, tm, gm, mv, cfg in data:
                S = pcen(P, s=s_from_tau(tau, cfg.frame_repetition_time_s), alpha=al)
                pds.append(pd_at_pfa(roc(S[w:], tm[w:], gm[w:], mv[w:]), a.pfa))
            grid[i, j] = np.nanmean(pds)
            print(f"alpha={al:4.2f} tau={tau:5.2f}s  Pd@{a.pfa:g}={grid[i, j]:.3f}")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    im = ax.imshow(grid, origin="lower", cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(a.taus))); ax.set_xticklabels([f"{t:g}" for t in a.taus])
    ax.set_yticks(range(len(a.alphas))); ax.set_yticklabels([f"{x:g}" for x in a.alphas])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                    color="w" if grid[i, j] < 0.6 else "k", fontsize=8)
    ax.set(xlabel="tau [s]", ylabel="alpha", title=f"PCEN Pd at Pfa={a.pfa:g} (clutter={a.clutter})")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(out / "sweep_tau_alpha.png", dpi=130)
    np.savetxt(out / "sweep_tau_alpha.csv", grid, delimiter=",",
               header="rows=alpha " + str(a.alphas) + " cols=tau " + str(a.taus))
    print("saved", out / "sweep_tau_alpha.png")


if __name__ == "__main__":
    main()
