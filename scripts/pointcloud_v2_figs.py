"""Dynamic trails, fan session 2 (results/pointcloud_v2/points.npz). Block: push-pull, hand 0.8 m directly in
front of the running fan (C10_near_on_pushpull). Row 1: range-time (8 repetitions; dashed: fan gate, shaded:
hand gate). Row 2: top view of the points in gesture windows, grey = empty room with fan on. Row 3: front view
(lateral vs height) of the hand-gate points in gesture windows, colour = time after GO."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/pointcloud_v2")
blk = sys.argv[2] if len(sys.argv) > 2 else "C10_near_on_pushpull"
emp = {"near_on": "C9_near_on_empty", "front_on": "B5_front_on_empty", "near_off": "D13_near_off_empty",
       "front_off": "A1_front_off_empty"}["_".join(blk.split("_")[1:3])]
z = np.load(root / "points.npz"); res = json.loads((root / "results.json").read_text())
fp = res["_params"]["frame_period_s"]
lab = json.loads(str(np.load(next(Path("/mnt/user-data/uploads/fan/P02").glob(blk + ".npz")))["labels"]))
cues = lab["cue_frames"]; hg = lab["hand_gate_m"]; fg = lab["fan_gate_m"]
M = [("published", "published"), ("ca_cfar", "CA-CFAR"), ("global", "global"), ("clutter_map", "clutter map"),
     ("pcen", "PCEN"), ("cpcen", "C-PCEN"), ("pcen_ca", "PCEN+CA")]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8})
fig, ax = plt.subplots(3, len(M), figsize=(7.2, 5.0), gridspec_kw={"hspace": 0.6, "wspace": 0.12})


def xyz(p):
    r, az, el = p[:, 1], p[:, 2], p[:, 3]
    return r * np.cos(el) * np.sin(az), -r * np.sin(el), r * np.cos(el) * np.cos(az)


t0, t1 = cues[4] - 10, cues[12]
gwin = lambda p: np.any([(p[:, 0] >= c + int(0.2 / fp)) & (p[:, 0] < c + int(1.7 / fp)) for c in cues], axis=0)
for j, (m, name) in enumerate(M):
    p = z[f"{blk}__{m}"]
    a = ax[0, j]; q = p[(p[:, 0] >= t0) & (p[:, 0] < t1)]
    a.scatter((q[:, 0] - t0) * fp, q[:, 1], s=0.8, c="#2a78d6", lw=0, rasterized=True)
    a.axhspan(*hg, color="#1baf7a", alpha=0.10, lw=0)
    for y in fg:
        a.axhline(y, color="#eb6834", lw=0.6, ls="--")
    a.set_ylim(0.2, 1.6); a.set_xlim(0, (t1 - t0) * fp); a.set_title(name); a.set_xlabel("time [s]")
    e = z[f"{emp}__{m}"]; ex, ey, ez = xyz(e)
    g = p[gwin(p)] if len(p) else p; x, y, zz = xyz(g)
    b = ax[1, j]
    b.scatter(ex, ez, s=0.6, c="#b8b6ae", lw=0, rasterized=True)
    b.scatter(x, zz, s=0.8, c="#2a78d6", lw=0, rasterized=True)
    b.set_xlim(-0.8, 0.8); b.set_ylim(0.2, 1.6); b.set_aspect("equal"); b.set_xticks([-0.5, 0.5]); b.set_xlabel("lateral [m]")
    h = (g[:, 1] >= hg[0]) & (g[:, 1] <= hg[1]) if len(g) else np.zeros(0, bool)
    tc = np.array([((t - np.array(cues))[(t - np.array(cues)) >= 0].min()) * fp for t in g[h, 0]]) if h.any() else []
    c = ax[2, j]
    c.scatter(x[h], y[h], s=1.0, c=tc, cmap="viridis", vmin=0.2, vmax=1.7, lw=0, rasterized=True)
    c.set_xlim(-0.5, 0.5); c.set_ylim(-0.5, 0.5); c.set_aspect("equal"); c.set_xticks([-0.3, 0.3]); c.set_xlabel("lateral [m]")
    for row, labl in ((a, "range [m]"), (b, "forward [m]"), (c, "height [m]")):
        if j == 0:
            row.set_ylabel(labl)
        else:
            row.set_yticklabels([])
fig.suptitle(f"Fan session 2, {blk} (all 3 Rx). Grey: empty room, fan on. Colour: time after GO", y=0.995, fontsize=8)
fig.savefig(root / f"trails_{blk}.png", dpi=220, bbox_inches="tight"); fig.savefig(root / f"trails_{blk}.pdf", bbox_inches="tight")
print("ok")
