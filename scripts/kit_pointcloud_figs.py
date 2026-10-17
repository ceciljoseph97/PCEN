"""3-D gesture trails on KIT (results/pointcloud_kit/points.npz).

Columns: front ends. Row 1: range-time trail of one recording (setup 4v1, window 1.0 m behind the hand).
Row 2: top view (lateral x vs forward z) of all points of that recording, colour = time.
Row 3: front view (lateral x vs height y) of the hand-gate points (0.15-0.60 m), colour = time.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/pointcloud_kit")
rec = sys.argv[2] if len(sys.argv) > 2 else None
z = np.load(root / "points.npz")
res = json.loads((root / "results.json").read_text())
fp = res["_params"]["frame_period_s"]
names = sorted({k.split("__")[0] for k in z.files})
rec = rec or next(n for n in names if "_4v1_" in n and n.startswith("GM_2_"))
M = [("published", "published"), ("ca_cfar", "CA-CFAR"), ("global", "global"), ("clutter_map", "clutter map"),
     ("pcen", "PCEN"), ("cpcen", "C-PCEN"), ("pcen_ca", "PCEN+CA")]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8})
fig, ax = plt.subplots(3, len(M), figsize=(7.2, 4.9), gridspec_kw={"hspace": 0.6, "wspace": 0.12})


def xyz(p):
    r, az, el = p[:, 1], p[:, 2], p[:, 3]
    return r * np.cos(el) * np.sin(az), -r * np.sin(el), r * np.cos(el) * np.cos(az)


T = max(int(z[f"{rec}__{m}"][:, 0].max()) if len(z[f"{rec}__{m}"]) else 0 for m, _ in M) + 1
for j, (m, name) in enumerate(M):
    p = z[f"{rec}__{m}"]; p = p[p[:, 0] >= 10]
    t = p[:, 0] * fp
    a = ax[0, j]
    a.scatter(t, p[:, 1], s=1, c="#2a78d6", lw=0, rasterized=True)
    a.axhspan(0.15, 0.60, color="#1baf7a", alpha=0.08, lw=0)
    a.set_ylim(0.1, 1.37); a.set_xlim(0, T * fp); a.set_title(name)
    a.set_xlabel("time [s]")
    x, y, zz = xyz(p)
    b = ax[1, j]
    b.scatter(x, zz, s=1, c=t, cmap="viridis", vmin=0, vmax=T * fp, lw=0, rasterized=True)
    b.set_xlim(-0.8, 0.8); b.set_ylim(0, 1.37); b.set_aspect("equal"); b.set_xticks([-0.5, 0.5])
    b.set_xlabel("lateral [m]")
    h = (p[:, 1] >= 0.15) & (p[:, 1] <= 0.60)
    c = ax[2, j]
    c.scatter(x[h], y[h], s=1.5, c=t[h], cmap="viridis", vmin=0, vmax=T * fp, lw=0, rasterized=True)
    c.set_xlim(-0.4, 0.4); c.set_ylim(-0.4, 0.4); c.set_aspect("equal"); c.set_xticks([-0.2, 0.2])
    c.set_xlabel("lateral [m]")
    for row, lab in ((a, "range [m]"), (b, "forward [m]"), (c, "height [m]")):
        if j == 0:
            row.set_ylabel(lab)
        else:
            row.set_yticklabels([])
setup = rec.split("_")[2]
bg = {"4v1": "window at 1.0 m", "7v1": "free space"}.get(setup, "")
fig.suptitle(f"KIT {rec} (setup {setup}, {bg}); colour = time", y=0.995, fontsize=8)
fig.savefig(root / f"trails_{rec}.png", dpi=220, bbox_inches="tight")
fig.savefig(root / f"trails_{rec}.pdf", bbox_inches="tight")
print("ok", rec)
