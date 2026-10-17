"""Dynamic-trail figures from results/pointcloud/points.npz.

Top row: range-time trail beside the running fan (push, side position, fan on; 8 repetitions), one panel per
front end; dashed lines = fan gate. Bottom row: points of all gesture windows beside the running fan
(swipe, push, circle) in the range-elevation plane (forward z, height y), grey = empty room with fan on.
Rx1 delivered no data in this session, so there is no azimuth (no top view).
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/pointcloud")
z = np.load(root / "points.npz")
res = json.loads((root / "results.json").read_text())
fp = res["_params"]["frame_period_s"]
M = [("published", "published"), ("ca_cfar", "CA-CFAR"), ("global", "global"), ("clutter_map", "clutter map"),
     ("pcen", "PCEN"), ("cpcen", "C-PCEN"), ("pcen_ca", "PCEN+CA")]
cues = list(range(100, 100 + 80 * 20, 80))
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8})
fig, ax = plt.subplots(1, len(M), figsize=(7.2, 1.9), gridspec_kw={"wspace": 0.12})
ax = ax[None, :]
t0, t1 = cues[2] - 20, cues[10]
for j, (m, name) in enumerate(M):
    p = z[f"C11_side_on_push__{m}"]
    q = p[(p[:, 0] >= t0) & (p[:, 0] < t1)]
    a = ax[0, j]
    a.scatter((q[:, 0] - t0) * fp, q[:, 1], s=1.2, c="#2a78d6", lw=0, rasterized=True)
    for c in cues:
        if t0 <= c < t1:
            a.axvline((c - t0) * fp, color="#9c9a93", lw=0.4, ls=":")
    for y in (0.85, 1.25):
        a.axhline(y, color="#eb6834", lw=0.6, ls="--")
    a.set_ylim(0.2, 2.0); a.set_xlim(0, (t1 - t0) * fp)
    a.set_title(name, fontsize=8)
    a.set_xlabel("time [s]")
    if j == 0:
        a.set_ylabel("range [m]")
    else:
        a.set_yticklabels([])
fig.suptitle("Fan session, push beside the running fan (Rx2+Rx3; dashed: fan gate)", y=1.04, fontsize=8)
fig.savefig(root / "trails.png", dpi=220, bbox_inches="tight")
fig.savefig(root / "trails.pdf", bbox_inches="tight")
print("ok")
