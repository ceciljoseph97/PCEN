"""EXPLORATORY (post hoc) re-analysis of the hold blocks of fan session 2.

The pre-stated hold windows (still 1.5-3.5 s vs hand lowered 4.0-4.9 s) put the LOWERING movement into the
negative window (mid beep at 3.5 s + reaction time), which inverts the AUC for every front end. Here:
  positive = still phase 1.5-3.5 s after GO, negative = hand down and still 5.0-5.9 s (before the next GO);
  score = mean over the window of the summed zero-Doppler (+-1 bin) statistic in the hand gate
  (a still hand appears at zero Doppler; front ends without static removal).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pcen_radar.io import load_recording
from pcen_radar.processing import rd_sequence
from analyze_fan import auc
import analyze_fan_v2 as V

data = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
files = sorted(data.glob("*.npz"))
offs = [f for f in files if "_off_empty" in f.stem]
fp = float(np.median([json.loads(str(np.load(f)["meta"]))["measured_frame_period_s"] for f in files]))
w0 = int(2 / fp)
ref = float(np.median(np.concatenate([rd_sequence(load_recording(f)["frames"], clutter="none")[w0:].ravel() for f in offs])))
res = {}
for f in [f for f in files if f.stem.endswith("_hold")]:
    d = load_recording(f); lab = d["labels"]; cfg = d["config"]
    P = rd_sequence(d["frames"], clutter="none"); R, D = P.shape[1:]
    rr = cfg.range_axis(R); hg = (rr >= lab["hand_gate_m"][0]) & (rr <= lab["hand_gate_m"][1])
    zd = slice(D // 2 - 1, D // 2 + 2)
    S = V.stats(P, "raw", fp, ref)
    pos, neg = {m: [] for m in S}, {m: [] for m in S}
    for c in lab["cue_frames"]:
        a1, b1 = c + int(1.5 / fp), c + int(3.5 / fp); a2, b2 = c + int(5.0 / fp), c + int(5.9 / fp)
        if b2 > len(P):
            continue
        for m, X in S.items():
            sc = X[:, hg][:, :, zd].sum(axis=(1, 2))
            pos[m].append(float(sc[a1:b1].mean())); neg[m].append(float(sc[a2:b2].mean()))
    res[f.stem] = {m: {"auc": auc(np.array(pos[m]), np.array(neg[m])), "n": len(pos[m])} for m in S}
    print(f.stem, {m: round(v["auc"], 3) for m, v in res[f.stem].items()}, flush=True)
(out / "hold_posthoc.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
