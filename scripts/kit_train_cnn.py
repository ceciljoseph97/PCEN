"""Learnable PCEN front end + CNN on the KIT FC-FMCW gesture data (needs PyTorch).

  conda activate lecture_radar_systems
  python scripts/kit_train_cnn.py                    # default: data in results/kit/maps, 3 seeds
  python scripts/kit_train_cnn.py --seeds 5 --epochs 40

PRE-REGISTERED (fixed before running)
-------------------------------------
Input: per recording, micro-Doppler (64 pooled Doppler channels) and range-time (64 range bins)
maps resampled to 256 time steps, stacked into a (128 channels x 256 time) power map.
  'raw' = no static removal, 'sr' = static removal (mean over the chirps of each window).
Front ends (all followed by the same SmallCNN from pcen_radar/nn.py):
  log_raw         log10(X / median)                              on raw maps
  log_sr          log10(X / median)                              on sr maps
  pcen_fixed_sr   PCEN, tau = 1 s, alpha .98, delta 2, r .5      on sr maps (not trained)
  pcen_learn_sr   PCEN, per-channel s, alpha, delta, r learned   on sr maps (init = fixed values)
  pcen_learn_raw  same learnable PCEN                            on raw maps
Training: Adam, lr 1e-3 (front-end parameters 1e-2), weight decay 1e-4, batch 32, 40 epochs,
no augmentation, no early stopping (no test-set peeking). Seeds 0..N-1.
Splits:
  E1 within-setup : part 2 setup 1, stratified 80/20 split (seed 0)
  E2 cross-setup  : MAIN. train part 2 setup 1 (240) -> test setups 4/5/6/7 (231)
  E3 cross-angle  : part 1, train 3v1 (0 deg) -> test 3v2 (40 deg)
Reported: test accuracy mean +- std over seeds; paired per-seed differences vs. log_sr and log_raw.

CONFIRMATION TEST (pre-registered 2026-10-15, after the E2 runs, before running this):
  The learnable PCEN trained on E2 (part 2) moved tau from 1 s to ~0.13 s. Hypothesis: a FIXED PCEN
  with tau = 0.13 s (alpha .98, delta 2, r .5) beats the audio default tau = 1 s on data the value
  was not learned from: KIT part 1 (other participants, other room), cross-angle in both directions
  (E3_cross_angle: 3v1 -> 3v2, E3_cross_angle_rev: 3v2 -> 3v1), 10 seeds, paired Wilcoxon.
  Front end name: pcen_fixed013_sr.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FRONTENDS = ["log_raw", "log_sr", "pcen_fixed_sr", "pcen_learn_sr", "pcen_learn_raw"]


def load(maps_dir: Path):
    meta = np.load(maps_dir / "meta.npz")
    X = {}
    for tag in ["raw", "sr"]:
        md = np.load(maps_dir / f"md_{tag}.npz")["log10_power"].astype(np.float32)   # (N, 256, 64)
        rt = np.load(maps_dir / f"rt_{tag}.npz")["log10_power"].astype(np.float32)   # (N, 256, 64)
        P = np.power(10.0, np.concatenate([md, rt], axis=2))                         # (N, 256, 128)
        X[tag] = np.ascontiguousarray(P.transpose(0, 2, 1))                          # (N, 128, 256)
    return X, meta["gesture"], meta["setup"], meta["part"]


def splits(g, su, part):
    s1 = np.where((part == "part2") & (su == "1"))[0]
    rng = np.random.default_rng(0)
    tr1, te1 = [], []
    for c in np.unique(g[s1]):                      # stratified 80/20
        idx = rng.permutation(s1[g[s1] == c])
        k = int(round(0.8 * len(idx)))
        tr1 += list(idx[:k]); te1 += list(idx[k:])
    return {
        "E1_within": (np.array(tr1), np.array(te1)),
        "E2_cross_setup": (s1, np.where((part == "part2") & (su != "1"))[0]),
        "E3_cross_angle": (np.where((part == "part1") & (su == "3v1"))[0],
                           np.where((part == "part1") & (su == "3v2"))[0]),
        "E3_cross_angle_rev": (np.where((part == "part1") & (su == "3v2"))[0],
                               np.where((part == "part1") & (su == "3v1"))[0]),
    }


def run_one(fe, X, y, tr, te, n_classes, seed, a, dev):
    import torch
    import torch.nn as nn
    from pcen_radar.nn import SmallCNN, LogFrontEnd, FixedPCEN, LearnablePCEN

    torch.manual_seed(seed); np.random.seed(seed)
    tag = "raw" if fe.endswith("raw") else "sr"
    data = torch.tensor(X[tag])
    s = 1.0 - math.exp(-a.dt / a.tau)
    C = data.shape[1]
    if fe.startswith("log"):
        front = LogFrontEnd()
    elif fe == "pcen_fixed_sr":
        front = FixedPCEN(C, s=s, alpha=0.98, delta=2.0, r=0.5)
    elif fe == "pcen_fixed013_sr":
        front = FixedPCEN(C, s=1.0 - math.exp(-a.dt / a.tau_fast), alpha=0.98, delta=2.0, r=0.5)
    else:
        front = LearnablePCEN(C, s=s, alpha=0.98, delta=2.0, r=0.5)
    model = SmallCNN(n_classes, front).to(dev)
    fe_params = [p for p in model.frontend.parameters() if p.requires_grad]
    body = [p for n_, p in model.named_parameters() if not n_.startswith("frontend.")]
    groups = [{"params": body, "lr": a.lr}]
    if fe_params:
        groups.append({"params": fe_params, "lr": a.lr * 10})
    opt = torch.optim.Adam(groups, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    ytr = torch.as_tensor(y[tr], dtype=torch.long); Xtr = data[tr]
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(tr))
        for i in range(0, len(perm), a.batch):
            b = perm[i:i + a.batch]
            loss = lossf(model(Xtr[b].to(dev)), ytr[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(data[te][i:i + 64].to(dev)).argmax(1).cpu() for i in range(0, len(te), 64)])
    acc = float((pred.numpy() == y[te]).mean())
    learned = None
    if fe.startswith("pcen_learn"):
        learned = {k: getattr(model.frontend, "log_" + k).exp().detach().cpu().numpy().tolist()
                   for k in ["s", "alpha", "delta", "r"]}
    return acc, learned


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", default=str(Path(__file__).resolve().parents[1] / "results" / "kit" / "maps"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "kit_cnn"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--tau-fast", dest="tau_fast", type=float, default=0.13,
                    help="time constant of the fixed fast PCEN (pcen_fixed013_sr)")
    ap.add_argument("--dt", type=float, default=0.0235, help="time step after resampling (~6 s / 256)")
    ap.add_argument("--splits", nargs="+", default=["E2_cross_setup", "E1_within", "E3_cross_angle"])
    ap.add_argument("--frontends", nargs="+", default=FRONTENDS)
    a = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("torch", torch.__version__, "device", dev)
    X, g, su, part = load(Path(a.maps))
    classes = sorted(set(g))
    y = np.array([classes.index(c) for c in g], dtype=np.int64)
    S = splits(g, su, part)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {}
    t0 = time.time()
    for sp in a.splits:
        tr, te = S[sp]
        ncls = len(classes)
        print(f"\n== {sp}: train {len(tr)}, test {len(te)}, classes in test {sorted(set(g[te]))}")
        res[sp] = {}
        for fe in a.frontends:
            accs, learned = [], []
            for seed in range(a.seeds):
                acc, lp = run_one(fe, X, y, tr, te, ncls, seed, a, dev)
                accs.append(acc); learned.append(lp)
                print(f"  {fe:15s} seed {seed}: acc {acc:.3f}   ({time.time() - t0:6.0f} s)", flush=True)
            res[sp][fe] = {"accs": accs, "mean": float(np.mean(accs)), "std": float(np.std(accs))}
            if learned[0] is not None:
                res[sp][fe]["learned_params_per_seed"] = learned      # per-channel values
                res[sp][fe]["learned_params_mean"] = {k: float(np.mean([np.mean(l[k]) for l in learned]))
                                                      for k in ["s", "alpha", "delta", "r"]}
        for ref in ["log_sr", "log_raw"]:
            if ref not in res[sp]:
                continue
            for fe in res[sp]:
                if fe == ref:
                    continue
                d = np.array(res[sp][fe]["accs"]) - np.array(res[sp][ref]["accs"])
                res[sp][fe][f"diff_vs_{ref}"] = {"mean": float(d.mean()), "per_seed": d.tolist()}
        json.dump(res, open(out / "results.json", "w"), indent=1)

    lines = ["| Front end | " + " | ".join(a.splits) + " |", "|---|" + "---|" * len(a.splits)]
    for fe in a.frontends:
        lines.append(f"| {fe} | " + " | ".join(f"{res[sp][fe]['mean']:.3f} ± {res[sp][fe]['std']:.3f}"
                                               for sp in a.splits) + " |")
    lines.append(f"\n{a.seeds} seeds, {a.epochs} epochs. Majority-class baseline E2 = "
                 f"{max(np.mean(g[S['E2_cross_setup'][1]] == c) for c in classes):.3f}.")
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print("saved", out)


if __name__ == "__main__":
    main()
