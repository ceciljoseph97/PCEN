"""Experiment B: PCEN as a (learnable) front end for a micro-Doppler CNN classifier.

Compares front ends on the SAME network and data split:
  log         : log10 of median-normalised power (standard)
  pcen_fixed  : PCEN with fixed default parameters
  pcen_learn  : PCEN with per-Doppler-bin learnable s, alpha, delta, r

Robustness test = cross-session split: train on some rooms/days, test on unseen ones.

  python scripts/train_classifier.py data/synth --test-sessions roomB
  python scripts/train_classifier.py data --test-sessions roomC --epochs 40 --clutter none

Requires PyTorch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcen_radar.dataset import build_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--test-sessions", nargs="+", required=True)
    ap.add_argument("--frontends", nargs="+", default=["log", "pcen_fixed", "pcen_learn"])
    ap.add_argument("--clutter", default="mean", choices=["none", "mean"],
                    help="static removal before building the micro-Doppler map")
    ap.add_argument("--win", type=int, default=64)
    ap.add_argument("--hop", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=3, help="repeat with different seeds")
    ap.add_argument("--out", default="results/classifier")
    a = ap.parse_args()

    import torch
    import torch.nn as nn
    from pcen_radar.nn import SmallCNN, LogFrontEnd, FixedPCEN, LearnablePCEN

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    X, y, S, classes = build_dataset(a.root, a.win, a.hop, a.clutter,
                                     cache=out / f"features_{a.clutter}_{a.win}_{a.hop}.npz")
    test = np.isin(S, a.test_sessions)
    print(f"{len(X)} windows, classes={classes}, train={(~test).sum()} test={test.sum()} "
          f"(test sessions {a.test_sessions})")
    Xtr, ytr, Xte, yte = map(torch.tensor, (X[~test], y[~test], X[test], y[test]))
    n_ch = X.shape[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def make(fe):
        f = {"log": lambda: LogFrontEnd(), "pcen_fixed": lambda: FixedPCEN(n_ch),
             "pcen_learn": lambda: LearnablePCEN(n_ch)}[fe]()
        return SmallCNN(len(classes), f).to(dev)

    results = {}
    for fe in a.frontends:
        accs, cms = [], []
        for seed in range(a.seeds):
            torch.manual_seed(seed); np.random.seed(seed)
            model = make(fe)
            fe_params = list(model.frontend.parameters())
            other = [p for n, p in model.named_parameters() if not n.startswith("frontend.")]
            opt = torch.optim.Adam([{"params": other, "lr": a.lr},
                                    {"params": [p for p in fe_params if p.requires_grad], "lr": a.lr * 10}])
            lossf = nn.CrossEntropyLoss()
            for ep in range(a.epochs):
                model.train()
                perm = torch.randperm(len(Xtr))
                tot = 0.0
                for i in range(0, len(perm), a.batch):
                    idx = perm[i:i + a.batch]
                    xb, yb = Xtr[idx].to(dev), ytr[idx].to(dev)
                    loss = lossf(model(xb), yb)
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot += loss.item() * len(idx)
                if (ep + 1) % 10 == 0:
                    print(f"  [{fe} seed {seed}] epoch {ep + 1}: train loss {tot / len(Xtr):.3f}")
            model.eval()
            with torch.no_grad():
                pred = torch.cat([model(Xte[i:i + 256].to(dev)).argmax(1).cpu()
                                  for i in range(0, len(Xte), 256)])
            acc = (pred == yte).float().mean().item()
            cm = np.zeros((len(classes), len(classes)), int)
            for t_, p_ in zip(yte.numpy(), pred.numpy()):
                cm[t_, p_] += 1
            accs.append(acc); cms.append(cm)
            if fe == "pcen_learn":
                fr = model.frontend
                learned = {k: getattr(fr, "log_" + k).exp().detach().cpu().numpy().round(4).tolist()
                           for k in ["s", "alpha", "delta", "r"]}
                (out / f"learned_pcen_seed{seed}.json").write_text(json.dumps(learned, indent=1), encoding="utf-8")
        results[fe] = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                       "accs": accs, "confusion": np.sum(cms, 0).tolist()}
        print(f"==> {fe:11s} test acc {np.mean(accs):.3f} ± {np.std(accs):.3f}")

    results["_params"] = vars(a) | {"classes": classes}
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = ["| Front end | Test accuracy (unseen sessions) |", "|---|---|"]
    lines += [f"| {k} | {v['acc_mean']:.3f} ± {v['acc_std']:.3f} |" for k, v in results.items()
              if not k.startswith("_")]
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
