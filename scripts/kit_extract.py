"""Extract micro-Doppler / range-time maps from KIT .mat recordings (one .npz per recording).

  python scripts/kit_extract.py <dataset folder containing part1/ part2/> <out folder> [--selection results/kit/selection.json]

With --selection only the 681 recordings used in results/kit are processed (otherwise all).
Then: python scripts/kit_experiments.py <out folder> results/kit
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcen_radar.kit import load_mat, extract, parse_name, KNOWN_DUPLICATES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--selection")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.dataset).glob("part*/*.mat"))
    if a.selection:
        sel = json.loads(Path(a.selection).read_text())
        keep = {n for k in sel for n in sel[k]}
        files = [f for f in files if f.name in keep]
    done = 0
    for f in files:
        meta = parse_name(f.name)
        if f.stat().st_size < 1e5 or meta["id"] in KNOWN_DUPLICATES:
            continue                          # empty files / legend duplicates
        o = out / (f.stem + ".npz")
        if o.exists():
            continue
        x, m = load_mat(f)
        np.savez_compressed(o, **extract(x, m), part=f.parent.name, **meta)
        done += 1
        if done % 50 == 0:
            print(done, "done")
    print(f"extracted {done} recordings -> {out}")


if __name__ == "__main__":
    main()
