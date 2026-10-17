"""Guided recording: seated person breathing at 1.0 m with a running fan behind (BGT60TR13C), plan v3.

  conda activate lecture_radar_systems
  python scripts/record_breathing_session.py --root C:/radar_data/breath --participant P03 --person-range 1.0 --fan-range 1.3
  python scripts/record_breathing_session.py --only R2_fan_on --redo ...      # repeat one block
  python scripts/record_breathing_session.py --dry-run --yes --root /tmp/breath_sim   # no radar: simulated

Scene
  radar on the table/tripod at chest height, chair with its front edge so that the chest is at --person-range
  (1.0 m). Fan behind the chair at --fan-range (about 1.3 m from the radar), shifted ~20 cm to one side so the
  body does not hide it, blades facing the radar, fixed speed, oscillation OFF.

Blocks (ABBA over the fan state): R1 fan OFF, R2 fan ON, R3 fan ON, R4 fan OFF.
One block = ONE continuous acquisition (~4.8 min), so temporal front ends see the whole sequence:
   0- 30 s  EMPTY      nobody in front of the radar (stand behind it, out of the beam)
             3 short beeps = walk in and sit down, hands in the lap
  30- 50 s  SETTLE     sit still
  50-290 s  NORMAL     breathe normally (no beeps); end beep = you can move
(--paced: metronome version with paced 10/15/20 breaths/min, free breathing and a 15-s breath hold.)
Stay still the whole time (no talking, no hand movements; arm movements look like huge breaths).
Files: <root>/<participant>/<block>_partNN.npz (60-s parts of one acquisition; signed int16 raw IF, no clipping)
plus <block>_labels.json. pcen_radar.io.load_parts() joins the parts.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcen_radar import RadarConfig

PHASES_NORMAL = [("empty", 30.0, None), ("settle", 20.0, None), ("free", 240.0, None)]
PHASES_PACED = [("empty", 30.0, None), ("settle", 20.0, None), ("paced_10", 60.0, 10.0), ("paced_15", 60.0, 15.0),
                ("paced_20", 60.0, 20.0), ("free", 30.0, None), ("hold", 15.0, None), ("free2", 15.0, None)]
PHASES = PHASES_NORMAL                      # default: normal breathing (--paced for the metronome version)
BLOCKS = [("R1_fan_off", "off"), ("R2_fan_on", "on"), ("R3_fan_on", "on"), ("R4_fan_off", "off")]
PART_S = 60.0


def beep(freq, ms):
    def _b():
        try:
            import winsound
            winsound.Beep(int(freq), int(ms))
        except Exception:
            print("\a", end="", flush=True)
    threading.Thread(target=_b, daemon=True).start()


def schedule():
    """phases in seconds and the cue events [(t_s, kind, text)] in wall-clock time (beeps must follow real time:
    the measured frame period is ~46 ms, not the nominal 50 ms)."""
    phases, ev, t = [], [], 0.0
    for name, dur, rate in PHASES:
        phases.append({"name": name, "start_s": t, "dur_s": dur, "rate_bpm": rate})
        if name == "settle":
            ev.append((t, "sit", "SIT DOWN now, hands in the lap, stay still"))
        elif name.startswith("paced"):
            ev.append((t, "info", f"PACED {int(rate)} breaths/min: high beep = IN, low beep = OUT"))
            per = 60.0 / rate
            k = 0
            while k * per < dur - 1e-6:
                ev.append((t + k * per, "in", None)); ev.append((t + k * per + per / 2, "out", None)); k += 1
        elif name in ("free", "free2"):
            ev.append((t, "long" if PHASES is PHASES_PACED else "info", "breathe NORMALLY, stay still"))
        elif name == "hold":
            ev.append((t, "long", "HOLD your breath"))
        t += dur
    ev.append((t - 0.3, "end", "END - you can move"))
    return phases, sorted(ev, key=lambda e: e[0]), t


def frames_of(phases, ts):
    """add start/end frame indices from the recorded timestamps."""
    for p in phases:
        p["start_frame"] = int(np.searchsorted(ts, p["start_s"]))
        p["end_frame"] = int(np.searchsorted(ts, p["start_s"] + p["dur_s"]))
    return phases


def cue(kind, text):
    if kind == "sit":
        for i in range(3):
            threading.Timer(0.25 * i, beep, (1000, 120)).start()
    elif kind == "in":
        beep(880, 150)
    elif kind == "out":
        beep(440, 150)
    elif kind in ("long", "end"):
        beep(700, 600)
    if text:
        print(f"\n  >>> {text}", flush=True)


def simulated_frames(cfg, n, phases, fan_on, person_r, fan_r, seed):
    from pcen_radar.simulate import simulate, Scene, _beat
    fp = cfg.frame_repetition_time_s
    sit = phases[1]["start_s"]

    def person(f, tf, tc):
        t = tf + tc
        if t[0] < sit:
            return 0.0
        rate = 0.25
        for p in phases:
            if p["start_s"] <= t[0] < p["start_s"] + p["dur_s"]:
                rate = (p["rate_bpm"] or 14) / 60.0 if p["name"] != "hold" else 0.0
        R = person_r + 0.004 * np.sin(2 * np.pi * rate * t)
        return _beat(cfg, R, 0.8, 0.0, tc)

    sc = Scene(n_frames=n, static=[(0.15, 25.0), (2.4, 15.0), (3.6, 10.0)], fan_range_m=fan_r, fan_amp=1.2,
               fan_frames=(0, n) if fan_on else (0, 0), target_path="none", limbs=False, seed=seed)
    frames, _, _ = simulate(cfg, sc, extra_fn=person)
    return frames, np.arange(n) * fp


def record_block(cfg, total_s, ev):
    from pcen_radar.acquire import open_device, next_frame, close_device
    nmax = int(total_s / 0.035) + 50
    q = np.empty((nmax, cfg.num_rx, cfg.num_chirps, cfg.num_samples), np.int16)   # int16 in RAM
    ts = np.empty(nmax); stats = {"dropped": 0}; drop_at = []; zeros = np.zeros(cfg.num_rx); lo = np.inf; hi = -np.inf
    dev = open_device(cfg); k_ev = 0; i = 0
    try:
        for _ in range(10):
            next_frame(dev, {})
        t0 = time.perf_counter()
        while True:
            el = time.perf_counter() - t0
            if el >= total_s or i >= nmax:
                break
            while k_ev < len(ev) and ev[k_ev][0] <= el:      # cues in real time
                cue(ev[k_ev][1], ev[k_ev][2]); k_ev += 1
            d0 = stats["dropped"]
            f = next_frame(dev, stats)
            ts[i] = time.perf_counter() - t0
            if stats["dropped"] > d0:
                drop_at.append(i)
            x = np.round(f * 4095.0)
            lo, hi = min(lo, x.min()), max(hi, x.max())
            q[i] = np.clip(x, -32768, 32767).astype(np.int16)
            zeros += (f == 0).reshape(cfg.num_rx, -1).mean(axis=1)
            if i % 20 == 0:
                print(f"\r  t = {ts[i]:6.1f} s", end="", flush=True)
            i += 1
    finally:
        close_device(dev)
    print()
    if lo < -32768 or hi > 32767:
        print(f"  WARNING: ADC values out of int16 range ({lo} .. {hi}) were clipped")
    return q[:i], ts[:i], drop_at, zeros / max(i, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/breath")
    ap.add_argument("--participant", default="P03")
    ap.add_argument("--person-range", dest="person_range", type=float, default=1.0)
    ap.add_argument("--fan-range", dest="fan_range", type=float, default=1.3)
    ap.add_argument("--fan-offset", dest="fan_offset", type=float, default=0.2, help="lateral fan offset [m]")
    ap.add_argument("--only", nargs="+"); ap.add_argument("--redo", action="store_true")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--yes", action="store_true")
    ap.add_argument("--notes", default="")
    ap.add_argument("--paced", action="store_true", help="metronome version (paced 10/15/20 + breath hold)")
    a = ap.parse_args()
    global PHASES
    PHASES = PHASES_PACED if a.paced else PHASES_NORMAL
    cfg = RadarConfig(); fp = cfg.frame_repetition_time_s
    out = Path(a.root) / a.participant; out.mkdir(parents=True, exist_ok=True)
    phases0, ev, total = schedule()
    print(cfg.summary()); print(f"one block: {total:.0f} s -> {out.resolve()}")
    for bi, (name, fan) in enumerate(BLOCKS):
        if a.only and name not in a.only:
            continue
        if (out / f"{name}_labels.json").exists() and not a.redo:
            print(f"[skip] {name} exists"); continue
        print("\n" + "=" * 72)
        print(f"BLOCK {bi + 1}/4: {name}   fan {fan.upper()}   ({total / 60:.1f} min)")
        phases = [dict(p) for p in phases0]
        print(f"  person at {a.person_range:.2f} m, fan at {a.fan_range:.2f} m behind the chair "
              f"({a.fan_offset:.2f} m to the side), fan {'ON fixed speed, oscillation OFF' if fan == 'on' else 'OFF'}")
        print("  Start: stand BEHIND the radar (out of the beam). 3 beeps = sit down, then breathe normally and stay "
              "still until the end beep." if not a.paced else
              "  Start: stand BEHIND the radar (out of the beam). 3 beeps = sit down. Then follow the beeps.")
        if not a.yes:
            k = input("  Enter = start,  s = skip,  q = quit > ").strip().lower()
            if k == "q":
                break
            if k == "s":
                continue
        if a.dry_run:
            n = int(round(total / fp))
            fr, ts = simulated_frames(cfg, n, phases, fan == "on", a.person_range, a.fan_range, bi)
            q = np.round(fr * 4095.0).astype(np.int16); drop_at = []; zeros = (fr == 0).reshape(n, cfg.num_rx, -1).mean(axis=(0, 2))
        else:
            for i in range(3, 0, -1):
                print(f"  starting in {i} ...", flush=True); time.sleep(1)
            q, ts, drop_at, zeros = record_block(cfg, total, ev)
        n = len(q); phases = frames_of(phases, ts)
        dts = np.diff(ts); fpm = float(np.median(dts)) if dts.size else fp
        stds = [float(q[::20, i].std()) for i in range(cfg.num_rx)]
        if max(zeros) > 0.05 or min(stds) == 0:
            print(f"  WARNING: ADC check: zero share per Rx {np.round(zeros, 3).tolist()}, std per Rx {stds}")
        labels = {"block": name, "fan": fan, "participant": a.participant, "person_range_m": a.person_range,
                  "fan_range_m": a.fan_range, "fan_offset_m": a.fan_offset, "phases": phases, "plan": "v3_breathing_paced" if a.paced else "v3_breathing_normal",
                  "frame_period_s": fp, "dry_run": a.dry_run}
        meta = {"sensor": "BGT60TR13C", "recorded": time.strftime("%Y%m%d_%H%M%S"), "notes": a.notes,
                "measured_frame_period_s": fpm, "late_frames": int((dts > 1.5 * fp).sum()), "dropped_before_frames": drop_at,
                "n_frames": n}
        per = int(round(PART_S / fp)); parts = []
        for k, s0 in enumerate(range(0, n, per)):
            f = out / f"{name}_part{k:02d}.npz"
            np.savez_compressed(f, frames_i16=q[s0:s0 + per], adc_scale=np.float32(4095.0), t=ts[s0:s0 + per],
                                config=cfg.to_json(), labels=json.dumps(dict(labels, part=k, part_start_frame=s0)),
                                meta=json.dumps(meta))
            parts.append(f.name)
        (out / f"{name}_labels.json").write_text(json.dumps(dict(labels, meta=meta, parts=parts), indent=1), encoding="utf-8")
        print(f"  saved {len(parts)} parts ({sum((out / p).stat().st_size for p in parts) / 1e6:.0f} MB)  "
              f"frame period {fpm * 1e3:.1f} ms  late frames {meta['late_frames']}  dropped {len(drop_at)}")
    print("\nDone. Tell Claude the files are in", out)


if __name__ == "__main__":
    main()
